import torch
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from time import time
import torch.nn.functional as F

from dataset import VideoDataset
from models import LanguageBind_model, CLIP_CLAP_model
from utils import set_random_seed, load_data
from eval_metrics import calculate_metrices_LLP, print_metrices
from export_segments import export_segments_to_txt



def merge_consecutive_segments(events):
    # events: list of [start, end]
    if not events:
        return []
    events.sort()
    merged = [events[0]]
    for current in events[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1: # Merge if consecutive (end+1 == start) or overlapping
             last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    return merged

def get_binary_events(similarities, thresholds_val):
    # similarities: (10, 25)
    # Return: list of (label_idx, start, end)
    
    # 1. Binary mask
    # print(similarities.shape)
    bin_mask = (similarities > thresholds_val).int() # (10, 25)
    
    events = []
    num_classes = bin_mask.shape[1]
    
    for cls_idx in range(num_classes):
        # Extract segments for this class
        seq = bin_mask[:, cls_idx].cpu().numpy()
        # Find continuous runs of 1s
        # Pad with 0 to detect edges
        diff = np.diff(np.concatenate(([0], seq, [0])))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        
        # In original code, segments are 1-based or 0-based? 
        # "frame_number - 1" logic in original suggests 1-based frames to 0-based index?
        # Dataset returns 10s. We split into 10 chunks. Index 0..9.
        # Let's say start=0, end=1 means 1st second.
        
        for s, e in zip(starts, ends):
            events.append({"class_idx": cls_idx, "start": int(s), "end": int(e)})
            
    return events

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', required=True, type=str)
    parser.add_argument('--audio_dir', required=True, type=str)
    parser.add_argument('--backbone', default='language_bind', type=str, choices=['language_bind', 'clip_clap'])
    parser.add_argument('--dataset', default='LLP', type=str)
    parser.add_argument('--threshold', default=0.75, type=float)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--use_tci', action='store_true', default=False,
                        help='Enable Temporal Context Injection (HAN g_sa)')
    parser.add_argument('--use_cross_modal', action='store_true', default=False,
                        help='Enable audio-visual cross-modal guidance')
    parser.add_argument('--use_filtering', action='store_true', default=False,
                        help='Enable video-level label filtering before per-segment detection')
    parser.add_argument('--filter_threshold', default=0.5, type=float,
                        help='Threshold for video-level filtering (norm_similarities scale, 0-1)')
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    
    # Load Data
    subset, labels = load_data(args.dataset)
    dataset = VideoDataset(args.video_dir, args.audio_dir, args.backbone, subset=subset)
    
    # Load Model
    if args.backbone == 'language_bind':
        model = LanguageBind_model(device)
    else:
        model = CLIP_CLAP_model(device, use_tci=args.use_tci, use_cross_modal=args.use_cross_modal)

    # Main Loop
    predictions = {
        "combined": {}, "video": {}, "audio": {}
    }
    similarity_data = {}  # {video_id: {'image': tensor, 'audio': tensor}}
    
    print(f"Starting inference with {args.backbone}, Threshold: {args.threshold}, TCI: {args.use_tci}, Filtering: {args.use_filtering} (thr={args.filter_threshold})")
    
    for i in tqdm(range(len(dataset))):
        decord_vr, waveform_and_sr, video_id = dataset[i]
        
        # Transforms
        # Audio: Split into 10 chunks (1s each)
        # Note: Model internal logic handles transformations usually if we pass raw?
        # Our `models.py` expects `audio_transformed`.
        # We need to instantiate transforms here? Or move transform logic to model?
        # In `models.py` I assumed `pixel_values` are passed.
        # Let's instantiate transforms here.
        
        from transforms import VisionTransform, AudioTransform
        vision_trans = VisionTransform(args.backbone) # default 8 frames
        audio_trans = AudioTransform(args.backbone)
        
        # 1. Transform Audio
        # For per-second granularity (10 chunks)
        audio_chunks = audio_trans.split_sample_audio(waveform_and_sr, sample_audio_sec=1) 
        
        # For global filtering (full 10s)
        audio_full = None
        if args.use_filtering:
            audio_full = audio_trans(waveform_and_sr)
            if isinstance(audio_full, torch.Tensor):
                audio_full = audio_full.to(device)
                if len(audio_full.shape) == 5:
                    audio_full = audio_full.squeeze(1)
        # shape: (10, 1, 3, 112, 1036) for LB or (10, ...) list for CLAP?
        
        if args.backbone == 'clip_clap' and isinstance(audio_chunks, list):
             audio_chunks = torch.stack(audio_chunks).to(device)
        elif isinstance(audio_chunks, torch.Tensor):
             audio_chunks = audio_chunks.to(device)
             if len(audio_chunks.shape) == 5: # (10, 1, ...)
                 audio_chunks = audio_chunks.squeeze(1)

        # 2. Transform Video (Global or per second?)
        # For simplicity and effectiveness, standard approach is:
        # - Video: Extract global features or features per second?
        # - Original `video_parser_optmizer` used `transform_type='image'` which extracts 10 images (1 per sec).
        video_frames = vision_trans(decord_vr, transform_type='image').to(device) # (10, C, H, W)
        
        # 3. Compute Similarities
        # Model __call__ expects (labels, vision, audio)
        # We process 'image' mode for frame-wise similarity.
        # We process 'audio' mode for chunk-wise similarity.
        
        # Note: `LanguageBind_model` in `models.py` handles "video_mode='video'" (avg) or 'image'.
        # We want frame-wise (10, 25).
        
        sims = model(labels, 
                     vision_transformed=video_frames, 
                     audio_transformed=audio_chunks, 
                     video_id=video_id,
                     similarity_type='audio-image', 
                     vision_mode='image', # output (10, 25) for image
                     start_time=0, end_time=10,
                     audio_transformed_full=audio_full)
        
        # sims['image'] : (10, K)
        # sims['audio'] : (10, K)
        
        # ── Label Filtering (Independent) ──
        vid_active_labels = labels
        aud_active_labels = labels
        
        if args.use_filtering:
            vid_sims_global = sims['image'].mean(dim=0)  # (K,)
            
            if 'global_audio' in sims:
                aud_sims_global = sims['global_audio'].squeeze(0) # (K,)
            else:
                aud_sims_global = sims['audio'].mean(dim=0)  # (K,)
            
            vid_pass = [i for i, s in enumerate(vid_sims_global) if s > args.filter_threshold]
            aud_pass = [i for i, s in enumerate(aud_sims_global) if s > args.filter_threshold]
            
            vid_filtered_labels = [labels[i] for i in vid_pass]
            aud_filtered_labels = [labels[i] for i in aud_pass]
            
            print(f"[FILTER] {video_id}: V {len(labels)}→{len(vid_filtered_labels)}, A {len(labels)}→{len(aud_filtered_labels)}")
            if vid_filtered_labels:
                print(f"[FILTER] V-Kept: {vid_filtered_labels}")
            if aud_filtered_labels:
                print(f"[FILTER] A-Kept: {aud_filtered_labels}")
            
            # Re-compute specifically for each modality if filtered
            if 0 < len(vid_filtered_labels) < len(labels):
                vid_active_labels = vid_filtered_labels
                sims_vid = model(vid_active_labels,
                                 vision_transformed=video_frames,
                                 audio_transformed=None, # only compute vision
                                 video_id=video_id,
                                 similarity_type='image',
                                 vision_mode='image',
                                 start_time=0, end_time=10)
                sims['image'] = sims_vid['image']
                if 'image_raw' in sims_vid:
                    sims['image_raw'] = sims_vid['image_raw']
            elif len(vid_filtered_labels) == 0:
                print(f"[FILTER] WARNING: All V-labels filtered out for {video_id}.")
                vid_active_labels = labels  # fallback
                
            if 0 < len(aud_filtered_labels) < len(labels):
                aud_active_labels = aud_filtered_labels
                sims_aud = model(aud_active_labels,
                                 vision_transformed=None, # only compute audio
                                 audio_transformed=audio_chunks,
                                 video_id=video_id,
                                 similarity_type='audio',
                                 start_time=0, end_time=10,
                                 audio_transformed_full=audio_full)
                sims['audio'] = sims_aud['audio']
                if 'audio_raw' in sims_aud:
                    sims['audio_raw'] = sims_aud['audio_raw']
            elif len(aud_filtered_labels) == 0:
                print(f"[FILTER] WARNING: All A-labels filtered out for {video_id}.")
                aud_active_labels = labels  # fallback
        
        # ── Thresholding & Event Detection ──
        vid_events_bin = get_binary_events(sims['image'], args.threshold)
        aud_events_bin = get_binary_events(sims['audio'], args.threshold)
        
        def events_to_mask(evt_list, num_classes, length=10):
            mask = np.zeros((length, num_classes), dtype=int)
            for e in evt_list:
                mask[e['start']:e['end'], e['class_idx']] = 1
            return mask
            
        vid_mask_active = events_to_mask(vid_events_bin, len(vid_active_labels))
        aud_mask_active = events_to_mask(aud_events_bin, len(aud_active_labels))
        
        # Project masks to full 25-class space for Fusion & Export
        vid_mask_full = np.zeros((10, len(labels)), dtype=int)
        aud_mask_full = np.zeros((10, len(labels)), dtype=int)
        
        for i, lbl in enumerate(vid_active_labels):
            idx = labels.index(lbl)
            vid_mask_full[:, idx] = vid_mask_active[:, i]
            
        for i, lbl in enumerate(aud_active_labels):
            idx = labels.index(lbl)
            aud_mask_full[:, idx] = aud_mask_active[:, i]
        
        fusion_mask_full = np.logical_and(vid_mask_full, aud_mask_full).astype(int)
        
        # Helper: convert mask → event list using full labels
        def mask_to_events(mask, label_list):
            events = []
            for cls_idx in range(len(label_list)):
                seq = mask[:, cls_idx]
                diff = np.diff(np.concatenate(([0], seq, [0])))
                starts = np.flatnonzero(diff == 1)
                ends = np.flatnonzero(diff == -1)
                for s, e in zip(starts, ends):
                    events.append({"event_label": label_list[cls_idx], "start": int(s), "end": int(e)})
            return events
        
        final_events = mask_to_events(fusion_mask_full, labels)
        vid_events = mask_to_events(vid_mask_full, labels)
        aud_events = mask_to_events(aud_mask_full, labels)

        predictions["combined"][video_id] = final_events
        predictions["video"][video_id] = vid_events
        predictions["audio"][video_id] = aud_events
        
        # Map filtered similarities back to the full 25 labels for segment analysis export
        # Map filtered similarities back to the full 25 labels for segment analysis export
        full_sims = {
            'image': torch.zeros((10, len(labels))),
            'audio': torch.zeros((10, len(labels))),
            'image_raw': torch.zeros((10, len(labels))),
            'audio_raw': torch.zeros((10, len(labels)))
        }
        for i, lbl in enumerate(vid_active_labels):
            idx = labels.index(lbl)
            full_sims['image'][:, idx] = sims['image'][:, i].cpu()
            if 'image_raw' in sims:
                full_sims['image_raw'][:, idx] = sims['image_raw'][:, i].cpu()
                
        for i, lbl in enumerate(aud_active_labels):
            idx = labels.index(lbl)
            full_sims['audio'][:, idx] = sims['audio'][:, i].cpu()
            if 'audio_raw' in sims:
                full_sims['audio_raw'][:, idx] = sims['audio_raw'][:, i].cpu()
                
        similarity_data[video_id] = full_sims  # Store per-video similarities
        
    # Save Results
    res_dir = "results"
    os.makedirs(res_dir, exist_ok=True)
    out_file = os.path.join(res_dir, f"candidates_{args.backbone}.json")
    
    # Needs to wrap in list? Original: {"combined": [ {vid: events}, ... ]}
    # My predictions["combined"] is {vid: events}.
    # Need to convert to list format for `eval_metrics.py` compatibility.
    
    formatted_preds = {
        "combined": [{k: v} for k, v in predictions["combined"].items()],
        "video": [{k: v} for k, v in predictions["video"].items()],
        "audio": [{k: v} for k, v in predictions["audio"].items()]
    }
    
    with open(out_file, 'w') as f:
        json.dump(formatted_preds, f, indent=4)
        
    seg_dir = "segment_analysis"
    os.makedirs(seg_dir, exist_ok=True)
    export_segments_to_txt(
        formatted_preds, args.video_dir, similarity_data,
        output_file=os.path.join(seg_dir, args.backbone,"segment_details.txt")
    )
    
    # Evaluation
    print("Running Evaluation...")
    # NOTE: `calculate_metrices_LLP` expects list format?
    # Original: pred["combined"] is list of dicts.
    # Yes, formatted_preds is correct.
    
    # We call the eval function
    metrics, _ = calculate_metrices_LLP(args.video_dir, formatted_preds, labels)
    print_metrices(metrics)

if __name__ == '__main__':
    main()
