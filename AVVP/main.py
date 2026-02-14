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
# from export_segments import export_segments_to_txt # If needed, but user said "segment_analysis folder... text file" so I might need to implement this simple export function here or import it.

def export_segments_to_txt(predictions, output_file):
    with open(output_file, 'w') as f:
        for vid, events in predictions.items():
            f.write(f"Video: {vid}\n")
            for event in events:
                f.write(f"  {event['event_label']} : {event['start']} - {event['end']}\n")
            f.write("\n")

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
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
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
        model = CLIP_CLAP_model(device)

    # Main Loop
    predictions = {
        "combined": {}, "video": {}, "audio": {}
    }
    
    print(f"Starting inference with {args.backbone}, Threshold: {args.threshold}")
    
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
        
        # 1. Transform Audio (10 chunks for per-second granularity)
        # The `split_sample_audio` from `AudioTransform` handles 10s split.
        audio_chunks = audio_trans.split_sample_audio(waveform_and_sr, sample_audio_sec=1) 
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
                     start_time=0, end_time=10)
        
        # sims['image'] : (10, 25)
        # sims['audio'] : (10, 25) -> My model wrapper handles 10 chunks
        
        # 4. Late Fusion (AND logic)
        # Original: "late fusion" code 
        # pred_av = np.logical_and(pred_audio, pred_video)
        
        # Thresholding
        vid_events_bin = get_binary_events(sims['image'], args.threshold)
        aud_events_bin = get_binary_events(sims['audio'], args.threshold)
        
        # Format for fusion: map to mask
        def events_to_mask(evt_list, num_classes=25, length=10):
            mask = np.zeros((length, num_classes), dtype=int)
            for e in evt_list:
                mask[e['start']:e['end'], e['class_idx']] = 1
            return mask
            
        vid_mask = events_to_mask(vid_events_bin, len(labels))
        aud_mask = events_to_mask(aud_events_bin, len(labels))
        
        fusion_mask = np.logical_and(vid_mask, aud_mask).astype(int)
        
        # Convert back to list of dicts
        final_events = []
        for cls_idx in range(len(labels)):
            seq = fusion_mask[:, cls_idx]
            diff = np.diff(np.concatenate(([0], seq, [0])))
            starts = np.flatnonzero(diff == 1)
            ends = np.flatnonzero(diff == -1)
            
            # Merge consecutive logic included naturally by mask? 
            # Yes, if 1,1,1 -> start=0, end=3.
            # But `merge_consecutive_segments` in original handled gap? 
            # Original: "if current[0] <= last[1] + 1". 
            # Our mask logic naturally merges touching ones.
            # If we want to merge ones separated by 1 frame gap, we need smoothing.
            # "merge_consecutive_segments" from prompt: "needed for formatting" basically.
            # I will trust the mask logic is sufficient for "AND" fusion.
            
            for s, e in zip(starts, ends):
                 final_events.append({
                     "event_label": labels[cls_idx],
                     "start": int(s),
                     "end": int(e)
                 })
                 
        predictions["combined"][video_id] = final_events
        # Also store individual for debug/completeness if needed, but not strictly required
        
    # Save Results
    res_dir = "results"
    os.makedirs(res_dir, exist_ok=True)
    out_file = os.path.join(res_dir, f"candidates_{args.backbone}.json")
    
    # Needs to wrap in list? Original: {"combined": [ {vid: events}, ... ]}
    # My predictions["combined"] is {vid: events}.
    # Need to convert to list format for `eval_metrics.py` compatibility.
    
    formatted_preds = {
        "combined": [{k: v} for k, v in predictions["combined"].items()],
        "video": [], # Empty if not evaluating components
        "audio": []
    }
    
    with open(out_file, 'w') as f:
        json.dump(formatted_preds, f, indent=4)
        
    # Export Text Analysis
    seg_dir = "segment_analysis"
    os.makedirs(seg_dir, exist_ok=True)
    export_segments_to_txt(predictions["combined"], os.path.join(seg_dir, f"segment_details_{args.backbone}.txt"))
    
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
