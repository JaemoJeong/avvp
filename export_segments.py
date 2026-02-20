"""
Segment-level GT and Prediction export utility
main.py에서 호출되어 자동으로 segment별 정보를 txt로 저장
"""
import numpy as np
import pandas as pd
import os
import torch

def export_segments_to_txt(predictions, video_dir_path, similarity_data, thresholds_data=None, output_file="segment_analysis.txt"):
    """
    각 segment별로 GT와 prediction을 txt 파일로 저장
    
    Args:
        predictions: main.py에서 생성된 predictions dict
        video_dir_path: 비디오 디렉토리 경로
        similarity_data: VideoParserOptimizer.similarities_data - dict of {video_id: {'video': tensor, 'audio': tensor}}
        thresholds_data: dict of {video_id: {'video': thresholds, 'audio': thresholds}} for dynamic thresholds
        output_file: 출력 txt 파일명
    """
    # Categories
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                'Clapping']
    
    id_to_idx = {id: index for index, id in enumerate(categories)}
    
    # Convert predictions to dict
    pred_combined = {list(d.keys())[0]: list(d.values())[0] for d in predictions["combined"]}
    pred_video = {list(d.keys())[0]: list(d.values())[0] for d in predictions["video"]}
    pred_audio = {list(d.keys())[0]: list(d.values())[0] for d in predictions["audio"]}
    
    # similarity_data is now a nested dict: {video_id: {'video': tensor, 'audio': tensor}}
    # We need to handle this correctly
    
    # Load GT data
    # print("Loading ground truth data for segment export...")
    # Updated paths to absolute paths used in AVVP project
    df_a = pd.read_csv("/mnt/hdd4tb/jaemo/data/LLP/AVVP_eval_audio.csv", header=0, sep='\t')
    df_v = pd.read_csv("/mnt/hdd4tb/jaemo/data/LLP/AVVP_eval_visual.csv", header=0, sep='\t')
    
    # Get valid video IDs
    download_videos_ids = [video_id.replace(".mp4", "") for video_id in os.listdir(video_dir_path) 
                          if os.path.splitext(os.path.join(video_dir_path, video_id))[1] == '.mp4']
    
    # Filter to predicted video IDs
    predicted_video_ids = set(pred_combined.keys())
    
    # Get all unique filenames
    all_filenames = sorted(set(df_a['filename'].tolist() + df_v['filename'].tolist()))
    all_filenames = [f for f in all_filenames if '_'.join(f.split('_')[:-2]) in predicted_video_ids]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Open output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("SEGMENT-LEVEL GROUND TRUTH vs PREDICTIONS\n")
        f.write("=" * 120 + "\n")
        f.write("NOTE: Similarity values are from filtered labels used in optimization.\n")
        f.write("      Columns may not match full 25 categories if label filtering was applied.\n")
        f.write("=" * 120 + "\n\n")
        
        total_segments = 0
        
        # Process each file
        for file_name in all_filenames:
            parts = file_name.split('_')
            video_id = '_'.join(parts[:-2])
            
            if video_id not in download_videos_ids:
                continue
            
            # Initialize matrices (25 categories x 10 segments)
            GT_a = np.zeros((25, 10))
            GT_v = np.zeros((25, 10))
            SO_a = np.zeros((25, 10))
            SO_v = np.zeros((25, 10))
            SO_av = np.zeros((25, 10))
            
            # Load GT Audio
            df_vid_a = df_a.loc[df_a['filename'] == file_name]
            for i in range(len(df_vid_a)):
                x1 = int(df_vid_a.iloc[i]['onset'])
                x2 = int(df_vid_a.iloc[i]['offset'])
                event = df_vid_a.iloc[i]['event_labels']
                idx = id_to_idx[event]
                GT_a[idx, x1:x2] = 1
            
            # Load GT Video
            df_vid_v = df_v.loc[df_v['filename'] == file_name]
            for i in range(len(df_vid_v)):
                x1 = int(df_vid_v.iloc[i]['onset'])
                x2 = int(df_vid_v.iloc[i]['offset'])
                event = df_vid_v.iloc[i]['event_labels']
                idx = id_to_idx[event]
                GT_v[idx, x1:x2] = 1
            
            GT_av = GT_a * GT_v
            
            # Load Predictions
            if video_id in pred_combined:
                for pred_dict in pred_combined[video_id]:
                    # Handle both 'event_label' from AVVP format and generic format
                    lbl = pred_dict.get("event_label", pred_dict.get("label", ""))
                    if lbl:
                        idx = id_to_idx[lbl.capitalize()] if lbl.capitalize() in id_to_idx else id_to_idx.get(lbl, -1)
                        if idx != -1:
                            x1, x2 = pred_dict["start"], pred_dict["end"]
                            SO_av[idx, x1:x2] = 1
            
            if video_id in pred_video:
                for pred_dict in pred_video[video_id]:
                    lbl = pred_dict.get("event_label", pred_dict.get("label", ""))
                    if lbl:
                        idx = id_to_idx[lbl.capitalize()] if lbl.capitalize() in id_to_idx else id_to_idx.get(lbl, -1)
                        if idx != -1:
                            x1, x2 = pred_dict["start"], pred_dict["end"]
                            SO_v[idx, x1:x2] = 1
            
            if video_id in pred_audio:
                for pred_dict in pred_audio[video_id]:
                    lbl = pred_dict.get("event_label", pred_dict.get("label", ""))
                    if lbl:
                        idx = id_to_idx[lbl.capitalize()] if lbl.capitalize() in id_to_idx else id_to_idx.get(lbl, -1)
                        if idx != -1:
                            x1, x2 = pred_dict["start"], pred_dict["end"]
                            SO_a[idx, x1:x2] = 1
            
            # Get similarity data for this video (nested dict structure)
            video_sim = None
            audio_sim = None
            combined_sim = None
            
            if similarity_data and video_id in similarity_data:
                # Expecting tensor or numpy array. If tensor, move to cpu
                def to_cpu(x):
                    if hasattr(x, 'cpu'): return x.cpu()
                    return x

                video_sim = to_cpu(similarity_data[video_id].get('video', None))
                if video_sim is None: video_sim = to_cpu(similarity_data[video_id].get('image', None)) # Fallback (Final used score)
                
                # Check for VCD keys
                video_sim_original = to_cpu(similarity_data[video_id].get('image_original', None))
                video_sim_vcd = to_cpu(similarity_data[video_id].get('image_vcd', None))
                
                # Check for raw keys
                video_sim_original_raw = to_cpu(similarity_data[video_id].get('image_original_raw', None))
                video_sim_vcd_raw = to_cpu(similarity_data[video_id].get('image_vcd_raw', None))
                audio_sim_raw = to_cpu(similarity_data[video_id].get('audio_raw', None))
                
                has_vcd = (video_sim_vcd is not None)
                
                audio_sim = to_cpu(similarity_data[video_id].get('audio', None))
                # combined_sim = similarity_data[video_id].get('combined', None)
            
            # Write header for this video
            f.write("\n" + "=" * 120 + "\n")
            f.write(f"VIDEO: {video_id}\n")
            f.write("=" * 120 + "\n\n")
            
            # Write segment-by-segment analysis
            for seg_idx in range(10):
                # Check if this segment has any GT or prediction
                has_gt = (GT_a[:, seg_idx].sum() > 0 or GT_v[:, seg_idx].sum() > 0)
                has_pred = (SO_a[:, seg_idx].sum() > 0 or SO_v[:, seg_idx].sum() > 0 or SO_av[:, seg_idx].sum() > 0)
                
                if not has_gt and not has_pred:
                    continue  # Skip empty segments
                
                total_segments += 1
                
                f.write(f"  SEGMENT {seg_idx}\n")
                f.write("  " + "-" * 140 + "\n")
                
                # Find categories with GT or prediction in this segment
                active_categories = []
                for cat_idx, category in enumerate(categories):
                    if (GT_a[cat_idx, seg_idx] > 0 or GT_v[cat_idx, seg_idx] > 0 or 
                        SO_a[cat_idx, seg_idx] > 0 or SO_v[cat_idx, seg_idx] > 0 or SO_av[cat_idx, seg_idx] > 0):
                        active_categories.append((cat_idx, category))
                
                if active_categories:
                    # Header for categories
                    if has_vcd:
                        f.write(f"  {'Category':<25} | {'GT_A':^5} {'GT_V':^5} {'GT_AV':^5} | {'Pred_A':^5} {'Pred_V':^5} {'Pred_AV':^5} | {'V_Orig':^6} {'V_VCD':^6} {'A_Sim':^6} | {'V_Orig':^8} {'V_VCD':^8} {'A_Sim':^8} (raw)")
                    else:
                        f.write(f"  {'Category':<25} | {'GT_A':^5} {'GT_V':^5} {'GT_AV':^5} | {'Pred_A':^5} {'Pred_V':^5} {'Pred_AV':^5} | {'V_Sim':^6} {'A_Sim':^6}")
                    f.write("\n")
                    f.write("  " + "-" * 150 + "\n")
                    
                    # Write each active category
                    for cat_idx, category in active_categories:
                        gt_a = int(GT_a[cat_idx, seg_idx])
                        gt_v = int(GT_v[cat_idx, seg_idx])
                        gt_av = int(GT_av[cat_idx, seg_idx])
                        pred_a = int(SO_a[cat_idx, seg_idx])
                        pred_v = int(SO_v[cat_idx, seg_idx])
                        pred_av = int(SO_av[cat_idx, seg_idx])
                        
                        v_sim_str, a_sim_str = "-", "-"
                        
                        # Get Similarity
                        # video/audio sim shape: (10, 25) ideally
                        if video_sim is not None:
                            try:
                                # Check shape
                                if len(video_sim.shape) == 2 and video_sim.shape[1] == 25:
                                    val = video_sim[seg_idx, cat_idx]
                                    v_sim_str = f"{val.item():.3f}" if hasattr(val, 'item') else f"{val:.3f}"
                            except: pass
                            
                        if audio_sim is not None:
                            try:
                                if len(audio_sim.shape) == 2 and audio_sim.shape[1] == 25:
                                    val = audio_sim[seg_idx, cat_idx]
                                    a_sim_str = f"{val.item():.3f}" if hasattr(val, 'item') else f"{val:.3f}"
                            except: pass

                        if has_vcd:
                            # Parse V_Orig and V_VCD (normalized)
                            v_orig_str = "-"
                            v_vcd_str = "-"
                            try:
                                if video_sim_original is not None:
                                    val = video_sim_original[seg_idx, cat_idx]
                                    v_orig_str = f"{val.item():.3f}" if hasattr(val, 'item') else f"{val:.3f}"
                                if video_sim_vcd is not None:
                                    val = video_sim_vcd[seg_idx, cat_idx]
                                    v_vcd_str = f"{val.item():.3f}" if hasattr(val, 'item') else f"{val:.3f}"
                            except: pass
                            
                            # Parse raw values
                            v_orig_raw_str, v_vcd_raw_str, a_raw_str = "-", "-", "-"
                            try:
                                if video_sim_original_raw is not None:
                                    val = video_sim_original_raw[seg_idx, cat_idx]
                                    v_orig_raw_str = f"{val.item():.3f}" if hasattr(val, 'item') else f"{val:.3f}"
                                if video_sim_vcd_raw is not None:
                                    val = video_sim_vcd_raw[seg_idx, cat_idx]
                                    v_vcd_raw_str = f"{val.item():.3f}" if hasattr(val, 'item') else f"{val:.3f}"
                                if audio_sim_raw is not None:
                                    val = audio_sim_raw[seg_idx, cat_idx]
                                    a_raw_str = f"{val.item():.3f}" if hasattr(val, 'item') else f"{val:.3f}"
                            except: pass
                            
                            f.write(f"  {category:<25} | {gt_a:^5} {gt_v:^5} {gt_av:^5} | {pred_a:^5} {pred_v:^5} {pred_av:^5} | {v_orig_str:^6} {v_vcd_str:^6} {a_sim_str:^6} | {v_orig_raw_str:^8} {v_vcd_raw_str:^8} {a_raw_str:^8}")
                        else:
                            f.write(f"  {category:<25} | {gt_a:^5} {gt_v:^5} {gt_av:^5} | {pred_a:^5} {pred_v:^5} {pred_av:^5} | {v_sim_str:^6} {a_sim_str:^6}")
                        f.write("\n")
                
                f.write("\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write(f"TOTAL ACTIVE SEGMENTS: {total_segments}\n")
        f.write("=" * 120 + "\n")
    
    print(f"✅ Segment analysis exported to: {output_file}")
