from torch.utils.data import Dataset
import os
import numpy as np
import torchaudio
import decord
from decord import VideoReader, cpu
import librosa

decord.bridge.set_bridge('torch')

class VideoDataset(Dataset):
    def __init__(self, video_dir_path, audio_dir_path, backbone, subset=None, video_ext=".mp4", audio_ext=".wav"):
        self.video_dir_path = video_dir_path
        self.audio_dir_path = audio_dir_path
        self.backbone = backbone
        self.video_ext = video_ext
        self.audio_ext = audio_ext
        
        self.videos_ids = [video_id.replace(".mp4", "") for video_id in os.listdir(video_dir_path) 
                           if video_id.endswith(".mp4")]

        if subset is not None:
            self.videos_ids = [vid for vid in self.videos_ids if vid in subset]

        print(f"Dataset Loaded: {len(self.videos_ids)} videos.")

    def __len__(self):
        return len(self.videos_ids)

    def __getitem__(self, idx):
        video_id = self.videos_ids[idx]
        video_path = os.path.join(self.video_dir_path, video_id)
        audio_path = os.path.join(self.audio_dir_path, video_id)
        
        # Load Audio
        if self.backbone == 'language_bind':
            waveform_and_sr = torchaudio.load(f"{audio_path}{self.audio_ext}")
        else:
            audio_data, sr = librosa.load(f"{audio_path}{self.audio_ext}", sr=48000)
            audio_data = np.expand_dims(audio_data, axis=0)
            waveform_and_sr = (audio_data, sr)

        # Load Video
        decord_vr = VideoReader(f"{video_path}{self.video_ext}", ctx=cpu(0))
        decord_vr = normalize_vr_to_10s(decord_vr)

        return decord_vr, waveform_and_sr, video_id

def normalize_vr_to_10s(decord_vr, target_seconds=10.0):
    fps = decord_vr.get_avg_fps()
    target_frames = int(round(fps * target_seconds))
    n = len(decord_vr)

    if n >= target_frames:
        frame_idx = np.arange(target_frames)      # trim
    else:
        # Pad with last frame
        frame_idx = np.arange(target_frames)
        frame_idx = np.clip(frame_idx, 0, n - 1)

    class VR10s:
        def __init__(self, vr, idx):
            self.vr = vr
            self.idx = idx
        def __len__(self):
            return len(self.idx)
        def get_batch(self, indices):
            # indices are relative to the 10s timeline
            # map valid indices to original vr indices
            return self.vr.get_batch(self.idx[indices])
        def get_avg_fps(self):
            return fps

    return VR10s(decord_vr, frame_idx)
