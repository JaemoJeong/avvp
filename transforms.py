from torchvision import transforms
from torchvision.transforms import Compose, Lambda
import torchaudio
import numpy as np
import torch
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
from torchvision.transforms import InterpolationMode

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
BICUBIC = InterpolationMode.BICUBIC

# Image Transforms
language_bind_image_transform = Compose([
    Lambda(lambda x: x / 255.0),
    transforms.Resize((224, 224), interpolation=BICUBIC), # Changed to Resize (224, 224)
    transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
])

clip_image_transforms = Compose([
    transforms.Resize((224, 224), interpolation=BICUBIC), # Changed to Resize (224, 224)
    Lambda(lambda x: x / 255.0),
    transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
])

# Video Transforms
language_bind_video_transform = Compose([
    Lambda(lambda x: x / 255.0),
    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
    transforms.Resize((224, 224), interpolation=BICUBIC), # Changed to Resize (224, 224) instead of Resize + CenterCrop
])

class VisionTransform:
    def __init__(self, model, num_frames=8, images_num=10):
        self.video_num_frames = num_frames
        self.images_num = images_num
        self.image_transforms = clip_image_transforms if model == "clip_clap" else language_bind_image_transform
        self.video_transforms = None if model == "clip_clap" else language_bind_video_transform
        self.model = model

    def __call__(self, decord_vr, transform_type, start=None, end=None):
        fps = len(decord_vr) // 10
        frames_indices = list(range(len(decord_vr)))
        
        if start is not None and end is not None:
            frames_indices = frames_indices[int(start * fps): int(end * fps)]
        
        if transform_type == "video":
            frames_indices = np.linspace(0, len(frames_indices) - 1, self.video_num_frames, dtype=int)
        else:
            frames_indices = np.linspace(0, len(frames_indices) - 1, self.images_num, dtype=int)

        if start is not None and end is not None:
            # Re-adjust indices if we sliced (though the slicing above already handles it, logic from original code might double count if not careful. 
            # In original code: frames_indicis = [frame_idx + start * fps for frame_idx in frames_indicis] was used when slicing logic was different.
            # Here: we sliced frames_indices first, then linspace. Correct.
            pass

        images = decord_vr.get_batch(frames_indices)
        
        if transform_type == "video": 
            if self.model == "clip_clap":
                return self.image_transforms(images.permute(0, 3, 1, 2))
            
            images = images.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            return self.video_transforms(images).unsqueeze(0)
        
        elif transform_type == "image":
            images = images.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            return self.image_transforms(images)
        
        else:
            raise ValueError("transform_type is not defined !!!")


DEFAULT_AUDIO_FRAME_SHIFT_MS = 10

class AudioTransform:
    def __init__(self, model):
        self.sample_rate = 16000
        self.num_mel_bins = 112
        self.target_length = 1036
        self.audio_mean = -4.2677393
        self.audio_std = 4.5689974
        self.model = model

    def __call__(self, audio_data_and_origin_sr, start_sec=None, end_sec=None):
        audio_data, origin_sr = audio_data_and_origin_sr

        if self.model == "clip_clap" and start_sec is None and end_sec is None:
            return torch.as_tensor(audio_data)

        if start_sec is not None and end_sec is not None:
            audio_data = self.crop_audio(audio_data, origin_sr, start_sec, end_sec)
            if self.model == "clip_clap":
                return torch.as_tensor(audio_data)
            
        if self.sample_rate != origin_sr:
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=origin_sr, new_freq=self.sample_rate)
            
        waveform_melspec = self.waveform2melspec(audio_data)
        return waveform_melspec.unsqueeze(0)
    
    def crop_audio(self, waveform, sample_rate, start_sec, end_sec):
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        end_sample = min(end_sample, waveform.shape[1])
        return waveform[:, start_sample:end_sample]

    def split_sample_audio(self, audio_data_and_origin_sr, sample_audio_sec, target_seconds=10):
        origin_audio_data, origin_sr = audio_data_and_origin_sr
        
        # Normalize audio to target_seconds
        target_samples = int(target_seconds * origin_sr)
        current_samples = origin_audio_data.shape[1]
        
        if current_samples >= target_samples:
            normalized_audio = origin_audio_data[:, :target_samples]
        else:
            if isinstance(origin_audio_data, np.ndarray):
                origin_audio_data = torch.from_numpy(origin_audio_data)
            n_repeat = (target_samples // current_samples) + 1
            repeated_audio = origin_audio_data.repeat(1, n_repeat)
            normalized_audio = repeated_audio[:, :target_samples]
        
        num_chunks = target_seconds // sample_audio_sec
        output = []
        for t in range(0, target_seconds, sample_audio_sec):
            audio_data = self.crop_audio(normalized_audio, origin_sr, t, t + sample_audio_sec)
            if self.model == "language_bind":
                output.append(self((audio_data, origin_sr)))
            else:
                output.append(torch.as_tensor(audio_data).squeeze(0))

        return torch.stack(output) if self.model == "language_bind" else output

    def waveform2melspec(self, audio_data):
        mel = self.get_mel(audio_data)
        
        if mel.shape[0] > self.target_length:
            chunk_frames = self.target_length
            total_frames = mel.shape[0]
            ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
            
            if len(ranges[1]) == 0: ranges[1] = [0]
            if len(ranges[2]) == 0: ranges[2] = [0]
            
            # Simple random split logic from original
            idx_front = np.random.choice(ranges[0])
            idx_middle = np.random.choice(ranges[1])
            idx_back = np.random.choice(ranges[2])
            
            mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
            mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
            mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
            
            mel_fusion = torch.stack([mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
        elif mel.shape[0] < self.target_length:
            n_repeat = int(self.target_length / mel.shape[0]) + 1
            mel = mel.repeat(n_repeat, 1)[:self.target_length, :]
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
        else:
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
            
        mel_fusion = mel_fusion.transpose(1, 2)
        mel_fusion = (mel_fusion - self.audio_mean) / (self.audio_std * 2)
        return mel_fusion

    def get_mel(self, audio_data):
        audio_data -= audio_data.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            audio_data,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
        )
        return mel
