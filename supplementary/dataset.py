import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from scipy.signal import butter, filtfilt
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import random
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


#####################################################################


class SensorTransform:
    def __init__(self, target_len, filter_btype='low', filter_order=4, cutoff_freq=0.1, 
                interpolation_mode='linear', mean=None, std=None):
        """
        Args:
            target_len (int): The target length of the sequence.
            filter_btype (str): The type of filter ('low', 'high', 'band').
            filter_order (int): The order of the filter.
            cutoff_freq (float or list): The cutoff frequency for the filter.
            interpolation_mode (str): The interpolation mode for resizing.
            mean (np.array, optional): Pre-computed mean for normalization.
            std (np.array, optional): Pre-computed std for normalization.
        """
        self.target_len = target_len
        self.interpolation_mode = interpolation_mode
        self.mean = mean


        self.std = std
        
        # Define Butterworth filter coefficients
        self.b, self.a = butter(filter_order, cutoff_freq, btype=filter_btype, analog=False)

    def fit(self, data_list):
        """
        Calculates mean and std from a list of numpy arrays (training data).
        Args:
            data_list (list): A list of sensor data arrays, each with shape (C, T).
        """
        # Concatenate all data along the time axis
        all_data = np.concatenate(data_list, axis=1)
        # Calculate mean and std channel-wise
        self.mean = np.mean(all_data, axis=1)
        self.std = np.std(all_data, axis=1)
        print("Mean and Std calculated and stored.")
   
    def _apply_filter(self, data):
        return filtfilt(self.b, self.a, data, axis=1)

    def _apply_normalization(self, data):
        if self.mean is not None and self.std is not None:
            mean = self.mean[:, np.newaxis]
            std = self.std[:, np.newaxis]
            data=(data-mean*data/(data+1e-8) )/(std+1e-8) 
        return data

    def _resize_to_target_len(self, data, device):
        data_tensor = torch.from_numpy(data.copy()).float().to(device)
        data_tensor = data_tensor.unsqueeze(0)
        
        interpolated_tensor = F.interpolate(
            data_tensor, 
            size=self.target_len, 
            mode=self.interpolation_mode, 
            align_corners=False if self.interpolation_mode != 'linear' else None # linear 모드는 align_corners 지원
        )
        return interpolated_tensor.squeeze(0)

    def __call__(self, sensor_data, device='cpu'):
        """
        Args:
            sensor_data (np.array): Input sensor data with shape (C, T).
            device (str): The device to move the final tensor to ('cpu' or 'cuda').
        Returns:
            torch.Tensor: Processed sensor data.
        """
        data_copy = sensor_data.copy()
        
        # 1. Apply normalization
        normalized_data = self._apply_normalization(data_copy)
        
        # 2. Apply filtering
        filtered_data = self._apply_filter(normalized_data)
        
        # 3. Resize the signal
        processed_data = self._resize_to_target_len(filtered_data, device)
        
        return processed_data


#####################################################################


class ClipConsistentTransforms:
    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        
        jitter_params = T.ColorJitter.get_params(
            brightness=(0.6, 1.4), contrast=(0.6, 1.4),
            saturation=(0.6, 1.4), hue=(-0.1, 0.1)
        )

        sigma = random.uniform(0.1, 2.0)

        tensor_frames = []
        for frame in clip:
            frame = T.Resize(self.size, antialias=True)(frame) 

            fn_indices, brightness_factor, contrast_factor, saturation_factor, hue_factor = jitter_params

            for fn_id in fn_indices:
                if fn_id == 0 and brightness_factor is not None:
                    frame = TF.adjust_brightness(frame, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    frame = TF.adjust_contrast(frame, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    frame = TF.adjust_saturation(frame, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    frame = TF.adjust_hue(frame, hue_factor)

            frame = TF.gaussian_blur(frame, kernel_size=[5, 5], sigma=sigma)

            tensor_frames.append(T.ToTensor()(frame))

        clip_tensor = torch.stack(tensor_frames, dim=0)

        # Normalize
        clip_tensor /= 255.0

        return clip_tensor


#####################################################################


class VideoSensorDataset(Dataset):
    def __init__(self, json_path: str, data_root: str, num_frames: int, transform, sensor_transform, threshold_epoch, start_index, end_index, cache_dir):
        super().__init__()
        
        self.data_root = data_root
        self.num_frames = num_frames
        self.transform = transform
        self.sensor_transform = sensor_transform
        self.current_epoch = 0
        self.threshold_epoch = threshold_epoch
        self.start_index = start_index
        self.end_index = end_index
        self.samples = []
        self.cache_dir = cache_dir
        self.use_cache = False

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for item in json_data['data']:

            relative_path_video = item['frame_path']
            relative_path_sensor = item['sensor_path']
            
            video_path = os.path.join(self.data_root, relative_path_video)
            sensor_path = os.path.join(self.data_root, relative_path_sensor)
            
            label = item['label']
            item_id = item['video_id']
            if 'optical_flow_dir' in item:
                relative_path_optical_flow = item['optical_flow_dir']
                optical_flow_path = os.path.join(self.data_root, relative_path_optical_flow)
            else:
                optical_flow_path = None
            
            self.samples.append((video_path, sensor_path, label, item_id, optical_flow_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, sensor_path, label, item_id, flow_path = self.samples[idx]
        
        if self.current_epoch <= self.threshold_epoch:  # only sensor data for threshold epochs
            dummy_clip = [Image.new('RGB', (224, 224)) for _ in range(self.num_frames)]
            frames_tensor = self.transform(dummy_clip)

        else:
            # Load and preprocess video frames
            parts = video_path.split(os.sep)

            if len(parts) >= 2:
                last_two_parts = '/'.join(parts[-2:])
            cache_dir = os.path.join(self.cache_dir, "videos")
            cache_path=os.path.join(cache_dir, last_two_parts)
            cache_path = cache_path.rsplit('.', 1)[0] + '.pt'

            cache_dir_for_file = os.path.dirname(cache_path)
            os.makedirs(cache_dir_for_file, exist_ok=True) 
            if os.path.exists(cache_path):
                with torch.serialization.safe_globals({Image.Image}):
                    frames = torch.load(cache_path)
            else:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"Video file not found at the constructed path: {video_path}")
                    raise IOError(f"Cannot open video file, it may be corrupted or in an unsupported format: {video_path}")
                    
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames > 1:
                    frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
                else:
                    frame_indices = np.zeros(self.num_frames, dtype=int)
                
                frames = []
                successful_reads = 0
                last_successful_frame = None

                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        successful_reads += 1
                        # OpenCV(BGR) -> RGB -> PIL Image
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        last_successful_frame = frame_pil.copy()
                        frames.append(frame_pil)
                        
                    else:
                        if last_successful_frame is not None:
                            frames.append(last_successful_frame.copy())
                        else:
                            frames.append(Image.new('RGB', (224, 224)))
                try:
                    # caching video clip atomically
                    import fcntl
                    lock_path = cache_path + ".lock"   
                    with open(lock_path, 'w') as f_lock:
                        fcntl.flock(f_lock, fcntl.LOCK_EX)
                        temp_cache_path = cache_path + ".tmp"
                        
                        if os.path.exists(cache_path):
                            frames = torch.load(cache_path, weights_only=False)
                        else:
                            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                            torch.save(frames, temp_cache_path)
                            os.rename(temp_cache_path, cache_path) 
                            print("clip is cached", cache_path)
                            
                except Exception as e:
                    print(f"Error during atomic cache write for {cache_path}: {e}")
                    if os.path.exists(temp_cache_path):
                        os.remove(temp_cache_path)
                    pass
                finally:
                    if os.path.exists(lock_path):
                        os.remove(lock_path)

                cap.release()

            if self.transform:
                frames_tensor = self.transform(frames)
            else:
                frames_tensor = torch.stack([T.ToTensor()(frame) for frame in frames])

        # Load and preprocess sensor data
        df = pd.read_csv(sensor_path)

        selected_indices = np.r_[self.start_index:self.end_index+1]
        selected_df = df.iloc[:, selected_indices].copy() 

        for col in selected_df.columns:
            if selected_df[col].isnull().sum() / len(selected_df) > 0.5:
                selected_df[col].fillna(0, inplace=True)

        selected_df.interpolate(method='linear', limit_direction='both', inplace=True)

        raw = selected_df.values

        sensor_data = raw.T # (C, T)

        if self.sensor_transform:
            sensor_data = self.sensor_transform(sensor_data)

        if flow_path is not None:
            try:
                flow_path = os.path.join(flow_path, "flow.npy")
                flow = np.load(flow_path) 
                flow = torch.from_numpy(flow).float()  # [T, 2, H, W]
                T, C, H, W = flow.shape
                if H > 224 or W > 224:
                    flow = F.interpolate(
                        flow, size=(224, 224),
                        mode="bilinear", align_corners=False
                    )
            except: 
                flow = {}        
        else:
            flow = {}
        return frames_tensor, sensor_data, label, [idx, item_id], flow
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch