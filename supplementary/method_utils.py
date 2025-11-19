import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
####################################################################


def calculate_sensor_stats(dataset):
    """
    Calculates mean and std for each sensor channel across the entire dataset.
    """
    all_sensor_data = []
    print("Calculating sensor statistics...")
    
    for i in tqdm(range(len(dataset)), desc="Collecting sensor data"):
        _, sensor_data, _, _, _ = dataset[i]     
        sensor_data = np.asarray(sensor_data, dtype=float)   
        all_sensor_data.append(sensor_data)
    
    concatenated_data = np.concatenate(all_sensor_data, axis=1).astype(float)
    
    mean = np.mean(concatenated_data, axis=1)
    std = np.std(concatenated_data, axis=1)
    
    print("Calculation complete.")
    return {'mean': mean, 'std': std}


#################################################################


def save_stats(stats, path):
    """
    Saves the calculated statistics to a .npy file.
    """
    np.save(path, stats)
    print(f"Sensor statistics saved to {path}")


#################################################################


def load_stats(path):
    """
    Loads statistics from a .npy file.
    """
    stats = np.load(path, allow_pickle=True).item()
    print(f"Sensor statistics loaded from {path}")
    return stats


#################################################################

def time_warp(x, sigma=0.2, num_knots=4):
    
    B, C, T = x.shape
    device = x.device
    
    warped_x = torch.zeros_like(x)
    for i in range(B):
        time_indices = torch.arange(T, device=device, dtype=torch.float32)
        perturb = sigma * T * torch.sin(torch.linspace(0, 4*3.14, T, device=device))
        perturb = perturb * torch.rand(1, device=device)  
        
        warped_indices = time_indices + perturb
        warped_indices = torch.clamp(warped_indices, 0, T-1)
        
        warped_indices_low = warped_indices.floor().long()
        warped_indices_high = warped_indices.ceil().long()
        warped_indices_high = torch.clamp(warped_indices_high, 0, T-1)
        
        weight_high = warped_indices - warped_indices_low.float()
        weight_low = 1.0 - weight_high
        
        x_batch = x[i]  # (C, T)
        warped_x_batch = torch.zeros_like(x_batch)
        
        for t in range(T):
            low_idx = warped_indices_low[t]
            high_idx = warped_indices_high[t]
            warped_x_batch[:, t] = weight_low[t] * x_batch[:, low_idx] + weight_high[t] * x_batch[:, high_idx]
        
        warped_x[i] = warped_x_batch
    
    return warped_x

def gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all replicas into a single tensor."""

    """
    Differentiable all_gather version (autograd supported)
    """
    from torch.distributed.nn.functional import all_gather
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    tensor = tensor.cuda()
    gathered = all_gather(tensor) 
    return torch.cat(gathered, dim=0)
