import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from .dataset import SensorTransform

SENSOR_NAMES = [
    'Ktch_B4_Cupboard',
    'Ktch_Motion_1',
    'Ktch_Motion_2',
    'Ktch_T1_Cupboard',
    'Ktch_T2_Cupboard',
    'Ktch_T3_Cupboard',
]


def _pick_column_indices(df_cols: List[str], wanted: List[str]) -> List[int]:
    idxs = []
    for name in wanted:
        if name not in df_cols:
            raise ValueError(f"CSV columns missing required sensor: {name}. Found={df_cols}")
        idxs.append(df_cols.index(name))
    return idxs

@dataclass
class SeqItem:
    sequence_id: str
    class_name: str
    class_idx: int
    window_paths: List[str]


class SequenceDataset(Dataset):
    def __init__(self, json_path: str, data_root: str, class_to_idx: Dict[str, int],
                 dtype: torch.dtype = torch.float32, sensor_transform: SensorTransform = None):
        super().__init__()
        self.data_root = data_root
        self.class_to_idx = class_to_idx
        self.dtype = dtype
        self.sensor_transform = sensor_transform

        with open(json_path, 'r', encoding='utf-8') as f:
            j = json.load(f)

        self.items: List[SeqItem] = []
        for seq in j["data"]:
            if "sequence_id" not in seq or "windows" not in seq:
                raise ValueError(f"Invalid sequence entry in {json_path}: {seq.keys()}")

            seq_id = seq["sequence_id"]
            windows = seq["windows"]
            if len(windows) == 0:
                continue

            class_names = set([w["class_name"] for w in windows])
            if len(class_names) != 1:
                raise ValueError(f"[{seq_id}] sequence has multiple class_name: {class_names}.")
            class_name = list(class_names)[0]

            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
            class_idx = self.class_to_idx[class_name]

            window_csvs = [os.path.join(self.data_root, w["sensor_path"]) for w in windows]
            self.items.append(SeqItem(seq_id, class_name, class_idx, window_csvs))

    def __len__(self):
        return len(self.items)

    def _load_window_csv(self, csv_path: str) -> np.ndarray:
        df = pd.read_csv(csv_path)
        cols = list(df.columns)
        selected_indices = _pick_column_indices(cols, SENSOR_NAMES)
        selected_df = df.iloc[:, selected_indices].fillna(0)
        data = selected_df.values.astype(np.float32).T  # (C, T)
        if data.shape[1] < 100:
            return None
        return data

    def __getitem__(self, idx: int):
        item = self.items[idx]
        tensors: List[torch.Tensor] = []

        for csv_path in item.window_paths:
            arr = self._load_window_csv(csv_path)
            if arr is None:
                continue

            if self.sensor_transform is not None:
                arr = self.sensor_transform(arr)  # torch.Tensor (C, target_len)
                if not isinstance(arr, torch.Tensor):
                    arr = torch.from_numpy(np.asarray(arr))
            else:
                arr = torch.from_numpy(arr)

            tensors.append(arr.to(dtype=self.dtype))

        if len(tensors) == 0:
            print("[WARN] too small window!!!")
            C = len(SENSOR_NAMES)
            target_len = getattr(self.sensor_transform, "target_len", 128)
            tensors.append(torch.zeros((C, target_len), dtype=self.dtype))
    
        seq_windows = torch.stack(tensors, dim=0).contiguous()  # (S, C, T)
        if seq_windows.shape[1] != 6 or seq_windows.shape[2] != 128:
            print(f"[WARN] Irregular shape in seq {item.sequence_id}: {seq_windows.shape}")

        target = torch.tensor(item.class_idx, dtype=torch.long)
        meta = {"sequence_id": item.sequence_id, "length": seq_windows.shape[0]}
        return seq_windows, target, meta

def collate_variable_length(batch):
    sequences, targets, metas = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    B = len(sequences)
    S_max = int(max(lengths).item())
    C = sequences[0].shape[1]
    T = sequences[0].shape[2]

    padded = torch.zeros((B, S_max, C, T), dtype=sequences[0].dtype)
    for i, s in enumerate(sequences):
        padded[i, :s.shape[0]] = s

    targets = torch.stack(targets, dim=0)
    return padded, lengths, targets, list(metas)