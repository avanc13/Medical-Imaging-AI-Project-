#!/usr/bin/env python3
"""


- indexes by (subject, slice_index) pairs: because each subject has multiple 2D slices(30)
- Loads each subject’s 3D volumes only once and caches in mem
- Returns 2D slice tensors shaped (C, H, W) for 2D U-Net input
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class FlashMRIDataset(Dataset):
    def __init__(self, subject_ids, data_dir, echo_indices=None, mode="train", cache=True):
        """
        Args:
            subject_ids (list[str]):  ['sub-04570', 'sub-04620']
            data_dir (str): path to processed dir with .npy files
            echo_indices (list[int]): which echoes to load (default: all 4)
            mode (str): 'train' or 'val' — controls random slice selection
            cache (bool): if True, keeps loaded volumes in RAM
        """
        self.subject_ids = subject_ids
        self.data_dir = data_dir
        self.echo_indices = echo_indices or [1, 2, 3, 4]
        self.mode = mode
        self.cache = cache

        # small cache: {sub_id: np.ndarray shape (H, W, D, N_echoes)}
        self._volume_cache = {}

        # build index of (subject, slice_idx)
        self.index = []
        for subj_id in subject_ids:
            first_echo_path = os.path.join(data_dir, f"{subj_id}_echo{self.echo_indices[0]}.npy")
            if not os.path.exists(first_echo_path):
                raise FileNotFoundError(f"Missing echo file for {subj_id}: {first_echo_path}")
            depth = np.load(first_echo_path, mmap_mode="r").shape[2]
            for z in range(depth):
                self.index.append((subj_id, z))

        print(f"[{mode.upper()}] dataset built with {len(self.index)} slices "
              f"from {len(subject_ids)} subjects.")

    def __len__(self):
        return len(self.index)

    def _load_subject(self, subj_id):
        """Load and stack all requested echoes for one subject."""
        vols = []
        for echo in self.echo_indices:
            path = os.path.join(self.data_dir, f"{subj_id}_echo{echo}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing echo file: {path}")
            vol = np.load(path)
            vols.append(vol)
        stacked = np.stack(vols, axis=-1)  # (H, W, D, N_echoes)
        return stacked

    def __getitem__(self, idx):
        subj_id, z = self.index[idx]

        # load from cache or disk
        if self.cache and subj_id in self._volume_cache:
            stacked = self._volume_cache[subj_id]
        else:
            stacked = self._load_subject(subj_id)
            if self.cache:
                self._volume_cache[subj_id] = stacked  # keep for reuse

        slice_2d = stacked[:, :, z, :]  # (H, W, N_echoes)
        x = torch.from_numpy(slice_2d).permute(2, 0, 1).float()  # (C, H, W)
        return x, subj_id
