import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, X_dir: str = "data", y_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.X_dir = X_dir
        self.y_dir = y_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(X_dir, f"{split}_X", "*.npy")))
        print(self.num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.X_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
#       X = torch.from_numpy(np.load(X_path))

        X = np.load(X_path)
        X = X.astype(np.float32)  # filter のために変更
        X = torch.from_numpy(X)
        
        subject_idx_path = os.path.join(self.y_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.y_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            
            return X, y, subject_idx
        else:
            return X, subject_idx
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.X_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.X_dir, f"{self.split}_X", "00000.npy")).shape[1]