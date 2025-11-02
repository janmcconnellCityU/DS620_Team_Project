"""
dataset.py
----------
PyTorch Dataset for loading Mel-spectrogram features and labels from index.csv.

Behavior:
- Reads rows from data/processed/index.csv (columns: path,label)
- Loads each .npy spectrogram file as float32
- Applies simple per-sample standardization (z-score)
- Returns tensors shaped [1, n_mels, time] and integer labels

Usage:
    from collab.zsolt.dataset import MelDigits
    ds = MelDigits("data/processed/index.csv")
    x, y = ds[0]
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class MelDigits(Dataset):
    """
    Dataset that maps paths from index.csv to spectrogram tensors and labels.
    """

    def __init__(self, index_csv: str | Path, transform=None):
        self.index_csv = Path(index_csv)
        self.paths, self.labels = [], []

        # Read the CSV manually to avoid pandas dependency
        with self.index_csv.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            if header != ["path", "label"]:
                raise ValueError(
                    "index.csv must have header: path,label. Got: " + ",".join(header)
                )
            for line in f:
                p, y = line.strip().split(",")
                self.paths.append(Path(p))
                self.labels.append(int(y))

        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        # Load spectrogram: expected shape [n_mels, time], dtype float32
        x = np.load(self.paths[i]).astype(np.float32)

        # Simple per-sample normalization improves training stability
        x = (x - x.mean()) / (x.std() + 1e-6)

        # Add a channel dimension to become [1, n_mels, time]
        x = torch.from_numpy(x).unsqueeze(0)

        # Integer class label 0..9
        y = torch.tensor(self.labels[i], dtype=torch.long)

        # Optional user transform hook (e.g., time masking, noise)
        if self.transform:
            x = self.transform(x)

        return x, y
