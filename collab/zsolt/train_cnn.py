"""
train_cnn.py
------------
Minimal training script for the SmallCNN on AudioMNIST Mel-spectrogram features.

Features:
- Loads dataset from data/processed/index.csv
- Splits into train/val/test
- Trains with CrossEntropyLoss and Adam
- Saves best validation checkpoint to checkpoints/cnn.pt
- Reports final test accuracy

Run:
    python -m collab.zsolt.train_cnn --index data/processed/index.csv --epochs 10

Dependencies:
    pip install torch torchvision torchaudio
"""

# collab/zsolt/train_cnn.py
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (slightly slower, safer for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# Dataset
# -----------------------------
class NpySpecDataset(Dataset):
    """
    Loads spectrograms saved as .npy arrays.
    Expects each row in the index CSV to contain:
      - path: path to .npy (required)
      - label: integer class 0..9 (required)
      - speaker: optional string/ID for group split
    Shapes accepted: (F, T) or (1, F, T). Returned: (1, F, max_frames)
    """
    def __init__(self, df: pd.DataFrame, max_frames: int = 128, pad_value: float = -80.0):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.max_frames = int(max_frames)
        self.pad_value = float(pad_value)

    def __len__(self):
        return len(self.paths)

    def _pad_or_trim(self, x: np.ndarray) -> np.ndarray:
        # x shape: (F, T)
        F, T = x.shape
        if T < self.max_frames:
            pad = np.full((F, self.max_frames - T), self.pad_value, dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        else:
            x = x[:, : self.max_frames]
        return x

    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.labels[idx]

        x = np.load(path)  # expected (F, T) or (1, F, T)
        if x.ndim == 3:
            # assume (1, F, T) or (F, T, 1)
            if x.shape[0] == 1:
                x = x[0]
            elif x.shape[-1] == 1:
                x = x[..., 0]
            else:
                raise ValueError(f"Unexpected 3D shape for {path}: {x.shape}")
        if x.ndim != 2:
            raise ValueError(f"Expected 2D (F,T) after squeeze, got {x.shape} for {path}")

        # pad/trim time axis
        x = x.astype(np.float32)
        x = self._pad_or_trim(x)

        # add channel dim -> (1, F, T)
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x)  # float32
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# -----------------------------
# Simple CNN model
# -----------------------------
class SmallCnn(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        # Input: (B, 1, F=128, T=128) by default
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (16, F/2, T/2)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (32, F/4, T/4)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # -> (64, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x

# -----------------------------
# Train / Eval
# -----------------------------
def run_epoch(model, loader, loss_fn, opt, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            opt.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                loss.backward()
                opt.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

# -----------------------------
# Split helpers
# -----------------------------
def speaker_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    speakers = sorted(df["speaker"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    n_val = max(1, int(len(speakers) * val_ratio))
    val_speakers = set(speakers[:n_val])
    tr = df[~df["speaker"].isin(val_speakers)].reset_index(drop=True)
    va = df[df["speaker"].isin(val_speakers)].reset_index(drop=True)
    return tr, va

def random_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * val_ratio)
    va = df.iloc[:n_val].reset_index(drop=True)
    tr = df.iloc[n_val:].reset_index(drop=True)
    return tr, va

# -----------------------------
# Main
# -----------------------------
def main(index_csv: str, epochs: int, batch_size: int, lr: float, seed: int,
         max_frames: int, num_workers: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir("checkpoints")

    # Expect CSV with at least: path,label
    # Optional: speaker
    df = pd.read_csv(index_csv)
    required = {"path", "label"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"{index_csv} must contain columns: {required}. Found: {list(df.columns)}")

    # Normalize/clean paths
    df["path"] = df["path"].apply(lambda p: os.path.normpath(p))

    # Choose split strategy
    if "speaker" in df.columns:
        df_train, df_val = speaker_split(df, val_ratio=0.2, seed=seed)
    else:
        df_train, df_val = random_split(df, val_ratio=0.2, seed=seed)

    train_ds = NpySpecDataset(df_train, max_frames=max_frames, pad_value=-80.0)
    val_ds   = NpySpecDataset(df_val,   max_frames=max_frames, pad_value=-80.0)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)

    model = SmallCnn(n_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -1.0
    best_path = os.path.join("checkpoints", "zsolt_cnn.pt")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_ld, loss_fn, opt, device, train=True)
        va_loss, va_acc = run_epoch(model, val_ld,   loss_fn, opt, device, train=False)

        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model_state": model.state_dict(),
                        "val_acc": best_val,
                        "config": {
                            "max_frames": max_frames,
                            "batch_size": batch_size,
                            "lr": lr,
                            "seed": seed
                        }},
                       best_path)
            print(f"  Saved new best to {best_path} (val_acc={best_val:.4f})")

    print(f"Done. Best val acc: {best_val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on AudioMNIST spectrograms (.npy) with padding.")
    parser.add_argument("--index", type=str, required=True,
                        help="CSV with columns: path,label[,speaker]")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=128,
                        help="Pad/trim time dimension to this many frames.")
    parser.add_argument("--num-workers", type=int, default=0,  # Windows-safe
                        help="DataLoader workers. Use 0 on Windows.")
    args = parser.parse_args()

    main(args.index, args.epochs, args.batch_size, args.lr, args.seed,
         args.max_frames, args.num_workers)
