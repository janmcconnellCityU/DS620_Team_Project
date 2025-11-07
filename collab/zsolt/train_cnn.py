"""
train_cnn.py
------------
Training script for spoken-digit classification using Mel-spectrogram features.

Includes:
- Dataset loading from index.csv
- Speaker-aware data split (prevents same voice in train/val/test)
- Collate function for variable-length spectrograms
- SpecAugment regularization during training
- Dropout-regularized CNN
- LR scheduler + weight decay
- Checkpoint saving for best validation model
"""

from pathlib import Path
import argparse
import json
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from collab.zsolt.dataset import MelDigits
from collab.zsolt.cnn_model import SmallCNN


# --------------------------------------------------------------------------
# Collate: pad variable-length spectrograms to the longest in the batch
# --------------------------------------------------------------------------
def pad_collate(batch):
    """
    Right-pads each spectrogram in a batch to the max time length.
    Input:  list of (x, y) where x:[1, n_mels, T_i], y:int
    Output: X:[B,1,n_mels,T_max], Y:[B]
    """
    xs, ys = zip(*batch)
    max_T = max(x.shape[-1] for x in xs)
    padded = []
    for x in xs:
        T = x.shape[-1]
        if T < max_T:
            x = F.pad(x, (0, max_T - T, 0, 0, 0, 0))
        padded.append(x)
    X = torch.stack(padded, 0)
    Y = torch.stack(ys, 0).long()
    return X, Y


# --------------------------------------------------------------------------
# SpecAugment: light frequency/time masking
# --------------------------------------------------------------------------
def spec_augment(x, max_freq_mask=16, max_time_mask=24, p=0.5):
    """
    Randomly masks frequency and time stripes in the spectrogram batch.
    x: [B,1,n_mels,T]
    """
    if torch.rand(1).item() > p:
        return x
    B, C, Fm, T = x.shape

    # frequency mask
    f = torch.randint(0, max_freq_mask + 1, (1,)).item()
    f0 = torch.randint(0, max(1, Fm - f + 1), (1,)).item()
    if f > 0:
        x[:, :, f0:f0 + f, :] = 0

    # time mask
    t = torch.randint(0, max_time_mask + 1, (1,)).item()
    t0 = torch.randint(0, max(1, T - t + 1), (1,)).item()
    if t > 0:
        x[:, :, :, t0:t0 + t] = 0

    return x


# --------------------------------------------------------------------------
# Speaker-based split (prevents speaker leakage between splits)
# --------------------------------------------------------------------------
def speaker_from_path(p: Path) -> str:
    """Extract speaker ID from filename such as '9_jackson_32.npy' -> 'jackson'."""
    name = p.stem
    m = re.search(r"[0-9]_([A-Za-z]+)_[0-9]+$", name)
    if m:
        return m.group(1).lower()
    parent = p.parent.name
    if parent and not parent.isdigit():
        return parent.lower()
    return name.lower()


def split_by_speaker(paths, val=0.15, test=0.15, seed=42):
    rng = np.random.default_rng(seed)
    speakers = [speaker_from_path(p) for p in paths]
    unique_speakers = np.array(sorted(set(speakers)))
    rng.shuffle(unique_speakers)
    n = len(unique_speakers)
    n_test = int(n * test)
    n_val = int(n * val)
    test_spk = set(unique_speakers[:n_test])
    val_spk = set(unique_speakers[n_test:n_test + n_val])
    train_spk = set(unique_speakers[n_test + n_val:])

    tr_idx, va_idx, te_idx = [], [], []
    for i, spk in enumerate(speakers):
        if spk in train_spk:
            tr_idx.append(i)
        elif spk in val_spk:
            va_idx.append(i)
        else:
            te_idx.append(i)
    return tr_idx, va_idx, te_idx


# --------------------------------------------------------------------------
# One epoch of training or evaluation
# --------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, opt, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            opt.zero_grad()

        with torch.set_grad_enabled(train):
            # augment only during training
            if train:
                x = spec_augment(x)
            logits = model(x)
            loss = loss_fn(logits, y)

            if train:
                loss.backward()
                opt.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# --------------------------------------------------------------------------
# Main training function
# --------------------------------------------------------------------------
def main(index_csv, epochs=10, batch_size=32, lr=3e-4, seed=42, resume=None, checkpoint_dir="checkpoints"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset and split by speaker
    full = MelDigits(index_csv)
    tr_idx, va_idx, te_idx = split_by_speaker(full.paths, val=0.15, test=0.15, seed=seed)
    train_ds = Subset(full, tr_idx)
    val_ds   = Subset(full, va_idx)
    test_ds  = Subset(full, te_idx)

    # DataLoaders
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, collate_fn=pad_collate, pin_memory=False)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=0, collate_fn=pad_collate, pin_memory=False)
    test_ld  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=0, collate_fn=pad_collate, pin_memory=False)

    # Model, optimizer, scheduler
    model = SmallCNN(n_classes=10).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2
    )

    best_val, history = -1.0, []
    ckpt = checkpoint_dir / "zsolt_cnn.pt"

    # -------------------------------
    # Resume checkpoint if provided
    # -------------------------------


    start_epoch = 1
    best_val_acc = 0.0

    if resume is not None and Path(resume).is_file():
        print(f"[resume] Loading from {resume}")
        checkpoint = torch.load(resume, map_location=device)

        # Case A: full checkpoint dict
        if isinstance(checkpoint, dict) and any(k in checkpoint for k in ["model_state", "model_state_dict"]):
            model_state = checkpoint.get("model_state") or checkpoint.get("model_state_dict")
            model.load_state_dict(model_state, strict=False)

            if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
                opt.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None and scheduler:
                scheduler.load_state_dict(checkpoint["scheduler_state"])

            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
            print(f"[resume] Continuing at epoch {start_epoch}")
        else:
            # Case B: weights-only file
            model.load_state_dict(checkpoint)
            print("[resume] Weights-only checkpoint loaded; optimizer will start fresh")

    
# ------------------------ TRAINING LOOP ------------------------
    for epoch in range(start_epoch, epochs + 1):
        # One full epoch on training data
        tr_loss, tr_acc = run_epoch(model, train_ld, loss_fn, opt, device, train=True)
        # Validation pass (no gradient)
        va_loss, va_acc = run_epoch(model, val_ld, loss_fn, opt, device, train=False)

        # Step the scheduler based on validation accuracy
        scheduler.step(va_acc)

        # Show current LR each epoch
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch:02d} | LR {current_lr:.6f} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        # Save metrics for plotting later
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc
        })

        # ---- best-weights for backwards-compat ----
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved new best to {ckpt} (val_acc={va_acc:.4f})")

        # ---- full checkpoints ----
        ckpt_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_val_acc": best_val_acc,
        }
        torch.save(ckpt_dict, checkpoint_dir / "last.pt")
        torch.save(ckpt_dict, checkpoint_dir / f"ckpt_epoch{epoch}.pt")
        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            torch.save(ckpt_dict, checkpoint_dir / "best.pt")

        # Optional: weights-only file
        torch.save(model.state_dict(), checkpoint_dir / "zsolt_cnn.pt")

    # ------------------------ TEST PHASE ------------------------
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_loss, test_acc = run_epoch(model, test_ld, loss_fn, opt, device, train=False)
    print(f"Done. Best val acc: {best_val:.4f} | Test acc: {test_acc:.4f}")

    # Final write of history (NOTE: this overwrites; switch to append if you want accumulation)
    with (checkpoint_dir / "zsolt_cnn_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/processed/index.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None, help="path to checkpoint (.pt) to resume from")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="where to save checkpoints")
    args = ap.parse_args()

    main(
        index_csv=args.index,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
    )
