"""
matting/train.py — Train the MattingUNet on the AISegment dataset.

AISegment dataset layout expected:
    data/aisegment/
        clip_img/
            YYYYMMDD/
                *.jpg (RGB images)
        matting/
            YYYYMMDD/
                *.png (alpha mattes, single-channel or RGBA)

Usage:
    python matting/train.py --data data/aisegment --epochs 30
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

# Allow relative import when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from model import MattingUNet


# ──────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────
class MattingLoss(nn.Module):
    """L1 (alpha) + Dice (boundary) combination."""

    def __init__(self, l1_weight: float = 0.7, dice_weight: float = 0.3):
        super().__init__()
        self.l1_w   = l1_weight
        self.dice_w = dice_weight

    def dice_loss(self, pred, target, eps=1e-6):
        pred   = pred.view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum()
        return 1 - (2 * inter + eps) / (pred.sum() + target.sum() + eps)

    def forward(self, pred, target):
        l1   = F.l1_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.l1_w * l1 + self.dice_w * dice


# ──────────────────────────────────────────────────────────────
# IoU metric
# ──────────────────────────────────────────────────────────────
def compute_iou(pred: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> float:
    pred_bin   = (pred  > threshold).float()
    target_bin = (target > threshold).float()
    inter = (pred_bin * target_bin).sum()
    union = (pred_bin + target_bin).clamp(max=1).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


import cv2
import torchvision.transforms as T

# ──────────────────────────────────────────────────────────────
# Dataset Helpers
# ──────────────────────────────────────────────────────────────
def _collect_pairs(clip_root: Path, matte_root: Path, max_pairs: int = None):
    pairs = []
    clip_root_str  = str(clip_root)
    matte_root_str = str(matte_root)
    n_checked  = 0

    for dirpath, _, filenames in os.walk(clip_root_str):
        jpg_files = sorted(f for f in filenames if f.lower().endswith(".jpg"))
        if not jpg_files:
            continue

        rel_dir = os.path.relpath(dirpath, clip_root_str)
        parts   = rel_dir.split(os.sep)
        if len(parts) >= 2:
            parts[1] = parts[1].replace("clip_", "matting_")
        matte_dir = os.path.join(matte_root_str, *parts)

        for fname in jpg_files:
            img_path   = os.path.join(dirpath, fname)
            matte_path = os.path.join(matte_dir, os.path.splitext(fname)[0] + ".png")
            n_checked += 1

            if os.path.exists(matte_path):
                pairs.append((img_path, matte_path))

            if max_pairs and len(pairs) >= max_pairs:
                return pairs

    return pairs

# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class AISegmentDataset(Dataset):
    def __init__(self, clip_root: str, matte_root: str, split: str = "train",
                 img_size: tuple = (256, 256), n_train: int = 5000,
                 n_val: int = 500, n_test: int = 500, seed: int = 42):
        self.split = split
        self.img_size = img_size

        max_needed = n_train + n_val + n_test
        all_pairs = _collect_pairs(Path(clip_root), Path(matte_root), max_pairs=max_needed * 2)
        
        rng = random.Random(seed)
        rng.shuffle(all_pairs)

        total = n_train + n_val + n_test
        all_pairs = all_pairs[:total]
        
        if split == "train":
            self.pairs = all_pairs[:n_train]
        elif split == "val":
            self.pairs = all_pairs[n_train : n_train + n_val]
        else:
            self.pairs = all_pairs[n_train + n_val : n_train + n_val + n_test]

        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, matte_path = self.pairs[idx]

        img       = cv2.imread(img_path,   cv2.IMREAD_COLOR)
        matte_raw = cv2.imread(matte_path, cv2.IMREAD_UNCHANGED)

        if img is None or matte_raw is None:
            raise FileNotFoundError(f"Could not read: {img_path} or {matte_path}")

        if matte_raw.ndim == 2:
            matte = matte_raw
        elif matte_raw.shape[2] == 4:
            matte = matte_raw[:, :, 3]
        else:
            matte = cv2.cvtColor(matte_raw, cv2.COLOR_BGR2GRAY)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H, W = self.img_size
        img   = cv2.resize(img,   (W, H), interpolation=cv2.INTER_LINEAR)
        matte = cv2.resize(matte, (W, H), interpolation=cv2.INTER_LINEAR)

        img   = torch.from_numpy(img.astype(np.float32)   / 255.0).permute(2, 0, 1)
        matte = torch.from_numpy(matte.astype(np.float32) / 255.0).unsqueeze(0)

        if self.split == "train":
            if random.random() > 0.5:
                img   = TF.hflip(img)
                matte = TF.hflip(matte)
            img = self.color_jitter(img)
            crop_frac = random.uniform(0.85, 1.0)
            ch = int(H * crop_frac)
            cw = int(W * crop_frac)
            top  = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            img   = TF.resized_crop(img,   top, left, ch, cw, (H, W))
            matte = TF.resized_crop(matte, top, left, ch, cw, (H, W))

        # Normalize img (ImageNet stats)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return img, matte


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clip_root  = os.path.join(args.data, "clip_img")
    matte_root = os.path.join(args.data, "matting")

    train_ds = AISegmentDataset(clip_root, matte_root, split="train", img_size=(args.size, args.size))
    val_ds   = AISegmentDataset(clip_root, matte_root, split="val",   img_size=(args.size, args.size))

    if len(train_ds) == 0:
        print("[error] Dataset is empty. Cannot train matting model.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Model, optimiser, loss
    model     = MattingUNet(pretrained=True).to(device)
    criterion = MattingLoss(l1_weight=0.7, dice_weight=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    # Logging
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "matting_log.csv"
    csv_f    = open(csv_path, "w", newline="")
    csv_w    = csv.writer(csv_f)
    csv_w.writerow(["epoch", "train_loss", "val_loss", "val_iou"])

    best_iou          = 0.0
    best_weights_path = out_dir / "matting_best.pth"
    patience          = args.patience
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for imgs, mattes in train_loader:
            imgs, mattes = imgs.to(device), mattes.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, mattes)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────
        model.eval()
        va_loss, va_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, mattes in val_loader:
                imgs, mattes = imgs.to(device), mattes.to(device)
                preds  = model(imgs)
                loss   = criterion(preds, mattes)
                va_loss += loss.item()
                for p, t in zip(preds, mattes):
                    va_iou += compute_iou(p, t)
        va_loss /= len(val_loader)
        va_iou  /= max(1, len(val_ds))

        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Ep {epoch:03d}/{args.epochs} | "
              f"tr_loss={tr_loss:.4f} | va_loss={va_loss:.4f} | "
              f"IoU={va_iou:.4f} | lr={lr:.2e}")
        csv_w.writerow([epoch, tr_loss, va_loss, va_iou])

        if va_iou > best_iou:
            best_iou          = va_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best IoU={best_iou:.4f})")
                break

    csv_f.close()
    torch.save(model.state_dict(), out_dir / "matting_final.pth")
    print(f"\n  Best IoU : {best_iou:.4f}")
    print(f"  Weights  : {best_weights_path}")

    if best_iou < 0.85:
        print("  [warn] IoU < 0.85 target. Consider: more epochs, "
              "lower LR, bigger dataset split, or heavier augmentation.")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/aisegment",
                        help="Root of AISegment dataset")
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--size",    type=int, default=256,
                        help="Resize input frames to size×size")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out",     default="matting/weights",
                        help="Directory to save weights and logs")
    args = parser.parse_args()
    train(args)
