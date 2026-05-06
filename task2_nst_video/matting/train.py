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


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class AISegmentDataset(Dataset):
    """
    Pairs RGB images with their alpha mattes.
    Accepts any image size; resizes to target_size.
    """

    def __init__(self, img_dir: str, matte_dir: str,
                 target_size: int = 256,
                 augment: bool = False):
        self.img_paths    = sorted(Path(img_dir).rglob("*.jpg"))
        self.matte_dir    = Path(matte_dir)
        self.target_size  = target_size
        self.augment      = augment

        # Build matte path lookup: same relative stem, .png extension
        self.matte_paths  = []
        for p in self.img_paths:
            # Try same relative path in matte directory with .png
            rel   = p.relative_to(img_dir)
            mpath = self.matte_dir / rel.with_suffix(".png")
            if not mpath.exists():
                # Try flat search
                mpath = self.matte_dir / (p.stem + ".png")
            self.matte_paths.append(mpath)

        # Filter out missing mattes
        valid = [(i, m) for i, m in zip(self.img_paths, self.matte_paths)
                 if m.exists()]
        self.img_paths   = [v[0] for v in valid]
        self.matte_paths = [v[1] for v in valid]

        if len(self.img_paths) == 0:
            print("[warn] No valid image/matte pairs found. "
                  "Check your AISegment dataset path.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img   = Image.open(self.img_paths[idx]).convert("RGB")
        matte = Image.open(self.matte_paths[idx])

        # Convert matte to single-channel float [0,1]
        if matte.mode == "RGBA":
            matte = matte.split()[3]   # alpha channel
        elif matte.mode != "L":
            matte = matte.convert("L")

        sz = self.target_size

        # ── Augmentation ──────────────────────────────────────
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img   = TF.hflip(img)
                matte = TF.hflip(matte)
            # Color jitter on image only
            img = TF.adjust_brightness(img, 0.8 + 0.4 * random.random())
            img = TF.adjust_saturation(img, 0.8 + 0.4 * random.random())
            # Random crop
            i, j, h, w = self._random_crop_params(img, sz)
            img   = TF.resized_crop(img,   i, j, h, w, (sz, sz), Image.BILINEAR)
            matte = TF.resized_crop(matte, i, j, h, w, (sz, sz), Image.NEAREST)
        else:
            img   = TF.resize(img,   (sz, sz), Image.BILINEAR)
            matte = TF.resize(matte, (sz, sz), Image.NEAREST)

        # ── To tensor ─────────────────────────────────────────
        img_t   = TF.to_tensor(img)                          # (3, H, W) [0,1]
        img_t   = TF.normalize(img_t,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        matte_t = torch.from_numpy(
            np.array(matte, dtype=np.float32) / 255.0
        ).unsqueeze(0)                                        # (1, H, W) [0,1]

        return img_t, matte_t

    @staticmethod
    def _random_crop_params(img, output_size):
        w, h   = img.size
        scale  = random.uniform(0.8, 1.0)
        new_h  = int(h * scale)
        new_w  = int(w * scale)
        top    = random.randint(0, max(0, h - new_h))
        left   = random.randint(0, max(0, w - new_w))
        return top, left, new_h, new_w


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    img_dir   = os.path.join(args.data, "clip_img")
    matte_dir = os.path.join(args.data, "matting")

    full_ds = AISegmentDataset(img_dir, matte_dir,
                               target_size=args.size, augment=True)
    val_n   = max(1, int(len(full_ds) * 0.1))
    train_n = len(full_ds) - val_n
    g       = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [train_n, val_n], generator=g)

    # Disable augmentation for val split via a wrapper
    val_ds.dataset.augment = False  # type: ignore

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4)

    print(f"Train: {train_n}  Val: {val_n}")

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
        va_iou  /= val_n

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
