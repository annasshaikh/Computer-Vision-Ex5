"""
data.py — Seeds dataset loader with augmentation.

Directory layout expected:
    seeds/
        1.jpg
        2.jpg
        ...
        150.jpg
"""

import os
import random
import numpy as np
import yaml
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


# ──────────────────────────────────────────────────────────────
# Seed everything
# ──────────────────────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class SeedsDataset(Dataset):
    """One image per class; label = filename stem (1-indexed)."""

    def __init__(self, root: str, image_size: int = 128,
                 transform=None, indices=None):
        self.root = Path(root)
        self.transform = transform
        self.image_size = image_size

        # Collect all images
        all_paths = sorted(
            self.root.glob("*.jpg"),
            key=lambda p: int(p.stem)
        )
        if not all_paths:
            raise FileNotFoundError(
                f"No .jpg images found in '{root}'. "
                "Check that the seeds/ folder is present."
            )

        self.paths = [all_paths[i] for i in indices] if indices is not None else all_paths
        # Labels for regression (normalized to 0-1 range based on max index 150)
        self.labels = [float(p.stem) / 150.0 for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Return label as float tensor for regression
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return img, label


# ──────────────────────────────────────────────────────────────
# Transform factories
# ──────────────────────────────────────────────────────────────
def get_train_transform(cfg: dict) -> T.Compose:
    aug = cfg["augmentation"]
    sz  = cfg["dataset"]["image_size"]
    zoom = aug["zoom"]

    transforms = [
        T.RandomResizedCrop(sz, scale=(1 - zoom, 1 + zoom)),
        T.RandomHorizontalFlip() if aug["horizontal_flip"] else T.Lambda(lambda x: x),
        T.RandomVerticalFlip()   if aug["vertical_flip"]   else T.Lambda(lambda x: x),
        T.RandomRotation(degrees=aug["rotation_degrees"]),
        T.ColorJitter(brightness=aug["brightness"]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(transforms)


def get_val_transform(cfg: dict) -> T.Compose:
    sz = cfg["dataset"]["image_size"]
    return T.Compose([
        T.Resize((sz, sz)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────
# Split builder
# ──────────────────────────────────────────────────────────────
def build_dataloaders(cfg: dict):
    """
    Returns (train_loader, val_loader, test_loader, num_outputs).
    """
    ds_cfg   = cfg["dataset"]
    root     = ds_cfg["root"]
    img_size = ds_cfg["image_size"]
    seed     = cfg["random_seed"]
    num_outputs = 1

    rng = np.random.RandomState(seed)

    # All indices 0..N-1
    # Use the actual number of files found in the directory for splitting
    all_paths = sorted(Path(root).glob("*.jpg"))
    total = len(all_paths)
    
    if total == 0:
        raise FileNotFoundError(f"No .jpg images found in '{root}'.")

    idx = np.arange(total)
    rng.shuffle(idx)

    val_n  = max(1, int(total * ds_cfg["val_split"]))
    test_n = max(1, int(total * ds_cfg["test_split"]))
    train_n = total - val_n - test_n

    train_idx = idx[:train_n].tolist()
    val_idx   = idx[train_n:train_n + val_n].tolist()
    test_idx  = idx[train_n + val_n:].tolist()

    train_ds = SeedsDataset(root, img_size, get_train_transform(cfg), train_idx)
    val_ds   = SeedsDataset(root, img_size, get_val_transform(cfg),   val_idx)
    test_ds  = SeedsDataset(root, img_size, get_val_transform(cfg),   test_idx)

    bs = cfg["training"]["batch_size"]
    g  = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=min(bs, len(train_ds)),
                              shuffle=True,  num_workers=2, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=min(bs, len(val_ds)),
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=min(bs, len(test_ds)),
                              shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, num_outputs


# ──────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config("config.yaml")
    seed_everything(cfg["random_seed"])
    train_loader, val_loader, test_loader, n_cls = build_dataloaders(cfg)
    print(f"Classes : {n_cls}")
    print(f"Train   : {len(train_loader.dataset)}")
    print(f"Val     : {len(val_loader.dataset)}")
    print(f"Test    : {len(test_loader.dataset)}")
    imgs, lbls = next(iter(train_loader))
    print(f"Batch   : {imgs.shape}, labels {lbls[:5]}")
