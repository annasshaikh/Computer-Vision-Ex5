"""
nst.py — Neural Style Transfer (Gatys, Ecker, Bethge 2015).

Usage:
    # Single transfer
    python nst.py --content content/frame_01.jpg --style style/starry_night.jpg

    # β/α ratio sweep
    python nst.py --content content/frame_01.jpg --style style/starry_night.jpg \
                  --sweep_ratios

    # Layer ablation (shallow vs deep)
    python nst.py --content content/frame_01.jpg --style style/starry_night.jpg \
                  --layer_ablation

    # 5×3 grid (5 content × 3 style)
    python nst.py --grid

    # Feature-map visualisation
    python nst.py --feature_maps --image content/frame_01.jpg
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ──────────────────────────────────────────────────────────────
# Config defaults
# ──────────────────────────────────────────────────────────────
CONTENT_LAYER = "relu4_2"
STYLE_LAYERS  = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
SHALLOW_LAYERS = ["relu1_1", "relu2_1"]
DEEP_LAYERS    = ["relu4_1", "relu5_1"]
IMG_SIZE       = 512
N_STEPS        = 500
ALPHA          = 1.0          # content weight
DEFAULT_BETA   = 1e5          # style weight

# ImageNet stats
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ──────────────────────────────────────────────────────────────
# Image I/O
# ──────────────────────────────────────────────────────────────
def load_image(path: str, size: int = IMG_SIZE,
               device: torch.device = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    t   = TF.to_tensor(img).unsqueeze(0)
    t   = TF.normalize(t.squeeze(0), [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]).unsqueeze(0)
    if device:
        t = t.to(device)
    return t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Un-normalise and convert to PIL."""
    t = t.squeeze(0).cpu()
    t = t * STD + MEAN
    t = t.clamp(0, 1)
    return TF.to_pil_image(t)


def save_image(t: torch.Tensor, path: str):
    tensor_to_pil(t).save(path)


# ──────────────────────────────────────────────────────────────
# VGG19 feature extractor
# ──────────────────────────────────────────────────────────────
# Mapping from friendly name → VGG19 sequential layer index
VGG_LAYER_MAP = {
    "relu1_1":  1,  "relu1_2":  3,
    "relu2_1":  6,  "relu2_2":  8,
    "relu3_1": 11,  "relu3_2": 13, "relu3_3": 15, "relu3_4": 17,
    "relu4_1": 20,  "relu4_2": 22, "relu4_3": 24, "relu4_4": 26,
    "relu5_1": 29,  "relu5_2": 31, "relu5_3": 33, "relu5_4": 35,
}


class VGGFeatures(nn.Module):
    def __init__(self, layers: list[str], device):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features.to(device)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        self.slices = nn.ModuleList()
        self.layer_names = layers
        indices = sorted(set([VGG_LAYER_MAP[l] for l in layers]))

        # Build slices
        prev = 0
        self._slice_to_layer = {}
        for idx in indices:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:idx + 1]))
            # Record which layer names map to this slice
            for ln in layers:
                if VGG_LAYER_MAP[ln] == idx:
                    self._slice_to_layer[len(self.slices) - 1] = ln
            prev = idx + 1

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for i, s in enumerate(self.slices):
            x = s(x)
            if i in self._slice_to_layer:
                out[self._slice_to_layer[i]] = x
        return out


# ──────────────────────────────────────────────────────────────
# Gram matrix
# ──────────────────────────────────────────────────────────────
def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)
    G = torch.bmm(f, f.transpose(1, 2))
    return G / (C * H * W)


# ──────────────────────────────────────────────────────────────
# NST core
# ──────────────────────────────────────────────────────────────
def run_nst(content_path: str, style_path: str,
            output_path: str,
            alpha: float = ALPHA,
            beta:  float = DEFAULT_BETA,
            n_steps: int = N_STEPS,
            img_size: int = IMG_SIZE,
            device: torch.device = None,
            init_tensor: torch.Tensor = None,
            style_layers: list[str] = None,
            verbose: bool = True) -> tuple[torch.Tensor, list[float]]:
    """
    Run NST and save result. Returns (generated_tensor, loss_history).

    init_tensor: if provided, used as initialization (temporal consistency).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_layers = list(set([CONTENT_LAYER] + (style_layers or STYLE_LAYERS)))
    extractor  = VGGFeatures(all_layers, device)

    content_img = load_image(content_path, img_size, device)
    style_img   = load_image(style_path,   img_size, device)

    with torch.no_grad():
        content_feats = extractor(content_img)
        style_feats   = extractor(style_img)
        style_grams   = {l: gram_matrix(style_feats[l])
                         for l in (style_layers or STYLE_LAYERS)}

    # Initialize generated image
    if init_tensor is not None:
        gen = init_tensor.clone().to(device).requires_grad_(True)
    else:
        gen = content_img.clone().requires_grad_(True)

    optimizer = optim.LBFGS([gen], max_iter=20)

    step = [0]
    loss_history = []

    def closure():
        optimizer.zero_grad()
        gen.data.clamp_(
            ((0 - torch.tensor([0.485, 0.456, 0.406])) /
             torch.tensor([0.229, 0.224, 0.225])).min().item(),
            ((1 - torch.tensor([0.485, 0.456, 0.406])) /
             torch.tensor([0.229, 0.224, 0.225])).max().item()
        )
        feats = extractor(gen)

        # Content loss
        c_loss = F.mse_loss(feats[CONTENT_LAYER],
                            content_feats[CONTENT_LAYER].detach())

        # Style loss
        s_loss = 0.0
        sl = style_layers or STYLE_LAYERS
        for l in sl:
            if l not in feats:
                continue
            G_gen    = gram_matrix(feats[l])
            G_target = style_grams[l].detach()
            s_loss  += F.mse_loss(G_gen, G_target)
        s_loss /= len(sl)

        loss = alpha * c_loss + beta * s_loss
        loss.backward()

        loss_history.append(loss.item())
        step[0] += 1
        if verbose and step[0] % 50 == 0:
            print(f"    step {step[0]:4d}  total={loss.item():.2e}  "
                  f"c={c_loss.item():.2e}  s={s_loss.item():.2e}")
        return loss

    import torch.nn.functional as F  # noqa: re-import for closure scope

    for _ in range(n_steps // 20):
        optimizer.step(closure)

    # Final clamp
    with torch.no_grad():
        gen.clamp_(
            ((0 - 0.485) / 0.229),
            ((1 - 0.406) / 0.225)
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_image(gen, output_path)
        if verbose:
            print(f"  Saved → {output_path}")

    return gen.detach(), loss_history


# ──────────────────────────────────────────────────────────────
# β/α sweep
# ──────────────────────────────────────────────────────────────
def sweep_beta_alpha(content_path, style_path, out_dir, ratios=(1e3, 1e5, 1e7)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgs = []
    for r in ratios:
        out_path = str(out_dir / f"ba_ratio_{r:.0e}.png")
        run_nst(content_path, style_path, out_path, beta=r, device=device)
        imgs.append((r, out_path))

    # Composite
    fig, axes = plt.subplots(1, len(ratios), figsize=(len(ratios) * 5, 5))
    content_pil = Image.open(content_path).resize((512, 512))
    for ax, (r, p) in zip(axes, imgs):
        ax.imshow(Image.open(p))
        ax.set_title(f"β/α = {r:.0e}", fontsize=10)
        ax.axis("off")
    plt.suptitle("β/α Ratio Ablation", fontsize=14)
    plt.tight_layout()
    ablation_path = str(out_dir / "beta_alpha_ablation.png")
    plt.savefig(ablation_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  β/α ablation → {ablation_path}")


# ──────────────────────────────────────────────────────────────
# Layer ablation
# ──────────────────────────────────────────────────────────────
def layer_ablation(content_path, style_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variants = {
        "shallow_layers": SHALLOW_LAYERS,
        "deep_layers":    DEEP_LAYERS,
        "all_layers":     STYLE_LAYERS,
    }
    paths = {}
    for name, layers in variants.items():
        p = str(out_dir / f"layer_{name}.png")
        run_nst(content_path, style_path, p, style_layers=layers, device=device)
        paths[name] = p

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = {
        "shallow_layers": "Shallow layers\n(relu1_1, relu2_1)",
        "deep_layers":    "Deep layers\n(relu4_1, relu5_1)",
        "all_layers":     "All layers\n(relu1-5)",
    }
    for ax, (name, p) in zip(axes, paths.items()):
        ax.imshow(Image.open(p))
        ax.set_title(titles[name], fontsize=10)
        ax.axis("off")
    plt.suptitle("Layer Ablation Study", fontsize=14)
    plt.tight_layout()
    ablation_path = str(out_dir / "layer_ablation.png")
    plt.savefig(ablation_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Layer ablation → {ablation_path}")


# ──────────────────────────────────────────────────────────────
# 5×3 grid
# ──────────────────────────────────────────────────────────────
def build_grid(content_dir, style_dir, out_dir):
    content_paths = sorted(Path(content_dir).glob("*.jpg"))[:5]
    style_paths   = sorted(Path(style_dir).glob("*"))
    style_paths   = [p for p in style_paths
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")][:3]

    if not content_paths:
        print(f"[warn] No content images found in {content_dir}")
        return
    if not style_paths:
        print(f"[warn] No style images found in {style_dir}")
        return

    out_dir = Path(out_dir)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows, cols = len(content_paths), len(style_paths)
    fig, axes  = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    for i, cp in enumerate(content_paths):
        for j, sp in enumerate(style_paths):
            tag = f"c{i+1}_s{j+1}"
            op  = str(out_dir / f"grid_{tag}.png")
            run_nst(str(cp), str(sp), op, device=device, verbose=False)
            axes[i][j].imshow(Image.open(op))
            axes[i][j].axis("off")
            if i == 0:
                axes[i][j].set_title(sp.stem, fontsize=8)
            if j == 0:
                axes[i][j].set_ylabel(cp.stem, fontsize=8)

    plt.suptitle("NST Grid: 5 Content × 3 Style", fontsize=14)
    plt.tight_layout()
    grid_path = str(out_dir / "grid.png")
    plt.savefig(grid_path, dpi=80, bbox_inches="tight")
    plt.close()
    print(f"  Grid saved → {grid_path}")


# ──────────────────────────────────────────────────────────────
# Feature-map visualisation
# ──────────────────────────────────────────────────────────────
def visualise_feature_maps(image_path: str, out_path: str,
                            shallow_layer="relu1_1", deep_layer="relu4_1",
                            n_channels: int = 8):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = VGGFeatures([shallow_layer, deep_layer], device)
    img       = load_image(image_path, IMG_SIZE, device)

    with torch.no_grad():
        feats = extractor(img)

    fig, axes = plt.subplots(2, n_channels, figsize=(n_channels * 2, 5))
    for row, (layer, title) in enumerate([(shallow_layer, f"Shallow ({shallow_layer})"),
                                          (deep_layer,    f"Deep ({deep_layer})")]):
        feat = feats[layer][0].cpu().numpy()  # (C, H, W)
        for col in range(n_channels):
            ch  = feat[col % feat.shape[0]]
            ch  = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            axes[row][col].imshow(ch, cmap="viridis")
            axes[row][col].set_title(f"ch{col}", fontsize=7)
            axes[row][col].axis("off")
        axes[row][0].set_ylabel(title, fontsize=9)

    plt.suptitle(f"Feature maps: {Path(image_path).name}", fontsize=11)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Feature maps → {out_path}")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch.nn.functional as F  # noqa

    parser = argparse.ArgumentParser()
    parser.add_argument("--content",       default="content/frame_01.jpg")
    parser.add_argument("--style",         default="style/starry_night.jpg")
    parser.add_argument("--output",        default="outputs/stylized.png")
    parser.add_argument("--beta",          type=float, default=DEFAULT_BETA)
    parser.add_argument("--steps",         type=int,   default=N_STEPS)
    parser.add_argument("--size",          type=int,   default=IMG_SIZE)
    parser.add_argument("--sweep_ratios",  action="store_true")
    parser.add_argument("--layer_ablation",action="store_true")
    parser.add_argument("--grid",          action="store_true")
    parser.add_argument("--feature_maps",  action="store_true")
    parser.add_argument("--image",         default="content/frame_01.jpg",
                        help="Image for feature-map viz")
    parser.add_argument("--out_dir",       default="outputs")
    args = parser.parse_args()

    if args.sweep_ratios:
        sweep_beta_alpha(args.content, args.style, args.out_dir)
    elif args.layer_ablation:
        layer_ablation(args.content, args.style, args.out_dir)
    elif args.grid:
        build_grid("content", "style", args.out_dir)
    elif args.feature_maps:
        visualise_feature_maps(
            args.image,
            os.path.join(args.out_dir, "feature_maps.png")
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_nst(args.content, args.style, args.output,
                beta=args.beta, n_steps=args.steps,
                img_size=args.size, device=device)
