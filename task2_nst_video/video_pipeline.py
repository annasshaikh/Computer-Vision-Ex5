"""
video_pipeline.py — Full video stylization pipeline.

Steps:
  1. Decode input_video.mp4 into frames via OpenCV.
  2. For each frame:
       a. Run MattingUNet to obtain alpha matte α_t.
       b. Run NST to obtain stylized frame S_t
          (initialized from previous stylized frame for temporal consistency).
  3. Composite:
       • Variant 1 (bg_stylized)  : O = α·F + (1-α)·S
       • Variant 2 (fg_stylized)  : O = α·S + (1-α)·F
       • Variant 3 (full_stylized): O = S
  4. Re-encode each variant to .mp4.

Usage:
    python video_pipeline.py \
        --video   input_video.mp4     \
        --style   style/starry_night.jpg \
        --matting matting/weights/matting_best.pth \
        --out_dir outputs
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from matting.model import MattingUNet
from nst import run_nst, tensor_to_pil, load_image, STYLE_LAYERS

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def frames_from_video(video_path: str) -> tuple[list[np.ndarray], float]:
    """Returns (frames_bgr, fps)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"  Decoded {len(frames)} frames @ {fps:.1f} fps")
    return frames, fps


def frames_to_video(frames_rgb: list[np.ndarray], out_path: str, fps: float):
    """Encode list of RGB uint8 frames to mp4."""
    if not frames_rgb:
        print(f"  [warn] No frames to encode for {out_path}")
        return
    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames_rgb:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Video saved → {out_path}")


def bgr_to_tensor(frame_bgr: np.ndarray, size: int,
                  device: torch.device) -> torch.Tensor:
    """BGR uint8 → normalised float tensor (1, 3, size, size)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((size, size), Image.LANCZOS)
    t   = TF.to_tensor(pil)
    t   = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return t.unsqueeze(0).to(device)


def tensor_to_rgb_np(t: torch.Tensor) -> np.ndarray:
    """Normalised (1,3,H,W) tensor → RGB uint8 (H,W,3)."""
    t = t.squeeze(0).cpu()
    t = (t * STD.squeeze(0) + MEAN.squeeze(0)).clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_matting_overlay(frames_rgb, mattes, out_path, n=5):
    """Save n-frame matting visualisation: original | matte | cutout."""
    n = min(n, len(frames_rgb))
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]

    import matplotlib.pyplot as plt  # local import to keep top clean
    for i in range(n):
        frame  = frames_rgb[i]
        alpha  = (mattes[i] * 255).astype(np.uint8)
        cutout = frame.copy()
        bg_mask = mattes[i] < 0.5
        cutout[bg_mask] = 0

        axes[i][0].imshow(frame);   axes[i][0].set_title("Frame");   axes[i][0].axis("off")
        axes[i][1].imshow(alpha, cmap="gray"); axes[i][1].set_title("Alpha"); axes[i][1].axis("off")
        axes[i][2].imshow(cutout);  axes[i][2].set_title("Cutout");  axes[i][2].axis("off")

    plt.suptitle("Human Matting Visualisation", fontsize=12)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Matting overlay → {out_path}")


# ──────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────
def run_pipeline(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load matting model ────────────────────────────────────
    matting_model = MattingUNet(pretrained=False).to(device)
    matting_model.load_state_dict(
        torch.load(args.matting, map_location=device)
    )
    matting_model.eval()
    print("  Matting model loaded.")

    # ── Decode video ──────────────────────────────────────────
    frames_bgr, fps = frames_from_video(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each frame ───────────────────────────────────
    frames_orig_rgb: list[np.ndarray] = []
    mattes_np:       list[np.ndarray] = []
    frames_stylized: list[np.ndarray] = []

    prev_stylized_tensor = None   # temporal consistency seed

    for idx, frame_bgr in enumerate(frames_bgr):
        print(f"\r  Processing frame {idx+1}/{len(frames_bgr)}", end="", flush=True)

        # Save temp content frame for NST
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames_orig_rgb.append(rgb_frame)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            content_tmp = tf.name
        Image.fromarray(rgb_frame).save(content_tmp)

        # ── Matting ──────────────────────────────────────────
        frame_tensor = bgr_to_tensor(frame_bgr, args.mat_size, device)
        with torch.no_grad():
            alpha_t = matting_model(frame_tensor)          # (1,1,H,W) [0,1]
        alpha_np = alpha_t.squeeze().cpu().numpy()         # (H,W)
        # Resize to original frame size
        orig_h, orig_w = frame_bgr.shape[:2]
        alpha_np = cv2.resize(alpha_np, (orig_w, orig_h), cv2.INTER_LINEAR)
        mattes_np.append(alpha_np)

        # ── NST ──────────────────────────────────────────────
        stylized_tensor, _ = run_nst(
            content_path  = content_tmp,
            style_path    = args.style,
            output_path   = None,
            beta          = args.beta,
            n_steps       = args.nst_steps,
            img_size      = args.nst_size,
            device        = device,
            init_tensor   = prev_stylized_tensor,   # temporal consistency
            verbose       = False,
        )
        prev_stylized_tensor = stylized_tensor.clone()
        os.unlink(content_tmp)

        # Resize stylized to original size
        stylized_np = tensor_to_rgb_np(stylized_tensor)
        stylized_np = cv2.resize(stylized_np, (orig_w, orig_h), cv2.INTER_LINEAR)
        frames_stylized.append(stylized_np)

    print()  # newline after progress

    # ── Matting overlay visualisation ─────────────────────────
    save_matting_overlay(frames_orig_rgb, mattes_np,
                         str(out_dir / "matting_overlay.png"))

    # ── Composite ─────────────────────────────────────────────
    alpha_3ch_list = [np.stack([m, m, m], axis=-1) for m in mattes_np]

    out_bg, out_fg, out_full = [], [], []
    for F_t, S_t, A_t in zip(frames_orig_rgb, frames_stylized, alpha_3ch_list):
        F_f = F_t.astype(np.float32) / 255.0
        S_f = S_t.astype(np.float32) / 255.0

        # Variant 1: bg stylized, subject natural
        bg   = (A_t * F_f + (1 - A_t) * S_f).clip(0, 1)
        out_bg.append((bg * 255).astype(np.uint8))

        # Variant 2: subject stylized, bg natural
        fg   = (A_t * S_f + (1 - A_t) * F_f).clip(0, 1)
        out_fg.append((fg * 255).astype(np.uint8))

        # Variant 3: full frame stylized (no matting)
        out_full.append(S_t)

    # ── Encode ───────────────────────────────────────────────
    frames_to_video(out_bg,   str(out_dir / "stylized_background.mp4"), fps)
    frames_to_video(out_fg,   str(out_dir / "stylized_subject.mp4"),    fps)
    frames_to_video(out_full, str(out_dir / "stylized_full.mp4"),       fps)

    # ── Branded poster (best frame) ───────────────────────────
    mid_idx = len(out_bg) // 2
    poster  = Image.fromarray(out_bg[mid_idx]).resize((1024, 1024), Image.LANCZOS)
    poster.save(str(out_dir / "branded_poster.png"))
    print(f"  Branded poster → {out_dir / 'branded_poster.png'}")


# ──────────────────────────────────────────────────────────────
# Content frame extractor
# ──────────────────────────────────────────────────────────────
def extract_content_frames(video_path: str, out_dir: str, n: int = 5):
    """Extract n evenly-spaced frames from the video."""
    frames, fps = frames_from_video(video_path)
    total = len(frames)
    indices = np.linspace(0, total - 1, n, dtype=int)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, idx in enumerate(indices):
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(str(out_dir / f"frame_{k+1:02d}.jpg"))
    print(f"  Extracted {n} content frames → {out_dir}")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",    default="input_video.mp4")
    parser.add_argument("--style",    default="style/starry_night.jpg")
    parser.add_argument("--matting",  default="matting/weights/matting_best.pth")
    parser.add_argument("--out_dir",  default="outputs")
    parser.add_argument("--beta",     type=float, default=1e5)
    parser.add_argument("--nst_steps",type=int,   default=300)
    parser.add_argument("--nst_size", type=int,   default=512)
    parser.add_argument("--mat_size", type=int,   default=256)
    parser.add_argument("--extract_frames", action="store_true",
                        help="Only extract content frames from video")
    parser.add_argument("--n_frames", type=int, default=5)
    args = parser.parse_args()

    if args.extract_frames:
        extract_content_frames(args.video, "task2_nst_video/content", args.n_frames)
    else:
        run_pipeline(args)
