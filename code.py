#!/usr/bin/env python3
"""
run_assignment5.py

End-to-end execution of Assignment 5:
  - Task 1: CNN regression on Seeds dataset (BaselineCNN + DeepRegCNN)
  - Task 2: Neural Style Transfer, ablation studies, human matting, video pipeline

Usage:
    python run_assignment5.py [--quick] [--epochs N] [--skip_task1] [--skip_task2]

Options:
    --quick          Use only 5 epochs and small NST steps for a fast test
    --epochs N       Override training epochs (default from config.yaml)
    --skip_task1     Skip CNN regression part
    --skip_task2     Skip Neural Style Transfer & video part
"""

import sys
import os
import argparse
import subprocess
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

# ----------------------------------------------------------------------
# 0. argument parsing & device setup
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="fast demo mode")
parser.add_argument("--epochs", type=int, default=None, help="override training epochs")
parser.add_argument("--skip_task1", action="store_true")
parser.add_argument("--skip_task2", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# ----------------------------------------------------------------------
# 1. Task 1 – CNN regression (Seeds)
# ----------------------------------------------------------------------
if not args.skip_task1:
    print("\n" + "="*60)
    print("TASK 1 – CNN Regression on Seeds Dataset")
    print("="*60)

    # Add task1 module to path
    sys.path.insert(0, "task1_cnn")
    from task1_cnn.data import load_config, seed_everything, build_dataloaders
    from task1_cnn.models import BaselineCNN, DeepRegCNN, build_model
    import task1_cnn.train as trainer

    # Load configuration
    cfg = load_config("task1_cnn/config.yaml")
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    elif args.quick:
        cfg["training"]["epochs"] = 5
        cfg["training"]["early_stopping_patience"] = 3

    seed_everything(cfg["random_seed"])

    # Build data loaders
    train_loader, val_loader, test_loader, n_out = build_dataloaders(cfg)
    print(f"Output dimension: {n_out}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    # ---- 1.1 Optimizer sweep on BaselineCNN (Model A) ----
    print("\n--- Optimizer sweep (Model A, short run) ---")
    opts = ["adam_fast", "adam_standard", "adam_stable", "sgd_fast", "sgd_standard", "sgd_stable"]
    sweep_results = []
    for opt in opts:
        print(f"Training with {opt} ...")
        res = trainer.train(cfg, model_key="model_a", opt_override=opt)
        sweep_results.append(res)
        print(f"  Test MSE: {res['test_mse']:.6f}  MAE: {res['test_mae']:.6f}")

    # ---- 1.2 Train deeper model (Model B) ----
    print("\n--- Training DeepRegCNN (Model B) ---")
    res_b = trainer.train(cfg, model_key="model_b", opt_override="adam_standard")
    print(f"Model B test MSE: {res_b['test_mse']:.6f}")

    # ---- 1.3 Full evaluation & scatter plot ----
    # Use best model A from the sweep (we'll take the one trained with adam_standard)
    # Reload best model A weights
    from task1_cnn import evaluate as ev_mod   # requires evaluate.py in task1_cnn

    best_a_path = Path("task1_cnn/weights_regression/BaselineCNN_adam_standard_best.pth")
    if best_a_path.exists():
        model_a = build_model(cfg, "model_a", n_out).to(device)
        model_a.load_state_dict(torch.load(best_a_path, map_location=device))
        y_true, y_pred = ev_mod.get_predictions(model_a, test_loader, device)
        ev_mod.plot_regression_scatter(y_true, y_pred, "task1_cnn/cnn_outputs_reg/scatter_A.png")
        print("Scatter plot saved: task1_cnn/cnn_outputs_reg/scatter_A.png")

    # ---- 1.4 Build comparison table ----
    # Collect results from sweep and model B
    all_results = {}
    for r in sweep_results:
        all_results[r["optimizer"]] = {"mse": r["test_mse"], "mae": r["test_mae"]}
    all_results["DeepRegCNN"] = {"mse": res_b["test_mse"], "mae": res_b["test_mae"]}

    table_rows = []
    for name, metrics in all_results.items():
        table_rows.append({
            "Method": name,
            "MSE": f"{metrics['mse']:.6f}",
            "MAE": f"{metrics['mae']:.6f}",
        })
    # Use pandas to display table if available, otherwise print CSV
    try:
        import pandas as pd
        df = pd.DataFrame(table_rows)
        print("\n--- Final Regression Comparison ---")
        print(df.to_string(index=False))
        df.to_csv("task1_cnn/cnn_outputs_reg/comparison_table.csv", index=False)
    except ImportError:
        print("\nMethod, MSE, MAE")
        for row in table_rows:
            print(f"{row['Method']}, {row['MSE']}, {row['MAE']}")

    print("Task 1 completed.")

# ----------------------------------------------------------------------
# 2. Task 2 – Neural Style Transfer + Matting + Video Pipeline
# ----------------------------------------------------------------------
if not args.skip_task2:
    print("\n" + "="*60)
    print("TASK 2 – Neural Style Transfer & Video Stylization")
    print("="*60)

    sys.path.insert(0, "task2_nst_video")
    from task2_nst_video.nst import run_nst, sweep_beta_alpha, layer_ablation, build_grid
    from task2_nst_video.nst import visualise_feature_maps
    from task2_nst_video.matting.model import MattingUNet

    # Paths – adjust if needed
    CONTENT_DIR = Path("task2_nst_video/content")
    STYLE_DIR   = Path("task2_nst_video/style")
    OUTPUT_DIR  = Path("task2_nst_video/outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Example images (first content and first style)
    content_files = list(CONTENT_DIR.glob("*.jpg")) + list(CONTENT_DIR.glob("*.png"))
    style_files   = [p for p in STYLE_DIR.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")]
    if not content_files or not style_files:
        print("WARNING: No content or style images found. Skipping NST parts.")
    else:
        CONTENT = content_files[0]
        STYLE   = style_files[0]

        # ---- 2.1 NST sanity check ----
        print("\n--- NST single transfer (sanity) ---")
        out_path = OUTPUT_DIR / "test_stylized.png"
        if args.quick:
            run_nst(str(CONTENT), str(STYLE), str(out_path), beta=1e5, n_steps=100)
        else:
            run_nst(str(CONTENT), str(STYLE), str(out_path), beta=1e5, n_steps=200)
        print(f"Saved stylized image: {out_path}")

        # ---- 2.2 beta/alpha sweep ----
        print("\n--- Beta/Alpha sweep ---")
        sweep_beta_alpha(str(CONTENT), str(STYLE), out_dir=str(OUTPUT_DIR),
                         ratios=(1e3, 1e5, 1e7))
        print(f"Ablation plot: {OUTPUT_DIR}/beta_alpha_ablation.png")

        # ---- 2.3 Layer ablation ----
        print("\n--- Layer ablation (shallow vs deep) ---")
        layer_ablation(str(CONTENT), str(STYLE), out_dir=str(OUTPUT_DIR))
        print(f"Layer ablation plot: {OUTPUT_DIR}/layer_ablation.png")

        # ---- 2.4 5x3 NST grid ----
        if len(content_files) >= 5 and len(style_files) >= 3:
            print("\n--- 5x3 NST grid ---")
            build_grid(str(CONTENT_DIR), str(STYLE_DIR), str(OUTPUT_DIR))
            print(f"Grid saved: {OUTPUT_DIR}/grid.png")
        else:
            print(f"Skipping 5x3 grid: need 5 content (found {len(content_files)}) "
                  f"and 3 style (found {len(style_files)}) images.")

    # ---- 2.5 Human Matting – load / train / visualise ----
    MATTING_WEIGHTS = Path("task2_nst_video/matting/weights/matting_best.pth")
    mat_model = None

    # If weights not found and user wants quick test, we can train a very minimal matting model?
    # Notebook shows training via external script; we call that script if desired.
    if not MATTING_WEIGHTS.exists():
        print("\n--- Matting weights not found. Training a small MattingUNet (30 epochs, quick mode) ---")
        # Call train.py if it exists
        train_script = Path("task2_nst_video/matting/train.py")
        if train_script.exists() and not args.quick:
            subprocess.run([sys.executable, str(train_script),
                            "--data", "data/aisegment",
                            "--epochs", "30",
                            "--out", "task2_nst_video/matting/weights"], check=False)
        else:
            print("  [skip] No matting weights and training script not found/quick mode.")
    else:
        print("\n--- Loading pretrained matting model ---")
        mat_model = MattingUNet(pretrained=False).to(device)
        mat_model.load_state_dict(torch.load(MATTING_WEIGHTS, map_location=device))
        mat_model.eval()
        print("Matting model loaded.")

    # ---- 2.6 Matting visualisation on video frames ----
    if mat_model is not None and content_files:
        from PIL import Image
        import torchvision.transforms.functional as TF

        print("\n--- Matting visualisation on sample frames ---")
        sample_frames = content_files[:5]
        n_rows = len(sample_frames)
        fig, axes = plt.subplots(n_rows, 3, figsize=(9, n_rows*3))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for i, fp in enumerate(sample_frames):
            img = Image.open(fp).convert("RGB").resize((256, 256))
            t = TF.to_tensor(img)
            t = TF.normalize(t, [0.485,0.456,0.406], [0.229,0.224,0.225]).unsqueeze(0).to(device)
            with torch.no_grad():
                alpha = mat_model(t).squeeze().cpu().numpy()
            orig = np.array(img)
            cutout = orig.copy()
            cutout[alpha < 0.5] = 0
            axes[i,0].imshow(orig); axes[i,0].axis("off"); axes[i,0].set_title("Frame")
            axes[i,1].imshow(alpha, cmap="gray"); axes[i,1].axis("off"); axes[i,1].set_title("Alpha")
            axes[i,2].imshow(cutout); axes[i,2].axis("off"); axes[i,2].set_title("Cutout")
        plt.suptitle("Human Matting on Video Frames")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "matting_overlay.png")
        plt.show()
        print(f"Matting overlay saved: {OUTPUT_DIR}/matting_overlay.png")

    # ---- 2.7 Feature-map visualisation (VGG19) ----
    if content_files:
        print("\n--- Feature-map visualisation (VGG19) ---")
        frame_img = content_files[0]
        out_feat = OUTPUT_DIR / "feature_maps_video_frame.png"
        visualise_feature_maps(str(frame_img), str(out_feat))
        print(f"Feature maps saved: {out_feat}")

    # ---- 2.8 Full video pipeline (if video file and matting weights exist) ----
    VIDEO_PATH = Path("task2_nst_video/input_video.mp4")
    if VIDEO_PATH.exists() and MATTING_WEIGHTS.exists():
        print("\n--- Running full video stylization pipeline ---")
        pipeline_script = Path("task2_nst_video/video_pipeline.py")
        if pipeline_script.exists():
            # first extract frames (optional, pipeline may do it)
            if args.quick:
                nsteps = 100
            else:
                nsteps = 300
            cmd = [
                sys.executable, str(pipeline_script),
                "--video", str(VIDEO_PATH),
                "--style", str(STYLE),
                "--matting", str(MATTING_WEIGHTS),
                "--out_dir", str(OUTPUT_DIR),
                "--nst_steps", str(nsteps),
                "--beta", "1e5"
            ]
            subprocess.run(cmd, check=False)
            print("Video pipeline completed. Check output videos in:", OUTPUT_DIR)

            # Show thumbnails
            import cv2
            variants = [
                (OUTPUT_DIR / "stylized_background.mp4", "BG Stylized"),
                (OUTPUT_DIR / "stylized_subject.mp4", "Subject Stylized"),
                (OUTPUT_DIR / "stylized_full.mp4", "Full Stylized"),
            ]
            thumbs = []
            for vpath, title in variants:
                if vpath.exists():
                    cap = cv2.VideoCapture(str(vpath))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        thumbs.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), title))
            if thumbs:
                fig, axs = plt.subplots(1, len(thumbs), figsize=(15, 5))
                if len(thumbs) == 1:
                    axs = [axs]
                for ax, (frame, title) in zip(axs, thumbs):
                    ax.imshow(frame); ax.set_title(title); ax.axis("off")
                plt.suptitle("Stylized Video Variants (frame 30)")
                plt.tight_layout()
                plt.show()
        else:
            print("video_pipeline.py not found – cannot run full pipeline.")
    else:
        print("\nSkipping full video pipeline: need input_video.mp4 and trained matting weights.")

    print("Task 2 completed.")

print("\nAssignment 5 finished successfully.")