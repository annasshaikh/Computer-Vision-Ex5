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
from PIL import Image

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

# Global output directory (e.g. for Kaggle)
OUTPUT_DIR = Path("/kaggle/working/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Helpers for Task 2
# ----------------------------------------------------------------------
def plot_matting_curves(log_csv, out_path):
    """Plots training/validation loss and IoU for the matting model."""
    if not os.path.exists(log_csv):
        print(f"  [warn] Matting log not found: {log_csv}")
        return
    epochs, tr_loss, va_loss, va_iou = [], [], [], []
    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"]))
            va_loss.append(float(row["val_loss"]))
            va_iou.append(float(row["val_iou"]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, tr_loss, label="Train Loss")
    ax1.plot(epochs, va_loss, label="Val Loss")
    ax1.set_title("Matting Training Loss")
    ax1.set_xlabel("Epoch"); ax1.legend()
    
    ax2.plot(epochs, va_iou, label="Val IoU", color="tab:green")
    ax2.set_title("Matting Validation IoU")
    ax2.set_xlabel("Epoch"); ax2.legend()
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Matting curves saved: {out_path}")

def plot_nst_loss(loss_history, out_path):
    """Plots the optimization loss curve for NST."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.title("NST Optimization Loss (log scale)")
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"NST loss curve saved: {out_path}")

# ----------------------------------------------------------------------
# Helpers for Task 1
# ----------------------------------------------------------------------
def plot_inference_examples(model, dataset, device, out_path, num_samples=5):
    """Visualizes model predictions on sample images from the dataset."""
    model.eval()
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 4))
    if num_samples == 1: axes = [axes]
    
    for i in range(num_samples):
        img_tensor, true_val = dataset[i]
        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(device)).item()
        
        # Un-normalize for visualization (assuming ImageNet stats used in task1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_show = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_show)
        axes[i].set_title(f"True: {true_val.item():.2f}\nPred: {pred:.2f}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Inference examples saved: {out_path}")

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
    log_dir = Path(cfg["logging"]["log_dir"])
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

    # Plot Model B training curves
    from task1_cnn import evaluate as ev_mod
    ev_mod.plot_training_curves(str(log_dir / "DeepRegCNN_adam_standard" / "training_log.csv"), 
                                str(OUTPUT_DIR / "DeepRegCNN_curves.png"))

    # ---- 1.3 Full evaluation & scatter plot ----
    # Use best model A from the sweep (we'll take the one trained with adam_standard)
    # Reload best model A weights
    from task1_cnn import evaluate as ev_mod   # requires evaluate.py in task1_cnn

    wt_dir = Path(cfg["logging"]["weights_dir"])
    best_a_path = wt_dir / "BaselineCNN_adam_standard_best.pth"
    if best_a_path.exists():
        model_a = build_model(cfg, "model_a", n_out).to(device)
        model_a.load_state_dict(torch.load(best_a_path, map_location=device))
        y_true, y_pred = ev_mod.get_predictions(model_a, test_loader, device)
        scatter_path = OUTPUT_DIR / "scatter_A.png"
        ev_mod.plot_regression_scatter(y_true, y_pred, str(scatter_path))
        print(f"Scatter plot saved: {scatter_path}")

        # Plot Model A curves
        ev_mod.plot_training_curves(str(log_dir / "BaselineCNN_adam_standard" / "training_log.csv"), 
                                    str(OUTPUT_DIR / "BaselineCNN_curves.png"))

        # Save inference examples for 5 test images
        plot_inference_examples(model_a, test_loader.dataset, device, 
                                str(OUTPUT_DIR / "test_inference_samples.png"), num_samples=5)

        # Feature-map visualization for a Seed image (for Task 2 requirement)
        print("\n--- Feature-map visualization (Seed Image) ---")
        # Try to find task2's visualization function
        try:
            from task2_nst_video.nst import visualise_feature_maps
            seed_img_path = Path("seeds/1.jpg")
            if seed_img_path.exists():
                out_feat_seed = OUTPUT_DIR / "feature_maps_seed.png"
                visualise_feature_maps(str(seed_img_path), str(out_feat_seed))
                print(f"Seed feature maps saved: {out_feat_seed}")
        except ImportError:
            print("  [skip] task2_nst_video.nst not imported yet.")

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
        df.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)
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
    
    # Kaggle dataset paths (robust check)
    possible_ais = [
        Path("/kaggle/input/aisegmentcom-matting-human-datasets"),
        Path("/kaggle/input/datasets/laurentmih/aisegmentcom-matting-human-datasets")
    ]
    KAGGLE_AIS_DATA = next((p for p in possible_ais if p.exists()), Path("data/aisegment"))

    possible_vid = [
        Path("/kaggle/input/input4cv/input.mp4"),
        Path("/kaggle/input/datasets/muhammadannasshaikh/input4cv/input.mp4")
    ]
    KAGGLE_VIDEO = next((p for p in possible_vid if p.exists()), Path("task2_nst_video/input_video.mp4"))

    # Example images (all styles and available content)
    content_files = list(CONTENT_DIR.glob("*.jpg")) + list(CONTENT_DIR.glob("*.png"))
    
    # If content frames are missing, try to extract them from the video
    if not content_files and KAGGLE_VIDEO.exists():
        print(f"\n--- Extracting frames from {KAGGLE_VIDEO} ---")
        pipeline_script = Path("task2_nst_video/video_pipeline.py")
        if pipeline_script.exists():
            subprocess.run([sys.executable, str(pipeline_script), 
                            "--extract_frames", "--video", str(KAGGLE_VIDEO)], check=False)
            content_files = list(CONTENT_DIR.glob("*.jpg")) + list(CONTENT_DIR.glob("*.png"))

    style_files = [p for p in STYLE_DIR.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")]
    
    # If no styles found, download a couple of public domain ones automatically
    if not style_files:
        print("\n--- Downloading default style images ---")
        STYLE_DIR.mkdir(parents=True, exist_ok=True)
        import urllib.request
        styles_to_dl = {
            "starry_night.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
            "great_wave.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/The_Great_Wave_off_Kanagawa.jpg/1024px-The_Great_Wave_off_Kanagawa.jpg",
            "composition_viii.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Vassily_Kandinsky%2C_1923_-_Composition_8%2C_huile_sur_toile%2C_140_cm_x_201_cm%2C_Mus%C3%A9e_Guggenheim%2C_New_York.jpg/1024px-Vassily_Kandinsky%2C_1923_-_Composition_8%2C_huile_sur_toile%2C_140_cm_x_201_cm%2C_Mus%C3%A9e_Guggenheim%2C_New_York.jpg"
        }
        for name, url in styles_to_dl.items():
            try:
                urllib.request.urlretrieve(url, str(STYLE_DIR / name))
                print(f"Downloaded {name}")
            except Exception as e:
                print(f"Failed to download {name}: {e}")
        style_files = [p for p in STYLE_DIR.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")]

    if not content_files or not style_files:
        print("WARNING: No content or style images found. Skipping NST parts.")
    else:
        CONTENT = content_files[0]
        STYLE   = style_files[0]

        # ---- 2.1 NST sanity check ----
        print("\n--- NST single transfer (sanity) ---")
        out_path = OUTPUT_DIR / "test_stylized.png"
        if args.quick:
            _, losses = run_nst(str(CONTENT), str(STYLE), str(out_path), beta=1e5, n_steps=100)
        else:
            _, losses = run_nst(str(CONTENT), str(STYLE), str(out_path), beta=1e5, n_steps=200)
        print(f"Saved stylized image: {out_path}")
        
        # Plot NST loss curve
        plot_nst_loss(losses, str(OUTPUT_DIR / "nst_loss_curve.png"))

        # ---- 2.2 beta/alpha sweep ----
        print("\n--- Beta/Alpha sweep ---")
        sweep_beta_alpha(str(CONTENT), str(STYLE), out_dir=str(OUTPUT_DIR),
                         ratios=(1e3, 1e5, 1e7))
        print(f"Ablation plot: {OUTPUT_DIR}/beta_alpha_ablation.png")

        # ---- 2.3 Layer ablation ----
        print("\n--- Layer ablation (shallow vs deep) ---")
        layer_ablation(str(CONTENT), str(STYLE), out_dir=str(OUTPUT_DIR))
        print(f"Layer ablation plot: {OUTPUT_DIR}/layer_ablation.png")

        # ---- 2.4 NST Grid (All Styles x First 5 Content) ----
        if len(content_files) >= 1 and len(style_files) >= 1:
            print(f"\n--- NST Grid ({len(content_files[:5])} content x {len(style_files)} style) ---")
            build_grid(str(CONTENT_DIR), str(STYLE_DIR), str(OUTPUT_DIR))
            print(f"Grid saved: {OUTPUT_DIR}/grid.png")
        else:
            print(f"Skipping grid: need at least 1 content and 1 style image.")

    # ---- 2.5 Human Matting – load / train / visualise ----
    # Try to find weights in repo first, otherwise use OUTPUT_DIR
    REPO_MATTING_WEIGHTS = Path("task2_nst_video/matting/weights/matting_best.pth")
    if REPO_MATTING_WEIGHTS.exists():
        MATTING_WEIGHTS = REPO_MATTING_WEIGHTS
    else:
        MATTING_WEIGHTS = OUTPUT_DIR / "matting_best.pth"
    mat_model = None

    # If weights not found and user wants quick test, we can train a very minimal matting model?
    # Notebook shows training via external script; we call that script if desired.
    if not MATTING_WEIGHTS.exists():
        print("\n--- Matting weights not found. Training a small MattingUNet (30 epochs, quick mode) ---")
        # Call train.py if it exists
        train_script = Path("task2_nst_video/matting/train.py")
        if train_script.exists() and not args.quick:
            ais_path = KAGGLE_AIS_DATA if KAGGLE_AIS_DATA.exists() else Path("data/aisegment")
            subprocess.run([sys.executable, str(train_script),
                            "--data", str(ais_path),
                            "--epochs", "30",
                            "--out", str(OUTPUT_DIR)], check=False)
            # Plot training curves if log exists
            plot_matting_curves(str(OUTPUT_DIR / "matting_log.csv"), 
                                str(OUTPUT_DIR / "matting_curves.png"))
        else:
            print("  [skip] No matting weights and training script not found/quick mode.")
    else:
        print("\n--- Loading pretrained matting model ---")
        mat_model = MattingUNet(pretrained=False).to(device)
        mat_model.load_state_dict(torch.load(MATTING_WEIGHTS, map_location=device))
        mat_model.eval()
        print("Matting model loaded.")
        
        # Plot curves if log exists in the same directory as weights
        plot_matting_curves(str(Path(MATTING_WEIGHTS).parent / "matting_log.csv"), 
                            str(OUTPUT_DIR / "matting_curves.png"))

    # ---- 2.6 Sample NST Inference on multiple frames ----
    if content_files and style_files:
        print("\n--- NST inference on multiple samples ---")
        samples = content_files[:min(5, len(content_files))]
        fig, axes = plt.subplots(1, len(samples), figsize=(len(samples) * 4, 4))
        if len(samples) == 1: axes = [axes]
        
        for i, cp in enumerate(samples):
            out_p = OUTPUT_DIR / f"sample_stylized_{i}.png"
            # Use reduced steps for fast inference visualization
            run_nst(str(cp), str(STYLE), str(out_p), n_steps=100, verbose=False)
            axes[i].imshow(Image.open(out_p))
            axes[i].set_title(f"Frame {i}")
            axes[i].axis("off")
        
        plt.suptitle(f"NST Inference Samples (Style: {STYLE.name})")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "nst_inference_samples.png")
        plt.close()
        print(f"NST inference samples saved: {OUTPUT_DIR}/nst_inference_samples.png")

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
    VIDEO_PATH = KAGGLE_VIDEO if KAGGLE_VIDEO.exists() else Path("task2_nst_video/input_video.mp4")
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
                plt.savefig(OUTPUT_DIR / "video_variants_inference.png")
                plt.show()
                print(f"Video variant thumbnails saved: {OUTPUT_DIR}/video_variants_inference.png")
        else:
            print("video_pipeline.py not found – cannot run full pipeline.")
    else:
        print("\nSkipping full video pipeline: need input_video.mp4 and trained matting weights.")

    # ---- 2.9 Branded Poster (1024x1024 Stylized Still) ----
    if content_files and style_files:
        print("\n--- Generating 1024x1024 Branded Poster ---")
        poster_path = OUTPUT_DIR / "branded_poster.png"
        # Cherry-pick frame 0 or similar
        run_nst(str(CONTENT), str(STYLE), str(poster_path), 
                n_steps=200 if not args.quick else 50, 
                img_size=1024, verbose=True)
        print(f"Branded poster saved: {poster_path}")

    print("Task 2 completed.")

print("\nAssignment 5 finished successfully.")