"""
evaluate.py — Evaluation for Task 1.

Produces:
  • Accuracy, MAE, RMSE via calculate_metrics()
  • Confusion matrix PNG
  • Failure-case analysis against failure_cases.json from Assignment 2
  • Comparison table CSV
  • Prediction visualisation on failure cases
  • Loss / accuracy curve plots

Usage:
    python evaluate.py --model a --weights weights/BaselineCNN_adam_best.pth
    python evaluate.py --model b --weights weights/DeepRegCNN_adam_best.pth
    python evaluate.py --compare   # produce full comparison table
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data   import build_dataloaders, load_config, seed_everything
from models import build_model


# ──────────────────────────────────────────────────────────────
# Metrics (matches Assignment-2 interface)
# ──────────────────────────────────────────────────────────────
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes MSE and MAE for regression.
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"mse": mse, "mae": mae}


# ──────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        preds = model(imgs).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(targets.numpy())
    return np.concatenate(all_labels).flatten(), np.concatenate(all_preds).flatten()


# ──────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────
def plot_regression_scatter(y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5)
    # Perfect prediction line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Regression: Predicted vs True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Scatter plot → {out_path}")


def plot_training_curves(log_csv: str, out_path: str):
    if not os.path.exists(log_csv):
        print(f"  [warn] No training log found at {log_csv}")
        return
    epochs, tr_mse, va_mse, va_mae = [], [], [], []
    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr_mse.append(float(row["train_mse"]))
            va_mse.append(float(row["val_mse"]))
            va_mae.append(float(row["val_mae"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, tr_mse, label="Train MSE")
    ax1.plot(epochs, va_mse, label="Val MSE")
    ax1.set_title("Training MSE")
    ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, va_mae, label="Val MAE", color="tab:green")
    ax2.set_title("Validation MAE")
    ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Training curves → {out_path}")


def plot_failure_case_predictions(model, failure_indices, dataset, device,
                                  out_path, max_show=12):
    """
    Show original images with predicted vs true labels.
    failure_indices: list of 0-indexed dataset positions that were
                     in the failure_cases.json from Assignment 2.
    """
    import torchvision.transforms.functional as TF
    from data import get_val_transform

    model.eval()
    show_idx = failure_indices[:max_show]
    cols = min(6, len(show_idx))
    rows = (len(show_idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(-1)

    for k, idx in enumerate(show_idx):
        img_tensor, true_target = dataset[idx]
        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(device)).item()
        # Un-normalize for display if desired, or show raw normalized
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_show = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[k].imshow(img_show)
        true_val = true_target.item()
        axes[k].set_title(f"T:{true_val:.2f}\nP:{pred:.2f}", fontsize=8)
        axes[k].axis("off")

    for k in range(len(show_idx), len(axes)):
        axes[k].axis("off")

    plt.suptitle("Failure Cases from A2 — CNN Predictions", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Failure case plot → {out_path}")


# ──────────────────────────────────────────────────────────────
# Failure-cases helper
# ──────────────────────────────────────────────────────────────
def load_failure_cases(path: str = "failure_cases.json") -> list[int]:
    """
    Load A2 failure_cases.json.
    Expected format: list of dicts with 'image_id' (1-indexed class id).
    Returns list of 0-indexed class labels.
    """
    if not os.path.exists(path):
        print(f"  [warn] {path} not found — failure-case analysis skipped.")
        return []
    with open(path) as f:
        data = json.load(f)
    # Support two common formats
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            return [int(d.get("image_id", d.get("class_id", 1))) - 1
                    for d in data]
        else:
            return [int(v) - 1 for v in data]
    return []


def count_failure_cases_fixed(y_true, y_pred, failure_class_labels: list) -> str:
    """
    Among the failure cases (identified by class label), how many does
    the current method now get right?
    """
    if not failure_class_labels:
        return "N/A"
    fixed, total = 0, 0
    for cls_label in failure_class_labels:
        mask = y_true == cls_label
        if mask.sum() == 0:
            continue
        total += mask.sum()
        fixed += (y_pred[mask] == y_true[mask]).sum()
    return f"{int(fixed)} / {int(total)}"


# ──────────────────────────────────────────────────────────────
# Per-model evaluation driver
# ──────────────────────────────────────────────────────────────
def evaluate_model(cfg, model_key, weights_path, out_dir,
                   failure_cls_labels=None):
    seed_everything(cfg["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, n_cls = build_dataloaders(cfg)
    model = build_model(cfg, model_key, n_cls).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    y_true, y_pred = get_predictions(model, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred)
    print(f"\n  {cfg[model_key]['name']} \u2014 {metrics}")

    # Scatter plot
    plot_regression_scatter(y_true, y_pred,
                            out_dir / f"{cfg[model_key]['name']}_scatter.png")

    return {
        "model": cfg[model_key]["name"],
        "mse":   f"{metrics['mse']:.6f}",
        "mae":   f"{metrics['mae']:.6f}",
    }


# ──────────────────────────────────────────────────────────────
# Comparison table
# ──────────────────────────────────────────────────────────────
def build_comparison_table(rows: list[dict], out_path: str):
    fieldnames = ["Method", "MSE", "MAE"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Comparison table \u2192 {out_path}")
    # Pretty-print
    col_w = [25, 12, 12]
    header = "  " + "".join(h.ljust(w) for h, w in zip(fieldnames, col_w))
    print(header)
    print("  " + "-" * sum(col_w))
    for row in rows:
        line = "  " + "".join(
            str(row[k]).ljust(w)
            for k, w in zip(fieldnames, col_w)
        )
        print(line)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["a", "b"], default="a")
    parser.add_argument("--weights", default=None,
                        help="Path to .pth weights (auto-detected if omitted)")
    parser.add_argument("--compare", action="store_true",
                        help="Build full comparison table for all methods")
    parser.add_argument("--log_csv", default=None,
                        help="Path to training_log.csv for curve plots")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--failure_cases", default="failure_cases.json")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    out_dir = Path(cfg["logging"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    failure_labels = load_failure_cases(args.failure_cases)

    if args.compare:
        rows = []
        for mkey in ["model_a", "model_b"]:
            wt_dir  = Path(cfg["logging"]["weights_dir"])
            run_name = cfg[mkey]["name"] + "_adam_standard"
            wt_path  = wt_dir / f"{run_name}_best.pth"
            if not wt_path.exists():
                 wt_path = wt_dir / f"{cfg[mkey]['name']}_best.pth"
            if wt_path.exists():
                r = evaluate_model(cfg, mkey, str(wt_path), out_dir)
                rows.append({
                    "Method": r["model"],
                    "MSE":    r["mse"],
                    "MAE":    r["mae"],
                })
            else:
                print(f"  [warn] Weights not found for {mkey}: {wt_path}")
        build_comparison_table(rows, out_dir / "comparison_table.csv")

    else:
        mkey    = "model_a" if args.model == "a" else "model_b"
        wt_path = args.weights
        if wt_path is None:
            wt_dir   = Path(cfg["logging"]["weights_dir"])
            run_name = cfg[mkey]["name"] + "_adam_standard"
            wt_path  = str(wt_dir / f"{run_name}_best.pth")

        evaluate_model(cfg, mkey, wt_path, out_dir)

        # Training curves
        log_csv = args.log_csv or str(
            Path(cfg["logging"]["log_dir"]) /
            f"{cfg[mkey]['name']}_adam_standard" / "training_log.csv"
        )
        plot_training_curves(log_csv,
                             str(out_dir / f"{cfg[mkey]['name']}_curves.png"))


if __name__ == "__main__":
    main()
