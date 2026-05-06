"""
train.py — End-to-end training for Task 1.

Usage:
    python train.py --model a          # train Model A (adam vs sgd comparison)
    python train.py --model b          # train Model B with winner optimizer
    python train.py --model a --opt sgd
    python train.py --all              # train everything in sequence
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from data   import build_dataloaders, load_config, seed_everything
from models import build_model


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def build_optimizer(model, cfg, model_key: str, opt_override: str = None):
    mcfg    = cfg[model_key]
    opt_name = opt_override or mcfg.get("optimizer", "adam_standard")
    
    if opt_name in cfg["optimizers"]:
        ocfg = cfg["optimizers"][opt_name]
        if "adam" in opt_name:
            return optim.Adam(model.parameters(),
                              lr=ocfg["lr"],
                              weight_decay=ocfg.get("weight_decay", 0.0))
        elif "sgd" in opt_name:
            return optim.SGD(model.parameters(),
                             lr=ocfg["lr"],
                             momentum=ocfg.get("momentum", 0.9))
    
    raise ValueError(f"Unknown optimizer: {opt_name}. Ensure it is defined in config.yaml.")


# --------------------------------------------------------------
# Train one epoch
# --------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total = 0.0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
    return total_loss / total


# --------------------------------------------------------------
# Evaluate
# --------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae, total = 0.0, 0.0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss  = criterion(preds, targets)
        total_loss += loss.item() * imgs.size(0)
        total_mae  += F.l1_loss(preds, targets).item() * imgs.size(0)
        total      += imgs.size(0)
    return total_loss / total, total_mae / total


# --------------------------------------------------------------
# --------------------------------------------------------------
# Main training function
# --------------------------------------------------------------
def train(cfg, model_key: str = "model_a", opt_override: str = None):
    seed_everything(cfg["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'-'*60}")
    print(f"  Training {cfg[model_key]['name']}  |  opt={opt_override or cfg[model_key].get('optimizer','adam_standard')}  |  device={device}")
    print(f"{'-'*60}")

    # Data
    train_loader, val_loader, test_loader, n_out = build_dataloaders(cfg)

    # Model
    model = build_model(cfg, model_key, n_out).to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, cfg, model_key, opt_override)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3)

    # Logging setup
    opt_tag   = opt_override or cfg[model_key].get("optimizer", "adam")
    run_name  = f"{cfg[model_key]['name']}_{opt_tag}"
    log_dir   = Path(cfg["logging"]["log_dir"]) / run_name
    wt_dir    = Path(cfg["logging"]["weights_dir"])
    wt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    csv_path = log_dir / "training_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_mse", "val_mse", "val_mae", "lr"])

    # Early stopping
    patience   = cfg["training"]["early_stopping_patience"]
    best_val_loss = float("inf")
    best_epoch    = 0
    epochs_no_improve = 0
    best_weights_path = wt_dir / f"{run_name}_best.pth"

    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        tr_mse = train_epoch(model, train_loader, criterion, optimizer, device)
        va_mse, va_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_mse)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"  Ep {epoch:03d}/{cfg['training']['epochs']}  "
              f"| tr_mse={tr_mse:.6f} | va_mse={va_mse:.6f} "
              f"| va_mae={va_mae:.6f} | lr={lr:.2e} ({elapsed:.1f}s)")

        writer.add_scalars("Loss", {"train": tr_mse, "val": va_mse}, epoch)
        writer.add_scalar("MAE", va_mae, epoch)
        csv_writer.writerow([epoch, tr_mse, va_mse, va_mae, lr])

        history["train_loss"].append(tr_mse)
        history["val_loss"].append(va_mse)
        history["val_mae"].append(va_mae)

        # Save best
        if va_mse < best_val_loss:
            best_val_loss = va_mse
            best_epoch    = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(best val_mse={best_val_loss:.6f} @ epoch {best_epoch})")
            break

    csv_file.close()
    writer.close()

    # Save final weights
    final_path = wt_dir / f"{run_name}_final.pth"
    torch.save(model.state_dict(), final_path)

    # Test evaluation with best weights
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    te_mse, te_mae = evaluate(model, test_loader, criterion, device)
    print(f"\n  Test — MSE={te_mse:.6f}  MAE={te_mae:.6f}")

    result = {
        "run_name":      run_name,
        "model":         cfg[model_key]["name"],
        "optimizer":     opt_tag,
        "best_epoch":    best_epoch,
        "best_val_mse":  best_val_loss,
        "test_mse":      te_mse,
        "test_mae":      te_mae,
        "history":       history,
    }

    result_path = log_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Weights saved to : {best_weights_path}")
    print(f"  Logs saved to    : {log_dir}")
    return result


# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b"], default="a")
    parser.add_argument("--opt",   choices=["adam", "sgd"], default=None)
    parser.add_argument("--all",   action="store_true",
                        help="Train Model A (adam+sgd) then Model B")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.all:
        opts = ["adam_fast", "adam_standard", "adam_stable", 
                "sgd_fast", "sgd_standard", "sgd_stable"]
        results = []
        for opt in opts:
            res = train(cfg, model_key="model_a", opt_override=opt)
            results.append(res)
        
        print("\n  -- Optimizer Exploration (Model A) --")
        for r in results:
            print(f"  {r['optimizer']:15s} | Test MSE: {r['test_mse']:.6f} | MAE: {r['test_mae']:.6f}")
        
        # Train Model B with best from previous (here we just run it with standard)
        train(cfg, model_key="model_b")
    else:
        key = "model_a" if args.model == "a" else "model_b"
        train(cfg, model_key=key, opt_override=args.opt)


if __name__ == "__main__":
    main()
