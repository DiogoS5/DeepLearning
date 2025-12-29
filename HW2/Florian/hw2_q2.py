import os
import csv
import math
import time
import copy
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from utils import (
    load_rnacompete_data,
    masked_mse_loss,
    masked_spearman_correlation,
    configure_seed,
)

# -----------------------------
# 0) Repro + device
# -----------------------------
configure_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1) Data
# -----------------------------
train_dataset = load_rnacompete_data(protein_name="RBFOX1", split="train")
val_dataset   = load_rnacompete_data(protein_name="RBFOX1", split="val")
test_dataset  = load_rnacompete_data(protein_name="RBFOX1", split="test")

def make_loaders(batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# -----------------------------
# 2) CNN Model A
# -----------------------------
class CNNRBP(nn.Module):
    def __init__(
        self,
        channels=(32, 64, 128),
        kernel_size=5,
        dropout=0.2,
    ):
        super().__init__()
        c1, c2, c3 = channels

        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(4, c1, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=kernel_size, padding=pad)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(dropout)

        # Robust pooling: avoids “conv_out_len guessing”
        # Output becomes [B, c3, 1]
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(c3, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: [B, 41, 4] -> [B, 4, 41]
        x = x.permute(0, 2, 1)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.drop(x)

        x = self.pool(self.relu(self.conv2(x)))
        x = self.drop(x)

        x = self.pool(self.relu(self.conv3(x)))
        x = self.drop(x)

        x = self.global_pool(x).squeeze(-1)  # [B, c3]
        x = self.drop(self.relu(self.fc1(x)))
        out = self.fc2(x)                    # [B, 1]
        return out

# -----------------------------
# 3) Train/eval epoch
# -----------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_spear = 0.0
    n_batches = 0

    for x, y, mask in loader:
        x = x.to(device).float()
        y = y.to(device).float()
        mask = mask.to(device).float()

        preds = model(x)
        loss = masked_mse_loss(preds, y, mask)
        spear = masked_spearman_correlation(preds, y, mask)

        total_loss += loss.item()
        total_spear += spear.item()
        n_batches += 1

    return total_loss / n_batches, total_spear / n_batches

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    total_spear = 0.0
    n_batches = 0

    for x, y, mask in loader:
        x = x.to(device).float()
        y = y.to(device).float()
        mask = mask.to(device).float()

        optimizer.zero_grad()
        preds = model(x)
        loss = masked_mse_loss(preds, y, mask)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            spear = masked_spearman_correlation(preds, y, mask)

        total_loss += loss.item()
        total_spear += spear.item()
        n_batches += 1

    return total_loss / n_batches, total_spear / n_batches

# -----------------------------
# 4) Single run (with early stopping + logging)
# -----------------------------
def run_experiment(
    exp_name,
    batch_size=64,
    lr=1e-3,
    weight_decay=0.0,
    channels=(32, 64, 128),
    kernel_size=5,
    dropout=0.2,
    max_epochs=50,
    patience=8,
    out_dir="runs_cnn",
):
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(batch_size)

    model = CNNRBP(channels=channels, kernel_size=kernel_size, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_val_spear = -1e9
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_spear = train_one_epoch(model, train_loader, optimizer)
        va_loss, va_spear = evaluate(model, val_loader)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_spearman": tr_spear,
            "val_spearman": va_spear,
        })

        print(
            f"[{exp_name}] Epoch {epoch:02d} | "
            f"Train loss {tr_loss:.4f}, Val loss {va_loss:.4f} | "
            f"Train ρ {tr_spear:.3f}, Val ρ {va_spear:.3f}"
        )

        # Early stopping on validation Spearman (main metric)
        if va_spear > best_val_spear:
            best_val_spear = va_spear
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{exp_name}] Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # Load best checkpoint and test once
    model.load_state_dict(best_state)
    te_loss, te_spear = evaluate(model, test_loader)
    print(f"[{exp_name}] BEST epoch {best_epoch} | Test loss {te_loss:.4f}, Test Spearman {te_spear:.3f}")

    # Save CSV history
    csv_path = os.path.join(out_dir, f"{exp_name}_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Plot losses
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("masked MSE")
    plt.title(f"{exp_name}: loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_loss.png"), dpi=150)
    plt.close()

    # Plot Spearman
    train_s = [h["train_spearman"] for h in history]
    val_s = [h["val_spearman"] for h in history]

    plt.figure()
    plt.plot(epochs, train_s, label="train")
    plt.plot(epochs, val_s, label="val")
    plt.xlabel("epoch")
    plt.ylabel("Spearman")
    plt.title(f"{exp_name}: Spearman curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_spearman.png"), dpi=150)
    plt.close()

    # Save best summary
    summary = {
        "exp_name": exp_name,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "channels": str(channels),
        "kernel_size": kernel_size,
        "dropout": dropout,
        "best_epoch": best_epoch,
        "best_val_spearman": float(best_val_spear),
        "test_loss": float(te_loss),
        "test_spearman": float(te_spear),
    }

    return summary

# -----------------------------
# 5) CNN hyperparameter tuning (simple grid)
# -----------------------------
def main():
    out_dir = "runs_cnn"
    os.makedirs(out_dir, exist_ok=True)

    # A simple, defensible tuning strategy:
    # - small grid (few runs), pick best on val Spearman, evaluate test once.
    # This respects “don’t tune on test”.[file:1][file:2]
    grid = [
        {"lr": 1e-3, "dropout": 0.2, "kernel_size": 5, "channels": (32, 64, 128)},
        {"lr": 5e-4, "dropout": 0.2, "kernel_size": 5, "channels": (32, 64, 128)},
        {"lr": 1e-3, "dropout": 0.3, "kernel_size": 5, "channels": (32, 64, 128)},
        {"lr": 1e-3, "dropout": 0.2, "kernel_size": 7, "channels": (32, 64, 128)},
        {"lr": 1e-3, "dropout": 0.2, "kernel_size": 5, "channels": (64, 128, 256)},
    ]

    all_summaries = []
    for i, cfg in enumerate(grid, start=1):
        exp_name = f"cnn_run{i}_lr{cfg['lr']}_do{cfg['dropout']}_k{cfg['kernel_size']}_ch{cfg['channels'][0]}"
        summary = run_experiment(
            exp_name=exp_name,
            batch_size=64,
            lr=cfg["lr"],
            weight_decay=0.0,
            channels=cfg["channels"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
            max_epochs=50,
            patience=8,
            out_dir=out_dir,
        )
        all_summaries.append(summary)

    # Save tuning summary CSV
    summary_path = os.path.join(out_dir, "cnn_tuning_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)

    # Print best by validation Spearman
    best = max(all_summaries, key=lambda d: d["best_val_spearman"])
    print("\nBEST CNN CONFIG (by val Spearman):")
    for k, v in best.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
