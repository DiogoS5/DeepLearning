import os
import csv
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import (
    configure_seed,
    load_rnacompete_data,
    masked_mse_loss,
    masked_spearman_correlation,
    plot,
)

# -----------------------------
# 0) Seed + device
# -----------------------------
configure_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# -----------------------------
# 1) Switch: grid search OR one run
# -----------------------------
RUN_GRID_SEARCH = True  # Set to False to run single config

SINGLE_CONFIG = dict(
    lr=1e-3,
    dropout=0.2,
    kernel_size=5,
    channels=(64, 128, 256),

)

# Early stopping settings (see L05)
PATIENCE = 8  # epochs to wait before stopping if no improvement
MIN_DELTA = 1e-4  # require at least this improvement to reset patience

# -----------------------------
# 2) Data
# -----------------------------
protein_name = "RBFOX1"
train_dataset = load_rnacompete_data(protein_name=protein_name, split="train")
val_dataset   = load_rnacompete_data(protein_name=protein_name, split="val")
test_dataset  = load_rnacompete_data(protein_name=protein_name, split="test")

def make_dataloaders(batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return dict(train=train_loader, val=val_loader), test_loader

# -----------------------------
# 3) CNN model
# -----------------------------
class CNNRBP(nn.Module):
    def __init__(self, channels=(32, 64, 128), kernel_size=5, dropout=0.2):
        super().__init__()
        c1, c2, c3 = channels # channels for conv layers
        pad = kernel_size // 2 # padding to maintain length before pooling

        self.conv1 = nn.Conv1d(4,  c1, kernel_size=kernel_size, padding=pad) # input has 4 channels (A,C,G,U)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad) # second conv layer
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=kernel_size, padding=pad) # third conv layer

        self.relu = nn.ReLU() # non-linear activation
        self.pool = nn.MaxPool1d(kernel_size=2) # downsample by factor of 2 using max pooling
        self.drop = nn.Dropout(dropout) # dropout for regularization

        self.global_pool = nn.AdaptiveMaxPool1d(1) # global max pooling to get fixed-size output

        self.fc1 = nn.Linear(c3, 256) # fully connected layer
        self.fc2 = nn.Linear(256, 1) # output layer for regression

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B,41,4] -> [B,4,41]

        x = self.relu(self.conv1(x)) 
        x = self.pool(x) 
        x = self.drop(x) 

        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.drop(x)

        x = self.global_pool(x).squeeze(-1) # [B,C,1] -> [B,C]
        x = self.drop(self.relu(self.fc1(x))) # FC + ReLU + Dropout
        return self.fc2(x)



# -----------------------------
# 4) Train/val with early stopping on val Spearman
# -----------------------------
def train_val_model_rna(
    model,
    optimizer,
    dataloaders,
    num_epochs=50,
    scheduler=None,
    log_interval=1,
    patience=8,
    min_delta=0.0,
):
    """
    Tried to maintain similar structure to practical 07 with added early stopping
      - train/val phases
      - store curves
      - keep best model weights on validation metric
      - early stopping on validation Spearman with a patience window
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_spear = -1e9
    best_epoch = -1
    bad_epochs = 0

    losses = dict(train=[], val=[])
    spears = dict(train=[], val=[])

    for epoch in range(num_epochs):
        if log_interval is not None and epoch % log_interval == 0:
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_spear = 0.0
            nsamples = 0

            for x, y, mask in dataloaders[phase]:
                x = x.to(device).float()
                y = y.to(device).float()
                mask = mask.to(device).float()
                nsamples += x.size(0)

                optimizer.zero_grad() # reset gradients

                with torch.set_grad_enabled(phase == "train"): # only track gradients in train
                    preds = model(x)
                    loss = masked_mse_loss(preds, y, mask)

                    if phase == "train": # backprop + optimize
                        loss.backward()
                        optimizer.step()

                with torch.no_grad(): # compute Spearman without tracking gradients
                    spear = masked_spearman_correlation(preds, y, mask)

                running_loss += loss.item() * x.size(0)
                running_spear += spear.item() * x.size(0)

            if scheduler is not None and phase == "train": # step learning-rate scheduler only in train
                scheduler.step()

            epoch_loss = running_loss / nsamples
            epoch_spear = running_spear / nsamples

            losses[phase].append(epoch_loss)
            spears[phase].append(epoch_spear)

            if log_interval is not None and epoch % log_interval == 0:
                print(f"{phase} Loss: {epoch_loss:.4f} Spearman: {epoch_spear:.3f}")

            # Only update early stopping based on validation
            if phase == "val":
                if epoch_spear > best_val_spear + min_delta:
                    best_val_spear = epoch_spear
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1

        if log_interval is not None and epoch % log_interval == 0:
            print(f"Bad epochs: {bad_epochs}/{patience}\n")

        # Early stopping trigger (after completing the val phase)
        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch} (best epoch {best_epoch}).")
            break

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Spearman: {best_val_spear:.3f} (epoch {best_epoch})")

    model.load_state_dict(best_model_wts)
    return model, losses, spears, best_val_spear, best_epoch


@torch.no_grad() # No gradients needed for evaluation
def evaluate_test(model, test_loader):
    model.eval()
    total_loss = 0.0
    total_spear = 0.0
    nsamples = 0

    for x, y, mask in test_loader:
        x = x.to(device).float()
        y = y.to(device).float()
        mask = mask.to(device).float()
        nsamples += x.size(0)

        preds = model(x)
        loss = masked_mse_loss(preds, y, mask)
        spear = masked_spearman_correlation(preds, y, mask)

        total_loss += loss.item() * x.size(0)
        total_spear += spear.item() * x.size(0)

    return total_loss / nsamples, total_spear / nsamples


# -----------------------------
# 5) One run wrapper
# -----------------------------
def run_one_config(cfg, out_dir="runs_cnn", batch_size=64, num_epochs=50, patience=8, min_delta=0.0):
    os.makedirs(out_dir, exist_ok=True)

    exp_name = cfg.get(
        "exp_name",
        f"cnn_lr{cfg['lr']}_do{cfg['dropout']}_k{cfg['kernel_size']}_ch{cfg['channels'][0]}"
    )

    dataloaders, test_loader = make_dataloaders(batch_size=batch_size)

    model = CNNRBP(
        channels=cfg["channels"],
        kernel_size=cfg["kernel_size"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    model, losses, spears, best_val_spear, best_epoch = train_val_model_rna(
        model=model,
        optimizer=optimizer,
        dataloaders=dataloaders,
        num_epochs=num_epochs,
        log_interval=1,
        patience=patience,
        min_delta=min_delta,
    )

    # Test only once, after model selection on validation
    test_loss, test_spear = evaluate_test(model, test_loader)
    print(f"[{exp_name}] Test loss: {test_loss:.4f} | Test Spearman: {test_spear:.3f}")

    # Curves length might be shorter due to early stopping
    epochs = list(range(len(losses["train"])))

    csv_path = os.path.join(out_dir, f"{exp_name}_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_spearman", "val_spearman"])
        for e in epochs:
            writer.writerow([e, losses["train"][e], losses["val"][e], spears["train"][e], spears["val"][e]])

    # Plotting helper from utils.py
    plot(
        epochs,
        {"train": losses["train"], "val": losses["val"]},
        filename=os.path.join(out_dir, f"{exp_name}_loss.png"),
    )
    plot(
        epochs,
        {"train": spears["train"], "val": spears["val"]},
        filename=os.path.join(out_dir, f"{exp_name}_spearman.png"),
        ylim=(-0.1, 1.0),
    )

    summary = dict(
        exp_name=exp_name,
        batch_size=batch_size,
        lr=cfg["lr"],
        dropout=cfg["dropout"],
        kernel_size=cfg["kernel_size"],
        channels=str(cfg["channels"]),
        best_epoch=int(best_epoch),
        best_val_spearman=float(best_val_spear),
        test_loss=float(test_loss),
        test_spearman=float(test_spear),
        patience=int(patience),
        min_delta=float(min_delta),
    )
    return summary


# -----------------------------
# 6) Main
# -----------------------------
def main():
    out_dir = "runs_cnn"
    os.makedirs(out_dir, exist_ok=True) # create output dir if needed

    # Single run
    if not RUN_GRID_SEARCH:
        cfg = dict(SINGLE_CONFIG)
        cfg["exp_name"] = "cnn_single_run"
        summary = run_one_config(
            cfg,
            out_dir=out_dir,
            batch_size=64,
            num_epochs=50,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
        )

        print("\nSINGLE RUN SUMMARY")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return

    grid = [
        {"lr": 1e-3, "dropout": 0.2, "kernel_size": 5, "channels": (32, 64, 128)},
        {"lr": 5e-4, "dropout": 0.2, "kernel_size": 5, "channels": (32, 64, 128)},
        {"lr": 1e-3, "dropout": 0.3, "kernel_size": 5, "channels": (32, 64, 128)},
        {"lr": 1e-3, "dropout": 0.2, "kernel_size": 7, "channels": (32, 64, 128)},
        {"lr": 1e-3, "dropout": 0.2, "kernel_size": 5, "channels": (64, 128, 256)},
    ]

    all_summaries = []
    for i, cfg in enumerate(grid, start=1):
        cfg = dict(cfg)
        cfg["exp_name"] = f"cnn_run{i}"
        summary = run_one_config(
            cfg,
            out_dir=out_dir,
            batch_size=64,
            num_epochs=50,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
        )
        all_summaries.append(summary)

    summary_path = os.path.join(out_dir, "cnn_tuning_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)

    best = max(all_summaries, key=lambda d: d["best_val_spearman"])
    print("\nBEST CNN CONFIG (by val Spearman):")
    for k, v in best.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
