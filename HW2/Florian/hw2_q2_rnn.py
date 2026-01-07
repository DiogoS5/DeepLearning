import os
import csv
import time
import copy
from itertools import product

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
# 3) RNN model
# -----------------------------
class RNNRBP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.fc1 = nn.Linear(rnn_out_dim, 256)
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 41, 4)
        rnn_out, h_n = self.rnn(x)

        if self.rnn.bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_last = h_n[-1]

        # Regression head
        x = self.drop(self.relu(self.fc1(h_last)))
        out = self.fc2(x)
        return out


# -----------------------------
# 4) Train/val with early stopping on val Spearman
# -----------------------------
def train_val_model_rna(
    model,
    optimizer,
    dataloaders,
    num_epochs=50,
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
        # Print only the epoch number, updating in place
        print(f"{epoch+1}", end='\r')

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

            epoch_loss = running_loss / nsamples
            epoch_spear = running_spear / nsamples

            losses[phase].append(epoch_loss)
            spears[phase].append(epoch_spear)



            # Only update early stopping based on validation
            if phase == "val":
                if epoch_spear > best_val_spear + min_delta:
                    best_val_spear = epoch_spear
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1




        # Early stopping trigger (after completing the val phase)
        if bad_epochs >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} (best epoch {best_epoch}).")
            break


    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
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
def run_one_config(cfg, out_dir="runs_rnn", batch_size=64, num_epochs=50, patience=8, min_delta=0.0):
    os.makedirs(out_dir, exist_ok=True)

    exp_name = cfg.get(
        "exp_name",
        f"rnn_lr{cfg['lr']}_dropout{cfg['dropout']}_layers{cfg['num_layers']}_hidden{cfg['hidden_dim']}_bi{cfg['bidirectional']}",
    )

    dataloaders, test_loader = make_dataloaders(batch_size=batch_size)

    model = RNNRBP(
        input_dim=4,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    model, losses, spears, best_val_spear, best_epoch = train_val_model_rna(
        model=model,
        optimizer=optimizer,
        dataloaders=dataloaders,
        num_epochs=num_epochs,
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
        num_layers=cfg["num_layers"],
        hidden_dim=cfg["hidden_dim"],
        bidirectional=cfg["bidirectional"],
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
    out_dir = "runs_rnn"
    os.makedirs(out_dir, exist_ok=True) # create output dir if needed
    
    # Grid creation
    lrs = [1e-3, 5e-4]
    dropouts = [0.2, 0.3]
    num_layers_list = [1, 2]
    hidden_dims = [128, 256]
    bidirectionals = [True, False]

    grid = [
        dict(
            lr=lr,
            dropout=dropout,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            bidirectional=bidirectional,
        )
        for lr, dropout, num_layers, hidden_dim, bidirectional
        in product(lrs, dropouts, num_layers_list, hidden_dims, bidirectionals)
    ]


    all_summaries = []
    total_runs = len(grid)
    for i, cfg in enumerate(grid, start=1):
        print(f"\n=== Starting run {i}/{total_runs} ===")
        cfg = dict(cfg)
        cfg["exp_name"] = f"rnn_run{i}"
        summary = run_one_config(
            cfg,
            out_dir=out_dir,
            batch_size=64,
            num_epochs=50,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
        )
        print(f"=== Finished run {i}/{total_runs} ===\n")
        all_summaries.append(summary)

    summary_path = os.path.join(out_dir, "rnn_tuning_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)

    best = max(all_summaries, key=lambda d: d["best_val_spearman"])
    print("\nBEST RNN CONFIG (by val Spearman):")
    for k, v in best.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
