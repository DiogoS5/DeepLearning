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
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=True, n_heads=4):
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

        # Attention
        self.mha = nn.MultiheadAttention(embed_dim=rnn_out_dim, num_heads=n_heads, batch_first=True)
        self.query_vector = nn.Parameter(torch.randn(1, 1, rnn_out_dim))

        self.fc1 = nn.Linear(rnn_out_dim, 256)
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        rnn_out, _ = self.rnn(x) # (Batch, 41, rnn_out_dim)

        #Expand query vector to match batch size
        query = self.query_vector.expand(x.size(0), -1, -1)

        #Compare query agaist keys. Extract values (weighthed sum of rnn_out)
        attn_output, _weights = self.mha(query, rnn_out, rnn_out)

        # Regression head       
        x = self.drop(self.relu(self.fc1(attn_output.squeeze(1))))
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
        print(f"Epoch {epoch+1}", end='\r')

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
        f"rnn_lr{cfg['lr']}_dropout{cfg['dropout']}_layers{cfg['num_layers']}_hidden{cfg['hidden_dim']}_bi{cfg['bidirectional']}_n_heads{cfg['n_heads']}",
    )

    dataloaders, test_loader = make_dataloaders(batch_size=batch_size)

    model = RNNRBP(
        input_dim=4,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
        n_heads=cfg["n_heads"],
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
# Main
# -----------------------------
def main():
    out_dir = "runs_rnn_attention"
    os.makedirs(out_dir, exist_ok=True) # create output dir if needed

    SINGLE_CONFIG = dict(
        lr=5e-4,
        dropout=0.2,
        num_layers=1,
        hidden_dim=256,
        bidirectional=True,
        n_heads=4,
    )
    
    cfg = dict(SINGLE_CONFIG)
    cfg["exp_name"] = "rnn_attention"
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

if __name__ == "__main__":
    main()
