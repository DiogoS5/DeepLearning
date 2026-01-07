# -*- coding: utf-8 -*-

# https://github.com/MedMNIST/MedMNIST

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from medmnist import BloodMNIST, INFO

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import time
print("both")
# ---------------- Hyperparameters ----------------

LEARNING_RATE = 0.001
NUM_CLASSES = 8
BATCH_SIZE = 64
EPOCHS = 200

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------- Data Loading ----------------

data_flag = 'bloodmnist'
info = INFO[data_flag]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

print("Loading data...")
train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Model ----------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, use_softmax=False):
        super().__init__()
        self.use_softmax = use_softmax

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)

        if self.use_softmax:
            return self.softmax(logits)
        return logits

# ---------------- Training ----------------

def train_epoch(loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ---------------- Evaluation ----------------

def evaluate(loader, model):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.tolist()

    return accuracy_score(targets, preds)

# ---------------- Experiment ----------------

def run_experiment(use_softmax):
    tag = "Softmax" if use_softmax else "No Softmax"
    print(f"\nRunning experiment: {tag}")

    model = SimpleCNN(NUM_CLASSES, use_softmax).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []

    total_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        train_loss = train_epoch(train_loader, model, criterion, optimizer)
        val_acc = evaluate(val_loader, model)

        train_losses.append(train_loss)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time/Epoch: {epoch_time:.2f}s")

    total_time = time.time() - total_start
    test_acc = evaluate(test_loader, model)

    print(f"Test Accuracy ({tag}): {test_acc:.4f}")
    print(f"Total Training Time ({tag}): {total_time/60:.2f} min")

    return train_losses, val_accs, test_acc

# ---------------- Main ----------------

loss_no_softmax, acc_no_softmax, test_no_softmax = run_experiment(use_softmax=False)
loss_softmax, acc_softmax, test_softmax = run_experiment(use_softmax=True)

epochs = list(range(1, EPOCHS + 1))

print("\nFinal Comparison")
print(f"Test Accuracy (No Softmax): {test_no_softmax:.4f}")
print(f"Test Accuracy (Softmax):    {test_softmax:.4f}")

# ---------------- Final Plots ----------------

# ---- Training Loss Comparison ----
plt.figure(figsize=(7, 5))
plt.plot(epochs, loss_no_softmax, label='No Softmax', linewidth=2)
plt.plot(epochs, loss_softmax, label='Softmax', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('CNN-training-loss-comparison.pdf', bbox_inches='tight')
plt.close()

# ---- Validation Accuracy Comparison ----
plt.figure(figsize=(7, 5))
plt.plot(epochs, acc_no_softmax, label='No Softmax', linewidth=2)
plt.plot(epochs, acc_softmax, label='Softmax', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('CNN-validation-accuracy-comparison.pdf', bbox_inches='tight')
plt.close()

