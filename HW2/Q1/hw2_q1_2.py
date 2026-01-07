# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST, INFO

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import time

# ---------------- Hyperparameters ----------------

LEARNING_RATE = 0.001
NUM_CLASSES = 8
BATCH_SIZE = 64
EPOCHS = 200

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------- Data ----------------

data_flag = 'bloodmnist'
info = INFO[data_flag]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

print("Loading data...")
train_ds = BloodMNIST(split='train', transform=transform, download=True, size=28)
val_ds   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
test_ds  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Model with MaxPooling ----------------

class SimpleCNN_MaxPool(nn.Module):
    def __init__(self, num_classes, use_softmax=False):
        super().__init__()
        self.use_softmax = use_softmax

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.softmax = nn.Softmax(dim=1)

        # 28 → 14 → 7 → 3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)

        if self.use_softmax:
            return self.softmax(logits)
        return logits

# ---------------- Training & Evaluation ----------------

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

def evaluate(loader, model):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            preds.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    return accuracy_score(targets, preds)

# ---------------- Experiment ----------------

def run_experiment(use_softmax):
    tag = "Softmax + MaxPool" if use_softmax else "Logits + MaxPool"
    print(f"\nRunning experiment: {tag}")

    model = SimpleCNN_MaxPool(NUM_CLASSES, use_softmax).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []

    start_time = time.time()

    for epoch in range(EPOCHS):
        loss = train_epoch(train_loader, model, criterion, optimizer)
        val_acc = evaluate(val_loader, model)

        train_losses.append(loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    total_time = time.time() - start_time
    test_acc = evaluate(test_loader, model)

    print(f"Test Accuracy ({tag}): {test_acc:.4f}")
    print(f"Training Time ({tag}): {total_time/60:.2f} min")

    return train_losses, val_accs, test_acc, total_time

# ---------------- Main ----------------

loss_logits, acc_logits, test_logits, time_logits = run_experiment(use_softmax=False)
loss_softmax, acc_softmax, test_softmax, time_softmax = run_experiment(use_softmax=True)

epochs = list(range(1, EPOCHS + 1))

print("\nFinal Results (MaxPool)")
print(f"Logits + MaxPool  | Acc: {test_logits:.4f} | Time: {time_logits/60:.2f} min")
print(f"Softmax + MaxPool | Acc: {test_softmax:.4f} | Time: {time_softmax/60:.2f} min")

# ---------------- Final Plots ----------------

# ---- Training Loss Comparison ----
plt.figure(figsize=(7, 5))
plt.plot(epochs, loss_logits, label='Logits + MaxPool', linewidth=2)
plt.plot(epochs, loss_softmax, label='Softmax + MaxPool', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison (MaxPool)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('CNN-training-loss-maxpool-comparison.pdf', bbox_inches='tight')
plt.close()

# ---- Validation Accuracy Comparison ----
plt.figure(figsize=(7, 5))
plt.plot(epochs, acc_logits, label='Logits + MaxPool', linewidth=2)
plt.plot(epochs, acc_softmax, label='Softmax + MaxPool', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison (MaxPool)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('CNN-validation-accuracy-maxpool-comparison.pdf', bbox_inches='tight')
plt.close()

