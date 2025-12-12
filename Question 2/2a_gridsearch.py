#!/usr/bin/env python

# Deep Learning Homework 1

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
import time
import utils

import multiprocessing as mp
import itertools

# Set matplotlib backend to non-interactive to prevent crashes in threads
plt.switch_backend('Agg')

class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """ Define a vanilla multiple-layer FFN with `layers` hidden layers """
        super().__init__()

        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        
        self.hidden = nn.ModuleList()

        # First layer (input to hidden)
        if layers > 0:
            self.hidden.append(nn.Linear(n_features, hidden_size))
        for _ in range(layers - 1):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))

        self.output = nn.Linear(hidden_size if layers > 0 
                                else n_features, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        for layer in self.hidden:
            Z = layer(x)
            P = self.activation(Z)
            x = self.dropout(P)
            
        return self.output(x)
    
    
def train_batch(X, y, model, optimizer, criterion, **kwargs):
    model.train()
    optimizer.zero_grad()
    y_hat = model(X)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def predict(model, X):
    model.eval()
    y_hat = model(X)
    _, preds = torch.max(y_hat, dim=1) 
    return preds


@torch.no_grad()
def evaluate(model, X, y, criterion):
    loss = criterion(model(X), y).item()
    preds = predict(model, X)
    accuracy = (preds == y).float().mean().item()
    return (loss, accuracy)


def plot(epochs, plottables, filename=None, ylim=None):
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plot_epochs = torch.cat([torch.tensor([0]), epochs]) 
        plt.plot(plot_epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def run_experiment(config, dataset_tensors, n_classes, n_feats, device):
    # Unpack configuration
    hidden_size, learning_rate, l2_decay, dropout = config
    
    # Unpack data
    train_X, train_y, dev_X, dev_y, test_X, test_y = dataset_tensors
    
    # Constants
    N_EPOCHS = 30
    batch_size = 64
    layers = 1
    activation = 'relu'
    optimizer_name = "SGD"
    
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))

    print(f"Start: HS={hidden_size}, LR={learning_rate}, L2={l2_decay}, DO={dropout}")

    # initialize the model
    model = FeedforwardNetwork(
        n_classes, n_feats, hidden_size, layers, activation, dropout
    ).to(device)

    optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=l2_decay
        )
    
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, N_EPOCHS + 1)
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    
    model.eval()
    initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
    initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
    train_losses.append(initial_train_loss)
    train_accs.append(initial_train_acc)
    valid_losses.append(initial_val_loss)
    valid_accs.append(initial_val_acc)

    for ii in epochs:
        epoch_train_losses = []
        model.train()
        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        model.eval()
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        _, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    
    # Plotting
    config_str = (
        f"batch-{batch_size}-lr-{learning_rate}-epochs-{N_EPOCHS}-"
        f"hidden-{hidden_size}-dropout-{dropout}-l2-{l2_decay}-"
        f"layers-{layers}-act-{activation}-opt-{optimizer_name}"
    )

    losses = {"Train Loss": train_losses, "Valid Loss": valid_losses}
    plot(epochs, losses, filename=f'training-loss-{config_str}.pdf')
    
    val_accuracy = { "Valid Accuracy": valid_accs }
    plot(epochs, val_accuracy, filename=f'validation-accuracy-{config_str}.pdf')

    # Return the string to write to CSV
    result_str = f"{hidden_size},{learning_rate},{l2_decay},{dropout},{max(valid_accs)},{test_acc}\n"
    print(f"Finished: {result_str.strip()}")
    return result_str


def main():
    # Params
    hidden_sizes = [16,32,64,128,256]
    learning_rates = [1e-4, 5e-4, 0.001, 0.01]
    l2_decays = [0.0, 1e-5]
    dropouts = [0.0 , 0.2]
    data_path = 'emnist-letters.npz'

    # Select device 
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Determine number of cores
    n_workers = mp.cpu_count()
    print(f"Parallelizing over {n_workers} CPU cores.")

    utils.configure_seed(seed=42)

    data = utils.load_dataset(data_path)
    dataset = utils.ClassificationDataset(data)
    
    # Load data onto CPU tensors
    train_X, train_y = dataset.X.to(device), dataset.y.to(device)
    dev_X, dev_y = dataset.dev_X.to(device), dataset.dev_y.to(device)
    test_X, test_y = dataset.test_X.to(device), dataset.test_y.to(device)
    
    # Pack tensors into a tuple
    dataset_tensors = (train_X, train_y, dev_X, dev_y, test_X, test_y)

    n_classes = torch.unique(dataset.y).shape[0] 
    n_feats = dataset.X.shape[1]

    # Generate all combinations of hyperparameters
    configs = list(itertools.product(hidden_sizes, learning_rates, l2_decays, dropouts))
    print(f"Total configurations to train: {len(configs)}")

    # Prepare arguments for starmap. Zip the configs with repeated instances of the static data
    args = zip(
        configs,
        itertools.repeat(dataset_tensors),
        itertools.repeat(n_classes),
        itertools.repeat(n_feats),
        itertools.repeat(device)
    )

    start_total = time.time()

    # Assign tasks to available cores
    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(run_experiment, args)

    # Write results to CSV after all parallel jobs are done
    with open('grid_search_accuracies.csv', 'w') as f:
        for res in results:
            f.write(res)

    elapsed = time.time() - start_total
    print(f'Total Grid Search took {elapsed//60:.0f}m {elapsed%60:.0f}s')

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()