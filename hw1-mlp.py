#!/usr/bin/env python

# Deep Learning Homework 1

# Multi-Layer Perceptron - Question 1 (3a)

# Command to run this file (remove #):
# python "hw1-mlp.py"


import argparse
import time
import pickle
import json

import numpy as np

import utils

debug = True # Set to True to enable debug prints


class MLP:
    """
    One-hidden-layer MLP for multi-class classification with:
      - hidden layer: 100 units, tanh activation
      - output layer: linear logits, softmax + cross-entropy loss
    Trained with stochastic gradient descent (batch size = 1).
    """

    def __init__(self, n_classes, n_features, n_hidden=100):
        # Weight shapes follow structure from practical_04_solution:
        # W1: (n_hidden, n_features), b1: (n_hidden,)
        # W2: (n_classes, n_hidden),  b2: (n_classes,)
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        # Initialize weights W ~ N(0, 0.1^2), biases = 0
        self.W1 = np.random.normal(loc=0.0, scale=0.1, size=(n_hidden, n_features))
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.normal(loc=0.0, scale=0.1, size=(n_classes, n_hidden))
        self.b2 = np.zeros(n_classes)

        if debug:
            print(f"Initialized MLP with W1 {self.W1.shape}, b1 {self.b1.shape}, "
                  f"W2 {self.W2.shape}, b2 {self.b2.shape}")

    def save(self, path):
        """
        Save MLP to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load MLP from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    # Forward pass and loss, following practical_04_solution structure

    # Easier than the implementation in practical_04_solution since we only have one hidden layer
    def forward(self, x):
        """
        Forward pass for a single example x (shape (n_features,)).
        Returns:
          logits: shape (n_classes,)
          h1: hidden activation (tanh), shape (n_hidden,)
        """
        # Hidden layer: z1 = W1 x + b1, h1 = tanh(z1)
        z1 = self.W1.dot(x) + self.b1
        h1 = np.tanh(z1)

        # Output layer: z2 = W2 h1 + b2 (logits)
        z2 = self.W2.dot(h1) + self.b2

        return z2, h1

    
    def compute_loss(self, output, y_index):
        """
        Adapted from practical_04_solution.compute_loss():
        y_index: integer label (not one-hot!)
        """
        # Softmax function as in logreg
        exps = np.exp(output - np.max(output))
        probs = exps / np.sum(exps)
        
        # Convert integer label to one-hot for compatibility
        y_one_hot = np.zeros(self.n_classes)
        y_one_hot[y_index] = 1.0
        
        # same as in practical_04_solution
        loss = -y_one_hot.dot(np.log(probs))

        return loss  

    def backward(self, x, y_index, output, h1):
        """
        Similar to practical_04_solution.backward():
        Returns: grad_weights [dW1, dW2], grad_biases [db1, db2]
        """
        # Softmax gradient (EXACT SAME as practical_04_solution)
        exps = np.exp(output - np.max(output))
        probs = exps / np.sum(exps)
        y_one_hot = np.zeros(self.n_classes)
        y_one_hot[y_index] = 1.0
        grad_z2 = probs - y_one_hot  # dL/dz2

        # Output layer gradients
        dW2 = grad_z2[:, None].dot(h1[None, :])  # (n_classes, n_hidden)
        db2 = grad_z2  # (n_classes,)

        # Backprop to hidden: dL/dh1 = W2^T dL/dz2
        grad_h1 = self.W2.T.dot(grad_z2)  # (n_hidden,)

        # Through tanh: dL/dz1 = dL/dh1 * (1 - h1^2)
        grad_z1 = grad_h1 * (1 - h1**2)  # (n_hidden,)

        # Hidden layer gradients
        dW1 = grad_z1[:, None].dot(x[None, :])  # (n_hidden, n_features)
        db1 = grad_z1  # (n_hidden,)

        return [dW1, dW2], [db1, db2]  # Returns lists for update_weights

    def update_weights(self, grad_weights, grad_biases, eta=0.001):
        self.W1 -= eta * grad_weights[0]
        self.b1 -= eta * grad_biases[0]
        self.W2 -= eta * grad_weights[1]
        self.b2 -= eta * grad_biases[1]

    def train_epoch(self, X, y, eta=0.001):
        total_loss = 0.0
        # For each observation and target
        for x_i, y_i in zip(X, y):
            output, h1 = self.forward(x_i)

            # Compute Loss and Update total loss
            loss = self.compute_loss(output, y_i)
            total_loss += loss
            # Compute backpropagation
            grad_weights, grad_biases = self.backward(x_i, y_i, output, h1)
            
            # Update weights
            self.W1 -= eta * grad_weights[0]
            self.b1 -= eta * grad_biases[0]
            self.W2 -= eta * grad_weights[1]
            self.b2 -= eta * grad_biases[1]
            
        return total_loss


    def predict(self, X):
        """
        X: (n_examples, n_features)
        Returns: predicted labels (n_examples,)
        """
        predicted_labels = []
        for x in X:
            # Compute forward pass and get the class with the highest probability
            output, _ = self.forward(x)
            y_hat = np.argmax(output)
            predicted_labels.append(y_hat)
        predicted_labels = np.array(predicted_labels)

        return predicted_labels

    def evaluate(self, X, y):
        """
        Returns classification accuracy on X,y.
        """
        y_hat = self.predict(X)
        return float(np.mean(y_hat == y))


def main(args):
    utils.configure_seed(seed=args.seed)

    if debug:
        print(f"Loading dataset from: {args.data_path}")
    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]

    if debug:
        print(f"Training data shape: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Validation data shape: {X_valid.shape}, y_valid: {y_valid.shape}")
        print(f"Test data shape: {X_test.shape}, y_test: {y_test.shape}")

    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    # Initialize model (single hidden layer, 100 units as specified in the homework)
    model = MLP(n_classes, n_feats, n_hidden=100)

    epochs = np.arange(1, args.epochs + 1)

    train_accs = []
    valid_accs = []
    train_losses = []

    eta = 0.001

    start = time.time()

    best_valid = 0.0
    best_epoch = -1

    for i in epochs:
        # Shuffle training data each epoch
        order = np.random.permutation(X_train.shape[0])
        X_train_shuf = X_train[order]
        y_train_shuf = y_train[order]

        # One epoch of SGD
        total_loss = model.train_epoch(X_train_shuf, y_train_shuf, eta=eta)
        avg_loss = total_loss / X_train.shape[0]

        # Evaluate accuracies
        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print("")
        print(f"Training epoch {i}")
        print(f"  loss: {avg_loss:.4f} | train acc: {train_acc:.4f} | val acc: {valid_acc:.4f}")

        # Save best model according to validation accuracy
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i
            print(f"  Validation accuracy improved, saving model at epoch {i}")
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print("")
    print(f"Training took {minutes} minutes and {seconds} seconds")

    print("")
    print("Reloading best checkpoint")
    best_model = MLP.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)
    print(f"Best model test acc: {test_acc:.4f}")

    # Plot accuracy curves (train/valid) over epochs
    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot,
    )

    # Plot training loss over epochs
    utils.plot(
        "Epoch", "Loss",
        {"train_loss": (epochs, train_losses)},
        filename=args.loss_plot,
    )

    # Save scores and timing
    with open(args.scores, "w") as f:
        json.dump(
            {
                "best_valid": float(best_valid),
                "selected_epoch": int(best_epoch),
                "test": float(test_acc),
                "time": float(elapsed_time),
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int,
                        help="Number of epochs to train for.")
    parser.add_argument("--data-path", type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="best_mlp.pkl")
    parser.add_argument("--accuracy-plot", default="Q1-mlp-accs.pdf")
    parser.add_argument("--loss-plot", default="Q1-mlp-loss.pdf")
    parser.add_argument("--scores", default="Q1-mlp-scores.json")
    args = parser.parse_args()
    main(args)
