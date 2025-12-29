#!/usr/bin/env python

# Deep Learning Homework 1

# Multi-class Perceptron Classifier - 1 (a)

# Command to run this file (remove #):
# python "C:\Users\flori\OneDrive\Desktop\Uni\IST\Deep Learning\Homework_1\hw1-perceptron.py" `
# --data-path "C:\Users\flori\OneDrive\Desktop\Uni\IST\Deep Learning\Homework_1\emnist-letters.npz" `
# --save-path "C:\Users\flori\OneDrive\Desktop\Uni\IST\Deep Learning\Homework_1\best_perceptron.pkl" `
# --accuracy-plot "C:\Users\flori\OneDrive\Desktop\Uni\IST\Deep Learning\Homework_1\Q1-perceptron-accs.pdf" `
# --scores "C:\Users\flori\OneDrive\Desktop\Uni\IST\Deep Learning\Homework_1\Q1-perceptron-scores.json"


import argparse
import time
import pickle
import json

import numpy as np

import utils

debug = False # Set to True to enable debug prints


class Perceptron:
    def __init__(self, n_classes, n_features):
        self.W = np.zeros((n_classes, n_features))
        if debug:
            print(f"Initialized weight matrix W with shape {self.W.shape}")

    def save(self, path):
        """
        Save perceptron to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load perceptron from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    # update_weight and train_epoch are derived from multi_class_perceptron_epoch in practical_02_solution
    # The given framework did not include any learning rate eta
    def update_weight(self, x_i, y_i):
        """
        x_i (n_features,): a single training example
        y_i (scalar): the gold label for that example
        """
        y_hat = np.argmax(self.W.dot(x_i))
        if y_hat != y_i:
            # Perceptron update.
            self.W[y_i, :] += x_i
            self.W[y_hat, :] -= x_i

    # This is basically just looping over the dataset with update_weight
    # In practical_02_solution multi_class_perceptron_epoch these two parts were combined
    def train_epoch(self, X, y):
        """
        X (n_examples, n_features): features for the whole dataset
        y (n_examples,): labels for the whole dataset
        """
        if debug:
            mistakes = 0
            for x_i, y_i in zip(X, y):
                prev_W = self.W.copy()
                self.update_weight(x_i, y_i)
                if not np.array_equal(self.W, prev_W):
                    mistakes += 1
            print(f"Number of weight updates in this epoch: {mistakes}")

        else:
            for x_i, y_i in zip(X, y):
                self.update_weight(x_i, y_i)

    # Derived from multi_class_classify in practical_02_solution (vectorized)
    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """        
        scores = self.W.dot(X.T)
        y_hat = np.argmax(scores, axis=0)
        return y_hat

    # Almost the same as evaluate in practical_02_solution
    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels

        returns classifier accuracy
        """
        y_hat = self.predict(X)
        acc = np.mean(y_hat == y)
        return acc


def main(args):
    utils.configure_seed(seed=args.seed)

    if debug:
        print(f"Loading dataset from: {args.data_path}")
    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]

    if debug:
        print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Unique classes in training set: {np.unique(y_train)} (total {len(np.unique(y_train))})")
        print(f"Validation data shape: X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
        print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    # initialize the model
    model = Perceptron(n_classes, n_feats)

    epochs = np.arange(1, args.epochs + 1)

    valid_accs = []
    train_accs = []

    start = time.time()

    best_valid = 0.0
    best_epoch = -1
    for i in epochs:

        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]
    
        model.train_epoch(X_train, y_train)

        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print("")
        print('Training epoch {}'.format(i))
        print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))

        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i
            print(f"Validation accuracy improved, saving model at epoch {i}")
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print("")
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print("")
    print("Reloading best checkpoint")
    # Load best model and evaluate on test
    best_model = Perceptron.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)

    print('Best model test acc: {:.4f}'.format(test_acc))

    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot
    )

    with open(args.scores, "w") as f:
        json.dump(
            {"best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": elapsed_time},
            f,
            indent=4
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="Number of epochs to train for.")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--accuracy-plot", default="Q1-perceptron-accs.pdf")
    parser.add_argument("--scores", default="Q1-perceptron-scores.json")
    args = parser.parse_args()
    main(args)
