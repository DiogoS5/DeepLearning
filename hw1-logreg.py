#!/usr/bin/env python

# Deep Learning Homework 1
# Logistic Regression - 2 (a)+(b)+(c)

# Command to run logreg from 2 (a)
# python "hw1-logreg.py" # set downsample = False in line 27

# Command to run logreg from 2 (b) with downsampled images
# python "hw1-logreg.py" # set downsample = True in line 27

# (or just run the file with the appropriate downsample setting,
# everything is set to default accordingly when in the right folder)

# Command to run grid search from 2 (c)
# python "hw1-logreg.py" --grid-search --grid-results "Q1-2c-grid-results.json"

import argparse
import time
import pickle
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # for nice table output

import utils

debug = False # Set to True to enable debug prints
downsample = False # Set to True to enable downsampling to 14x14 images (for question 2 (b))


class LogisticRegression:
    def __init__(self, n_classes, n_features):
        self.W = np.zeros((n_classes, n_features))
        if debug:
            print(f"Initialized weight matrix W with shape {self.W.shape}")

    def save(self, path):
        """
        Save logreg to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load logreg from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)


    # Function weight_update was omitted and integrated here following the structure 
    # of multi_class_lr_epoch in practical_03_solution
    def train_epoch(self, X, y, eta=0.0001, l2=0.00001):
        """
        One epoch of stochastic gradient descent for multiclass logistic regression
        with L2 regularization (weight decay).
        """
        n_classes = self.W.shape[0]
        # For each observation in data
        for x_i, y_i in zip(X, y):

            # Get probability scores according to the model (n_classes,).
            scores = self.W.dot(x_i)
            
            # One-hot encode true label (n_classes,).
            y_one_hot = np.zeros(n_classes)
            y_one_hot[y_i] = 1

            # Softmax function
            # This gives the label probabilities according to the model (n_classes,).
            # Goal: Mimimize negative log likelihood (NLL)
            exps = np.exp(scores - np.max(scores))
            probs = exps / np.sum(exps)

            # SGD update with NLL and L2 penalty. W is n_classes x n_features.
            # l2 regularization follows L03 page 31
            # NLL follows L03 page 36
            gradient = np.outer(probs - y_one_hot, x_i) + l2 * self.W
            self.W -= eta * gradient

    # Same as for perceptron
    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """        
        scores = self.W.dot(X.T)
        y_hat = np.argmax(scores, axis=0)
        return y_hat

    # Same as for perceptron
    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels

        returns classifier accuracy
        """
        y_hat = self.predict(X)
        acc = np.mean(y_hat == y)
        return acc
    

# Note: This function was created by AI as a helper and is not part of the main assignment.
def show_letters(X, y=None, n=10):
    """
    Display n handwritten letter images from dataset X.

    Args:
        X: numpy array of shape (num_examples, 784), each row is a flattened 28x28 image.
        y: Optional. numpy array of labels corresponding to X.
        n: number of images to display (default 10).

    This function plots n images in a row.
    """

    plt.figure(figsize=(n, 1.5))
    for i in range(n):
        if downsample:
            image = X[i].reshape(14, 14)
        else:
            image = X[i].reshape(28, 28)
        image = np.flipud(image)            # Flip vertically
        image = np.rot90(image, k=-1)      # Rotate 90 degrees right (clockwise)
        plt.subplot(1, n, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        if y is not None:
            plt.title(chr(y[i] + ord('A')))
    plt.tight_layout()
    plt.show()

# The next three functions enable downsampling from 28x28 to 14x14 images according to 
# question 2 (b) and where also implemented by AI
def vector_to_image(x_row):
    """
    x_row: 1D array of length 785 (bias + 784 pixels)
    returns: 2D array 28x28 (image only, bias removed)
    """
    img_flat = x_row[1:]           # drop bias term
    img = img_flat.reshape(28, 28)
    return img

def downsample_28_to_14(img):
    """
    img: 2D array (28x28)
    returns: 2D array (14x14) by averaging non-overlapping 2x2 blocks
    """
    # reshape into (14, 2, 14, 2) to group 2x2 blocks, then average over the 2 and 2 axes
    img_reshaped = img.reshape(14, 2, 14, 2)
    img_down = img_reshaped.mean(axis=(1, 3))  # average over small blocks
    return img_down

def build_downsampled_features(X):
    """
    X: (n_examples, 785) with bias + 784 pixels
    returns: (n_examples, 1 + 196) with bias + flattened 14x14 image
    """
    n_examples = X.shape[0]
    X_new = np.zeros((n_examples, 1 + 14*14), dtype=X.dtype)
    X_new[:, 0] = 1.0  # bias

    for i in range(n_examples):
        img = vector_to_image(X[i])
        img_down = downsample_28_to_14(img)
        X_new[i, 1:] = img_down.flatten()

    return X_new

def train_single_config(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                        eta, l2, feature_type):
    """
    Train one configuration and return:
      - best_valid_acc, test_acc
      - per-epoch train/valid accuracies (lists)
    """
    utils.configure_seed(seed=args.seed)

    # Copy data so we don't overwrite original arrays across configs
    X_train_c = X_train.copy()
    X_valid_c = X_valid.copy()
    X_test_c  = X_test.copy()

    # Prepare features
    if feature_type == "downsampled":
        X_train_c = build_downsampled_features(X_train_c)
        X_valid_c = build_downsampled_features(X_valid_c)
        X_test_c  = build_downsampled_features(X_test_c)

    n_classes = np.unique(y_train).size
    n_feats = X_train_c.shape[1]

    # initialize the model
    model = LogisticRegression(n_classes, n_feats)

    epochs = np.arange(1, args.epochs + 1)

    train_accs = []
    valid_accs = []

    best_valid = 0.0

    for i in epochs:

        train_order = np.random.permutation(X_train_c.shape[0])
        X_train_shuf = X_train_c[train_order]
        y_train_shuf = y_train[train_order]

        model.train_epoch(X_train_shuf, y_train_shuf, eta=eta, l2=l2)

        train_acc = model.evaluate(X_train_c, y_train)
        valid_acc = model.evaluate(X_valid_c, y_valid)

        # print("")
        # print('Training epoch {}'.format(i))
        # print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))

        # Tighter layout for grid search output
        print(f"    Epoch {i:2d}: train_acc = {train_acc:.4f}, valid_acc = {valid_acc:.4f}")

        if valid_acc > best_valid:
            best_valid = valid_acc
            model.save(args.save_path)

    # Load best model and evaluate on test
    best_model = LogisticRegression.load(args.save_path)
    test_acc = best_model.evaluate(X_test_c, y_test)

    return float(best_valid), float(test_acc)


def main(args):
    utils.configure_seed(seed=args.seed)
    
    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    
    if debug:
        print(f"Loading dataset from: {args.data_path}")
        print(f"Original shapes: train {X_train.shape}, valid {X_valid.shape}")
    
    # GRID SEARCH for 2(c): 3 etas × 2 l2 × 2 feature types = 12 configurations
    if args.grid_search:
        print("Starting grid search for Question 2(c)...")
        
        # Grid parameters as specified in homework
        etas = [0.00001, 0.0001, 0.001]      # 3 learning rates
        l2s = [0.00001, 0.0001]              # 2 L2 penalties  
        feature_types = ["raw", "downsampled"] # 2 feature representations
        
        results = []
        best_overall_valid = 0.0
        best_config = None
        
        for eta in etas:
            for l2_penalty in l2s:
                for feat_type in feature_types:
                    print(f"\nRunning config: eta={eta}, l2={l2_penalty}, features={feat_type}")
                    
                    start = time.time()
                    valid_acc, test_acc = train_single_config(
                        X_train, y_train, X_valid, y_valid, X_test, y_test,
                        eta=eta, l2=l2_penalty, feature_type=feat_type
                    )

                    elapsed_time = time.time() - start
                    
                    config = {
                        'eta': eta,
                        'l2': l2_penalty,
                        'features': feat_type,
                        'best_valid_acc': valid_acc,
                        'test_acc': test_acc,
                        'time_seconds': elapsed_time
                    }

                    results.append(config)
                    
                    print(f"  → Best valid: {valid_acc:.4f}, Test: {test_acc:.4f}, Time: {elapsed_time:.1f}s")
                    
                    if valid_acc > best_overall_valid:
                        best_overall_valid = valid_acc
        
        # Create results table
        df = pd.DataFrame(results)
        print("\nGrid search results:")
        print(df.to_string(index=False, float_format='%.5f'))
        
        # Save results
        df.to_json(args.grid_results, orient='records', indent=2)
        print(f"\nResults saved to {args.grid_results}")
        
        # Highlight best configuration
        best_idx = df['best_valid_acc'].idxmax()
        print(f"\nBest configuration (highest validation accuracy):")
        print(f"  eta: {df.loc[best_idx, 'eta']:.5f}")
        print(f"  l2: {df.loc[best_idx, 'l2']:.5f}")
        print(f"  features: {df.loc[best_idx, 'features']}")
        print(f"  Valid acc: {df.loc[best_idx, 'best_valid_acc']:.4f}")
        print(f"  Test acc: {df.loc[best_idx, 'test_acc']:.4f}")
        print("")
        
        return  # Exit after grid search
    
    # SINGLE RUN for 2(a) and (b)
    else:
        if downsample:
            X_train = build_downsampled_features(X_train)
            X_valid = build_downsampled_features(X_valid)
            X_test  = build_downsampled_features(X_test)

        if debug:
            print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"Unique classes in training set: {np.unique(y_train)} (total {len(np.unique(y_train))})")
            print(f"Validation data shape: X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
            print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            # Show 10 letters from training data, labels as letters A-Z
            show_letters(X_train[:, 1:], y_train, n=10)

        n_classes = np.unique(y_train).size
        n_feats = X_train.shape[1]

        # initialize the model
        model = LogisticRegression(n_classes, n_feats)

        epochs = np.arange(1, args.epochs + 1)

        train_accs = []
        valid_accs = []

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
        best_model = LogisticRegression.load(args.save_path)
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
    # Single run parameters (2(a)+(b))
    parser.add_argument('--epochs', default=20, type=int,
                        help="Number of epochs to train for.")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="best_logreg.pkl")
    if downsample: # change at the beginning of this file
        parser.add_argument("--accuracy-plot", default="Q1-2b-logreg-accs.pdf")
        parser.add_argument("--scores", default="Q1-2b-logreg-scores.json")
    else:
        parser.add_argument("--accuracy-plot", default="Q1-2a-logreg-accs.pdf")
        parser.add_argument("--scores", default="Q1-2a-logreg-scores.json")
    
    # Grid search parameters (2(c))
    parser.add_argument('--grid-search', action='store_true', 
                       help="Run grid search for 2(c) [3 etas x 2 l2 x 2 features = 12 configs]")
    parser.add_argument('--grid-results', default="Q1-2c-grid-results.json",
                       help="Output JSON for grid search results")
    
    args = parser.parse_args()
    main(args)
