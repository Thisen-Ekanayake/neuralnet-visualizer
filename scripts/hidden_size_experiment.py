#!/usr/bin/env python3
"""Train a single-hidden-layer MLP on MNIST for several hidden-layer sizes
and report test accuracy for each, to see how accuracy scales with the
number of hidden neurons.

Usage:
    python3 hidden_size_experiment.py
    python3 hidden_size_experiment.py --sizes 1 5 10 20 50 --epochs 15
"""
import argparse
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MNIST"


def read_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic} in {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows * cols)


def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic} in {path}"
        return np.frombuffer(f.read(), dtype=np.uint8)


class OneHiddenLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


def load_data(device):
    x_train = read_idx_images(DATA_DIR / "train-images.idx3-ubyte").astype(np.float32) / 255.0
    y_train = read_idx_labels(DATA_DIR / "train-labels.idx1-ubyte").astype(np.int64)
    x_test = read_idx_images(DATA_DIR / "t10k-images.idx3-ubyte").astype(np.float32) / 255.0
    y_test = read_idx_labels(DATA_DIR / "t10k-labels.idx1-ubyte").astype(np.int64)

    return (
        torch.from_numpy(x_train).to(device),
        torch.from_numpy(y_train).to(device),
        torch.from_numpy(x_test).to(device),
        torch.from_numpy(y_test).to(device),
    )


def train_and_eval(hidden_size, x_train, y_train, x_test, y_test, epochs, batch_size, lr, device, seed):
    torch.manual_seed(seed)
    model = OneHiddenLayerMLP(784, hidden_size, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = x_train.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = x_train[idx], y_train[idx]

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()
    n_params = sum(p.numel() for p in model.parameters())
    return accuracy, n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=Path, default=Path("hidden_size_accuracy.png"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    x_train, y_train, x_test, y_test = load_data(device)
    print(f"train: {tuple(x_train.shape)}  test: {tuple(x_test.shape)}\n")

    results = []
    for hidden_size in args.sizes:
        t0 = time.time()
        acc, n_params = train_and_eval(
            hidden_size, x_train, y_train, x_test, y_test,
            args.epochs, args.batch_size, args.lr, device, args.seed,
        )
        dt = time.time() - t0
        results.append((hidden_size, acc, n_params))
        print(f"hidden={hidden_size:>3}  params={n_params:>7,}  test_accuracy={acc*100:6.2f}%  ({dt:.1f}s)")

    print("\nsummary:")
    print(f"{'hidden neurons':>15} | {'params':>8} | {'accuracy':>9}")
    print("-" * 40)
    for hidden_size, acc, n_params in results:
        print(f"{hidden_size:>15} | {n_params:>8,} | {acc*100:>8.2f}%")

    try:
        import matplotlib.pyplot as plt

        sizes = [r[0] for r in results]
        accs = [r[1] * 100 for r in results]
        plt.figure(figsize=(6, 4))
        plt.plot(sizes, accs, marker="o")
        plt.xscale("log")
        plt.xlabel("hidden layer neurons (log scale)")
        plt.ylabel("test accuracy (%)")
        plt.title("MNIST test accuracy vs. hidden layer size\n(single hidden layer MLP)")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"\nplot saved to {args.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
