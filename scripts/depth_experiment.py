#!/usr/bin/env python3
"""Train an MLP on MNIST where every hidden layer has a fixed width (default
1 neuron) and only the number of hidden layers varies, to see how accuracy
scales with depth alone (as opposed to hidden_size_experiment.py, which
varies width of a single hidden layer).

Usage:
    python3 depth_experiment.py
    python3 depth_experiment.py --depths 1 2 3 4 5 --width 1 --epochs 15
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from hidden_size_experiment import load_data


class DeepNarrowMLP(nn.Module):
    def __init__(self, input_size, width, depth, output_size):
        super().__init__()
        layers = [nn.Linear(input_size, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_and_eval(depth, width, x_train, y_train, x_test, y_test, epochs, batch_size, lr, device, seed):
    torch.manual_seed(seed)
    model = DeepNarrowMLP(784, width, depth, 10).to(device)
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
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--width", type=int, default=1, help="neurons per hidden layer")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=Path, default=Path("depth_accuracy.png"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    x_train, y_train, x_test, y_test = load_data(device)
    print(f"train: {tuple(x_train.shape)}  test: {tuple(x_test.shape)}")
    print(f"hidden layer width fixed at {args.width}\n")

    results = []
    for depth in args.depths:
        t0 = time.time()
        acc, n_params = train_and_eval(
            depth, args.width, x_train, y_train, x_test, y_test,
            args.epochs, args.batch_size, args.lr, device, args.seed,
        )
        dt = time.time() - t0
        results.append((depth, acc, n_params))
        print(f"hidden_layers={depth}  params={n_params:>6,}  test_accuracy={acc*100:6.2f}%  ({dt:.1f}s)")

    print("\nsummary:")
    print(f"{'hidden layers':>14} | {'params':>7} | {'accuracy':>9}")
    print("-" * 38)
    for depth, acc, n_params in results:
        print(f"{depth:>14} | {n_params:>7,} | {acc*100:>8.2f}%")

    try:
        import matplotlib.pyplot as plt

        depths = [r[0] for r in results]
        accs = [r[1] * 100 for r in results]
        plt.figure(figsize=(6, 4))
        plt.plot(depths, accs, marker="o")
        plt.xlabel("number of hidden layers")
        plt.ylabel("test accuracy (%)")
        plt.xticks(depths)
        plt.title(f"MNIST test accuracy vs. depth\n(width={args.width} neuron(s) per hidden layer)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"\nplot saved to {args.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
