#!/usr/bin/env python3
"""View images/labels from MNIST IDX-format files.

Usage:
    python3 view_mnist.py --index 0
    python3 view_mnist.py --index 0 --set test
    python3 view_mnist.py --index 0 --png out.png
    python3 view_mnist.py --count 10 --png-dir out/
"""
import argparse
import struct
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MNIST"

FILES = {
    "train": (DATA_DIR / "train-images.idx3-ubyte", DATA_DIR / "train-labels.idx1-ubyte"),
    "test": (DATA_DIR / "t10k-images.idx3-ubyte", DATA_DIR / "t10k-labels.idx1-ubyte"),
}

ASCII_RAMP = " .:-=+*#%@"


def read_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected magic number {magic} in {path}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def to_ascii(image):
    lines = []
    for row in image:
        lines.append("".join(ASCII_RAMP[int(pixel) * (len(ASCII_RAMP) - 1) // 255] for pixel in row))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["train", "test"], default="train")
    parser.add_argument("--index", type=int, default=0, help="index of a single image to view")
    parser.add_argument("--count", type=int, help="export this many images starting at --index")
    parser.add_argument("--png", type=Path, help="save the single image as a PNG file")
    parser.add_argument("--png-dir", type=Path, help="save --count images as PNGs into this dir")
    args = parser.parse_args()

    images_path, labels_path = FILES[args.set]
    images = read_idx_images(images_path)
    labels = read_idx_labels(labels_path)

    if args.count:
        out_dir = args.png_dir or Path("mnist_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        for i in range(args.index, args.index + args.count):
            Image.fromarray(images[i]).save(out_dir / f"{args.set}_{i:05d}_label{labels[i]}.png")
        print(f"Wrote {args.count} PNGs to {out_dir}/")
        return

    image, label = images[args.index], labels[args.index]
    print(f"{args.set} set, index {args.index}, label={label}\n")
    print(to_ascii(image))

    if args.png:
        from PIL import Image

        Image.fromarray(image).save(args.png)
        print(f"\nSaved to {args.png}")


if __name__ == "__main__":
    main()
