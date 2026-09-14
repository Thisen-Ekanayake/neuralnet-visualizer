import struct
from pathlib import Path

import numpy as np
import torch

MNIST_DIR = Path(__file__).resolve().parent.parent / "data" / "MNIST"


def read_idx(path):
    with open(path, "rb") as f:
        zero, dtype_code, ndim = struct.unpack(">HBB", f.read(4))
        if zero != 0 or dtype_code != 0x08:
            raise ValueError(f"{path} is not an unsigned-byte IDX file")
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(device):
    """Return ((x_train, y_train), (x_test, y_test)) as tensors on `device`, pixels scaled to [0, 1]."""

    def split(images_file, labels_file):
        x = read_idx(MNIST_DIR / images_file).reshape(-1, 784).astype(np.float32) / 255.0
        y = read_idx(MNIST_DIR / labels_file).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    return (
        split("train-images.idx3-ubyte", "train-labels.idx1-ubyte"),
        split("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"),
    )
