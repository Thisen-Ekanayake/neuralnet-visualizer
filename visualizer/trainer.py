import math
import random
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
}
MAX_HIDDEN_LAYERS = 8
MAX_UNITS = 256
PROGRESS_INTERVAL = 0.1
SNAPSHOT_INTERVAL = 0.5
PACED_SNAPSHOT_INTERVAL = 0.25


class Network(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        sizes = [784] + [units for units, _ in hidden] + [10]
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.activations = nn.ModuleList(ACTIVATIONS[name]() for _, name in hidden)

    def trace(self, x):
        pre_acts, post_acts = [], []
        for linear, activation in zip(self.linears, self.activations):
            z = linear(x)
            x = activation(z)
            pre_acts.append(z)
            post_acts.append(x)
        return self.linears[-1](x), pre_acts, post_acts

    def forward(self, x):
        return self.trace(x)[0]


def neuron_health(activation, z):
    """Classify each neuron from its pre-activations `z` over the test set.

    Returns (kind, rate, mask): `rate` is the fraction of images where the neuron
    is active (ReLU family / GELU) or saturated (tanh / sigmoid); `mask` flags
    neurons that are dead/stuck/saturated. Linear layers have no such state.
    """
    if activation in ("relu", "leaky_relu"):
        rate = (z > 0).float().mean(0)
        return ("dead" if activation == "relu" else "stuck"), rate, rate == 0
    if activation == "gelu":
        rate = (z > -3).float().mean(0)
        return "dead", rate, rate == 0
    if activation in ("tanh", "sigmoid"):
        limit = 3.0 if activation == "tanh" else 5.0
        rate = (z.abs() > limit).float().mean(0)
        return "saturated", rate, rate >= 0.99
    return None, None, None


def clean(t, decimals=4):
    t = torch.nan_to_num(t.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6)
    return np.round(t.cpu().numpy(), decimals).tolist()


def finite(x):
    return x if math.isfinite(x) else None


def parse_hidden(hidden):
    if not isinstance(hidden, list) or len(hidden) > MAX_HIDDEN_LAYERS:
        raise ValueError(f"use 0 to {MAX_HIDDEN_LAYERS} hidden layers")
    parsed = []
    for layer in hidden:
        units, activation = int(layer["units"]), layer["activation"]
        if not 1 <= units <= MAX_UNITS:
            raise ValueError(f"each hidden layer needs 1 to {MAX_UNITS} neurons")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        parsed.append((units, activation))
    return parsed


class Session:
    """One browser connection: its model, its training thread, and what it streams back.

    Methods ending in `_locked` expect `self.lock` to be held by the caller.
    """

    def __init__(self, data, device, emit):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.device = device
        self.emit = emit
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.model = None
        self.hidden = []
        self.build_id = 0
        self.epoch = 0.0
        self.sample_index = 0

    def build(self, hidden, build_id):
        hidden = parse_hidden(hidden)
        self.stop_training()
        with self.lock:
            self.hidden = hidden
            self.build_id = int(build_id)
            self.model = Network(hidden).to(self.device)
            self.epoch = 0.0
        self.emit_snapshot()

    def start_training(self, epochs, lr, batch_size, optimizer, seconds_per_epoch=0):
        epochs, lr, batch_size = int(epochs), float(lr), int(batch_size)
        seconds_per_epoch = float(seconds_per_epoch)
        if not 0 <= seconds_per_epoch <= 600:
            raise ValueError("seconds per epoch must be between 0 and 600")
        if not 1 <= epochs <= 100:
            raise ValueError("epochs must be between 1 and 100")
        if not 0 < lr <= 10:
            raise ValueError("learning rate must be in (0, 10]")
        if not 1 <= batch_size <= 4096:
            raise ValueError("batch size must be between 1 and 4096")
        if optimizer not in ("adam", "sgd"):
            raise ValueError(f"unknown optimizer {optimizer!r}")
        if self.model is None:
            raise ValueError("no network built yet")
        self.stop_training()
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._train, args=(epochs, lr, batch_size, optimizer, seconds_per_epoch), daemon=True
        )
        self.thread.start()

    def stop_training(self):
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()
        self.thread = None

    def set_sample(self, index=None, mode=None):
        if self.model is None:
            raise ValueError("no network built yet")
        with self.lock:
            n = len(self.y_test)
            if mode == "random":
                index = random.randrange(n)
            elif mode == "misclassified":
                with torch.no_grad():
                    wrong = (self.model(self.x_test).argmax(1) != self.y_test).nonzero().flatten()
                index = wrong[random.randrange(len(wrong))].item() if len(wrong) else random.randrange(n)
            self.sample_index = max(0, min(n - 1, int(index)))
            message = self._forward_locked()
        self.emit(message)

    def emit_snapshot(self):
        with self.lock:
            messages = [self._weights_frame_locked(), self._stats_locked(), self._forward_locked()]
        for message in messages:
            self.emit(message)

    def _train(self, epochs, lr, batch_size, optimizer_name, seconds_per_epoch):
        model, build_id = self.model, self.build_id
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        n = len(self.y_train)
        steps = math.ceil(n / batch_size)
        start_epoch = self.epoch
        # A GPU runs these small nets far faster than the eye can follow, so
        # optionally pace steps to a wall-clock budget per epoch.
        seconds_per_step = seconds_per_epoch / steps
        snapshot_interval = PACED_SNAPSHOT_INTERVAL if seconds_per_step else SNAPSHOT_INTERVAL
        run_start = time.monotonic()
        steps_done = 0
        self.emit({"type": "status", "buildId": build_id, "training": True})

        loss_sum = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device)
        seen = 0
        last_progress = last_snapshot = time.monotonic()
        try:
            for epoch in range(epochs):
                perm = torch.randperm(n, device=self.device)
                for step in range(steps):
                    if self.stop_event.is_set():
                        return
                    idx = perm[step * batch_size : (step + 1) * batch_size]
                    xb, yb = self.x_train[idx], self.y_train[idx]
                    with self.lock:
                        logits = model(xb)
                        loss = F.cross_entropy(logits, yb)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        self.epoch = start_epoch + epoch + (step + 1) / steps
                    loss_sum += loss.detach() * len(idx)
                    correct += (logits.argmax(1) == yb).sum()
                    seen += len(idx)
                    steps_done += 1
                    if seconds_per_step:
                        delay = run_start + steps_done * seconds_per_step - time.monotonic()
                        if delay > 0:
                            self.stop_event.wait(delay)

                    now = time.monotonic()
                    if now - last_progress >= PROGRESS_INTERVAL:
                        self.emit({
                            "type": "progress",
                            "buildId": build_id,
                            "epoch": self.epoch,
                            "loss": finite(loss_sum.item() / seen),
                            "trainAcc": correct.item() / seen,
                        })
                        loss_sum.zero_()
                        correct.zero_()
                        seen = 0
                        last_progress = now
                    if now - last_snapshot >= snapshot_interval:
                        self.emit_snapshot()
                        last_snapshot = now
        except Exception as exc:
            self.emit({"type": "error", "message": f"training failed: {exc}"})
        finally:
            self.emit_snapshot()
            self.emit({"type": "status", "buildId": build_id, "training": False})

    def _weights_frame_locked(self):
        """Binary frame: uint32 [buildId, layerCount, (in, out) per layer], then float32 W, b per layer."""
        linears = self.model.linears
        header = [self.build_id, len(linears)]
        for linear in linears:
            header += [linear.in_features, linear.out_features]
        tensors = []
        for linear in linears:
            tensors += [linear.weight.detach().flatten(), linear.bias.detach()]
        values = torch.nan_to_num(torch.cat(tensors)).float().cpu().numpy()
        return np.array(header, dtype=np.uint32).tobytes() + values.tobytes()

    def _stats_locked(self):
        with torch.no_grad():
            logits, pre_acts, _ = self.model.trace(self.x_test)
            test_loss = F.cross_entropy(logits, self.y_test).item()
            test_acc = (logits.argmax(1) == self.y_test).float().mean().item()
            layers = []
            for (_, activation), z in zip(self.hidden, pre_acts):
                kind, rate, mask = neuron_health(activation, z)
                layers.append({
                    "activation": activation,
                    "kind": kind,
                    "rate": clean(rate, 3) if rate is not None else None,
                    "dead": mask.nonzero().flatten().tolist() if mask is not None else [],
                })
        return {
            "type": "stats",
            "buildId": self.build_id,
            "epoch": self.epoch,
            "testLoss": finite(test_loss),
            "testAcc": test_acc,
            "layers": layers,
        }

    def _forward_locked(self):
        i = self.sample_index
        x = self.x_test[i : i + 1]
        with torch.no_grad():
            logits, pre_acts, post_acts = self.model.trace(x)
            probs = torch.softmax(logits, 1)[0]
        return {
            "type": "forward",
            "buildId": self.build_id,
            "index": i,
            "label": int(self.y_test[i]),
            "pred": int(probs.argmax()),
            "probs": clean(probs),
            "pixels": (x[0] * 255).round().to(torch.uint8).tolist(),
            "preActivations": [clean(z[0]) for z in pre_acts],
            "activations": [clean(a[0]) for a in post_acts],
        }
