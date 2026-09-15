import copy
import json
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
OPTIMIZERS = ("adam", "sgd", "sgd_plain")
# Per-parameter optimizer state reported by a single step: {short name: PyTorch's state key}.
OPTIMIZER_STATE = {"adam": {"m": "exp_avg", "v": "exp_avg_sq"}, "sgd": {"buf": "momentum_buffer"}, "sgd_plain": {}}
MAX_HIDDEN_LAYERS = 8
MAX_UNITS = 256
PROGRESS_INTERVAL = 0.1
SNAPSHOT_INTERVAL = 0.5
PACED_SNAPSHOT_INTERVAL = 0.25
FRAME_WEIGHTS = 0
FRAME_STEP = 1
LR_CURVE_POINTS = 41
LR_CURVE_TEST_IMAGES = 1000
GRADCHECK_EPS = 1e-5


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


def trace_params(activations, params, x):
    """`Network.trace` with explicit [W1, b1, W2, b2, ...] tensors in place of the model's own."""
    pre_acts = []
    for k, activation in enumerate(activations):
        z = F.linear(x, params[2 * k], params[2 * k + 1])
        x = activation(z)
        pre_acts.append(z)
    return F.linear(x, params[-2], params[-1]), pre_acts


def activation_slope(activation, z):
    """σ'(z) elementwise, taken with autograd so it is exactly the slope backprop used."""
    with torch.enable_grad():
        z = z.detach().requires_grad_(True)
        (slope,) = torch.autograd.grad(activation(z).sum(), z)
    return slope


def make_optimizer(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.SGD(params, lr=lr, momentum=0.9 if name == "sgd" else 0.0)


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


def pack_frame(kind, header, blocks):
    """Binary frame: uint32 [kind, header bytes], a JSON header, then float32 data.

    `blocks` is a list of (name, tensor); the header lists them as [name, offset, length] in floats.
    The JSON is space-padded to a multiple of 4 bytes so the float32 data stays aligned.
    """
    index, offset = [], 0
    for name, tensor in blocks:
        index.append([name, offset, tensor.numel()])
        offset += tensor.numel()
    text = json.dumps({**header, "blocks": index}, separators=(",", ":"), allow_nan=False).encode()
    text += b" " * (-len(text) % 4)
    data = torch.cat([tensor.detach().float().flatten() for _, tensor in blocks])
    data = torch.nan_to_num(data).cpu().numpy()
    return np.array([kind, len(text)], dtype=np.uint32).tobytes() + text + data.tobytes()


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


def check_optimizer(lr, optimizer):
    if not 0 < lr <= 10:
        raise ValueError("learning rate must be in (0, 10]")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}")


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
        # Kept across training runs and single steps, so Adam/momentum state accumulates.
        self.optimizer = None
        self.optimizer_name = None
        self.single_steps = 0
        self.step_id = 0
        self.last_step = None  # the latest single step's inputs, for the gradient check
        self.undo = None  # the state before the latest single step, until training or a rebuild

    def build(self, hidden, build_id):
        hidden = parse_hidden(hidden)
        self.stop_training()
        with self.lock:
            self.hidden = hidden
            self.build_id = int(build_id)
            self.model = Network(hidden).to(self.device)
            self.epoch = 0.0
            self.optimizer = self.optimizer_name = None
            self.single_steps = 0
            self.last_step = self.undo = None
        self.emit_snapshot()

    def start_training(self, epochs, lr, batch_size, optimizer, seconds_per_epoch=0):
        epochs, lr, batch_size = int(epochs), float(lr), int(batch_size)
        seconds_per_epoch = float(seconds_per_epoch)
        if not 0 <= seconds_per_epoch <= 600:
            raise ValueError("seconds per epoch must be between 0 and 600")
        if not 1 <= epochs <= 100:
            raise ValueError("epochs must be between 1 and 100")
        if not 1 <= batch_size <= 4096:
            raise ValueError("batch size must be between 1 and 4096")
        check_optimizer(lr, optimizer)
        if self.model is None:
            raise ValueError("no network built yet")
        self.stop_training()
        with self.lock:
            self.undo = None
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

    def _require_idle(self, action):
        if self.model is None:
            raise ValueError("no network built yet")
        if self.thread is not None and self.thread.is_alive():
            raise ValueError(f"stop training before {action}")

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

    def train_image(self, index=None, mode=None):
        """Preview a training image (and the current prediction) before stepping on it."""
        if self.model is None:
            raise ValueError("no network built yet")
        with self.lock:
            index = self._train_index_locked(index, mode)
            x = self.x_train[index : index + 1]
            with torch.no_grad():
                pred = int(self.model(x).argmax(1))
            message = {
                "type": "trainImage",
                "buildId": self.build_id,
                "index": index,
                "label": int(self.y_train[index]),
                "pred": pred,
                "pixels": (x[0] * 255).round().to(torch.uint8).tolist(),
            }
        self.emit(message)

    def _train_index_locked(self, index, mode):
        n = len(self.y_train)
        if mode == "random":
            return random.randrange(n)
        if mode == "misclassified":
            with torch.no_grad():
                wrong = (self.model(self.x_train).argmax(1) != self.y_train).nonzero().flatten()
            return wrong[random.randrange(len(wrong))].item() if len(wrong) else random.randrange(n)
        return max(0, min(n - 1, int(index)))

    def emit_snapshot(self):
        with self.lock:
            messages = [self._weights_frame_locked(), self._stats_locked(), self._forward_locked()]
        for message in messages:
            self.emit(message)

    def _optimizer_locked(self, name, lr):
        if self.optimizer is None or self.optimizer_name != name:
            self.optimizer = make_optimizer(name, self.model.parameters(), lr)
            self.optimizer_name = name
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return self.optimizer

    def _train(self, epochs, lr, batch_size, optimizer_name, seconds_per_epoch):
        with self.lock:
            model, build_id = self.model, self.build_id
            optimizer = self._optimizer_locked(optimizer_name, lr)
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

    # ---------- single steps ----------

    def single_step(self, index=None, mode=None, lr=0.001, optimizer="adam"):
        """One optimizer step on one training image, reported in full as a step frame."""
        lr = float(lr)
        check_optimizer(lr, optimizer)
        self._require_idle("taking a single step")
        with self.lock:
            frame = self._single_step_locked(self._train_index_locked(index, mode), lr, optimizer)
        self.emit(frame)
        self.emit_snapshot()

    def undo_step(self):
        self._require_idle("undoing a step")
        with self.lock:
            record = self.undo
            if record is None:
                raise ValueError("nothing to undo")
            with torch.no_grad():
                for param, saved in zip(self.model.parameters(), record["params"]):
                    param.copy_(saved)
            if record["optimizer"] is None:
                self.optimizer = self.optimizer_name = None
            else:
                name, state = record["optimizer"]
                self.optimizer = make_optimizer(name, self.model.parameters(), state["param_groups"][0]["lr"])
                self.optimizer.load_state_dict(copy.deepcopy(state))
                self.optimizer_name = name
            self.epoch = record["epoch"]
            self.single_steps = record["single_steps"]
            self.undo = None
            message = {"type": "undone", "buildId": self.build_id, "stepId": self.step_id}
        self.emit(message)
        self.emit_snapshot()

    def gradcheck(self, step_id, layer, row, col):
        """Finite-difference ∂L/∂W[layer][row, col] at the recorded step, in float64."""
        layer, row, col = int(layer), int(row), int(col)
        with self.lock:
            record = self.last_step
            if record is None or record["id"] != int(step_id):
                raise ValueError("that step is no longer available; take a new step")
            params = [p.double() for p in record["params"]]
            k = 2 * layer
            if not (0 <= layer < len(params) // 2 and 0 <= row < params[k].shape[0] and 0 <= col < params[k].shape[1]):
                raise ValueError("no such weight")
            x, y = record["x"].double(), record["y"]

            def loss_at(shift):
                shifted = list(params)
                shifted[k] = params[k].clone()
                shifted[k][row, col] += shift
                with torch.no_grad():
                    logits, pre_acts = trace_params(self.model.activations, shifted, x)
                return F.cross_entropy(logits, y).item(), pre_acts

            loss_plus, pre_plus = loss_at(GRADCHECK_EPS)
            loss_minus, pre_minus = loss_at(-GRADCHECK_EPS)
            # A ReLU input that changes sign between w+ε and w−ε means the difference straddles a kink.
            kink = any(
                bool(((zp > 0) != (zm > 0)).any())
                for (_, activation), zp, zm in zip(self.hidden, pre_plus, pre_minus)
                if activation in ("relu", "leaky_relu")
            )
            message = {
                "type": "gradcheck",
                "buildId": self.build_id,
                "stepId": record["id"],
                "layer": layer,
                "row": row,
                "col": col,
                "eps": GRADCHECK_EPS,
                "weight": params[k][row, col].item(),
                "lossPlus": finite(loss_plus),
                "lossMinus": finite(loss_minus),
                "numeric": finite((loss_plus - loss_minus) / (2 * GRADCHECK_EPS)),
                "backprop": record["grads"][k][row, col].item(),
                "kink": kink,
            }
        self.emit(message)

    @staticmethod
    def _optimizer_state(optimizer, param, keys):
        """{short name: tensor} for one parameter; zeros where the optimizer has no state yet."""
        state = optimizer.state.get(param, {})
        return {
            short: state[key].detach().clone() if key in state else torch.zeros_like(param)
            for short, key in keys.items()
        }

    def _single_step_locked(self, index, lr, optimizer_name):
        model = self.model
        params = list(model.parameters())  # W1, b1, W2, b2, ...
        x, y = self.x_train[index : index + 1], self.y_train[index : index + 1]
        before = [p.detach().clone() for p in params]
        # Save the optimizer before _optimizer_locked can replace it, so undo brings back the old one.
        self.undo = {
            "params": before,
            "optimizer": None if self.optimizer is None
            else (self.optimizer_name, copy.deepcopy(self.optimizer.state_dict())),
            "epoch": self.epoch,
            "single_steps": self.single_steps,
        }
        optimizer = self._optimizer_locked(optimizer_name, lr)
        state_keys = OPTIMIZER_STATE[optimizer_name]
        state_before = [self._optimizer_state(optimizer, p, state_keys) for p in params]
        first_step = not optimizer.state.get(params[0])

        logits, pre_acts, post_acts = model.trace(x)
        for t in (*pre_acts, *post_acts, logits):
            t.retain_grad()
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad.detach().clone() for p in params]
        optimizer.step()
        after = [p.detach().clone() for p in params]
        state_after = [self._optimizer_state(optimizer, p, state_keys) for p in params]
        slopes = [activation_slope(act, z) for act, z in zip(model.activations, pre_acts)]

        with torch.no_grad():
            probs = torch.softmax(logits, 1)[0]
            logits_after = model(x)
            loss_after = F.cross_entropy(logits_after, y).item()
            probs_after = torch.softmax(logits_after, 1)[0]
            # The loss change that the gradient predicts for this update (a first-order Taylor estimate).
            first_order = sum(
                (g.double() * (a.double() - b.double())).sum().item() for g, a, b in zip(grads, after, before)
            )
            lr_curve = self._lr_curve_locked(before, after, lr, x, y)

        self.epoch += 1 / len(self.y_train)
        self.single_steps += 1
        self.step_id += 1
        self.last_step = {"id": self.step_id, "x": x, "y": y, "params": before, "grads": grads}

        group = optimizer.param_groups[0]
        info = {"name": optimizer_name, "firstStep": first_step}
        if optimizer_name == "adam":
            info.update(
                t=int(optimizer.state[params[0]]["step"].item()),
                beta1=group["betas"][0], beta2=group["betas"][1], eps=group["eps"],
            )
        else:
            info["momentum"] = group["momentum"]
        header = {
            "buildId": self.build_id,
            "stepId": self.step_id,
            "index": index,
            "label": int(y[0]),
            "pred": int(probs.argmax()),
            "loss": finite(loss.item()),
            "lr": lr,
            "optimizer": info,
            "after": {"loss": finite(loss_after), "pred": int(probs_after.argmax())},
            "firstOrder": finite(first_order),
            "lrCurve": lr_curve,
            "singleSteps": self.single_steps,
        }
        blocks = [
            ("input", x[0]),
            ("logits", logits[0]),
            ("probs", probs),
            ("deltaOut", logits.grad[0]),
            ("probsAfter", probs_after),
        ]
        for h, (z, a, slope) in enumerate(zip(pre_acts, post_acts, slopes)):
            blocks += [
                (f"hidden.{h}.z", z[0]),
                (f"hidden.{h}.a", a[0]),
                (f"hidden.{h}.slope", slope[0]),
                (f"hidden.{h}.dLda", a.grad[0]),
                (f"hidden.{h}.delta", z.grad[0]),
            ]
        for k in range(len(params)):
            blocks += [(f"param.{k}.before", before[k]), (f"param.{k}.grad", grads[k]), (f"param.{k}.after", after[k])]
            for short in state_keys:
                blocks += [
                    (f"param.{k}.{short}Before", state_before[k][short]),
                    (f"param.{k}.{short}After", state_after[k][short]),
                ]
        return pack_frame(FRAME_STEP, header, blocks)

    def _lr_curve_locked(self, before, after, lr, x, y):
        """Loss had the step used learning rate η: w − η·d, where d = (w − w′) / lr.

        For plain SGD, momentum and Adam the direction d doesn't depend on the learning rate,
        so every point is exactly the step that η would have taken. Measured on the step's image
        and on test images: one image's loss keeps falling as the step grows (the network just
        memorizes it), so overshooting only shows up on images the step didn't see.
        """
        activations = self.model.activations
        before64 = [b.double() for b in before]
        directions = [(b - a.double()) / lr for b, a in zip(before64, after)]
        x64 = x.double()
        x_test, y_test = self.x_test[:LR_CURVE_TEST_IMAGES], self.y_test[:LR_CURVE_TEST_IMAGES]
        lrs = np.geomspace(lr / 100, lr * 100, LR_CURVE_POINTS)
        losses, test_losses = [], []
        for eta in lrs:
            stepped = [b - float(eta) * d for b, d in zip(before64, directions)]
            logits, _ = trace_params(activations, stepped, x64)
            losses.append(finite(F.cross_entropy(logits, y).item()))
            logits, _ = trace_params(activations, [s.float() for s in stepped], x_test)
            test_losses.append(finite(F.cross_entropy(logits, y_test).item()))
        logits, _ = trace_params(activations, before, x_test)
        return {
            "lrs": lrs.tolist(),
            "losses": losses,
            "testLosses": test_losses,
            "testLossBefore": finite(F.cross_entropy(logits, y_test).item()),
            "testImages": len(y_test),
        }

    # ---------- snapshots ----------

    def _weights_frame_locked(self):
        """Binary frame: uint32 [FRAME_WEIGHTS, buildId, layerCount, (in, out) per layer], then float32 W, b per layer."""
        linears = self.model.linears
        header = [FRAME_WEIGHTS, self.build_id, len(linears)]
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
            "singleSteps": self.single_steps,
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
