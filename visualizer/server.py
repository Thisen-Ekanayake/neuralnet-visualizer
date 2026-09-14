#!/usr/bin/env python3
"""3D neural network visualizer: FastAPI server that trains MNIST models with PyTorch
and streams weights/activations to a three.js page.

Usage:
    python3 visualizer/server.py            # GPU if CUDA is available, else CPU; open http://127.0.0.1:8000
    python3 visualizer/server.py --gpu      # require the GPU
    python3 visualizer/server.py --cpu      # train on the CPU
    python3 visualizer/server.py --port 8080
"""
import argparse
import asyncio
import platform
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mnist_data import load_mnist
from trainer import ACTIVATIONS, MAX_HIDDEN_LAYERS, MAX_UNITS, Session

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEVICE = None
DEVICE_KIND = None
DEVICE_DETAILS = None
DATA = None


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def describe_device(device):
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"{props.name} · {props.total_memory / 1024**3:.1f} GB VRAM"
    return f"{cpu_name()} · {torch.get_num_threads()} threads"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


async def handle(session, msg):
    kind = msg.get("type")
    if kind == "build":
        await asyncio.to_thread(session.build, msg["hidden"], msg["buildId"])
    elif kind == "train":
        await asyncio.to_thread(
            session.start_training,
            msg["epochs"], msg["lr"], msg["batchSize"], msg["optimizer"], msg.get("secondsPerEpoch", 0),
        )
    elif kind == "stop":
        await asyncio.to_thread(session.stop_training)
    elif kind == "sample":
        await asyncio.to_thread(session.set_sample, msg.get("index"), msg.get("mode"))
    else:
        raise ValueError(f"unknown message type {kind!r}")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    outbox = asyncio.Queue()

    def emit(message):
        loop.call_soon_threadsafe(outbox.put_nowait, message)

    async def sender():
        while True:
            message = await outbox.get()
            if isinstance(message, bytes):
                await ws.send_bytes(message)
            else:
                await ws.send_json(message)

    sender_task = asyncio.create_task(sender())
    session = Session(DATA, DEVICE, emit)
    emit({
        "type": "hello",
        "device": DEVICE_DETAILS,
        "deviceKind": DEVICE_KIND,
        "maxLayers": MAX_HIDDEN_LAYERS,
        "maxUnits": MAX_UNITS,
        "activations": list(ACTIVATIONS),
    })
    try:
        while True:
            msg = await ws.receive_json()
            try:
                await handle(session, msg)
            except (KeyError, TypeError, ValueError) as exc:
                emit({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        await asyncio.to_thread(session.stop_training)
        sender_task.cancel()


def main():
    global DATA, DEVICE, DEVICE_KIND, DEVICE_DETAILS
    parser = argparse.ArgumentParser()
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--gpu", action="store_true", help="train on the GPU (default when CUDA is available)")
    device_group.add_argument("--cpu", action="store_true", help="train on the CPU")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.gpu and not torch.cuda.is_available():
        parser.error("--gpu was given but PyTorch cannot see a CUDA GPU; use --cpu")
    use_gpu = not args.cpu and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_gpu else "cpu")
    DEVICE_KIND = "gpu" if use_gpu else "cpu"
    DEVICE_DETAILS = describe_device(DEVICE)

    DATA = load_mnist(DEVICE)
    print(f"Training on {DEVICE_KIND.upper()}: {DEVICE_DETAILS}")
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
