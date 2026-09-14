#!/usr/bin/env python3
"""3D neural network visualizer: FastAPI server that trains MNIST models on the GPU
and streams weights/activations to a three.js page.

Usage:
    python3 visualizer/server.py            # then open http://127.0.0.1:8000
    python3 visualizer/server.py --port 8080
"""
import argparse
import asyncio
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mnist_data import load_mnist
from trainer import ACTIVATIONS, MAX_HIDDEN_LAYERS, MAX_UNITS, Session

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
DATA = None

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
        "device": DEVICE_NAME,
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
    global DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    DATA = load_mnist(DEVICE)
    print(f"MNIST loaded on {DEVICE_NAME}")
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
