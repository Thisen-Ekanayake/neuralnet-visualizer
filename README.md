# NeuralNet

Small experiments on how a fully connected neural network's **width** (neurons per hidden layer) and **depth** (number of hidden layers) affect MNIST accuracy, plus an interactive **3D visualizer** that trains networks on the GPU or CPU, shows their weights, activations, and dead neurons live, and walks through a single training step (forward pass, loss, backpropagation, chain rule, gradient descent, and the optimizer) as formulas and as the network's actual numbers.

![The visualizer: orbiting the 3D network, training it, backpropagation stage by stage, and a narrow network of dead ReLUs](demo/neuralnet-demo.gif)

## Contents

```
scripts/
  view_mnist.py              View MNIST digits as ASCII art or export them as PNGs
  hidden_size_experiment.py  Width study: 1 hidden layer with 1, 5, 10, 20, 50 neurons
  depth_experiment.py        Depth study: 1–5 hidden layers of a fixed width
visualizer/
  server.py                  FastAPI server: trains models with PyTorch, streams state over a WebSocket
  trainer.py                 Model definition, training thread, single traced steps, dead-neuron detection
  mnist_data.py              IDX-format MNIST loader
  static/                    The page: scene.js draws with WebGL, scene_cpu.js with a CPU canvas,
                             walkthrough.js is the Training step tab
                             (three.js and KaTeX are vendored in static/vendor, so no internet is needed)
*.png                        Plots and a sample digit produced by the scripts
```

## Setup

### 1. Python packages

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch that matches your driver (see https://pytorch.org/get-started/locally/). Everything also runs on the CPU, just slower.

Tested with Python 3.10.13, torch 2.10.0 (CUDA 12.8), numpy 2.2.6, matplotlib 3.10.8, Pillow 12.1.0, fastapi 0.128.0, uvicorn 0.40.0, websockets 16.0, on an RTX 4060 Laptop GPU.

### 2. MNIST dataset

The dataset is not committed. Put the four uncompressed IDX files in `data/MNIST/`, named exactly like this:

```
data/MNIST/train-images.idx3-ubyte
data/MNIST/train-labels.idx1-ubyte
data/MNIST/t10k-images.idx3-ubyte
data/MNIST/t10k-labels.idx1-ubyte
```

If the dataset already lives somewhere else, link to it instead of copying:

```bash
mkdir -p data
ln -s /path/to/your/MNIST data/MNIST
```

If your files end in `.gz`, decompress them first with `gunzip *.gz`. Files named with a dash (`train-images-idx3-ubyte`) have the same contents; rename them to the dotted names above.

## Viewing digits

```bash
python3 scripts/view_mnist.py --index 0                       # ASCII art of training image 0
python3 scripts/view_mnist.py --index 0 --set test            # from the test set
python3 scripts/view_mnist.py --index 0 --png digit.png       # save as PNG
python3 scripts/view_mnist.py --count 20 --png-dir out/       # export 20 PNGs (label in the file name)
```

## Width experiment

One hidden layer (`784 → h → 10`, ReLU), trained for 15 epochs with Adam (lr 0.001, batch size 128, seed 0) for each hidden size, then evaluated on the 10,000 test images.

```bash
python3 scripts/hidden_size_experiment.py
python3 scripts/hidden_size_experiment.py --sizes 1 5 10 20 50 100 --epochs 15
```

Prints a results table and saves `hidden_size_accuracy.png` in the current directory (change it with `--plot`).

| Hidden neurons | Parameters | Test accuracy |
|---:|---:|---:|
| 1 | 805 | 30.38% |
| 5 | 3,985 | 90.30% |
| 10 | 7,960 | 92.83% |
| 20 | 15,910 | 95.53% |
| 50 | 39,760 | 97.27% |

One neuron squeezes every image into a single number, which is too little to separate 10 digits. Five neurons already reach 90%, and each further increase helps less.

## Depth experiment

Stacks 1–5 hidden layers that all have the same width, with the same training setup.

```bash
python3 scripts/depth_experiment.py                  # width 1
python3 scripts/depth_experiment.py --width 2 --plot depth_accuracy_w2.png
```

| Hidden layers | Width 1 | Width 2 |
|---:|---:|---:|
| 1 | 30.38% | 65.59% |
| 2 | 11.35% | 11.35% |
| 3 | 11.35% | 32.42% |
| 4 | 11.35% | 11.35% |
| 5 | 11.35% | 11.35% |

11.35% means the network predicts "1" for every image, since 1 is the most common digit in the test set. At these widths, each extra layer is another chance for every ReLU in the layer to go dead, so no signal reaches the output. Extra depth only helps when each layer is wide enough to carry information forward. Results at widths 1–2 vary with the random seed because whether a layer dies is partly luck.

## 3D visualizer

```bash
python3 visualizer/server.py              # GPU if PyTorch can see one, otherwise CPU; then open http://127.0.0.1:8000
python3 visualizer/server.py --gpu        # train on the GPU, render with WebGL (exits with an error if CUDA isn't available)
python3 visualizer/server.py --cpu        # train on the CPU, render on the CPU
python3 visualizer/server.py --cpu --port 8080
```

| | Training (server) | 3D rendering (browser) |
|---|---|---|
| `--gpu`, or no flag with CUDA available | PyTorch on the GPU | WebGL on the GPU, redrawn every frame |
| `--cpu` | PyTorch on the CPU | Software Canvas 2D on the CPU, redrawn only when something changes |
| no flag, no CUDA | PyTorch on the CPU | WebGL on the GPU |

CPU rendering creates no WebGL context. It draws into a canvas created with `willReadFrequently`, a hint that makes Chrome/Chromium keep the canvas in CPU memory and rasterize it in software (verified in Chrome; other browsers may treat the hint differently). It uses the same layout, colors, and interactions as the WebGL view, but at 1× resolution, and it draws at most 40k connections by default (every connection of 784→50→10). Expect roughly 15–40 fps while orbiting, depending on network size. The "Max connections drawn" slider goes up to 150k in this mode (roughly 100–250 ms per frame, depending on how long the lines are on screen). The browser still uses the GPU to put the finished page on screen, which a web page can't control.

Leave the server running while you use the page. It loads MNIST onto the chosen device at startup, prints which device it is using (the GPU's name and memory, or the CPU's model and PyTorch thread count), and each browser tab gets its own model.

**Left panel: build and train**
- Pick a preset (including the networks from the experiments above), or set the number of hidden layers (0–8), the neurons in each layer (1–256), and each layer's activation: ReLU, Leaky ReLU, GELU, Tanh, Sigmoid, or Linear.
- The parameter count updates immediately, with a per-layer weights/biases table.
- **Train** runs on the server's device with Adam, SGD with momentum 0.9, or plain SGD. **Speed** defaults to about 5 seconds per epoch so you can watch the network learn; *Full speed* trains as fast as the device allows. **Train more** continues from the current weights and the optimizer's state (Adam's running averages, the momentum buffer), as do single steps from the Training step tab; switching optimizers starts a fresh one. **Reinitialize** starts over with new random weights, and any architecture change also starts fresh.

**Center: the 3D network**
- Drag to orbit, right-drag to pan, scroll to zoom. The buttons at the top jump to preset camera angles or turn on auto-rotation.
- The input layer is a 28×28 grid showing the current digit. Neurons glow by activation (amber positive, cyan negative). Output neurons grow with their predicted probability.
- Connections are blue for positive weights and orange for negative, brighter for larger weights.
- Hover a neuron for its activation, bias, and firing rate. Click it to highlight its connections and show its details on the right.
- The bottom-left panel shows what renders the view (the browser's GPU and frame rate, or the CPU and milliseconds per frame), which device the server trains on (GPU name and memory, or CPU model and thread count), and how many connections are drawn.

**Right panel, Inspect tab**
- **Connections:** color by weight, or by *signal* (weight × input for the current digit) to see which paths the digit actually uses, or, after a single step, by each weight's gradient or by how much the step changed it. You can hide weak weights, adjust brightness, and cap how many lines are drawn (large networks are randomly subsampled above the cap, 150k by default, up to 700k).
- **Forward pass:** step through test images, jump to a random misclassified one, and see all 10 output probabilities.
- **Training progress:** test accuracy, train loss, and a live chart.
- **Dead neurons:** per-layer counts, measured on all 10,000 test images. ReLU/GELU neurons are *dead* if they never produce a positive output (so they get no gradient), Leaky ReLU neurons are *stuck* if always negative, and Tanh/Sigmoid neurons are *saturated* if they sit in the flat tail on at least 99% of images. Dead neurons are drawn in red.
- **Selected neuron:** for a first-hidden-layer neuron, its 784 incoming weights are drawn as a 28×28 image, which shows the pattern it responds to. Deeper and output neurons show their incoming weights as a bar chart, and input pixels show their outgoing weights.

**Right panel, Training step tab**

Walks through one real training step, one stage at a time. Each stage shows the formula, rendered with KaTeX, then the same formula with this network's numbers filled in.
- Pick a training image (by index, random, or one the network currently gets wrong) and press **Take one step**. The server runs one optimizer step on that single image (a batch of 1), with the learning rate and optimizer from the Training panel, and records every intermediate value. Training must be stopped first.
- Go through the stages with **◀ Prev / Next ▶**, the stage buttons, or the arrow keys:
  - **Forward pass**, one layer at a time: the weighted sum and activation of one neuron, with its largest terms written out.
  - **Softmax** and the cross-entropy **loss**.
  - **Backpropagation**: the output error δ = p − y, then δ for each hidden layer, including the activation's slope σ′. Neurons with σ′ = 0 stop the error.
  - **Chain rule** for one traced weight: ∂L/∂w = δᵢ·aⱼ, the strongest single path from the weight to the loss, and a map of the gradient of every weight in its layer. In the first layer each neuron's tile is the digit itself, scaled by that neuron's δ.
  - **Gradient descent**: w ← w − η·∂L/∂w.
  - **Optimizer**: what Adam or momentum actually did to the traced weight, using its real stored state.
  - **Result**: the loss before and after, the first-order estimate from the gradient, and the loss for a range of learning rates on this image and on 1,000 test images.
- The traced weight starts as the one with the largest gradient. Pick another with the controls, by clicking a tile of the gradient map, or by clicking a neuron in the 3D view.
- **Check with finite differences** nudges the traced weight by ±ε on the server and compares the change in the loss with the backprop gradient.
- **Undo step** restores the weights and the optimizer state, so you can retry the same image with a different learning rate.
- While the tab is open, the 3D view shows the step's training image and follows the stage it is on. Layers the pass hasn't reached yet stay grey, neurons are colored by their error δ during backprop, connections show the gradient or the update, and the traced weight is drawn in white. Untick *3D view follows the walkthrough* to keep the Inspect view.

## Troubleshooting

- **CPU mode feels slow when orbiting.** Lower "Max connections drawn" or raise "Hide weak weights"; drawing time scales with the number of visible lines.
- **The visualizer renders on the integrated GPU.** The bottom-left panel shows the browser's GPU. On a hybrid-graphics Linux laptop, start the browser on the NVIDIA GPU, for example `prime-run google-chrome-stable` (from the `nvidia-prime` package on Arch).
- **The experiment scripts exit with code 139 and a `Gdk-CRITICAL` message.** This happens as matplotlib's GTK backend shuts down, after the results table and plot have already been written. The results are fine.
- **`FileNotFoundError` for an `.idx` file.** Check the dataset names and location in [MNIST dataset](#2-mnist-dataset).
- **The visualizer page says "reconnecting…".** The server isn't running, or it's on a different port than the one in the URL.
