// The "Training step" tab: one real optimizer step on one training image, walked through stage by
// stage as formulas and as the same formulas with the network's actual numbers.
import { el, ctx2d, drawWeightBars, drawTiles, drawMatrix } from './draw.js';
import { T, num, factor, sigFor, vec, expandSum, sumRows, normalCdf, normalPdf, renderTex, texHtml as m } from './tex.js';

export const FRAME_STEP = 1;
const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');
const RED = '#ff5d73';
const GREEN = '#4ade80';
const red = (tex) => T`\textcolor{${RED}}{${tex}}`;
const SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const sup = (n) => String(n).replace(/\d/g, (d) => SUPERSCRIPT[d]).replace('-', '⁻');

const ACT = {
  relu: { name: 'ReLU', f: T`\max(0,\,z)`, d: T`\begin{cases}1 & z>0\\ 0 & z\le 0\end{cases}` },
  leaky_relu: { name: 'Leaky ReLU', f: T`\max(0.01z,\,z)`, d: T`\begin{cases}1 & z>0\\ 0.01 & z\le 0\end{cases}` },
  gelu: { name: 'GELU', f: T`z\,\Phi(z)`, d: T`\Phi(z)+z\,\varphi(z)` },
  tanh: { name: 'Tanh', f: T`\tanh z`, d: T`1-\tanh^2 z = 1-a^2` },
  sigmoid: { name: 'Sigmoid', f: T`\dfrac{1}{1+e^{-z}}`, d: T`a\,(1-a)` },
  linear: { name: 'Linear', f: T`z`, d: T`1` },
};
const OPTIMIZER = {
  adam: { name: 'Adam', chip: 'Adam' },
  sgd: { name: 'SGD + momentum', chip: 'momentum' },
  sgd_plain: { name: 'SGD (plain)', chip: 'SGD' },
};

function buildStages(L) {
  const stages = [{ kind: 'input', group: 'Forward', chip: 'x' }];
  for (let l = 1; l <= L; l++) stages.push({ kind: 'forward', layer: l, group: 'Forward', chip: l === L ? 'Out' : `H${l}` });
  stages.push({ kind: 'softmax', group: 'Loss', chip: 'softmax' }, { kind: 'loss', group: 'Loss', chip: 'L' });
  stages.push({ kind: 'outputError', group: 'Backprop', chip: 'Out' });
  for (let l = L - 1; l >= 1; l--) stages.push({ kind: 'backprop', layer: l, group: 'Backprop', chip: `H${l}` });
  stages.push({ kind: 'chain', group: 'Chain rule', chip: '∂L/∂w' });
  stages.push({ kind: 'descent', group: 'Update', chip: 'GD' }, { kind: 'optimizer', group: 'Update', chip: 'optimizer' });
  stages.push({ kind: 'result', group: 'Result', chip: 'ΔL' });
  return stages;
}

/** Step frame: uint32 [FRAME_STEP, header bytes], a JSON header listing its float32 blocks, then the blocks. */
export function parseStepFrame(buffer) {
  const headerBytes = new Uint32Array(buffer, 4, 1)[0];
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerBytes)));
  const start = 8 + headerBytes;
  const trace = { ...header, hidden: [], params: [] };
  delete trace.blocks;
  for (const [name, offset, length] of header.blocks) {
    const values = new Float32Array(buffer, start + offset * 4, length);
    const [group, index, key] = name.split('.');
    if (group === 'hidden') (trace.hidden[index] ??= {})[key] = values;
    else if (group === 'param') (trace.params[index] ??= {})[key] = values;
    else trace[group] = values;
  }
  return trace;
}

/** Arranges a parsed trace by layer; returns null if it doesn't match the architecture. */
function prepare(t, sizes) {
  const L = sizes.length - 1;
  if (t.params.length !== 2 * L || t.hidden.length !== L - 1) return null;
  t.layers = [];
  for (let l = 0; l < L; l++) {
    const W = t.params[2 * l], b = t.params[2 * l + 1];
    if (W.before.length !== sizes[l] * sizes[l + 1]) return null;
    const update = new Float32Array(W.before.length);
    for (let k = 0; k < update.length; k++) update[k] = W.after[k] - W.before[k];
    t.layers.push({ inN: sizes[l], outN: sizes[l + 1], W, b, update });
  }
  t.acts = [t.input, ...t.hidden.map((h) => h.a)];         // a⁽⁰⁾ … a⁽ᴸ⁻¹⁾
  t.zs = [...t.hidden.map((h) => h.z), t.logits];           // z⁽¹⁾ … z⁽ᴸ⁾
  t.deltas = [...t.hidden.map((h) => h.delta), t.deltaOut]; // δ⁽¹⁾ … δ⁽ᴸ⁾
  t.forward = {
    input: t.input, activations: t.hidden.map((h) => h.a), preActivations: t.hidden.map((h) => h.z),
    probs: t.probs, label: t.label, pred: t.pred,
  };
  t.sceneWeights = t.layers.map(({ inN, outN, W, b }) => ({ inN, outN, W: W.before, b: b.before }));
  t.sceneStep = {
    grads: t.layers.map((x) => x.W.grad), updates: t.layers.map((x) => x.update),
    deltas: t.hidden.map((h) => h.delta), deltaOut: t.deltaOut,
  };
  return t;
}

function argmaxAbs(values) {
  let best = 0;
  for (let k = 1; k < values.length; k++) if (Math.abs(values[k]) > Math.abs(values[best])) best = k;
  return best;
}

function activationNumbers(act, z, a) {
  switch (act) {
    case 'relu': return T`\max(0,\ ${num(z)}) = ${num(a)}`;
    case 'leaky_relu': return T`\max(0.01\cdot${factor(z)},\ ${num(z)}) = ${num(a)}`;
    case 'gelu': return T`${factor(z)}\cdot\Phi(${num(z)}) = ${factor(z)}\cdot${num(normalCdf(z))} = ${num(a)}`;
    case 'tanh': return T`\tanh(${num(z)}) = ${num(a)}`;
    case 'sigmoid': return T`\frac{1}{1+e^{${num(-z)}}} = ${num(a)}`;
    default: return T`z = ${num(a)}`;
  }
}

function slopeNumbers(act, z, a, slope) {
  const s = slope === 0 ? red('0') : num(slope);
  switch (act) {
    case 'relu':
    case 'leaky_relu': return T`\sigma'(${num(z)}) = ${s} \qquad (\text{since } z ${z > 0 ? '>' : T`\le`} 0)`;
    case 'gelu': return T`\Phi(${num(z)}) + ${factor(z)}\cdot\varphi(${num(z)}) = ${num(normalCdf(z))} + ${factor(z)}\cdot${num(normalPdf(z))} = ${s}`;
    case 'tanh': return T`1-a^2 = 1-${factor(a)}^2 = ${s}`;
    case 'sigmoid': return T`a\,(1-a) = ${num(a)}\,(1-${num(a)}) = ${s}`;
    default: return s;
  }
}

function probBars(probs, label, pred) {
  const box = el('div', 'walk-probs');
  box.innerHTML = Array.from(probs, (p, d) => {
    const cls = ['prob', d === pred ? 'pred' : '', d === label ? 'label' : ''].join(' ');
    return `<div class="${cls}"><span>${d}</span><div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></div><span>${(p * 100).toFixed(1)}%</span></div>`;
  }).join('');
  return box;
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = ctx2d(canvas);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = '10px system-ui, sans-serif';
  return { ctx, w, h };
}

function gridLine(ctx, x0, y0, x1, y1) {
  ctx.strokeStyle = '#1c2432';
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
}

/** Builds the DOM of one stage. */
class StageUI {
  constructor(box) { this.box = box; }
  title(text) { this.box.append(el('h3', 'stage-title', text)); }
  text(html, className = 'explain') { const p = el('p', className); p.innerHTML = html; this.box.append(p); return p; }
  label(text) { this.box.append(el('div', 'tex-label', text)); }
  tex(tex) { const div = el('div', 'tex'); renderTex(div, tex); this.box.append(div); return div; }
  canvas(className) { const c = el('canvas', className); this.box.append(c); return c; }
  node(n) { this.box.append(n); return n; }
}

export class Walkthrough {
  constructor({ send, getConfig, onChange, onSelect }) {
    Object.assign(this, { send, getConfig, onChange, onSelect });
    this.active = false;
    this.connected = false;
    this.training = false;
    this.trainSize = 60000;
    this.singleSteps = 0;
    this.sizes = [784, 10];
    this.hidden = [];
    this.stages = buildStages(1);
    this.stage = 0;
    this.image = null;
    this.imageIndex = 0;
    this.trace = null;
    this.undone = false;
    this.stale = false; // the network has trained since the trace was recorded
    this.canUndo = false;
    this.pending = false;
    this.followed = null; // {layer: 1…L, i: to-neuron, j: from-neuron}
    this.followedByUser = false;
    this.focus = null; // a neuron picked by the user: {layer (scene index), index}
    this.gradchecks = new Map();
    this.follow3d = true;
    this.scene = null;
    this.bind();
  }

  get L() { return this.sizes.length - 1; }

  bind() {
    $('walk-index').addEventListener('change', (e) => {
      const index = parseInt(e.target.value, 10);
      if (!Number.isNaN(index)) this.requestImage({ index: Math.min(this.trainSize - 1, Math.max(0, index)) });
    });
    $('walk-prev').addEventListener('click', () => this.requestImage({ index: (this.imageIndex + this.trainSize - 1) % this.trainSize }));
    $('walk-next').addEventListener('click', () => this.requestImage({ index: (this.imageIndex + 1) % this.trainSize }));
    $('walk-random').addEventListener('click', () => this.requestImage({ mode: 'random' }));
    $('walk-wrong').addEventListener('click', () => this.requestImage({ mode: 'misclassified' }));
    $('walk-step').addEventListener('click', () => this.takeStep());
    $('walk-undo').addEventListener('click', () => {
      if (!this.canUndo) return;
      this.canUndo = false;
      this.send({ type: 'undo' });
      this.renderControls();
    });
    $('walk-follow').addEventListener('change', (e) => { this.follow3d = e.target.checked; this.sceneChanged(); });
    $('walk-back').addEventListener('click', () => this.go(this.stage - 1));
    $('walk-fwd').addEventListener('click', () => this.go(this.stage + 1));
  }

  // ---------- events from app.js ----------

  setActive(active) {
    this.active = active;
    if (active && this.connected && !this.image) this.requestImage({ index: this.imageIndex });
    if (active) this.render();
  }

  setConnected(connected) {
    this.connected = connected;
    this.pending = false;
    this.image = null;
    this.clearTrace(); // the server starts a fresh session (and model) for every connection
    if (connected && this.active) this.requestImage({ index: this.imageIndex });
    this.render();
  }

  setTrainSize(n) { this.trainSize = n; }

  setArchitecture(sizes, hidden) {
    this.sizes = sizes;
    this.hidden = hidden.map((layer) => ({ ...layer }));
    this.stages = buildStages(this.L);
    this.stage = 0;
    this.clearTrace();
    this.render();
  }

  setStats(stats) {
    this.singleSteps = stats.singleSteps ?? 0;
    // The preview's prediction belongs to a network; refresh it for a new one.
    if (this.active && this.connected && !this.training && this.image && this.image.buildId !== stats.buildId) {
      this.requestImage({ index: this.imageIndex });
    }
    this.renderControls();
  }

  setTraining(training) {
    const finished = this.training && !training;
    this.training = training;
    if (training) this.markStale();
    if (finished && this.connected && this.active) this.requestImage({ index: this.imageIndex });
    this.renderControls();
  }

  trainingRequested() { this.markStale(); }

  /**
   * Training moves the weights away from the recorded step: the server drops its undo record, and
   * the 3D view goes back to the live network. The trace stays readable as a record of that step.
   */
  markStale() {
    this.canUndo = false;
    if (this.trace && !this.stale) {
      this.stale = true;
      this.sceneChanged();
    }
    this.render();
  }

  configChanged() { this.renderControls(); }

  onTrainImage(msg) {
    this.image = msg;
    this.imageIndex = msg.index;
    this.renderControls();
  }

  onStepFrame(buffer, buildId) {
    this.pending = false;
    const trace = parseStepFrame(buffer);
    if (trace.buildId !== buildId || !prepare(trace, this.sizes)) { this.renderControls(); return; }
    this.trace = trace;
    this.undone = false;
    this.stale = false;
    this.canUndo = true;
    this.singleSteps = trace.singleSteps;
    this.imageIndex = trace.index;
    this.image = {
      buildId, index: trace.index, label: trace.label, pred: trace.after.pred,
      pixels: Array.from(trace.input, (v) => Math.round(v * 255)),
    };
    this.gradchecks.clear();
    if (!this.followedByUser || !this.followed) {
      this.followed = this.largestGradient(1);
      this.followedByUser = false;
    }
    this.render();
    this.sceneChanged();
  }

  onUndone(msg) {
    if (this.trace?.stepId === msg.stepId) this.undone = true;
    this.canUndo = false;
    if (this.active) this.requestImage({ index: this.imageIndex });
    this.render();
  }

  onGradcheck(msg) {
    this.gradchecks.set(`${msg.stepId}:${msg.layer + 1}:${msg.row}:${msg.col}`, msg);
    if (this.stages[this.stage]?.kind === 'chain') this.renderStage();
  }

  onError() {
    if (!this.pending) return;
    this.pending = false;
    this.renderControls();
  }

  /** A neuron picked in the 3D view (or on a bar chart): focus on it and follow its strongest weight. */
  selectNeuron(hit) {
    this.focus = hit;
    if (hit && this.trace) {
      if (hit.layer === 0) {
        const { W, outN } = this.trace.layers[0];
        let best = 0;
        for (let i = 1; i < outN; i++) if (Math.abs(W.grad[i * 784 + hit.index]) > Math.abs(W.grad[best * 784 + hit.index])) best = i;
        this.followed = { layer: 1, i: best, j: hit.index };
      } else {
        const { W, inN } = this.trace.layers[hit.layer - 1];
        const j = argmaxAbs(W.grad.subarray(hit.index * inN, (hit.index + 1) * inN));
        this.followed = { layer: hit.layer, i: hit.index, j };
      }
      this.followedByUser = true;
    }
    this.render();
    this.sceneChanged();
  }

  handleKey(e) {
    if (e.altKey || e.ctrlKey || e.metaKey || e.target.closest?.('input, select, textarea')) return;
    if (e.key === 'ArrowRight') { this.go(this.stage + 1); e.preventDefault(); }
    if (e.key === 'ArrowLeft') { this.go(this.stage - 1); e.preventDefault(); }
  }

  /** A recorded step that still matches the network's weights (nothing has trained since). */
  hasLiveStep() { return this.trace != null && !this.stale; }
  stepData() { return this.hasLiveStep() ? this.trace.sceneStep : null; }

  /** What the 3D view should show for the current stage, or null to leave it as the Inspect tab has it. */
  sceneData() {
    if (!this.hasLiveStep() || !this.follow3d) return null;
    if (this.scene) return this.scene;
    const t = this.trace, s = this.stages[this.stage], L = this.L, f = this.followed;
    const view = { edgeMode: 'signal', neuronMode: 'activation', phase: null, activeLayer: null, focusLink: null };
    switch (s.kind) {
      case 'input': Object.assign(view, { phase: 'forward', activeLayer: 0, focusLink: 0 }); break;
      case 'forward': Object.assign(view, { phase: 'forward', activeLayer: s.layer, focusLink: s.layer - 1 }); break;
      case 'softmax':
      case 'loss': Object.assign(view, { phase: 'forward', activeLayer: L, focusLink: L - 1 }); break;
      case 'outputError': Object.assign(view, { edgeMode: 'backward', neuronMode: 'delta', phase: 'backward', activeLayer: L, focusLink: L - 1 }); break;
      case 'backprop': Object.assign(view, { edgeMode: 'backward', neuronMode: 'delta', phase: 'backward', activeLayer: s.layer, focusLink: s.layer }); break;
      case 'chain':
      case 'descent': Object.assign(view, { edgeMode: 'gradient', neuronMode: 'delta', focusLink: f.layer - 1 }); break;
      default: Object.assign(view, { edgeMode: 'update', focusLink: f.layer - 1 });
    }
    const showWeight = ['chain', 'descent', 'optimizer', 'result'].includes(s.kind);
    this.scene = {
      forward: t.forward, weights: t.sceneWeights, step: t.sceneStep, view,
      highlight: showWeight ? { link: f.layer - 1, from: f.j, to: f.i } : null,
    };
    return this.scene;
  }

  // ---------- actions ----------

  requestImage(extra) { if (this.connected) this.send({ type: 'trainImage', ...extra }); }

  canStep() { return this.connected && !this.training && !this.pending; }

  takeStep() {
    if (!this.canStep()) return;
    const { lr, optimizer } = this.getConfig();
    this.pending = true;
    this.send({ type: 'step', index: this.imageIndex, lr, optimizer });
    this.renderControls();
  }

  go(index) {
    if (!this.trace) return;
    const next = Math.max(0, Math.min(this.stages.length - 1, index));
    if (next === this.stage) return;
    this.stage = next;
    this.renderChips();
    this.renderStage();
    this.sceneChanged();
  }

  follow(followed) {
    this.followed = followed;
    this.followedByUser = true;
    this.focus = null;
    this.render();
    this.sceneChanged();
  }

  pickNeuron(li, index) {
    const hit = { layer: li, index };
    this.onSelect(hit);
    this.selectNeuron(hit);
  }

  clearTrace() {
    this.trace = null;
    this.undone = false;
    this.stale = false;
    this.canUndo = false;
    this.followed = null;
    this.followedByUser = false;
    this.focus = null;
    this.gradchecks.clear();
    this.sceneChanged();
  }

  sceneChanged() {
    this.scene = null;
    this.onChange();
  }

  largestGradient(layer) {
    const { W, inN } = this.trace.layers[layer - 1];
    const k = argmaxAbs(W.grad);
    if (W.grad[k] !== 0) return { layer, i: Math.floor(k / inN), j: k % inN };
    // Every gradient in the layer is 0 (dead neurons downstream): follow the strongest input anyway,
    // so the chain rule stage shows which factor is 0 instead of blaming a black pixel.
    return { layer, i: 0, j: argmaxAbs(this.trace.acts[layer - 1]) };
  }

  /** The neuron a per-layer stage expands: the picked one, else an end of the followed weight, else the largest. */
  focusIn(li, values) {
    if (this.focus?.layer === li) return this.focus.index;
    const f = this.followed;
    if (f?.layer === li) return f.i;
    if (f && f.layer - 1 === li) return f.j;
    return li === this.L ? this.trace.label : argmaxAbs(values);
  }

  nodeName(li, index) {
    if (li === 0) return `pixel ${index} (row ${Math.floor(index / 28)}, column ${index % 28})`;
    return li === this.L ? `output ${index} (digit ${index})` : `neuron ${index} of H${li}`;
  }

  layerName(li) { return li === 0 ? 'the input' : li === this.L ? 'the output layer' : `H${li}`; }

  // ---------- rendering ----------

  render() {
    this.renderControls();
    if (!this.active) return;
    this.renderChips();
    this.renderStage();
  }

  renderControls() {
    const img = this.image;
    const ctx = ctx2d($('walk-digit'));
    if (img) {
      const data = ctx.createImageData(28, 28);
      img.pixels.forEach((p, k) => data.data.set([p, p, p, 255], k * 4));
      ctx.putImageData(data, 0, 0);
    } else {
      ctx.clearRect(0, 0, 28, 28);
    }
    $('walk-index').max = this.trainSize - 1;
    if (document.activeElement !== $('walk-index')) $('walk-index').value = this.imageIndex;
    $('walk-image-info').innerHTML = img
      ? `Label <b>${img.label}</b> · the network ${this.trace?.index === img.index && !this.undone ? 'now ' : ''}predicts <b class="${img.pred === img.label ? 'ok' : 'bad'}">${img.pred}</b>`
      : '';
    const { lr, optimizer } = this.getConfig();
    const steps = `${fmt(this.singleSteps)} single step${this.singleSteps === 1 ? '' : 's'} so far`;
    $('walk-config').textContent = this.training
      ? 'Training is running. Stop it to take a single step.'
      : `${OPTIMIZER[optimizer].name} · η = ${lr} (set in the Training panel) · batch of 1 · ${steps}`;
    $('walk-step').disabled = !this.canStep();
    $('walk-step').textContent = this.pending ? 'Stepping…' : 'Take one step';
    $('walk-undo').disabled = !(this.canUndo && this.connected && !this.training);
  }

  renderChips() {
    const box = $('walk-chips');
    box.replaceChildren();
    let group = null, groupEl = null;
    this.stages.forEach((s, k) => {
      if (s.group !== group) {
        group = s.group;
        groupEl = el('div', 'chip-group');
        groupEl.append(el('span', 'chip-label', group));
        box.append(groupEl);
      }
      const label = s.kind === 'optimizer' && this.trace ? OPTIMIZER[this.trace.optimizer.name].chip : s.chip;
      const chip = el('button', `chip${k === this.stage && this.trace ? ' current' : ''}`, label);
      chip.disabled = !this.trace;
      chip.addEventListener('click', () => this.go(k));
      groupEl.append(chip);
    });
    $('walk-back').disabled = !this.trace || this.stage === 0;
    $('walk-fwd').disabled = !this.trace || this.stage === this.stages.length - 1;
    $('walk-pos').textContent = this.trace ? `${this.stage + 1} / ${this.stages.length}` : '';
  }

  renderStage() {
    const box = $('walk-stage');
    box.replaceChildren();
    const ui = new StageUI(box);
    if (!this.trace) {
      ui.text(`Press <b>Take one step</b>. The server runs one real training step on the image above, using the learning rate and optimizer from the Training panel, and records every number along the way. You can then go through it here stage by stage: the forward pass, the loss, backpropagation, the chain rule, gradient descent and the optimizer.`);
      ui.text('Tip: click a neuron in the 3D view to focus the walkthrough on it.', 'note');
      return;
    }
    if (this.stale) {
      ui.node(el('div', 'banner', 'The network has trained since this step, so the 3D view shows the live network. The numbers here still describe the step as it happened. Take a new step to walk through the current weights.'));
    } else if (this.undone) {
      ui.node(el('div', 'banner', 'This step was undone: the weights and optimizer state are back to what they were before it. Take the step again, maybe with a different learning rate, to compare.'));
    }
    const s = this.stages[this.stage], t = this.trace;
    const render = {
      input: this.stageInput, forward: this.stageForward, softmax: this.stageSoftmax, loss: this.stageLoss,
      outputError: this.stageOutputError, backprop: this.stageBackprop, chain: this.stageChain,
      descent: this.stageDescent, optimizer: this.stageOptimizer, result: this.stageResult,
    }[s.kind];
    render.call(this, ui, t, s);
  }

  /** The vector of a layer as bars (and in full when short); clicking a bar focuses that neuron. */
  vectorBars(ui, values, focus, li, name) {
    if (values.length <= 10) ui.tex(T`${name} = ${vec(values, { head: 10 })}`);
    const canvas = ui.canvas('walk-bars');
    const pick = drawWeightBars(canvas, values, focus);
    canvas.title = 'Click a bar to focus on that neuron';
    canvas.addEventListener('click', (e) => this.pickNeuron(li, pick(e.offsetX)));
  }

  weightPicker(ui) {
    const f = this.followed;
    const row = el('div', 'picker');
    const layer = el('select');
    for (let l = 1; l <= this.L; l++) {
      const short = (li) => (li === 0 ? 'In' : li === this.L ? 'Out' : `H${li}`);
      layer.append(new Option(`W⁽${sup(l)}⁾ ${short(l - 1)} → ${short(l)}`, l, false, l === f.layer));
    }
    const { inN, outN } = this.trace.layers[f.layer - 1];
    const numberInput = (value, max, label) => {
      const input = el('input');
      Object.assign(input, { type: 'number', min: 0, max, value, title: label });
      input.setAttribute('aria-label', label);
      return input;
    };
    const to = numberInput(f.i, outN - 1, 'To neuron i');
    const from = numberInput(f.j, inN - 1, 'From neuron j');
    const best = el('button', 'btn small', 'Largest gradient');
    layer.addEventListener('change', () => this.follow(this.largestGradient(Number(layer.value))));
    const pick = () => {
      const i = Math.min(outN - 1, Math.max(0, parseInt(to.value, 10) || 0));
      const j = Math.min(inN - 1, Math.max(0, parseInt(from.value, 10) || 0));
      this.follow({ layer: f.layer, i, j });
    };
    to.addEventListener('change', pick);
    from.addEventListener('change', pick);
    best.addEventListener('click', () => this.follow(this.largestGradient(f.layer)));
    row.append(el('span', 'muted', 'Weight'), layer, el('span', 'muted', 'i'), to, el('span', 'muted', 'j'), from, best);
    ui.node(row);
  }

  /** ∂L/∂W or ΔW of a whole layer: 28×28 tiles for the first layer, a matrix otherwise. */
  weightMap(ui, l, values, what) {
    const { inN, outN } = this.trace.layers[l - 1];
    const f = this.followed;
    const highlight = f?.layer === l ? { row: f.i, col: f.j } : null;
    const canvas = ui.canvas('walk-map');
    const pick = inN === 784 ? drawTiles(canvas, values, outN, highlight) : drawMatrix(canvas, values, outN, inN, highlight);
    canvas.title = 'Click a weight to follow it';
    canvas.addEventListener('click', (e) => {
      const hit = pick(e.offsetX, e.offsetY);
      if (hit) this.follow({ layer: l, i: hit.row, j: hit.col });
    });
    const layout = inN === 784
      ? `Each tile is one ${outN === 10 && l === this.L ? 'output' : 'neuron'}'s 784 incoming weights, laid out like the image`
      : `Rows are the ${outN} neurons of ${this.layerName(l)}, columns the ${inN} of ${this.layerName(l - 1)}`;
    ui.text(`${layout}: ${what} Blue is positive, orange negative, on one shared scale; the followed weight is framed in white. Click any weight to follow it.`, 'note');
  }

  // ---------- stages ----------

  stageInput(ui, t) {
    const nonzero = t.input.reduce((n, v) => n + (v > 0), 0);
    ui.title(`Input: training image #${fmt(t.index)}`);
    ui.text(`The image's 784 pixels, from 0 (black) to 1 (white), form the input vector ${m(T`\mathbf{x}`)}, also written ${m(T`\mathbf{a}^{(0)}`)}. Its label is <b>${t.label}</b>, so the target ${m(T`\mathbf{y}`)} has a 1 in position ${t.label} and 0 everywhere else.`);
    ui.label('Formula');
    ui.tex(T`\mathbf{a}^{(0)} = \mathbf{x} \in [0,1]^{784}, \qquad \mathbf{y} = (0,\dots,\underset{${t.label}}{1},\dots,0)`);
    ui.label('With the numbers');
    const first = [];
    for (let j = 0; j < 784 && first.length < 3; j++) if (t.input[j] > 0) first.push(j);
    ui.tex(T`${first.map((j) => T`x_{${j}} = ${num(t.input[j])}`).join(T`,\ `)},\ \dots \qquad \#\{\,j : x_j > 0\,\} = ${nonzero}`);
    ui.text(`${784 - nonzero} of the 784 pixels are black (${m('x_j = 0')}). They add nothing to any weighted sum, so the weights attached to them get a gradient of exactly 0 from this image.`, 'note');
  }

  stageForward(ui, t, { layer: l }) {
    const out = l === this.L, { W, b, inN, outN } = t.layers[l - 1];
    const aPrev = t.acts[l - 1], z = t.zs[l - 1];
    const act = out ? null : this.hidden[l - 1].activation;
    const a = out ? null : t.acts[l];
    const from = l === 1 ? `${inN} pixels` : `${inN} outputs of H${l - 1}`;
    ui.title(out ? 'Forward pass: output scores' : `Forward pass: hidden layer ${l} (${ACT[act].name})`);
    ui.text(out
      ? `Each of the 10 output neurons takes a weighted sum of the ${from} and adds its bias. The results ${m('z')} are the scores (logits), one per digit. There is no activation function here; softmax comes next.`
      : `Each of the ${outN} neuron${outN === 1 ? '' : 's'} takes a weighted sum of the ${from} and adds its bias, giving ${m('z')}. Then ${ACT[act].name} turns ${m('z')} into the neuron's output ${m('a')}, which the next layer reads.`);
    ui.label('Formula');
    ui.tex(T`z^{(${l})}_i = \sum_{j=1}^{${inN}} W^{(${l})}_{ij}\,a^{(${l - 1})}_j + b^{(${l})}_i`
      + (out ? '' : T`, \qquad a^{(${l})}_i = \sigma\big(z^{(${l})}_i\big), \quad \sigma(z) = ${ACT[act].f}`));
    ui.text(`As matrices, ${m(T`\mathbf{z}^{(${l})} = W^{(${l})}\mathbf{a}^{(${l - 1})} + \mathbf{b}^{(${l})}`)}, where ${m(T`W^{(${l})}`)} is ${outN} × ${inN}.`, 'note');
    const i = this.focusIn(l, out ? z : a);
    const ex = expandSum(W.before.subarray(i * inN, (i + 1) * inN), aPrev, { bias: b.before[i], total: z[i] });
    ui.label(`With the numbers, for ${out ? `digit ${i}` : `neuron ${i}`}`);
    ui.tex(sumRows(T`z^{(${l})}_{${i}}`, ex, (j) => T`W^{(${l})}_{${i},${j}}\,a^{(${l - 1})}_{${j}}`, T`b^{(${l})}_{${i}}`));
    if (!out) ui.tex(T`a^{(${l})}_{${i}} = ${activationNumbers(act, z[i], a[i])}`);
    const zeros = ex.count - ex.nonZero;
    const listed = ex.restCount ? `It lists the ${ex.shown.length} largest terms; the brace adds up the other ${fmt(ex.restCount)}.` : '';
    if (zeros || listed) {
      ui.text(`${zeros ? `${fmt(zeros)} of the ${fmt(ex.count)} terms are exactly 0 (${l === 1 ? 'black pixels' : 'inputs that are 0'}) and are left out. ` : ''}${listed}`, 'note');
    }
    this.vectorBars(ui, out ? z : a, i, l, out ? T`\mathbf{z}^{(${l})}` : T`\mathbf{a}^{(${l})}`);
  }

  stageSoftmax(ui, t) {
    const z = t.logits, y = t.label, pred = t.pred;
    const exps = Array.from(z, Math.exp), total = exps.reduce((s, v) => s + v, 0);
    ui.title('Softmax: scores to probabilities');
    ui.text('Softmax turns the 10 scores into probabilities. It exponentiates each score, which makes them all positive, then divides by their total, so they add up to 1. A higher score always means a higher probability.');
    ui.label('Formula');
    ui.tex(T`p_k = \frac{e^{z_k}}{\sum_{m=0}^{9} e^{z_m}} \qquad \big(z_k = z^{(${this.L})}_k\big)`);
    ui.label('With the numbers');
    const terms = [0, 1, 2].map((k) => T`e^{${num(z[k])}}`).concat([T`\cdots`, T`e^{${num(z[9])}}`]);
    ui.tex(T`\sum_{m} e^{z_m} = ${terms.join(' + ')} = ${num(total)}`);
    ui.tex(T`p_{${y}} = \frac{e^{${num(z[y])}}}{${num(total)}} = \frac{${num(exps[y])}}{${num(total)}} = ${num(t.probs[y])}`);
    if (pred !== y) ui.tex(T`p_{${pred}} = \frac{e^{${num(z[pred])}}}{${num(total)}} = ${num(t.probs[pred])}`);
    ui.text(pred === y
      ? `The correct digit, ${y}, gets the highest probability, so the prediction is right.`
      : `The network's top guess is ${pred}, but the label is ${y}, so the prediction is wrong.`, 'note');
    ui.node(probBars(t.probs, y, pred));
  }

  stageLoss(ui, t) {
    const y = t.label, p = t.probs[y];
    ui.title('Loss: cross-entropy');
    ui.text(`The loss measures how wrong the network was: the negative log of the probability it gave the correct digit, ${y}. It is 0 when that probability is 1, and it grows without limit as the probability falls toward 0.`);
    ui.label('Formula');
    ui.tex(T`L = -\sum_{k=0}^{9} y_k \log p_k = -\log p_{y}`);
    ui.label('With the numbers');
    ui.tex(T`L = -\log p_{${y}} = -\log(${num(p)}) = ${num(t.loss)}`);
    ui.text(`For comparison, a network that spreads its bets evenly over the 10 digits gets ${m(T`L = -\log 0.1 \approx 2.303`)}.`, 'note');
    this.lossCurve(ui.canvas('walk-plot small'), p, t.loss);
    ui.text(`The curve is ${m(T`-\log p`)} for every probability ${m('p')} from 0 to 1; the dot is this image.`, 'note');
  }

  stageOutputError(ui, t) {
    const L = this.L, y = t.label;
    ui.title('Backpropagation starts: the output error δ');
    ui.text('Backpropagation works backward from the loss. First it asks how the loss would change if each output score changed a little. For softmax followed by cross-entropy the answer is simple: the predicted probability minus the target.');
    ui.label('Formula');
    ui.tex(T`\delta^{(${L})}_k = \frac{\partial L}{\partial z^{(${L})}_k} = p_k - y_k`);
    ui.label('With the numbers');
    const rows = Array.from(t.probs, (p, k) => {
      const d = k === y ? T`\textcolor{${GREEN}}{${k}}` : String(k);
      return T`${d} & ${num(p)} & ${k === y ? 1 : 0} & ${num(t.deltaOut[k])}`;
    });
    ui.tex(T`\begin{array}{c|ccc} k & p_k & y_k & \delta^{(${L})}_k \\ \hline ${rows.join(T` \\ `)} \end{array}`);
    ui.text(`The correct digit's ${m(T`\delta = ${num(t.deltaOut[y])}`)} is negative, so raising its score would lower the loss. Every other digit has a positive ${m(T`\delta`)}, so lowering its score would lower the loss. Gradient descent will nudge the scores both ways.`, 'note');
    this.vectorBars(ui, t.deltaOut, this.focusIn(L, t.deltaOut), L, T`\boldsymbol{\delta}^{(${L})}`);
  }

  stageBackprop(ui, t, { layer: l }) {
    const act = this.hidden[l - 1].activation, h = t.hidden[l - 1];
    const up = t.layers[l], n = this.sizes[l], nUp = this.sizes[l + 1];
    ui.title(`Backpropagation: hidden layer ${l}`);
    ui.text(`Each neuron in H${l} feeds ${nUp === 1 ? 'the one neuron' : `all ${nUp} neurons`} of ${this.layerName(l + 1)}. Adding up their errors ${m(T`\delta`)}, each weighted by the connecting weight, gives ${m(T`\partial L/\partial a`)}: how the loss changes with this neuron's output. Multiplying by the slope of ${ACT[act].name}, ${m(T`\sigma'(z)`)}, turns that into the neuron's own error ${m(T`\delta`)}.`);
    ui.label('Formula');
    ui.tex(T`\frac{\partial L}{\partial a^{(${l})}_i} = \sum_{k=1}^{${nUp}} W^{(${l + 1})}_{ki}\,\delta^{(${l + 1})}_k, \qquad \delta^{(${l})}_i = \frac{\partial L}{\partial a^{(${l})}_i}\;\sigma'\big(z^{(${l})}_i\big)`);
    ui.tex(T`\sigma'(z) = ${ACT[act].d}`);
    ui.text(`As vectors, ${m(T`\boldsymbol{\delta}^{(${l})} = \big(W^{(${l + 1})\top}\boldsymbol{\delta}^{(${l + 1})}\big)\odot\sigma'\big(\mathbf{z}^{(${l})}\big)`)}, where ${m(T`\odot`)} multiplies element by element.`, 'note');
    const i = this.focusIn(l, h.delta);
    const column = Float64Array.from({ length: nUp }, (_, k) => up.W.before[k * n + i]);
    const ex = expandSum(column, t.deltas[l], { total: h.dLda[i] });
    const slope = h.slope[i];
    ui.label(`With the numbers, for neuron ${i}`);
    ui.tex(sumRows(T`\frac{\partial L}{\partial a^{(${l})}_{${i}}}`, ex, (k) => T`W^{(${l + 1})}_{${k},${i}}\,\delta^{(${l + 1})}_{${k}}`));
    ui.tex(T`\sigma'\big(z^{(${l})}_{${i}}\big) = ${slopeNumbers(act, h.z[i], h.a[i], slope)}`);
    ui.tex(T`\delta^{(${l})}_{${i}} = ${factor(h.dLda[i])}\cdot${slope === 0 ? red('0') : factor(slope)} = ${slope === 0 ? red('0') : num(h.delta[i])}`);
    if (slope === 0) {
      ui.text(`This neuron's input was ${m(T`z = ${num(h.z[i])} \le 0`)}, where ${ACT[act].name} is flat, so its slope is 0. No error flows back through it, and none of its incoming weights will change from this image.`, 'note bad');
    }
    const blocked = h.slope.reduce((c, s) => c + (s === 0), 0);
    if (blocked) ui.text(`${blocked} of the ${n} neuron${n === 1 ? '' : 's'} in H${l} ${blocked === 1 ? 'has' : 'have'} ${m(T`\sigma' = 0`)} for this image, so the error stops there.`, 'note');
    this.vectorBars(ui, h.delta, i, l, T`\boldsymbol{\delta}^{(${l})}`);
  }

  stageChain(ui, t) {
    const f = this.followed, l = f.layer, { W, inN } = t.layers[l - 1];
    const g = W.grad[f.i * inN + f.j], aj = t.acts[l - 1][f.j], di = t.deltas[l - 1][f.i];
    ui.title('Chain rule: the gradient of one weight');
    ui.text(`Follow the weight from ${this.nodeName(l - 1, f.j)} to ${this.nodeName(l, f.i)}. Nudging it by a small amount changes that neuron's ${m('z')} by ${m('a_j')} times as much (${m('a_j')} is the value the weight multiplies), and backprop already found how ${m('z')} changes the loss: ${m(T`\delta_i`)}. The chain rule multiplies the two.`);
    this.weightPicker(ui);
    ui.label('Formula');
    ui.tex(T`\frac{\partial L}{\partial W^{(${l})}_{ij}} = \frac{\partial L}{\partial z^{(${l})}_i}\cdot\frac{\partial z^{(${l})}_i}{\partial W^{(${l})}_{ij}} = \delta^{(${l})}_i\;a^{(${l - 1})}_j`);
    ui.label('With the numbers');
    ui.tex(T`\frac{\partial L}{\partial W^{(${l})}_{${f.i},${f.j}}} = \delta^{(${l})}_{${f.i}}\,a^{(${l - 1})}_{${f.j}} = ${di === 0 ? red('0') : factor(di)}\cdot${aj === 0 ? red('0') : factor(aj)} = ${num(g)}`);
    if (aj === 0) ui.text(`${m(T`a_j = 0`)} (${l === 1 ? 'a black pixel' : 'a neuron that output 0'}), so this weight's gradient is 0 for this image.`, 'note bad');
    else if (di === 0) ui.text(`${m(T`\delta_i = 0`)}: no error reaches this neuron, so none of its incoming weights can change.${l < this.L ? ' The path below shows where the error stops.' : ''}`, 'note bad');
    if (l < this.L) this.strongestPath(ui, t, f, g);
    else ui.text('This weight feeds an output score directly, so there is only one path to the loss.', 'note');
    this.gradcheckBlock(ui, t, f);
    ui.label(`∂L/∂W⁽${sup(l)}⁾ for all ${fmt(W.grad.length)} weights of this layer`);
    ui.tex(T`\frac{\partial L}{\partial W^{(${l})}} = \boldsymbol{\delta}^{(${l})}\,\mathbf{a}^{(${l - 1})\top}, \qquad \frac{\partial L}{\partial \mathbf{b}^{(${l})}} = \boldsymbol{\delta}^{(${l})}`);
    const what = inN === 784
      ? `with one training image each tile is ${m(T`\delta_i\,\mathbf{x}`)}, the digit itself, scaled and signed by that neuron's error.`
      : `each cell is ${m(T`\delta_i\,a_j`)}.`;
    this.weightMap(ui, l, W.grad, what);
  }

  /** The single path from the followed weight to the loss with the largest |product| (a small DP). */
  strongestPath(ui, t, f, g) {
    const L = this.L;
    const best = [], next = [];
    best[L] = Float64Array.from(t.deltaOut);
    for (let layer = L - 1; layer >= f.layer; layer--) {
      const n = this.sizes[layer], nUp = this.sizes[layer + 1], Wup = t.layers[layer].W.before;
      const slope = t.hidden[layer - 1].slope;
      best[layer] = new Float64Array(n);
      next[layer] = new Int32Array(n);
      for (let q = 0; q < n; q++) {
        if (layer === f.layer && q !== f.i) continue;
        let top = 0, arg = 0;
        for (let k = 0; k < nUp; k++) {
          const v = Wup[k * n + q] * best[layer + 1][k];
          if (k === 0 || Math.abs(v) > Math.abs(top)) { top = v; arg = k; }
        }
        best[layer][q] = slope[q] * top;
        next[layer][q] = arg;
      }
    }
    const factors = [{ name: T`a^{(${f.layer - 1})}_{${f.j}}`, v: t.acts[f.layer - 1][f.j] }];
    let q = f.i;
    for (let layer = f.layer; layer < L; layer++) {
      factors.push({ name: T`\sigma'\big(z^{(${layer})}_{${q}}\big)`, v: t.hidden[layer - 1].slope[q] });
      const k = next[layer][q];
      factors.push({ name: T`W^{(${layer + 1})}_{${k},${q}}`, v: t.layers[layer].W.before[k * this.sizes[layer] + q] });
      q = k;
    }
    factors.push({ name: T`\delta^{(${L})}_{${q}}`, v: t.deltaOut[q] });
    const product = factors.reduce((p, x) => p * x.v, 1);
    const paths = this.sizes.slice(f.layer + 1).reduce((p, n) => p * n, 1);
    ui.label('Unrolled all the way to the loss');
    ui.text(`${m(T`\delta_i`)} was itself built from every route from this neuron to the loss. So the gradient is a sum over all ${fmt(paths)} path${paths === 1 ? '' : 's'}, each the product of the local derivatives along it. The strongest single path:`);
    const parts = factors.map((x) => T`\underbrace{${x.name}}_{${x.v === 0 ? red('0') : num(x.v)}}`);
    const rows = [];
    for (let k = 0; k < parts.length; k += 4) rows.push(parts.slice(k, k + 4).join(T`\cdot`));
    ui.tex(T`\begin{aligned} &${rows.join(T` \\ &\cdot `)} \\ &= ${num(product)} \end{aligned}`);
    const zero = factors.some((x) => x.v === 0);
    ui.text(paths === 1
      ? 'It is the only path, so it is the whole gradient.'
      : `This path alone contributes ${m(num(product))}; all ${fmt(paths)} paths together give ${m(num(g))}.${zero ? ' A factor of 0 (red) blocks a path completely; when every path is blocked, the weight gets no gradient at all, which is how dead ReLUs stop a deep, narrow network from learning.' : ''}`, 'note');
  }

  gradcheckBlock(ui, t, f) {
    const result = this.gradchecks.get(`${t.stepId}:${f.layer}:${f.i}:${f.j}`);
    ui.label('Check it numerically');
    const row = el('div', 'buttons tight');
    const button = el('button', 'btn small', result ? 'Check again' : 'Check with finite differences');
    button.disabled = !this.connected;
    button.addEventListener('click', () => this.send({ type: 'gradcheck', stepId: t.stepId, layer: f.layer - 1, row: f.i, col: f.j }));
    row.append(button);
    ui.node(row);
    if (!result) {
      ui.text(`The server nudges this weight up and down by a tiny ${m(T`\epsilon`)}, reruns the image, and measures how much the loss changed. No chain rule involved.`, 'note');
      return;
    }
    ui.tex(T`\frac{L(w+\epsilon) - L(w-\epsilon)}{2\epsilon} = \frac{${num(result.lossPlus, 10)} - ${num(result.lossMinus, 10)}}{2\cdot ${num(result.eps)}} = ${num(result.numeric, 6)}`);
    const diff = Math.abs(result.numeric - result.backprop);
    const rel = diff / Math.max(Math.abs(result.numeric), Math.abs(result.backprop), 1e-30);
    ui.tex(T`\text{backprop: } ${num(result.backprop, 6)} \qquad \text{relative difference: } ${num(rel, 2)}`);
    const agree = rel < 1e-4 || diff < 1e-9;
    ui.text(agree
      ? '✓ They agree: the chain rule gives the same slope as nudging the weight and measuring.'
      : result.kink
        ? 'They differ because the nudge pushed a ReLU input across 0, where the slope jumps (a kink). The chain rule uses the slope on one side of it.'
        : 'They differ by more than rounding would explain.', agree ? 'note ok' : 'note bad');
  }

  stageDescent(ui, t) {
    const f = this.followed, l = f.layer, { W, inN } = t.layers[l - 1];
    const k = f.i * inN + f.j, w = W.before[k], g = W.grad[k], lr = t.lr;
    const sig = sigFor(w, lr * g);
    ui.title('Gradient descent: a step against the gradient');
    ui.text(`The gradient points in the direction that increases the loss fastest, so gradient descent moves every weight a small step the other way. The learning rate ${m(T`\eta = ${num(lr)}`)} sets the size of the step.`);
    this.weightPicker(ui);
    ui.label('Formula');
    ui.tex(T`W^{(${l})}_{ij} \;\leftarrow\; W^{(${l})}_{ij} - \eta\,\frac{\partial L}{\partial W^{(${l})}_{ij}}`);
    ui.label('With the numbers');
    ui.tex(T`W^{(${l})}_{${f.i},${f.j}} \;\leftarrow\; ${num(w, sig)} - ${num(lr)}\cdot${factor(g)} = ${num(w - lr * g, sig)}`);
    ui.text(t.optimizer.name === 'sgd_plain'
      ? 'This is exactly the rule SGD (plain) applies, so the next stage shows the same update.'
      : `This is the textbook rule. ${OPTIMIZER[t.optimizer.name].name} doesn't apply it as is; the next stage shows what it does instead.`, 'note');
    ui.tex(T`W^{(l)} \leftarrow W^{(l)} - \eta\,\frac{\partial L}{\partial W^{(l)}} \quad \text{for every layer at once}`);
  }

  stageOptimizer(ui, t) {
    const f = this.followed, l = f.layer, { W, inN } = t.layers[l - 1];
    const k = f.i * inN + f.j, w0 = W.before[k], w1 = W.after[k], g = W.grad[k], lr = t.lr, o = t.optimizer;
    const dw = w1 - w0, sig = sigFor(w0, dw || lr * g);
    const wName = T`W^{(${l})}_{${f.i},${f.j}}`;
    const intro = {
      sgd_plain: 'Plain SGD applies the gradient descent rule directly: each weight moves by the learning rate times its gradient.',
      sgd: `Momentum keeps a velocity ${m('v')} for every weight. Each step it multiplies the old velocity by ${m(T`\mu = ${num(o.momentum)}`)}, adds the new gradient, and moves the weight along the velocity. Gradients that keep pointing the same way build up speed; ones that flip back and forth cancel out.`,
      adam: `Adam keeps two running averages for every weight: ${m('m')}, of its recent gradients (which way to go), and ${m('v')}, of its recent squared gradients (how big they usually are). It steps along ${m(T`m/\sqrt{v}`)}, so each weight moves at a pace set by its own history rather than by the size of today's gradient. The hats correct both averages for starting at 0.`,
    }[o.name];
    ui.title(o.name === 'adam' ? `Optimizer: Adam (step t = ${o.t})` : `Optimizer: ${OPTIMIZER[o.name].name}`);
    ui.text(intro);
    this.weightPicker(ui);
    ui.label('Formula');
    if (o.name === 'sgd_plain') {
      ui.tex(T`\Delta W = -\eta\,g, \qquad W \leftarrow W + \Delta W`);
      ui.label('With the numbers');
      ui.tex(T`\Delta W = -${num(lr)}\cdot${factor(g)} = ${num(-lr * g)}`);
    } else if (o.name === 'sgd') {
      const v0 = W.bufBefore[k], v1 = W.bufAfter[k];
      ui.tex(T`v_t = \mu\,v_{t-1} + g_t \;\;\big(\text{first step: } v_1 = g_1\big), \qquad W \leftarrow W - \eta\,v_t, \qquad \mu = ${num(o.momentum)}`);
      ui.label('With the numbers');
      ui.tex(o.firstStep ? T`v_1 = g_1 = ${num(v1)}` : T`v_t = ${num(o.momentum)}\cdot${factor(v0)} + ${factor(g)} = ${num(v1)}`);
      ui.tex(T`\Delta W = -${num(lr)}\cdot${factor(v1)} = ${num(-lr * v1)}`);
    } else {
      const { t: step, beta1: b1, beta2: b2, eps } = o;
      const m0 = W.mBefore[k], v0 = W.vBefore[k], m1 = W.mAfter[k], v1 = W.vAfter[k];
      const mHat = m1 / (1 - b1 ** step), vHat = v1 / (1 - b2 ** step);
      ui.tex(T`\begin{aligned} m_t &= \beta_1\,m_{t-1} + (1-\beta_1)\,g_t, & \hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, \\ v_t &= \beta_2\,v_{t-1} + (1-\beta_2)\,g_t^2, & \hat v_t &= \frac{v_t}{1-\beta_2^{\,t}}, \\ W &\leftarrow W - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} \end{aligned}`);
      ui.text(`${m(T`\beta_1 = ${num(b1)}`)}, ${m(T`\beta_2 = ${num(b2)}`)}, ${m(T`\epsilon = ${num(eps)}`)}; ${m('t')} counts this optimizer's steps.`, 'note');
      ui.label('With the numbers');
      ui.tex(T`m_{${step}} = ${num(b1)}\cdot${factor(m0)} + ${num(1 - b1)}\cdot${factor(g)} = ${num(m1)}`);
      ui.tex(T`v_{${step}} = ${num(b2)}\cdot${factor(v0)} + ${num(1 - b2)}\cdot${factor(g)}^2 = ${num(v1)}`);
      ui.tex(T`\hat m_{${step}} = \frac{${num(m1)}}{1-${num(b1)}^{${step}}} = ${num(mHat)}, \qquad \hat v_{${step}} = \frac{${num(v1)}}{1-${num(b2)}^{${step}}} = ${num(vHat)}`);
      ui.tex(T`\Delta W = -${num(lr)}\cdot\frac{${num(mHat)}}{\sqrt{${num(vHat)}} + ${num(eps)}} = ${num(-lr * mHat / (Math.sqrt(vHat) + eps))}`);
      if (step === 1) {
        ui.text(`On the first step ${m(T`\hat m = g`)} and ${m(T`\hat v = g^2`)}, so ${m(T`\Delta W \approx -\eta\,\mathrm{sign}(g)`)}: every weight with a non-zero gradient moves by almost exactly ${m(T`\eta`)}, however small its gradient.`, 'note');
      }
      if (g === 0 && dw !== 0) ui.text(`This weight's gradient is 0 for this image, yet it still moves: ${m('m')} remembers the gradients of earlier steps.`, 'note');
    }
    ui.tex(T`${wName} \;\leftarrow\; ${num(w0, sig)} + ${factor(dw, 4)} = ${num(w1, sig)}`);
    const plainStep = -lr * g;
    ui.text(`That is the new weight PyTorch stored. Plain gradient descent would have moved it by ${m(num(plainStep))}; ${OPTIMIZER[o.name].name} moved it by ${m(num(dw))}${plainStep && dw && o.name !== 'sgd_plain' ? `, ${m(num(Math.abs(dw / plainStep), 3))} times as far${Math.sign(dw) !== Math.sign(plainStep) ? ', and in the other direction' : ''}` : ''}.`, 'note');
    ui.label(`ΔW⁽${sup(l)}⁾: how every weight of this layer moved`);
    this.weightMap(ui, l, t.layers[l - 1].update, `each ${inN === 784 ? 'pixel' : 'cell'} is the change ${m(T`\Delta W_{ij}`)} of one weight.`);
  }

  stageResult(ui, t) {
    const y = t.label, dL = t.after.loss - t.loss;
    ui.title('Result: did the step help?');
    ui.text('The same image, run through the updated network:');
    ui.label('Before → after, on this image');
    ui.tex(T`L:\; ${num(t.loss)} \;\to\; ${num(t.after.loss)} \qquad \Delta L = ${num(dL)}`);
    ui.tex(T`p_{${y}}:\; ${num(t.probs[y])} \;\to\; ${num(t.probsAfter[y])}`);
    const verdict = (d) => (d === y ? `<b class="ok">${d}</b>` : `<b class="bad">${d}</b>`);
    ui.text(`Predicted digit: ${verdict(t.pred)} → ${verdict(t.after.pred)} (label ${y}).${dL < 0 ? '' : ' The loss went up: the step overshot, which a smaller learning rate would avoid.'}`, 'note');
    ui.label('What the gradient predicted');
    ui.tex(T`\Delta L \;\approx\; \sum_{\text{all weights}} \frac{\partial L}{\partial w}\,\Delta w = ${num(t.firstOrder)} \qquad \big(\text{actual: } ${num(dL)}\big)`);
    ui.text('The gradient describes the loss as if it were flat (a tilted plane) near the current weights, and this estimate follows that plane. The actual change differs because the loss surface curves, and the bigger the step, the bigger the gap.', 'note');
    ui.label('Loss vs learning rate');
    this.lrPlot(ui.canvas('walk-plot'), t);
    ui.text(`The loss had this same step used a different learning rate ${m(T`\eta`)} (log scale), on this image and on ${fmt(t.lrCurve.testImages)} test images the step never saw. The dots mark the ${m(T`\eta`)} you used; dashed lines are the losses before the step. Too small a step barely helps. This image's loss often keeps falling as the step grows, because the network simply memorizes it; the test loss shows where a step becomes too big for everything else.`, 'note');
  }

  // ---------- plots ----------

  lossCurve(canvas, p, loss) {
    const { ctx, w, h } = setupCanvas(canvas);
    const mg = { l: 30, r: 10, t: 8, b: 20 }, pw = w - mg.l - mg.r, ph = h - mg.t - mg.b, yMax = 7;
    const x = (v) => mg.l + v * pw, y = (v) => mg.t + ph - (Math.min(v, yMax) / yMax) * ph;
    ctx.lineWidth = 1;
    for (let v = 0; v <= yMax; v += 1) {
      gridLine(ctx, mg.l, y(v), mg.l + pw, y(v));
      ctx.fillStyle = '#6f7b91'; ctx.textAlign = 'right'; ctx.fillText(String(v), mg.l - 5, y(v) + 3);
    }
    ctx.textAlign = 'center';
    for (const v of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(String(v), x(v), h - 6);
    ctx.strokeStyle = '#ffc861';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let k = 0; k <= 200; k++) {
      const v = Math.max(1e-3, k / 200);
      if (k) ctx.lineTo(x(v), y(-Math.log(v))); else ctx.moveTo(x(v), y(-Math.log(v)));
    }
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.beginPath(); ctx.arc(x(p), y(loss), 4, 0, Math.PI * 2); ctx.fill();
    ctx.textAlign = p > 0.6 ? 'right' : 'left';
    ctx.fillText(` L = ${loss.toFixed(3)} `, x(p), y(loss) - 8);
  }

  lrPlot(canvas, t) {
    const { lrs, losses, testLosses, testLossBefore } = t.lrCurve;
    const mid = Math.floor(lrs.length / 2); // lrs[mid] is the learning rate used
    const { ctx, w, h } = setupCanvas(canvas);
    const mg = { l: 38, r: 12, t: 22, b: 22 }, pw = w - mg.l - mg.r, ph = h - mg.t - mg.b;
    const all = [...losses, ...testLosses, t.loss, testLossBefore].filter((v) => v != null);
    const top = Math.min(Math.max(...all) * 1.08, Math.max(t.loss, t.after.loss, testLossBefore ?? 0) * 2.5) || 1;
    const lo = Math.log10(lrs[0]), hi = Math.log10(lrs[lrs.length - 1]);
    const x = (eta) => mg.l + ((Math.log10(eta) - lo) / (hi - lo)) * pw;
    const y = (loss) => mg.t + ph - (Math.min(loss, top) / top) * ph;
    ctx.lineWidth = 1;
    for (let k = 0; k <= 4; k++) {
      const v = (top * k) / 4;
      gridLine(ctx, mg.l, y(v), mg.l + pw, y(v));
      ctx.fillStyle = '#6f7b91'; ctx.textAlign = 'right'; ctx.fillText(v.toFixed(v < 1 ? 2 : 1), mg.l - 5, y(v) + 3);
    }
    ctx.textAlign = 'center';
    for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) {
      gridLine(ctx, x(10 ** e), mg.t, x(10 ** e), mg.t + ph);
      ctx.fillStyle = '#6f7b91';
      ctx.fillText(`10${sup(e)}`, x(10 ** e), h - 6);
    }
    const series = [
      { values: losses, before: t.loss, color: '#ffc861', label: 'this image' },
      { values: testLosses, before: testLossBefore, color: '#6ea8ff', label: `${fmt(t.lrCurve.testImages)} test images` },
    ];
    for (const { values, before, color } of series) {
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(mg.l, y(before)); ctx.lineTo(mg.l + pw, y(before)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let drawing = false;
      lrs.forEach((eta, k) => {
        if (values[k] == null) { drawing = false; return; }
        if (drawing) ctx.lineTo(x(eta), y(values[k])); else ctx.moveTo(x(eta), y(values[k]));
        drawing = true;
      });
      ctx.stroke();
      if (values[mid] != null) {
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(x(lrs[mid]), y(values[mid]), 4, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.textAlign = 'left';
    let lx = mg.l + 4;
    for (const { color, label } of series) {
      ctx.fillStyle = color;
      ctx.fillRect(lx, 8, 12, 2);
      ctx.fillStyle = '#8a96ab';
      ctx.fillText(label, lx + 16, 12);
      lx += ctx.measureText(label).width + 32;
    }
    ctx.fillStyle = '#8a96ab';
    ctx.textAlign = 'right';
    ctx.fillText(`dots: η = ${t.lr}`, mg.l + pw, 12);
  }
}
