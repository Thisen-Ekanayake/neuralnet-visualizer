import { NetworkScene } from './scene.js';
import { TrainingChart } from './chart.js';

const ACTIVATIONS = [
  ['relu', 'ReLU'], ['leaky_relu', 'Leaky ReLU'], ['gelu', 'GELU'],
  ['tanh', 'Tanh'], ['sigmoid', 'Sigmoid'], ['linear', 'Linear (none)'],
];
const ACT_LABEL = Object.fromEntries(ACTIVATIONS);
const PRESETS = [
  ['width50', '784 → 50 → 10', [[50, 'relu']]],
  ['width1', '784 → 1 → 10', [[1, 'relu']]],
  ['deep1', '5 hidden × 1 neuron', Array(5).fill([1, 'relu'])],
  ['deep2', '5 hidden × 2 neurons', Array(5).fill([2, 'relu'])],
  ['classic', '784 → 128 → 64 → 10', [[128, 'relu'], [64, 'relu']]],
  ['none', 'No hidden layer', []],
];
const POSITIVE = [61, 139, 255];
const NEGATIVE = [255, 106, 61];
let MAX_LAYERS = 8;
let MAX_UNITS = 256;

const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');
const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;
const num = (v, digits = 3) => (v == null ? '—' : v.toFixed(digits));

const state = {
  hidden: [{ units: 50, activation: 'relu' }],
  buildId: 1,
  weights: null,
  forward: null,
  stats: null,
  lastProgress: null,
  selected: null,
  training: false,
  view: { edgeMode: 'weights', cutoff: 0, brightness: 1, maxEdges: 150000, showDead: true },
};

const sizes = () => [784, ...state.hidden.map((l) => l.units), 10];
const isOutput = (li) => li === sizes().length - 1;
const layerName = (li) => (li === 0 ? 'Input' : isOutput(li) ? 'Output' : `Hidden ${li}`);
const shortName = (li) => (li === 0 ? 'In' : isOutput(li) ? 'Out' : `H${li}`);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function clampInt(value, lo, hi) {
  const n = parseInt(value, 10);
  return Number.isNaN(n) ? null : Math.min(hi, Math.max(lo, n));
}

function activationSelect(value) {
  const select = el('select');
  for (const [key, label] of ACTIVATIONS) select.append(new Option(label, key, false, key === value));
  return select;
}

// ---------- 3D scene ----------

const scene = new NetworkScene($('scene'), {
  onHover: showTooltip,
  onSelect: (hit) => {
    const same = hit && state.selected && hit.layer === state.selected.layer && hit.index === state.selected.index;
    state.selected = same ? null : hit;
    renderSelected();
    refreshScene();
  },
});
const chart = new TrainingChart($('chart'));

let refreshPending = false;
function refreshScene() {
  if (refreshPending) return;
  refreshPending = true;
  requestAnimationFrame(() => {
    refreshPending = false;
    const { weights, forward, stats, selected, view } = state;
    scene.update({ weights, forward, stats, selected, view });
  });
}

let rebuildPending = false;
function rebuildScene() {
  if (rebuildPending) return;
  rebuildPending = true;
  requestAnimationFrame(() => {
    rebuildPending = false;
    scene.build(sizes(), state.hidden.map((l) => ACT_LABEL[l.activation]), state.view.maxEdges);
    refreshScene();
    updateHud();
  });
}

// ---------- architecture editor ----------

function renderLayerList() {
  const list = $('layer-list');
  list.replaceChildren(fixedRow('Input', '784', '28×28 pixels'));
  state.hidden.forEach((layer, i) => list.append(layerRow(layer, i)));
  list.append(fixedRow('Output', '10', 'softmax, digits 0–9'));
  $('layers-count').textContent = state.hidden.length;
  $('layers-dec').disabled = state.hidden.length === 0;
  $('layers-inc').disabled = state.hidden.length >= MAX_LAYERS;
}

function fixedRow(name, size, note) {
  const row = el('div', 'layer fixed');
  const left = el('span');
  left.append(el('b', null, name), ` · ${size}`);
  row.append(left, el('span', null, note));
  return row;
}

function layerRow(layer, i) {
  const row = el('div', 'layer');
  const top = el('div', 'layer-top');
  const units = el('input');
  Object.assign(units, { type: 'number', min: 1, max: MAX_UNITS, value: layer.units, title: 'Neurons in this layer' });
  const activation = activationSelect(layer.activation);
  activation.title = 'Activation function';
  const remove = el('button', 'remove', '×');
  remove.title = 'Remove this layer';
  remove.setAttribute('aria-label', `Remove hidden layer ${i + 1}`);
  top.append(el('span', 'tag', `H${i + 1}`), units, activation, remove);

  const slider = el('input');
  Object.assign(slider, { type: 'range', min: 1, max: MAX_UNITS, value: layer.units });
  slider.setAttribute('aria-label', `Neurons in hidden layer ${i + 1}`);

  const setUnits = (value, source) => {
    const n = clampInt(value, 1, MAX_UNITS);
    if (n == null || n === layer.units) return;
    layer.units = n;
    if (source !== units) units.value = n;
    if (source !== slider) slider.value = n;
    architectureChanged();
  };
  units.addEventListener('input', () => setUnits(units.value, units));
  units.addEventListener('change', () => { units.value = layer.units; });
  slider.addEventListener('input', () => setUnits(slider.value, slider));
  activation.addEventListener('change', () => { layer.activation = activation.value; architectureChanged(); });
  remove.addEventListener('click', () => { state.hidden.splice(i, 1); architectureChanged({ rerenderList: true }); });

  row.append(top, slider);
  return row;
}

function syncPreset() {
  const key = JSON.stringify(state.hidden.map((l) => [l.units, l.activation]));
  const match = PRESETS.find(([, , hidden]) => JSON.stringify(hidden) === key);
  $('preset').value = match ? match[0] : '';
}

function updateParams() {
  const s = sizes();
  let total = 0, biases = 0;
  const rows = ['<tr><th>Layer</th><th>Weights</th><th>Biases</th><th>Total</th></tr>'];
  for (let l = 0; l < s.length - 1; l++) {
    const w = s[l] * s[l + 1], b = s[l + 1];
    total += w + b;
    biases += b;
    rows.push(`<tr><td>${shortName(l)} → ${shortName(l + 1)}</td><td>${fmt(w)}</td><td>${fmt(b)}</td><td>${fmt(w + b)}</td></tr>`);
  }
  rows.push(`<tr class="total"><td>Total</td><td>${fmt(total - biases)}</td><td>${fmt(biases)}</td><td>${fmt(total)}</td></tr>`);
  $('param-total').textContent = fmt(total);
  $('param-sub').textContent = `trainable parameters · ${(total * 4 / 1024).toFixed(1)} KB as float32`;
  $('param-table').innerHTML = rows.join('');
}

function architectureChanged({ rerenderList = false } = {}) {
  state.buildId++;
  state.weights = state.forward = state.stats = state.lastProgress = null;
  state.selected = null;
  if (rerenderList) renderLayerList();
  syncPreset();
  updateParams();
  chart.reset();
  renderMetrics();
  renderTrainStatus();
  renderDead();
  renderForward();
  renderSelected();
  rebuildScene();
  scheduleBuild();
}

let buildTimer = null;
function scheduleBuild() {
  clearTimeout(buildTimer);
  buildTimer = setTimeout(flushBuild, 250);
}
function flushBuild() {
  clearTimeout(buildTimer);
  buildTimer = null;
  send({ type: 'build', buildId: state.buildId, hidden: state.hidden });
}

for (const [key, label] of PRESETS) $('preset').append(new Option(label, key));
$('preset').addEventListener('change', (e) => {
  const preset = PRESETS.find(([key]) => key === e.target.value);
  if (!preset) return;
  state.hidden = preset[2].map(([units, activation]) => ({ units, activation }));
  architectureChanged({ rerenderList: true });
});

for (const [key, label] of ACTIVATIONS) $('act-all').append(new Option(label, key));
$('act-all').addEventListener('change', (e) => {
  const value = e.target.value;
  e.target.value = '';
  if (!value || !state.hidden.length) return;
  state.hidden.forEach((l) => { l.activation = value; });
  architectureChanged({ rerenderList: true });
});

$('layers-dec').addEventListener('click', () => {
  state.hidden.pop();
  architectureChanged({ rerenderList: true });
});
$('layers-inc').addEventListener('click', () => {
  const last = state.hidden[state.hidden.length - 1];
  state.hidden.push({ units: last?.units ?? 16, activation: last?.activation ?? 'relu' });
  architectureChanged({ rerenderList: true });
});

// ---------- training ----------

$('train').addEventListener('click', () => {
  if (buildTimer !== null) flushBuild();
  send({
    type: 'train',
    epochs: Number($('epochs').value),
    lr: Number($('lr').value),
    batchSize: Number($('batch').value),
    optimizer: $('optimizer').value,
    secondsPerEpoch: Number($('pace').value),
  });
});
$('stop').addEventListener('click', () => send({ type: 'stop' }));
$('reinit').addEventListener('click', () => architectureChanged());

function updateTrainButtons() {
  $('train').disabled = state.training || !connected;
  $('stop').disabled = !state.training;
  $('train').textContent = (state.stats?.epoch ?? 0) > 0 ? 'Train more' : 'Train';
}

let errorTimer = null;
function showError(message) {
  const status = $('train-status');
  status.textContent = message;
  status.classList.add('error');
  clearTimeout(errorTimer);
  errorTimer = setTimeout(renderTrainStatus, 6000);
}

function renderTrainStatus() {
  clearTimeout(errorTimer);
  const status = $('train-status');
  status.classList.remove('error');
  const epoch = state.stats?.epoch ?? 0;
  const p = state.lastProgress;
  if (state.training) {
    const where = `Training on the ${serverDeviceKind.toUpperCase()}…`;
    status.textContent = p ? `${where} epoch ${p.epoch.toFixed(2)}` : where;
  } else if (epoch > 0) {
    status.textContent = `Trained for ${epoch.toFixed(2)} epochs. "Train more" continues from these weights.`;
  } else {
    status.textContent = 'Untrained: random initial weights.';
  }
  updateTrainButtons();
}

function renderMetrics() {
  const stats = state.stats, p = state.lastProgress;
  const tiles = [
    ['Test accuracy', stats ? pct(stats.testAcc, 2) : '—'],
    ['Train loss', p?.loss != null ? p.loss.toFixed(3) : '—'],
    ['Epochs', stats ? stats.epoch.toFixed(2) : '—'],
  ];
  $('metrics').replaceChildren(...tiles.map(([label, value]) => {
    const tile = el('div');
    tile.append(el('span', null, label), el('b', null, value));
    return tile;
  }));
}

// ---------- forward pass ----------

function requestSample(extra) { send({ type: 'sample', ...extra }); }
const currentIndex = () => state.forward?.index ?? (clampInt($('sample-index').value, 0, 9999) ?? 0);

$('sample-index').addEventListener('change', (e) => {
  const index = clampInt(e.target.value, 0, 9999);
  if (index != null) requestSample({ index });
});
$('sample-prev').addEventListener('click', () => requestSample({ index: (currentIndex() + 9999) % 10000 }));
$('sample-next').addEventListener('click', () => requestSample({ index: (currentIndex() + 1) % 10000 }));
$('sample-random').addEventListener('click', () => requestSample({ mode: 'random' }));
$('sample-wrong').addEventListener('click', () => requestSample({ mode: 'misclassified' }));

function renderForward() {
  const f = state.forward;
  const ctx = $('digit').getContext('2d');
  if (!f) {
    ctx.clearRect(0, 0, 28, 28);
    $('verdict').textContent = '';
    $('probs').replaceChildren();
    return;
  }
  const image = ctx.createImageData(28, 28);
  f.pixels.forEach((p, i) => image.data.set([p, p, p, 255], i * 4));
  ctx.putImageData(image, 0, 0);
  if (document.activeElement !== $('sample-index')) $('sample-index').value = f.index;

  const correct = f.pred === f.label;
  $('verdict').innerHTML = `Label <b>${f.label}</b> · predicted <b class="${correct ? 'ok' : 'bad'}">${f.pred}</b>
    <span class="muted">(${pct(f.probs[f.pred])} confident)</span> · <span class="${correct ? 'ok' : 'bad'}">${correct ? 'correct' : 'wrong'}</span>`;
  $('probs').innerHTML = f.probs.map((p, d) => {
    const cls = ['prob', d === f.pred ? 'pred' : '', d === f.label ? 'label' : ''].join(' ');
    return `<div class="${cls}"><span>${d}</span><div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></div><span>${pct(p)}</span></div>`;
  }).join('');
}

// ---------- dead neurons ----------

function renderDead() {
  const box = $('dead-list');
  if (!state.hidden.length) { box.innerHTML = '<div class="muted small">No hidden layers, so nothing can die.</div>'; return; }
  if (!state.stats) { box.innerHTML = '<div class="muted small">Waiting for the server…</div>'; return; }
  box.innerHTML = state.stats.layers.map((info, h) => {
    const n = state.hidden[h].units;
    const name = `H${h + 1} · ${ACT_LABEL[info.activation]}`;
    if (!info.kind) {
      return `<div class="dead-row"><div class="top"><span>${name}</span><span class="muted">not applicable</span></div></div>`;
    }
    const count = info.dead.length;
    return `<div class="dead-row"><div class="top"><span>${name}</span>
      <span class="count ${count ? 'bad' : ''}">${count} / ${n} ${info.kind}</span></div>
      <div class="bar"><i style="width:${(count / n) * 100}%"></i></div></div>`;
  }).join('');
}

// ---------- neuron details (tooltip + selection panel) ----------

function describeNeuron(layer, index) {
  const s = sizes(), f = state.forward, w = state.weights;
  const rows = [];
  if (layer === 0) {
    rows.push(['Pixel value', f ? num(f.input[index], 2) : '—']);
    rows.push(['Connections', `${fmt(s[1])} out`]);
    return { title: `Input pixel (row ${Math.floor(index / 28)}, col ${index % 28})`, rows };
  }
  const bias = w?.[layer - 1]?.b[index];
  if (isOutput(layer)) {
    rows.push(['Probability', f ? pct(f.probs[index]) : '—']);
    if (f) rows.push(['True label?', f.label === index ? 'yes' : 'no']);
    rows.push(['Bias', num(bias)]);
    rows.push(['Connections', `${fmt(s[layer - 1])} in`]);
    return { title: `Output · digit ${index}`, rows };
  }
  const h = layer - 1;
  rows.push(['Activation fn', ACT_LABEL[state.hidden[h].activation]]);
  rows.push(['Output', f ? num(f.activations[h][index]) : '—']);
  rows.push(['Pre-activation', f ? num(f.preActivations[h][index]) : '—']);
  rows.push(['Bias', num(bias)]);
  const info = state.stats?.layers[h];
  if (info?.kind) {
    const rate = info.rate[index];
    const rateText = info.kind === 'saturated' ? `saturated on ${pct(rate)} of test images` : `active on ${pct(rate)} of test images`;
    rows.push(['Status', info.dead.includes(index) ? `<span class="dead">${info.kind.toUpperCase()}</span> · ${rateText}` : rateText]);
  }
  rows.push(['Connections', `${fmt(s[layer - 1])} in · ${fmt(s[layer + 1])} out`]);
  return { title: `Hidden ${layer} · neuron ${index}`, rows };
}

function showTooltip(hit, client) {
  const tip = $('tooltip');
  if (!hit || !client) { tip.hidden = true; return; }
  const { title, rows } = describeNeuron(hit.layer, hit.index);
  tip.innerHTML = `<b>${title}</b>${rows.map(([k, v]) => `<div><span class="muted">${k}:</span> ${v}</div>`).join('')}`;
  tip.hidden = false;
  tip.style.left = `${Math.min(client[0] + 14, window.innerWidth - tip.offsetWidth - 8)}px`;
  tip.style.top = `${Math.min(client[1] + 14, window.innerHeight - tip.offsetHeight - 8)}px`;
}

function renderSelected() {
  const sel = state.selected, info = $('sel-info'), canvas = $('sel-canvas'), caption = $('sel-caption');
  canvas.hidden = true;
  caption.textContent = '';
  if (!sel) {
    info.className = 'muted small';
    info.textContent = 'Click a neuron in the 3D view.';
    return;
  }
  const { title, rows } = describeNeuron(sel.layer, sel.index);
  info.className = 'small';
  info.innerHTML = `<b>${title}</b><div class="kv">${rows.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join('')}</div>`;

  const weights = state.weights;
  if (!weights) return;
  if (sel.layer === 1) {
    const w = weights[0];
    drawWeightImage(canvas, w.W.subarray(sel.index * w.inN, (sel.index + 1) * w.inN));
    caption.textContent = 'Incoming weights drawn as a 28×28 image: blue pixels push this neuron up, orange pixels push it down. It is the pattern this neuron responds to.';
  } else if (sel.layer > 1) {
    const w = weights[sel.layer - 1];
    drawWeightBars(canvas, w.W.subarray(sel.index * w.inN, (sel.index + 1) * w.inN));
    caption.textContent = `Incoming weights from the ${fmt(w.inN)} neurons of ${layerName(sel.layer - 1)} (blue positive, orange negative).`;
  } else {
    const w = weights[0];
    const outgoing = Float32Array.from({ length: w.outN }, (_, o) => w.W[o * w.inN + sel.index]);
    drawWeightBars(canvas, outgoing);
    caption.textContent = `Outgoing weights to the ${fmt(w.outN)} neurons of ${layerName(1)}.`;
  }
}

function maxAbsOf(values) {
  let m = 0;
  for (const v of values) m = Math.max(m, Math.abs(v));
  return m || 1;
}

function drawWeightImage(canvas, values) {
  canvas.hidden = false;
  canvas.className = 'pixels';
  canvas.width = 28;
  canvas.height = 28;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(28, 28);
  const max = maxAbsOf(values);
  values.forEach((v, i) => {
    const t = Math.abs(v) / max, c = v >= 0 ? POSITIVE : NEGATIVE;
    image.data.set([c[0] * t, c[1] * t, c[2] * t, 255], i * 4);
  });
  ctx.putImageData(image, 0, 0);
}

function drawWeightBars(canvas, values) {
  canvas.hidden = false;
  canvas.className = 'bars';
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const max = maxAbsOf(values), mid = h / 2, step = w / values.length;
  ctx.fillStyle = '#222b3a';
  ctx.fillRect(0, mid, w, 1);
  values.forEach((v, i) => {
    const barH = (Math.abs(v) / max) * (mid - 4);
    ctx.fillStyle = v >= 0 ? `rgb(${POSITIVE})` : `rgb(${NEGATIVE})`;
    ctx.fillRect(i * step, v >= 0 ? mid - barH : mid + 1, Math.max(1, step - 1), barH);
  });
}

// ---------- view controls ----------

function setView(patch) {
  state.view = { ...state.view, ...patch };
  refreshScene();
}

$('edge-mode').addEventListener('change', (e) => setView({ edgeMode: e.target.value }));
$('cutoff').addEventListener('input', (e) => {
  $('cutoff-out').textContent = `${e.target.value}%`;
  setView({ cutoff: Number(e.target.value) / 100 });
});
$('brightness').addEventListener('input', (e) => {
  $('bright-out').textContent = `${Number(e.target.value).toFixed(1)}×`;
  setView({ brightness: Number(e.target.value) });
});
$('max-edges').addEventListener('input', (e) => { $('max-edges-out').textContent = `${Math.round(e.target.value / 1000)}k`; });
$('max-edges').addEventListener('change', (e) => {
  state.view = { ...state.view, maxEdges: Number(e.target.value) };
  scene.buildLinks(state.view.maxEdges);
  refreshScene();
  updateHud();
});
$('show-dead').addEventListener('change', (e) => setView({ showDead: e.target.checked }));
for (const button of document.querySelectorAll('[data-view]')) {
  button.addEventListener('click', () => scene.setView(button.dataset.view));
}
$('auto-rotate').addEventListener('change', (e) => scene.setAutoRotate(e.target.checked));

// ---------- server connection ----------

let ws = null;
let connected = false;
let serverDevice = '…';
let serverDeviceKind = 'server';

function send(message) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
}

function setConnected(value) {
  connected = value;
  const pill = $('conn');
  pill.textContent = value ? 'server connected' : 'reconnecting…';
  pill.className = `pill ${value ? 'ok' : 'bad'}`;
  updateTrainButtons();
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    setConnected(true);
    flushBuild();
  };
  ws.onclose = () => {
    setConnected(false);
    state.training = false;
    renderTrainStatus();
    setTimeout(connect, 1500);
  };
  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) onWeights(event.data);
    else onMessage(JSON.parse(event.data));
  };
}

function onWeights(buffer) {
  const [buildId, count] = new Uint32Array(buffer, 0, 2);
  if (buildId !== state.buildId) return;
  const dims = new Uint32Array(buffer, 8, count * 2);
  const s = sizes();
  if (count !== s.length - 1) return;
  let offset = 8 + count * 8;
  const layers = [];
  for (let l = 0; l < count; l++) {
    const inN = dims[l * 2], outN = dims[l * 2 + 1];
    if (inN !== s[l] || outN !== s[l + 1]) return;
    const W = new Float32Array(buffer, offset, inN * outN);
    offset += inN * outN * 4;
    const b = new Float32Array(buffer, offset, outN);
    offset += outN * 4;
    layers.push({ inN, outN, W, b });
  }
  state.weights = layers;
  refreshScene();
  if (state.selected) renderSelected();
}

function onMessage(msg) {
  if (msg.type === 'hello') {
    serverDevice = msg.device;
    serverDeviceKind = msg.deviceKind;
    MAX_LAYERS = msg.maxLayers;
    MAX_UNITS = msg.maxUnits;
    updateHud();
    return;
  }
  if (msg.type === 'error') { showError(msg.message); return; }
  if (msg.buildId !== state.buildId) return;

  if (msg.type === 'progress') {
    state.lastProgress = msg;
    chart.addTrain(msg.epoch, msg.loss, msg.trainAcc);
    renderMetrics();
    renderTrainStatus();
  } else if (msg.type === 'stats') {
    state.stats = msg;
    chart.addTest(msg.epoch, msg.testAcc);
    renderMetrics();
    renderDead();
    renderTrainStatus();
    refreshScene();
    if (state.selected) renderSelected();
  } else if (msg.type === 'forward') {
    msg.input = Float32Array.from(msg.pixels, (p) => p / 255);
    state.forward = msg;
    renderForward();
    refreshScene();
    if (state.selected) renderSelected();
  } else if (msg.type === 'status') {
    state.training = msg.training;
    renderTrainStatus();
  }
}

// ---------- HUD ----------

const gpuName = scene.gpuName.replace(/\s*\(0x[0-9a-f]+\)/i, '').slice(0, 90);
function updateHud() {
  const neurons = sizes().reduce((a, b) => a + b, 0);
  $('hud').textContent = [
    `Render  ${gpuName} · ${scene.fps} fps`,
    `Train   ${serverDeviceKind.toUpperCase()} · ${serverDevice} (PyTorch)`,
    `Scene   ${fmt(neurons)} neurons · ${fmt(scene.edgeDrawn)} of ${fmt(scene.edgeTotal)} connections drawn`,
  ].join('\n');
}
setInterval(updateHud, 500);

// ---------- start ----------

renderLayerList();
syncPreset();
updateParams();
renderMetrics();
renderTrainStatus();
renderDead();
rebuildScene();
connect();
