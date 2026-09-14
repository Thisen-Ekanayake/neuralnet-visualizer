import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const LAYER_GAP = 9;
const INPUT_SPACING = 0.26;
const HIDDEN_SPACING = 0.62;
const OUTPUT_SPACING = 0.8;

const COLORS = {
  background: new THREE.Color('#0b0f17'),
  idle: new THREE.Color('#465068'),
  positive: new THREE.Color('#ffc861'),
  negative: new THREE.Color('#4cc9f0'),
  dead: new THREE.Color('#ff3b5c'),
  selected: new THREE.Color('#ffffff'),
  pixelOff: new THREE.Color('#161c29'),
  pixelOn: new THREE.Color('#ffffff'),
  edgePositive: new THREE.Color('#3d8bff'),
  edgeNegative: new THREE.Color('#ff6a3d'),
  edgeIdle: new THREE.Color('#5b6b8c'),
};

const VIEWS = {
  angled: [-0.62, 0.38, 0.69],
  side: [0, 0.08, 1],
  front: [-1, 0.06, 0.02],
  top: [0, 1, 0.02],
};

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleIndices(total, count, seed) {
  const all = new Uint32Array(total);
  for (let i = 0; i < total; i++) all[i] = i;
  if (count >= total) return all;
  const rand = mulberry32(seed);
  for (let i = 0; i < count; i++) {
    const j = i + Math.floor(rand() * (total - i));
    const tmp = all[i]; all[i] = all[j]; all[j] = tmp;
  }
  return all.slice(0, count);
}

function layoutLayer(n, kind, x) {
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    let y, z;
    if (kind === 'input') {
      const row = Math.floor(i / 28), col = i % 28;
      y = (13.5 - row) * INPUT_SPACING;
      z = (col - 13.5) * INPUT_SPACING;
    } else if (kind === 'output') {
      y = (4.5 - i) * OUTPUT_SPACING;
      z = 0;
    } else {
      const cols = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols);
      const row = Math.floor(i / cols), col = i % cols;
      y = ((rows - 1) / 2 - row) * HIDDEN_SPACING;
      z = (col - (cols - 1) / 2) * HIDDEN_SPACING;
    }
    positions.set([x, y, z], i * 3);
  }
  return positions;
}

function textSprite(title, subtitle = '') {
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = subtitle ? 150 : 96;
  const ctx = canvas.getContext('2d');
  ctx.font = '600 58px system-ui, sans-serif';
  const width = Math.min(500, Math.max(ctx.measureText(title).width + 48, subtitle ? 330 : 0));
  ctx.fillStyle = 'rgba(11, 15, 23, 0.72)';
  ctx.beginPath();
  ctx.roundRect(256 - width / 2, 4, width, canvas.height - 8, 22);
  ctx.fill();
  ctx.textAlign = 'center';
  ctx.fillStyle = '#d8dfeb';
  ctx.font = '600 58px system-ui, sans-serif';
  ctx.fillText(title, 256, 64);
  if (subtitle) {
    ctx.fillStyle = '#8a96ab';
    ctx.font = '42px system-ui, sans-serif';
    ctx.fillText(subtitle, 256, 128);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false, depthTest: false }));
  sprite.renderOrder = 10;
  const height = subtitle ? 1.35 : 0.85;
  sprite.scale.set(height * canvas.width / canvas.height, height, 1);
  return sprite;
}

export class NetworkScene {
  constructor(canvas, { onHover, onSelect }) {
    this.canvas = canvas;
    this.onHover = onHover;
    this.onSelect = onSelect;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.background = COLORS.background;
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 5000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.autoRotateSpeed = 0.8;

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(-3, 6, 8);
    this.scene.add(sun);

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.edgeMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false });

    this.layers = [];
    this.links = [];
    this.data = null;
    this.hovered = null;
    this.edgeTotal = 0;
    this.edgeDrawn = 0;
    this.fps = 0;

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.pointerClient = null;
    this.needsPick = false;
    this.downAt = null;
    this.tmpMatrix = new THREE.Matrix4();
    this.tmpColor = new THREE.Color();
    this.tmpPos = new THREE.Vector3();
    this.tmpScale = new THREE.Vector3();
    this.identityQuat = new THREE.Quaternion();

    canvas.addEventListener('pointermove', (e) => this.onPointerMove(e));
    canvas.addEventListener('pointerleave', () => { this.pointerClient = null; this.setHovered(null); });
    canvas.addEventListener('pointerdown', (e) => { this.downAt = [e.clientX, e.clientY]; });
    canvas.addEventListener('pointerup', (e) => this.onPointerUp(e));
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();

    let frames = 0, last = performance.now();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      if (this.needsPick) this.pick();
      this.renderer.render(this.scene, this.camera);
      frames++;
      const now = performance.now();
      if (now - last >= 500) { this.fps = Math.round(frames * 1000 / (now - last)); frames = 0; last = now; }
    });
  }

  get gpuName() {
    const gl = this.renderer.getContext();
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  clear() {
    this.root.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material && obj.material !== this.edgeMaterial) {
        obj.material.map?.dispose();
        obj.material.dispose();
      }
    });
    this.root.clear();
    this.layers = [];
    this.links = [];
  }

  build(sizes, activationLabels, maxEdges) {
    const previousLayerCount = this.layers.length;
    this.clear();
    this.sizes = sizes;
    this.data = null;
    this.hovered = null;
    const span = (sizes.length - 1) * LAYER_GAP;

    this.layers = sizes.map((n, li) => {
      const kind = li === 0 ? 'input' : li === sizes.length - 1 ? 'output' : 'hidden';
      const x = li * LAYER_GAP - span / 2;
      const positions = layoutLayer(n, kind, x);
      const geometry = kind === 'input'
        ? new THREE.BoxGeometry(0.05, 0.22, 0.22)
        : new THREE.SphereGeometry(kind === 'output' ? 0.3 : 0.21, 20, 14);
      const mesh = new THREE.InstancedMesh(geometry, new THREE.MeshLambertMaterial({ color: 0xffffff }), n);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.root.add(mesh);

      let halfHeight = 0;
      for (let i = 0; i < n; i++) halfHeight = Math.max(halfHeight, Math.abs(positions[i * 3 + 1]));
      const title = kind === 'input' ? 'Input' : kind === 'output' ? 'Output' : `Hidden ${li}`;
      const subtitle = kind === 'input' ? '784 (28×28)' : kind === 'output' ? '10 digits'
        : `${n} · ${activationLabels[li - 1]}`;
      const label = textSprite(title, subtitle);
      label.position.set(x, -halfHeight - 1.4, 0);
      this.root.add(label);

      if (kind === 'output') {
        for (let d = 0; d < 10; d++) {
          const digit = textSprite(String(d));
          digit.scale.multiplyScalar(0.7);
          digit.position.set(x + 0.9, positions[d * 3 + 1], 0);
          this.root.add(digit);
        }
      }
      return { kind, n, x, positions, mesh, halfHeight };
    });

    this.buildLinks(maxEdges);
    this.refresh();
    this.layers.forEach((layer) => layer.mesh.computeBoundingSphere());
    if (previousLayerCount !== sizes.length) this.setView('angled');
  }

  buildLinks(maxEdges) {
    for (const link of this.links) {
      this.root.remove(link.lines);
      link.lines.geometry.dispose();
    }
    this.links = [];
    const sizes = this.sizes;
    let total = 0;
    for (let l = 0; l < sizes.length - 1; l++) total += sizes[l] * sizes[l + 1];
    const ratio = Math.min(1, maxEdges / total);
    this.edgeTotal = total;
    this.edgeDrawn = 0;

    for (let l = 0; l < sizes.length - 1; l++) {
      const inN = sizes[l], outN = sizes[l + 1], m = inN * outN;
      const count = Math.max(1, Math.round(m * ratio));
      const idx = sampleIndices(m, count, 1234 + l);
      const from = this.layers[l].positions, to = this.layers[l + 1].positions;
      const positions = new Float32Array(count * 6);
      for (let e = 0; e < count; e++) {
        const flat = idx[e], o = Math.floor(flat / inN), i = flat - o * inN;
        positions[e * 6] = from[i * 3]; positions[e * 6 + 1] = from[i * 3 + 1]; positions[e * 6 + 2] = from[i * 3 + 2];
        positions[e * 6 + 3] = to[o * 3]; positions[e * 6 + 4] = to[o * 3 + 1]; positions[e * 6 + 5] = to[o * 3 + 2];
      }
      const colors = new Float32Array(count * 8);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const colorAttr = new THREE.BufferAttribute(colors, 4).setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('color', colorAttr);
      const lines = new THREE.LineSegments(geometry, this.edgeMaterial);
      lines.renderOrder = -1;
      this.root.add(lines);
      this.links.push({
        l, inN, outN, idx, lines, colors, colorAttr,
        values: new Float32Array(count),
        // Overlap grows with line count, so per-line opacity must shrink with it to keep structure visible.
        density: Math.min(1, 500 / count) ** 0.75,
        fanIn: Math.min(1, 6 / Math.sqrt(inN)),
      });
      this.edgeDrawn += count;
    }
    this.colorEdges();
  }

  setView(name) {
    if (!this.layers.length) return;
    const first = this.layers[0].x, last = this.layers[this.layers.length - 1].x;
    const halfHeight = Math.max(...this.layers.map((l) => l.halfHeight)) + 1.5;
    const center = new THREE.Vector3((first + last) / 2, 0, 0);
    const radius = Math.hypot((last - first) / 2 + 2, halfHeight);
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const fit = radius / Math.sin(fov / 2) / Math.min(1, this.camera.aspect * 0.9);
    const distance = name === 'front' ? fit * 0.55 : fit * 0.8;
    // The angled camera sits on the input side, so perspective enlarges the input grid; bias the target toward it.
    const target = name === 'front' ? new THREE.Vector3(first, 0, 0)
      : name === 'angled' ? center.clone().setX(center.x - (last - first) * 0.12) : center;
    const dir = new THREE.Vector3(...VIEWS[name]).normalize();
    this.camera.position.copy(target).addScaledVector(dir, distance);
    this.controls.target.copy(target);
    this.controls.update();
  }

  setAutoRotate(on) { this.controls.autoRotate = on; }

  update(data) {
    const edgesChanged = !this.data
      || data.weights !== this.data.weights
      || data.selected !== this.data.selected
      || data.view !== this.data.view
      || (data.view.edgeMode === 'signal' && data.forward !== this.data.forward);
    this.data = data;
    this.refreshNeurons();
    if (edgesChanged) this.colorEdges();
  }

  refresh() {
    this.refreshNeurons();
    this.colorEdges();
  }

  refreshNeurons() {
    const data = this.data;
    const forward = data?.forward, stats = data?.stats, selected = data?.selected;
    const showDead = data?.view.showDead;
    const color = this.tmpColor;

    this.layers.forEach((layer, li) => {
      const { mesh, n, positions, kind } = layer;
      const acts = kind === 'hidden' ? forward?.activations[li - 1] : null;
      let maxAbs = 0;
      if (acts) for (const a of acts) maxAbs = Math.max(maxAbs, Math.abs(a));
      const dead = kind === 'hidden' && showDead && stats ? new Set(stats.layers[li - 1]?.dead) : null;

      for (let i = 0; i < n; i++) {
        let scale = 1;
        if (kind === 'input') {
          color.lerpColors(COLORS.pixelOff, COLORS.pixelOn, forward ? forward.input[i] : 0);
        } else if (kind === 'output') {
          const p = forward ? forward.probs[i] : 0;
          color.lerpColors(COLORS.idle, COLORS.positive, Math.sqrt(p));
          scale = 1 + 0.6 * p;
        } else if (acts && maxAbs > 0) {
          const a = acts[i];
          color.lerpColors(COLORS.idle, a >= 0 ? COLORS.positive : COLORS.negative, Math.abs(a) / maxAbs);
        } else {
          color.copy(COLORS.idle);
        }
        if (dead?.has(i)) color.copy(COLORS.dead);
        if (selected && selected.layer === li && selected.index === i) { color.copy(COLORS.selected); scale *= 1.5; }
        if (this.hovered && this.hovered.layer === li && this.hovered.index === i) scale *= 1.35;

        this.tmpPos.fromArray(positions, i * 3);
        this.tmpScale.setScalar(scale);
        mesh.setMatrixAt(i, this.tmpMatrix.compose(this.tmpPos, this.identityQuat, this.tmpScale));
        mesh.setColorAt(i, color);
      }
      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor.needsUpdate = true;
    });
  }

  colorEdges() {
    const data = this.data;
    const view = data?.view ?? { edgeMode: 'weights', cutoff: 0, brightness: 1 };
    const weights = data?.weights, forward = data?.forward, selected = data?.selected;

    for (const link of this.links) {
      const { lines, colors, idx, values, inN, l } = link;
      lines.visible = view.edgeMode !== 'off';
      if (!lines.visible) continue;

      const W = weights?.[l]?.W;
      const source = forward ? (l === 0 ? forward.input : forward.activations[l - 1]) : null;
      const signal = view.edgeMode === 'signal' && source;
      const incoming = selected && selected.layer === l + 1;
      const outgoing = selected && selected.layer === l;
      const gain = view.brightness * link.density;

      if (!W) {
        const alpha = Math.min(1, 0.4 * gain);
        for (let k = 0; k < colors.length; k += 4) {
          colors[k] = COLORS.edgeIdle.r; colors[k + 1] = COLORS.edgeIdle.g; colors[k + 2] = COLORS.edgeIdle.b;
          colors[k + 3] = alpha;
        }
        link.colorAttr.needsUpdate = true;
        continue;
      }

      let maxAbs = 0;
      for (let e = 0; e < idx.length; e++) {
        const flat = idx[e];
        let v = W[flat];
        if (signal) v *= source[flat % inN];
        values[e] = v;
        const a = Math.abs(v);
        if (a > maxAbs) maxAbs = a;
      }
      const inv = maxAbs > 0 ? 1 / maxAbs : 0;

      for (let e = 0; e < idx.length; e++) {
        const v = values[e], t = Math.abs(v) * inv;
        let b = t < view.cutoff ? 0 : t * gain;
        if (selected) {
          const flat = idx[e];
          const o = Math.floor(flat / inN), i = flat - o * inN;
          const connected = (incoming && o === selected.index) || (outgoing && i === selected.index);
          const focus = incoming ? link.fanIn : 1;
          b = connected ? (t < view.cutoff ? 0 : t * view.brightness * focus) : b * 0.2;
        }
        const c = v >= 0 ? COLORS.edgePositive : COLORS.edgeNegative;
        const k = e * 8;
        colors[k] = colors[k + 4] = c.r;
        colors[k + 1] = colors[k + 5] = c.g;
        colors[k + 2] = colors[k + 6] = c.b;
        colors[k + 3] = colors[k + 7] = Math.min(1, b);
      }
      link.colorAttr.needsUpdate = true;
    }
  }

  onPointerMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.pointerClient = [e.clientX, e.clientY];
    this.needsPick = true;
  }

  onPointerUp(e) {
    if (!this.downAt) return;
    const moved = Math.hypot(e.clientX - this.downAt[0], e.clientY - this.downAt[1]);
    this.downAt = null;
    if (moved < 5 && e.button === 0) this.onSelect(this.hitTest());
  }

  hitTest() {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.layers.map((l) => l.mesh), false);
    if (!hits.length) return null;
    const layer = this.layers.findIndex((l) => l.mesh === hits[0].object);
    return { layer, index: hits[0].instanceId };
  }

  pick() {
    this.needsPick = false;
    if (!this.pointerClient) return;
    this.setHovered(this.hitTest());
  }

  setHovered(hit) {
    const changed = (hit?.layer !== this.hovered?.layer) || (hit?.index !== this.hovered?.index);
    this.hovered = hit;
    if (changed) this.refreshNeurons();
    this.onHover(hit, this.pointerClient);
  }
}
