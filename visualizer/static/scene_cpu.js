import * as THREE from 'three';
import { COLORS, NetworkScene } from './scene.js';

const NEAR = 0.1;
const MIN_ALPHA = 1 / 512;
const EDGE_CHUNK = 64;
const BACKGROUND = `#${COLORS.background.getHexString()}`;
// Edge colors live in Float32Arrays, so match against float32-rounded palette values.
const EDGE_PALETTE = [COLORS.edgePositive, COLORS.edgeNegative, COLORS.edgeIdle].map((c) => ({
  r: Math.fround(c.r), g: Math.fround(c.g), style: `#${c.getHexString()}`,
}));

const toSrgb255 = (c) => Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055));

/**
 * Draws the same network as NetworkScene with a software Canvas 2D context instead of WebGL.
 * Layout, colors, picking and camera controls all come from the base class; only drawing differs.
 */
export class CpuNetworkScene extends NetworkScene {
  initRenderer(canvas) {
    // willReadFrequently makes the browser back this canvas with CPU memory and rasterize it in
    // software, instead of the default GPU-accelerated canvas.
    this.ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
    this.width = 0;
    this.height = 0;
    this.dirty = true;
    this.frameMs = 0;
    this.screen = [];
    this.labels = [];
    this.viewProjection = new THREE.Matrix4();
    this.neuronColor = new THREE.Color();
  }

  get rendererName() { return 'CPU · software Canvas 2D'; }

  get perfText() { return `${this.frameMs.toFixed(0)} ms/frame, redraws on change`; }

  invalidate() { this.dirty = true; }

  setSize(w, h) {
    // 1× resolution: CPU rasterization cost grows with pixel count, and 2× is ~4× slower.
    this.canvas.width = w;
    this.canvas.height = h;
    this.width = w;
    this.height = h;
  }

  build(...args) {
    super.build(...args);
    this.screen = this.layers.map((layer) => new Float32Array(layer.n * 3));
    this.labels = this.root.children.filter((obj) => obj.isSprite);
  }

  render(cameraMoved) {
    if (!cameraMoved && !this.dirty) return false;
    this.dirty = false;
    const t0 = performance.now();
    const { ctx } = this;

    ctx.globalAlpha = 1;
    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, this.width, this.height);
    if (this.screen.length === this.layers.length) {
      this.viewProjection.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
      this.projectNeurons();
      this.drawEdges();
      this.drawNeurons();
      this.drawLabels();
    }
    // Canvas draws are recorded lazily; reading one pixel forces rasterization so the timing is real.
    ctx.getImageData(0, 0, 1, 1);
    this.frameMs = performance.now() - t0;
    return true;
  }

  project(x, y, z, out, offset) {
    const m = this.viewProjection.elements;
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (w <= NEAR) {
      out[offset + 2] = -1;
      return;
    }
    out[offset] = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / w * 0.5 + 0.5) * this.width;
    out[offset + 1] = (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w * 0.5) * this.height;
    out[offset + 2] = w;
  }

  projectNeurons() {
    this.layers.forEach((layer, li) => {
      const pos = layer.positions, out = this.screen[li];
      for (let i = 0; i < layer.n; i++) this.project(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2], out, i * 3);
    });
  }

  drawEdges() {
    const { ctx } = this;
    ctx.lineWidth = 1;
    for (const link of this.links) {
      if (!link.lines.visible) continue;
      const from = this.screen[link.l], to = this.screen[link.l + 1];
      const { colors, idx, inN } = link;
      // Switching strokeStyle per line is slow, so draw one color at a time within small chunks;
      // chunking keeps positive and negative lines interleaved so neither color paints over the other.
      for (let start = 0; start < idx.length; start += EDGE_CHUNK) {
        const end = Math.min(idx.length, start + EDGE_CHUNK);
        for (const color of EDGE_PALETTE) {
          ctx.strokeStyle = color.style;
          for (let e = start; e < end; e++) {
            const k = e * 8, alpha = colors[k + 3];
            if (alpha < MIN_ALPHA || colors[k] !== color.r || colors[k + 1] !== color.g) continue;
            const flat = idx[e], o = Math.floor(flat / inN), i = flat - o * inN;
            if (from[i * 3 + 2] < 0 || to[o * 3 + 2] < 0) continue;
            ctx.globalAlpha = alpha;
            ctx.beginPath();
            ctx.moveTo(from[i * 3], from[i * 3 + 1]);
            ctx.lineTo(to[o * 3], to[o * 3 + 1]);
            ctx.stroke();
          }
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  drawNeurons() {
    const order = [];
    this.layers.forEach((layer, li) => {
      const s = this.screen[li];
      for (let i = 0; i < layer.n; i++) if (s[i * 3 + 2] > 0) order.push(li * 65536 + i);
    });
    const depth = (id) => this.screen[Math.floor(id / 65536)][(id % 65536) * 3 + 2];
    order.sort((a, b) => depth(b) - depth(a));

    const { ctx } = this;
    const focal = this.camera.projectionMatrix.elements[5] * this.height / 2;
    for (const id of order) {
      const li = Math.floor(id / 65536), i = id % 65536;
      const layer = this.layers[li];
      const colors = layer.mesh.instanceColor.array;
      const scale = layer.mesh.instanceMatrix.array[i * 16];
      const r = toSrgb255(colors[i * 3]), g = toSrgb255(colors[i * 3 + 1]), b = toSrgb255(colors[i * 3 + 2]);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      if (layer.kind === 'input') {
        this.fillPixel(layer, i);
        continue;
      }
      const s = this.screen[li];
      const x = s[i * 3], y = s[i * 3 + 1];
      const radius = Math.max(0.8, layer.radius * scale * focal / s[i * 3 + 2]);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      if (radius > 2.5) {
        ctx.strokeStyle = 'rgba(0,0,0,0.35)';
        ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.22)';
        ctx.beginPath();
        ctx.arc(x - radius * 0.32, y - radius * 0.32, radius * 0.42, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  fillPixel(layer, i) {
    const x = layer.positions[i * 3], y = layer.positions[i * 3 + 1], z = layer.positions[i * 3 + 2];
    const h = layer.radius;
    const corners = this.cornerBuffer ??= new Float32Array(12);
    this.project(x, y + h, z - h, corners, 0);
    this.project(x, y + h, z + h, corners, 3);
    this.project(x, y - h, z + h, corners, 6);
    this.project(x, y - h, z - h, corners, 9);
    if (corners[2] < 0 || corners[5] < 0 || corners[8] < 0 || corners[11] < 0) return;
    const { ctx } = this;
    ctx.beginPath();
    ctx.moveTo(corners[0], corners[1]);
    ctx.lineTo(corners[3], corners[4]);
    ctx.lineTo(corners[6], corners[7]);
    ctx.lineTo(corners[9], corners[10]);
    ctx.closePath();
    ctx.fill();
  }

  drawLabels() {
    const focal = this.camera.projectionMatrix.elements[5] * this.height / 2;
    const out = this.labelBuffer ??= new Float32Array(3);
    for (const sprite of this.labels) {
      this.project(sprite.position.x, sprite.position.y, sprite.position.z, out, 0);
      if (out[2] < 0) continue;
      const h = sprite.scale.y * focal / out[2], w = sprite.scale.x * focal / out[2];
      if (h < 3) continue;
      this.ctx.drawImage(sprite.material.map.image, out[0] - w / 2, out[1] - h / 2, w, h);
    }
  }
}
