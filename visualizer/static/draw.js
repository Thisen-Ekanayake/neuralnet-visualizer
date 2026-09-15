// Small 2D-canvas helpers shared by the inspect panel and the training-step walkthrough.

export const POSITIVE = [61, 139, 255];
export const NEGATIVE = [255, 106, 61];
const BACKGROUND = [11, 15, 23];
const HIGHLIGHT = [255, 255, 255];

let software = false;
/** With --cpu every canvas is created with willReadFrequently, which keeps it in CPU memory. */
export function setSoftwareCanvas(value) { software = value; }
export const ctx2d = (canvas) => canvas.getContext('2d', { willReadFrequently: software });

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

export function maxAbsOf(values) {
  let m = 0;
  for (const v of values) m = Math.max(m, Math.abs(v));
  return m || 1;
}

function shade(v, max) {
  const t = Math.min(1, Math.abs(v) / max), c = v >= 0 ? POSITIVE : NEGATIVE;
  return [c[0] * t, c[1] * t, c[2] * t, 255];
}

/** 784 values as a 28×28 image: blue positive, orange negative, black near zero. */
export function drawWeightImage(canvas, values) {
  canvas.width = 28;
  canvas.height = 28;
  const ctx = ctx2d(canvas);
  const image = ctx.createImageData(28, 28);
  const max = maxAbsOf(values);
  values.forEach((v, i) => image.data.set(shade(v, max), i * 4));
  ctx.putImageData(image, 0, 0);
}

/** One bar per value around a center line, sized to the canvas's CSS box; `highlight` is drawn white. */
export function drawWeightBars(canvas, values, highlight = -1) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = ctx2d(canvas);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const max = maxAbsOf(values), mid = h / 2, step = w / values.length;
  ctx.fillStyle = '#222b3a';
  ctx.fillRect(0, mid, w, 1);
  values.forEach((v, i) => {
    const barH = Math.max(i === highlight ? 2 : 0, (Math.abs(v) / max) * (mid - 4));
    ctx.fillStyle = i === highlight ? '#ffffff' : v >= 0 ? `rgb(${POSITIVE})` : `rgb(${NEGATIVE})`;
    ctx.fillRect(i * step, v >= 0 ? mid - barH : mid + 1, Math.max(1, step - 1), barH);
  });
  return (x) => Math.min(values.length - 1, Math.max(0, Math.floor(x / step)));
}

/**
 * A rows × 784 matrix as one 28×28 tile per row, on a shared color scale, with 1px frames.
 * `highlight` = {row, col} frames that tile in white and whitens that pixel.
 * Returns a function mapping a CSS-pixel click to {row, col} (or null).
 */
export function drawTiles(canvas, values, rows, highlight = null) {
  const perRow = Math.min(rows, Math.max(1, Math.ceil(Math.sqrt(rows * 1.6))));
  const tileRows = Math.ceil(rows / perRow);
  const size = 29;
  canvas.width = perRow * size + 1;
  canvas.height = tileRows * size + 1;
  canvas.style.width = `${Math.min(460, canvas.width * (rows <= 4 ? 4 : rows <= 16 ? 3 : 2))}px`;
  const ctx = ctx2d(canvas);
  const image = ctx.createImageData(canvas.width, canvas.height);
  const put = (x, y, rgba) => image.data.set(rgba, (y * canvas.width + x) * 4);
  for (let k = 0; k < image.data.length; k += 4) image.data.set([...BACKGROUND, 255], k);
  const max = maxAbsOf(values);
  for (let r = 0; r < rows; r++) {
    const ox = (r % perRow) * size + 1, oy = Math.floor(r / perRow) * size + 1;
    for (let p = 0; p < 784; p++) put(ox + (p % 28), oy + Math.floor(p / 28), shade(values[r * 784 + p], max));
    if (highlight?.row === r) {
      for (let k = -1; k <= 28; k++) {
        put(ox + k, oy - 1, [...HIGHLIGHT, 255]); put(ox + k, oy + 28, [...HIGHLIGHT, 255]);
        put(ox - 1, oy + k, [...HIGHLIGHT, 255]); put(ox + 28, oy + k, [...HIGHLIGHT, 255]);
      }
      if (highlight.col != null) put(ox + (highlight.col % 28), oy + Math.floor(highlight.col / 28), [...HIGHLIGHT, 255]);
    }
  }
  ctx.putImageData(image, 0, 0);
  return (x, y) => {
    const scale = canvas.width / canvas.clientWidth;
    const px = Math.floor(x * scale) - 1, py = Math.floor(y * scale) - 1;
    const tx = Math.floor(px / size), ty = Math.floor(py / size), ix = px % size, iy = py % size;
    const row = ty * perRow + tx;
    if (px < 0 || py < 0 || tx >= perRow || row >= rows || ix >= 28 || iy >= 28) return null;
    return { row, col: iy * 28 + ix };
  };
}

/** A rows × cols matrix heatmap (row = to-neuron, col = from-neuron) with the highlighted cell framed. */
export function drawMatrix(canvas, values, rows, cols, highlight = null) {
  const cellW = Math.max(1, Math.min(24, Math.floor(440 / cols)));
  const cellH = Math.max(1, Math.min(24, cellW, Math.floor(300 / rows)));
  canvas.width = cols * cellW;
  canvas.height = rows * cellH;
  canvas.style.width = `${canvas.width}px`;
  const ctx = ctx2d(canvas);
  const max = maxAbsOf(values);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = `rgba(${shade(values[r * cols + c], max).slice(0, 3)},1)`;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }
  }
  if (highlight) {
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = Math.max(1, Math.min(2, cellW / 4));
    ctx.strokeRect(highlight.col * cellW + 0.5, highlight.row * cellH + 0.5, Math.max(1, cellW - 1), Math.max(1, cellH - 1));
  }
  return (x, y) => {
    const scale = canvas.width / canvas.clientWidth;
    const col = Math.floor((x * scale) / cellW), row = Math.floor((y * scale) / cellH);
    return row >= 0 && row < rows && col >= 0 && col < cols ? { row, col } : null;
  };
}
