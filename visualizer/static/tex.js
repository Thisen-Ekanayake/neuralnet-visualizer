// Numbers → TeX for the walkthrough. Pure functions (so node can test them); only renderTex touches KaTeX.

export const T = String.raw;
const SIG = 4;
const trimZeros = (s) => (s.includes('.') ? s.replace(/\.?0+$/, '') : s);

/** `sig` significant digits; ×10^k below 0.001 or from 10,000 up (and whenever rounding would need it). */
export function num(v, sig = SIG) {
  if (v == null || Number.isNaN(v)) return T`\text{n/a}`;
  if (!Number.isFinite(v)) return v > 0 ? T`\infty` : T`-\infty`;
  if (v === 0) return '0';
  const abs = Math.abs(v);
  const fixed = abs >= 1e-3 && abs < 1e4 ? v.toPrecision(sig) : null;
  if (fixed && !fixed.includes('e')) return trimZeros(fixed);
  const [mantissa, exponent] = v.toExponential(sig - 1).split('e');
  const m = trimZeros(mantissa);
  const power = T`10^{${Number(exponent)}}`;
  return m === '1' ? power : m === '-1' ? `-${power}` : T`${m}\times${power}`;
}

/** A number used as a factor: negatives and ×10^k forms go in parentheses. */
export function factor(v, sig = SIG) {
  const s = num(v, sig);
  return v < 0 || s.includes('10^') ? `(${s})` : s;
}

/** Significant digits that show 3 digits of a change `delta` in `value` (for weights before/after a step). */
export function sigFor(value, delta) {
  if (!value || !delta) return SIG;
  const gap = Math.floor(Math.log10(Math.abs(value))) - Math.floor(Math.log10(Math.abs(delta)));
  return Math.min(9, Math.max(SIG, gap + 3));
}

/** A row vector, eliding the middle: [a, b, c, …, y, z]ᵀ. */
export function vec(values, { head = 3, tail = 2, sig = SIG } = {}) {
  const n = values.length;
  const fmt = (v) => num(v, sig);
  const items = n <= head + tail + 1 ? Array.from(values, fmt)
    : [...Array.from(values.slice(0, head), fmt), T`\cdots`, ...Array.from(values.slice(n - tail), fmt)];
  return T`\begin{bmatrix}${items.join(' & ')}\end{bmatrix}^{\!\top}`;
}

/**
 * Σ_j xs[j]·ys[j] (+ bias) split into the `top` largest |terms| and the rest.
 * `total` is the value to reconcile against (the server's number); the remainder is defined
 * as total − shown − bias, so shown + rest + bias = total exactly. Zero terms are left out.
 */
export function expandSum(xs, ys, { top = 3, bias = 0, total } = {}) {
  const terms = [];
  for (let j = 0; j < xs.length; j++) {
    const p = xs[j] * ys[j];
    if (p !== 0) terms.push({ j, x: xs[j], y: ys[j], p });
  }
  terms.sort((a, b) => Math.abs(b.p) - Math.abs(a.p));
  const shown = terms.slice(0, top);
  const sum = total ?? terms.reduce((s, t) => s + t.p, 0) + bias;
  const rest = sum - shown.reduce((s, t) => s + t.p, 0) - bias;
  return { shown, restCount: terms.length - shown.length, rest, bias, total: sum, nonZero: terms.length, count: xs.length };
}

/**
 * An expansion as rows of an `aligned` block: `lhs = symbolic terms`, `= the products in numbers`,
 * `+ the remainder and bias` (its own row, so a narrow panel doesn't clip it), `= total`.
 * `term(j)` names one product symbolically; `biasName`, when given, appends the bias.
 */
export function sumRows(lhs, ex, term, biasName = null) {
  const symbolic = ex.shown.map((t) => term(t.j));
  const numeric = ex.shown.map((t) => T`${factor(t.x)}\cdot${factor(t.y)}`);
  const rest = [];
  if (ex.restCount) {
    symbolic.push(T`\underbrace{\cdots}_{${ex.restCount}\text{ more}}`);
    rest.push(T`\underbrace{\cdots}_{=\,${num(ex.rest)}}`);
  }
  if (biasName) {
    symbolic.push(biasName);
    rest.push(factor(ex.bias));
  }
  // At most two products per row (the remainder and bias ride along on a row with one), so a
  // 480px panel never clips a row.
  const lines = [];
  for (let k = 0; k < numeric.length; k += 2) lines.push(numeric.slice(k, k + 2));
  if (rest.length) {
    if (lines.at(-1)?.length === 1) lines.at(-1).push(...rest);
    else lines.push(rest);
  }
  const rows = [T`${lhs} &= ${symbolic.join(' + ') || '0'}`];
  lines.forEach((items, k) => rows.push(k ? T`&\phantom{=}\; + ${items.join(' + ')}` : T`&= ${items.join(' + ')}`));
  rows.push(T`&= ${num(ex.total)}`);
  return T`\begin{aligned} ${rows.join(T` \\ `)} \end{aligned}`;
}

/** Standard normal CDF Φ and density φ (erf from Abramowitz & Stegun 7.1.26, |error| < 1.5e-7). */
export function normalCdf(z) {
  const x = Math.abs(z) / Math.SQRT2, t = 1 / (1 + 0.3275911 * x);
  const erf = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return 0.5 * (1 + (z >= 0 ? erf : -erf));
}
export const normalPdf = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

export function renderTex(target, tex, displayMode = true) {
  const katex = globalThis.katex;
  if (!katex) { target.textContent = tex; return; }
  katex.render(tex, target, { displayMode, throwOnError: false, strict: 'ignore' });
}

/** Inline math as an HTML string, for use inside explanatory text. */
export function texHtml(tex) {
  const katex = globalThis.katex;
  if (!katex) return tex.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
  return katex.renderToString(tex, { throwOnError: false, strict: 'ignore' });
}
