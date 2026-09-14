const LOSS_COLOR = '#ffc861';
const TRAIN_ACC_COLOR = 'rgba(110, 168, 255, 0.45)';
const TEST_ACC_COLOR = '#6ea8ff';

export class TrainingChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.pending = false;
    this.reset();
    new ResizeObserver(() => this.schedule()).observe(canvas);
  }

  reset() {
    this.train = [];
    this.test = [];
    this.schedule();
  }

  addTrain(epoch, loss, acc) {
    this.train.push({ epoch, loss, acc });
    this.schedule();
  }

  addTest(epoch, acc) {
    const last = this.test[this.test.length - 1];
    if (last && Math.abs(last.epoch - epoch) < 1e-9) last.acc = acc;
    else this.test.push({ epoch, acc });
    this.schedule();
  }

  schedule() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; this.draw(); });
  }

  draw() {
    const { canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const m = { l: 34, r: 36, t: 8, b: 20 };
    const pw = w - m.l - m.r, ph = h - m.t - m.b;
    const losses = this.train.map((p) => p.loss).filter((v) => v != null);
    const maxEpoch = Math.max(1, ...this.train.map((p) => p.epoch), ...this.test.map((p) => p.epoch));
    const maxLoss = losses.length ? Math.max(...losses) * 1.1 : 2.5;
    const x = (epoch) => m.l + (epoch / maxEpoch) * pw;
    const yLoss = (loss) => m.t + ph - (loss / maxLoss) * ph;
    const yAcc = (acc) => m.t + ph - acc * ph;

    ctx.font = '10px system-ui, sans-serif';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = m.t + (ph * i) / 4;
      ctx.strokeStyle = '#1c2432';
      ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + pw, y); ctx.stroke();
      ctx.fillStyle = '#6f7b91';
      ctx.textAlign = 'right';
      ctx.fillText((maxLoss * (1 - i / 4)).toFixed(2), m.l - 5, y + 3);
      ctx.textAlign = 'left';
      ctx.fillText(`${100 - i * 25}%`, m.l + pw + 5, y + 3);
    }
    ctx.textAlign = 'center';
    const tickStep = maxEpoch <= 10 ? 1 : Math.ceil(maxEpoch / 10);
    for (let e = 0; e <= maxEpoch + 1e-9; e += tickStep) ctx.fillText(String(e), x(e), h - 6);

    if (!this.train.length && !this.test.length) {
      ctx.fillStyle = '#8a96ab';
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('Press Train to start', m.l + pw / 2, m.t + ph / 2);
      return;
    }

    const line = (points, color, width, dashed) => {
      if (points.length < 1) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dashed ? [3, 3] : []);
      ctx.beginPath();
      points.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      ctx.stroke();
      ctx.setLineDash([]);
    };
    line(this.train.filter((p) => p.loss != null).map((p) => [x(p.epoch), yLoss(p.loss)]), LOSS_COLOR, 1.5);
    line(this.train.map((p) => [x(p.epoch), yAcc(p.acc)]), TRAIN_ACC_COLOR, 1.2, true);
    const test = this.test.map((p) => [x(p.epoch), yAcc(p.acc)]);
    line(test, TEST_ACC_COLOR, 2);
    ctx.fillStyle = TEST_ACC_COLOR;
    const lastTest = test[test.length - 1];
    if (lastTest) { ctx.beginPath(); ctx.arc(lastTest[0], lastTest[1], 3, 0, Math.PI * 2); ctx.fill(); }

    ctx.textAlign = 'left';
    const legend = [['train loss', LOSS_COLOR], ['train acc', TRAIN_ACC_COLOR], ['test acc', TEST_ACC_COLOR]];
    let lx = m.l + 6;
    for (const [text, color] of legend) {
      ctx.fillStyle = color;
      ctx.fillRect(lx, m.t + 5, 10, 2);
      ctx.fillStyle = '#8a96ab';
      ctx.fillText(text, lx + 14, m.t + 9);
      lx += ctx.measureText(text).width + 28;
    }
  }
}
