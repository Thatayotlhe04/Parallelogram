// The field: every record of the dataset as one cell of a skewed lattice.
// Drawing only. It knows states and positions, never the content of a record.

export const CLEAN = 0, WARN = 1, ERROR = 2, GONE = 3;
const SKEW = 5 / 22;                 // the lean of the mark: 5 across for 22 down
const MIN_PITCH = 3, MAX_PITCH = 26;
const ACCENT = "#fb923c", OK = "#4ade80";

const ease = (t) => 1 - Math.pow(1 - t, 3);
const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export class Field {
  constructor(canvas, { onHover, onSelect }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onHover = onHover;
    this.onSelect = onSelect;
    this.n = 0;
    this.arrange = "file";
    this.match = null;
    this.selected = -1;
    this.hover = -1;
    this.links = new Map();
    this.budget = 4096;
    this.order = new Int32Array(0);     // slot -> record index
    this.slotOf = new Int32Array(0);    // record index -> slot
    this.from = null;                   // positions the animation starts from
    this.t0 = 0;
    this.frame = 0;

    new ResizeObserver(() => this.layout(false)).observe(canvas);
    canvas.addEventListener("pointermove", (e) => this.point(e));
    canvas.addEventListener("pointerleave", () => this.setHover(-1));
    canvas.addEventListener("click", (e) => {
      const i = this.hit(e);
      if (i >= 0) this.onSelect(i);
    });
    canvas.addEventListener("keydown", (e) => this.key(e));
  }

  setData({ state, mask, tokens, gone, touched }, links, budget) {
    const grew = state.length !== this.n;
    this.n = state.length;
    this.state = state; this.mask = mask; this.tokens = tokens; this.gone = gone; this.touched = touched;
    this.links = links;
    this.budget = budget;
    if (grew) { this.from = null; this.selected = -1; this.hover = -1; this.match = null; }
    this.sort();
    this.layout(!grew);
  }

  setArrange(mode) { this.arrange = mode; this.sort(); this.layout(true); }
  setMatch(match) { this.match = match; this.draw(); }
  setSelected(index) { this.selected = index; this.draw(); }

  setHover(index, event) {
    if (index === this.hover && !event) return;
    this.hover = index;
    this.onHover(index, event);
    this.draw();
  }

  rank(i) {
    // findings first: what blocks, then warnings, then repaired, clean, removed.
    const s = this.state[i];
    if (s === ERROR) return 0;
    if (s === WARN) return 1;
    if (s === GONE) return 4;
    return this.touched[i] ? 2 : 3;
  }

  sort() {
    const idx = Array.from({ length: this.n }, (_, i) => i);
    if (this.arrange === "findings") {
      const low = (m) => (m ? 31 - Math.clz32(m & -m) : 99);
      idx.sort((a, b) => this.rank(a) - this.rank(b) || low(this.mask[a]) - low(this.mask[b]) || a - b);
    } else if (this.arrange === "tokens") {
      idx.sort((a, b) => this.tokens[b] - this.tokens[a] || a - b);
    }
    this.order = Int32Array.from(idx);
    this.slotOf = new Int32Array(this.n);
    for (let s = 0; s < this.n; s++) this.slotOf[this.order[s]] = s;
  }

  layout(animate) {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.w = Math.max(1, rect.width); this.h = Math.max(1, rect.height);
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.dpr = dpr;
    if (!this.n) { this.draw(); return; }

    const before = this.pos ? { pos: this.pos, per: this.per } : null;
    // Largest cell that fits every record; below the minimum, records share cells.
    let per = 1, pitch, cols, rows, cells;
    for (;;) {
      cells = Math.ceil(this.n / per);
      pitch = Math.min(MAX_PITCH, Math.floor(Math.sqrt(((this.w - SKEW * this.h) * this.h) / cells)));
      for (; pitch >= MIN_PITCH; pitch--) {
        rows = Math.floor(this.h / pitch);
        cols = Math.floor((this.w - SKEW * rows * pitch) / pitch);
        if (cols > 0 && cols * rows >= cells) break;
      }
      if (pitch >= MIN_PITCH) break;
      per = Math.ceil(per * 1.5 + 0.5);
    }
    // Do not leave a wide lattice two rows tall: use the rows it needs.
    rows = Math.ceil(cells / cols);
    Object.assign(this, { per, pitch, cols, rows, cells });
    this.gap = pitch >= 9 ? 2 : pitch >= 5 ? 1 : 0;
    this.height = rows * pitch;
    this.ox = Math.max(0, (this.w - (cols * pitch + SKEW * this.height)) / 2);
    this.oy = Math.max(0, (this.h - this.height) / 2);

    this.pos = new Float32Array(this.n * 2);
    for (let i = 0; i < this.n; i++) {
      const cell = Math.floor(this.slotOf[i] / per);
      this.pos[i * 2] = (cell % cols) * pitch;
      this.pos[i * 2 + 1] = Math.floor(cell / cols) * pitch;
    }
    const canAnimate = animate && before && before.per === 1 && per === 1 &&
      before.pos.length === this.pos.length && this.n <= 12000 && !reduced();
    if (canAnimate) { this.from = before.pos; this.t0 = performance.now(); }
    else this.from = null;
    this.draw();
  }

  color(i) {
    const s = this.state[i];
    if (this.arrange === "tokens" && s !== GONE) {
      const t = this.tokens[i] / this.budget;
      if (t > 1) return ACCENT;
      const v = Math.round(30 + Math.sqrt(Math.min(1, t)) * 190);
      return `rgb(${v},${v},${v})`;
    }
    if (s === ERROR) return ACCENT;
    if (s === WARN) return "#c4c4c4";
    if (s === GONE) return null;
    return "#232323";
  }

  draw() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => { this.frame = 0; this.paint(); });
  }

  paint() {
    const { ctx, dpr } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.n) return;
    // x' = x - SKEW * y, shifted so the top row leans right, as the mark does.
    ctx.setTransform(dpr, 0, -SKEW * dpr, dpr, (this.ox + SKEW * this.height) * dpr, this.oy * dpr);

    let k = 1;
    if (this.from) {
      k = Math.min(1, (performance.now() - this.t0) / 460);
      if (k >= 1) this.from = null;
    }
    const e = ease(k), size = this.pitch - this.gap, match = this.match;
    const at = (i) => {
      const x = this.pos[i * 2], y = this.pos[i * 2 + 1];
      if (!this.from) return [x, y];
      return [this.from[i * 2] + (x - this.from[i * 2]) * e, this.from[i * 2 + 1] + (y - this.from[i * 2 + 1]) * e];
    };

    if (this.per === 1) {
      const band = Math.max(2, Math.round(size / 4));
      for (let i = 0; i < this.n; i++) {
        const [x, y] = at(i);
        ctx.globalAlpha = match && !match[i] ? 0.16 : 1;
        const fill = this.color(i);
        if (fill) {
          ctx.fillStyle = fill;
          ctx.fillRect(x, y, size, size);
          if (this.touched[i] && this.state[i] === CLEAN && this.arrange !== "tokens" && size >= 5) {
            ctx.fillStyle = OK;
            ctx.fillRect(x, y + size - band, size, band);
          }
        } else if (size >= 5) {
          ctx.strokeStyle = "#3a3a3a"; ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
        } else {
          ctx.fillStyle = "#121212";
          ctx.fillRect(x, y, size, size);
        }
      }
    } else {
      // Shared cells show the worst record inside them.
      for (let c = 0; c < this.cells; c++) {
        let worst = -1, rank = 9, lit = !match;
        for (let s = c * this.per; s < Math.min(this.n, (c + 1) * this.per); s++) {
          const i = this.order[s], r = this.rank(i);
          if (r < rank) { rank = r; worst = i; }
          if (match && match[i]) lit = true;
        }
        ctx.globalAlpha = lit ? 1 : 0.16;
        ctx.fillStyle = this.color(worst) || "#121212";
        ctx.fillRect((c % this.cols) * this.pitch, Math.floor(c / this.cols) * this.pitch, size, size);
      }
    }
    ctx.globalAlpha = 1;

    const ring = (i, color, width) => {
      if (i < 0 || i >= this.n) return;
      const [x, y] = at(i);
      ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.strokeRect(x - 1.5, y - 1.5, size + 3, size + 3);
    };
    if (this.selected >= 0 && this.links.has(this.selected)) {
      // A duplicate points at the record it repeats.
      const j = this.links.get(this.selected);
      const [x1, y1] = at(this.selected), [x2, y2] = at(j);
      ctx.strokeStyle = "rgba(255,255,255,.75)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x1 + size / 2, y1 + size / 2); ctx.lineTo(x2 + size / 2, y2 + size / 2); ctx.stroke();
      ctx.setLineDash([]);
      ring(j, "rgba(255,255,255,.75)", 1);
    }
    ring(this.hover, "rgba(255,255,255,.6)", 1);
    ring(this.selected, "#fff", 2);
    if (this.from) this.draw();
  }

  // Pointer position -> record index (the worst record of a shared cell).
  hit(event) {
    if (!this.n) return -1;
    const rect = this.canvas.getBoundingClientRect();
    const y = event.clientY - rect.top - this.oy;
    const x = event.clientX - rect.left - this.ox - SKEW * (this.height - y);
    if (x < 0 || y < 0 || y >= this.height) return -1;
    const col = Math.floor(x / this.pitch), row = Math.floor(y / this.pitch);
    if (col >= this.cols) return -1;
    return this.pick(row * this.cols + col);
  }

  pick(cell) {
    if (cell < 0 || cell >= this.cells) return -1;
    let worst = -1, rank = 9;
    for (let s = cell * this.per; s < Math.min(this.n, (cell + 1) * this.per); s++) {
      const i = this.order[s], r = this.rank(i);
      if (r < rank) { rank = r; worst = i; }
    }
    return worst;
  }

  point(event) { this.setHover(this.hit(event), event); }

  key(event) {
    const step = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -this.cols, ArrowDown: this.cols }[event.key];
    if (!step || !this.n) return;
    event.preventDefault();
    const cell = this.selected < 0 ? 0 : Math.floor(this.slotOf[this.selected] / this.per) + step;
    const i = this.pick(Math.max(0, Math.min(this.cells - 1, cell)));
    if (i >= 0) this.onSelect(i);
  }

  // Where a record's cell sits on the page, for anchoring the hover card.
  place(i) {
    const rect = this.canvas.getBoundingClientRect();
    const x = this.pos[i * 2], y = this.pos[i * 2 + 1];
    return { x: rect.left + this.ox + x + SKEW * (this.height - y), y: rect.top + this.oy + y, size: this.pitch };
  }
}
