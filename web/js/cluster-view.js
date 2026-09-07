// The clustered view: the catalogue laid out by similarity.
//
// Nodes are the thumbnails themselves rather than coloured dots. The
// tiles already exist, they are what a reader recognises a surface by,
// and a field of them shows the shape of the collection in a way a
// scatter of dots cannot -- the minimal surfaces look like each other,
// and that is the point being made.
//
// TWO THINGS THAT WOULD GO WRONG NAIVELY.
//
// Decoding 473 PNGs at their stored 320x320 costs about 190 MB of RGBA
// and stalls the tab. Each is decoded once into a small ImageBitmap and
// the full-size image is dropped, which is ~8 MB at 64px and fast to
// draw. Loading is progressive: the layout runs immediately and tiles
// appear as they arrive, so nothing waits on the network.
//
// Drawing 473 images every frame while the layout settles is also
// wasteful. During settling the nodes are drawn as plain squares, and
// the thumbnails go in once it has come to rest.

import { knnGraph, SpringLayout } from './cluster.js';

const TILE_PX = 64;          // decode size; drawn smaller than this
const SETTLE_ITERS = 600;    // measured: separation stops improving after

export class ClusterView {
  constructor(canvas, entries, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.all = entries;
    this.thumbUrl = opts.thumbUrl;
    this.onSelect = opts.onSelect || (() => {});
    this.hasMesh = opts.hasMesh || (() => true);
    this.bitmaps = new Map();
    this.pending = new Set();
    this.selected = null;
    this.hover = -1;
    this.view = { x: 0, y: 0, k: 1 };
    this.entries = entries;
    this._bind();
  }

  /** Lay out the given subset (the catalogue's current filter). */
  show(entries) {
    this.entries = entries;
    this.selectedIdx = -1;
    const n = entries.length;
    this.edges = n > 1 ? knnGraph(entries) : [];
    const r = this.canvas.getBoundingClientRect();
    this.layout = new SpringLayout(n, this.edges,
                                   { width: r.width || 900, height: r.height || 600 });
    this.iters = 0;
    this.spaced = null;
    this.view = { x: 0, y: 0, k: 1 };
    this._loadTiles();
    this._run();
    return this;
  }

  /**
   * Node size falls as the field grows, so the total ink stays roughly
   * constant: with everything shown the tiles are stamp-sized and the
   * clusters read as texture; filter down to a dozen and they become
   * large enough to study individually.
   */
  nodeSize() {
    const n = Math.max(1, this.entries.length);
    const r = this.canvas.getBoundingClientRect();
    const area = (r.width || 900) * (r.height || 600);
    // sqrt(area / n) is the side of the square each tile would get if the
    // canvas were divided evenly, which is the largest size that can fit
    // without guaranteed overlap. 0.85 of it leaves the gaps that make
    // the clusters legible as clusters.
    return Math.max(12, Math.min(132, 0.85 * Math.sqrt(area / n)));
  }

  _loadTiles() {
    for (const e of this.entries) {
      if (this.bitmaps.has(e.slug) || this.pending.has(e.slug)) continue;
      if (!this.hasMesh(e.slug)) continue;      // no tile exists for it
      this.pending.add(e.slug);
      const img = new Image();
      img.decoding = 'async';
      img.onload = () => {
        // Downscale on decode; the 320px original is never kept.
        const done = (bm) => {
          this.bitmaps.set(e.slug, bm);
          this.pending.delete(e.slug);
          this.dirty = true;
        };
        if (window.createImageBitmap) {
          createImageBitmap(img, { resizeWidth: TILE_PX, resizeHeight: TILE_PX,
                                   resizeQuality: 'medium' })
            .then(done).catch(() => done(img));
        } else done(img);
      };
      img.onerror = () => this.pending.delete(e.slug);
      img.src = this.thumbUrl(e.slug);
    }
  }

  _run() {
    cancelAnimationFrame(this._raf);
    const frame = () => {
      if (this.iters < SETTLE_ITERS) {
        // A slice per frame, so the field is seen to organise itself
        // rather than appearing already solved after a second's freeze.
        this.layout.step(12);
        this.iters += 12;
        if (this.iters >= SETTLE_ITERS) this._space();
        this.dirty = true;
      }
      if (this.dirty) this._draw();
      this._raf = requestAnimationFrame(frame);
    };
    this._raf = requestAnimationFrame(frame);
  }

  /**
   * Ease the densest clumps apart, radially, without rearranging them.
   *
   * The springs optimise for similarity and know nothing about tile size,
   * so a tight cluster stacks its members and only the last one drawn is
   * visible -- 473 tiles showing as about forty. Three repairs were tried
   * and the first two are recorded so they are not tried again:
   *
   *   Axis-aligned pair separation snapped the field onto a lattice. It
   *   fixed the overlap by discarding the layout, which is the one thing
   *   the view exists to show.
   *
   *   Uniform scaling did nothing at all, and I should have seen why:
   *   fitted() normalises the bounding box to the canvas, so scaling
   *   every position by a constant is invariant under it. Worse, it was
   *   paired with a zoom-to-fit, and that DID have an effect -- tiles
   *   ended up 6 pixels across instead of the 31 the canvas affords.
   *
   * What works is local and radial: overlapping pairs push apart along
   * the line joining them, a fraction of the overlap per round, for a few
   * rounds. That redistributes density -- which fitted() does not undo --
   * while leaving each tile among the same neighbours the springs chose.
   */
  _space(rounds = 60) {
    const r = this.canvas.getBoundingClientRect();
    const p = this.layout.fitted(r.width, r.height, this.nodeSize() * 0.6 + 12);
    const n = this.entries.length;
    const x = Float64Array.from(p.x), y = Float64Array.from(p.y);
    const want = this.nodeSize() * 0.95;
    const damp = 0.5;
    for (let it = 0; it < rounds; it++) {
      let hits = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let ex = x[j] - x[i], ey = y[j] - y[i];
          let d = Math.hypot(ex, ey);
          if (d >= want) continue;
          if (d < 1e-6) {                 // exactly coincident: nudge
            ex = (i % 2 ? 1 : -1) * 1e-3;
            ey = (j % 2 ? 1 : -1) * 1e-3;
            d = Math.hypot(ex, ey);
          }
          const push = ((want - d) / d) * damp * 0.5;
          const ox = ex * push, oy = ey * push;
          x[i] -= ox; y[i] -= oy;
          x[j] += ox; y[j] += oy;
          hits++;
        }
      }
      if (!hits) break;
    }
    // Re-fit: the pushing grew the bounding box past the canvas.
    let lox = Infinity, loy = Infinity, hix = -Infinity, hiy = -Infinity;
    for (let i = 0; i < n; i++) {
      if (x[i] < lox) lox = x[i];
      if (y[i] < loy) loy = y[i];
      if (x[i] > hix) hix = x[i];
      if (y[i] > hiy) hiy = y[i];
    }
    const pad = this.nodeSize() * 0.55 + 6;
    const sc = Math.min((r.width - 2 * pad) / Math.max(1e-6, hix - lox),
                        (r.height - 2 * pad) / Math.max(1e-6, hiy - loy), 1);
    const ox = pad + (r.width - 2 * pad - (hix - lox) * sc) / 2;
    const oy = pad + (r.height - 2 * pad - (hiy - loy) * sc) / 2;
    for (let i = 0; i < n; i++) {
      x[i] = ox + (x[i] - lox) * sc;
      y[i] = oy + (y[i] - loy) * sc;
    }
    this.spaced = { x, y };
    this.view = { x: 0, y: 0, k: 1 };
  }

  _positions() {
    if (this.spaced) return this.spaced;
    const r = this.canvas.getBoundingClientRect();
    return this.layout.fitted(r.width, r.height, this.nodeSize() * 0.6 + 12);
  }

  _draw() {
    this.dirty = false;
    const c = this.ctx;
    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    if (this.canvas.width !== Math.round(r.width * dpr)) {
      this.canvas.width = Math.round(r.width * dpr);
      this.canvas.height = Math.round(r.height * dpr);
    }
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, r.width, r.height);
    if (!this.entries.length) return;

    const p = this._positions();
    this.pos = p;
    const v = this.view;
    c.save();
    c.translate(v.x, v.y);
    c.scale(v.k, v.k);

    const s = this.nodeSize();
    const settling = this.iters < SETTLE_ITERS;

    // Tiles are drawn throughout, including while the layout is still
    // moving. Squares were cheaper but told the reader nothing, and the
    // settling is the part worth watching -- surfaces visibly finding
    // their relatives. 473 drawImage calls of a 64px bitmap cost far less
    // than the spring iteration happening in the same frame.
    for (let i = 0; i < this.entries.length; i++) {
      const e = this.entries[i];
      const bm = this.bitmaps.get(e.slug);
      const x = p.x[i] - s / 2, y = p.y[i] - s / 2;
      if (bm) {
        c.drawImage(bm, x, y, s, s);
      } else {
        c.fillStyle = '#141820';
        c.fillRect(x, y, s, s);
      }
      if (e.slug === this.selected || i === this.hover) {
        c.strokeStyle = e.slug === this.selected ? '#6fb3f2' : '#9aa3af';
        c.lineWidth = 2 / v.k;
        c.strokeRect(x, y, s, s);
      }
    }
    c.restore();

    if (settling) this._progress(c, r);
    if (this.hover >= 0) this._label(c, r, p, s);
  }

  _progress(c, r) {
    c.fillStyle = '#6b7480';
    c.font = '12px system-ui, sans-serif';
    c.fillText(`arranging ${this.entries.length} surfaces by similarity…`,
               12, r.height - 12);
  }

  _label(c, r, p, s) {
    const e = this.entries[this.hover];
    if (!e) return;
    const v = this.view;
    const x = p.x[this.hover] * v.k + v.x;
    const y = p.y[this.hover] * v.k + v.y - (s * v.k) / 2 - 8;
    c.font = '12px system-ui, sans-serif';
    const w = c.measureText(e.name).width + 12;
    c.fillStyle = 'rgba(8,10,12,0.92)';
    c.fillRect(Math.min(Math.max(x - w / 2, 2), r.width - w - 2), y - 16, w, 20);
    c.fillStyle = '#e8eaee';
    c.textAlign = 'center';
    c.fillText(e.name, Math.min(Math.max(x, w / 2 + 2), r.width - w / 2 - 2), y - 2);
    c.textAlign = 'left';
  }

  _at(px, py) {
    if (!this.pos) return -1;
    const v = this.view;
    const s = this.nodeSize();
    const x = (px - v.x) / v.k, y = (py - v.y) / v.k;
    // Last drawn wins, so hit-test back to front.
    for (let i = this.entries.length - 1; i >= 0; i--) {
      if (Math.abs(x - this.pos.x[i]) <= s / 2 &&
          Math.abs(y - this.pos.y[i]) <= s / 2) return i;
    }
    return -1;
  }

  _bind() {
    const cv = this.canvas;
    let drag = null;
    cv.addEventListener('mousedown', (ev) => {
      drag = { x: ev.offsetX, y: ev.offsetY, vx: this.view.x, vy: this.view.y,
               moved: false };
    });
    addEventListener('mouseup', () => { drag = null; });
    cv.addEventListener('mousemove', (ev) => {
      if (drag) {
        const dx = ev.offsetX - drag.x, dy = ev.offsetY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        this.view.x = drag.vx + dx;
        this.view.y = drag.vy + dy;
        this.dirty = true;
        return;
      }
      const h = this._at(ev.offsetX, ev.offsetY);
      if (h !== this.hover) { this.hover = h; this.dirty = true; }
      cv.style.cursor = h >= 0 ? 'pointer' : 'grab';
    });
    cv.addEventListener('click', (ev) => {
      if (drag && drag.moved) return;             // a pan, not a click
      const i = this._at(ev.offsetX, ev.offsetY);
      if (i >= 0) this.onSelect(this.entries[i].slug);
    });
    cv.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      const f = Math.exp(-ev.deltaY * 0.0016);
      const k = Math.min(6, Math.max(0.4, this.view.k * f));
      // Zoom about the pointer, not the origin.
      this.view.x = ev.offsetX - (ev.offsetX - this.view.x) * (k / this.view.k);
      this.view.y = ev.offsetY - (ev.offsetY - this.view.y) * (k / this.view.k);
      this.view.k = k;
      this.dirty = true;
    }, { passive: false });
  }

  select(slug) {
    this.selected = slug;
    this.dirty = true;
  }

  destroy() {
    cancelAnimationFrame(this._raf);
    for (const bm of this.bitmaps.values()) if (bm.close) bm.close();
    this.bitmaps.clear();
  }
}
