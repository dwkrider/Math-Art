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
    this.relaxed = null;
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
    return Math.max(14, Math.min(132, 760 / Math.sqrt(n)));
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
        if (this.iters >= SETTLE_ITERS) this._relax();
        this.dirty = true;
      }
      if (this.dirty) this._draw();
      this._raf = requestAnimationFrame(frame);
    };
    this._raf = requestAnimationFrame(frame);
  }

  /**
   * Push overlapping tiles apart, once the springs have settled.
   *
   * The spring layout optimises for similarity and knows nothing about
   * how big a tile is, so a tight cluster piles its members on top of one
   * another -- exactly where the reader most wants to see them. This is a
   * few rounds of the standard pairwise separation: any two nodes closer
   * than a tile apart are pushed to just over a tile apart, each moving
   * half the distance.
   *
   * It runs in layout space, on the same neighbours the drawing uses, and
   * deliberately AFTER settling: doing it during would fight the springs
   * and stop the clusters forming in the first place.
   */
  _relax(rounds = 90) {
    const r = this.canvas.getBoundingClientRect();
    const p = this.layout.fitted(r.width, r.height, this.nodeSize() * 0.6 + 12);
    const n = this.entries.length;
    // The separation the tiles need, expressed back in layout units.
    const want = this.nodeSize() * 1.02;
    const x = p.x, y = p.y;
    for (let it = 0; it < rounds; it++) {
      let moved = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let ex = x[j] - x[i], ey = y[j] - y[i];
          // Square tiles: separate on whichever axis is least overlapped,
          // which keeps rows and columns tidy instead of rounding
          // everything into rings.
          const ox = want - Math.abs(ex), oy = want - Math.abs(ey);
          if (ox <= 0 || oy <= 0) continue;
          if (ox < oy) {
            const s = (ex >= 0 ? 1 : -1) * ox / 2;
            x[i] -= s; x[j] += s;
          } else {
            const s = (ey >= 0 ? 1 : -1) * oy / 2;
            y[i] -= s; y[j] += s;
          }
          moved++;
        }
      }
      if (!moved) break;
    }
    this.relaxed = { x, y };
    this._frame();
  }

  /**
   * Zoom out so the relaxed field fits the canvas.
   *
   * Relaxation pushes tiles apart and therefore outside the box the
   * spring layout was fitted to, so the edges were being clipped. The fix
   * is the view transform, not the positions: rescaling the positions
   * would shrink the gaps the relaxation just created and put the tiles
   * back on top of each other.
   */
  _frame() {
    const r = this.canvas.getBoundingClientRect();
    const p = this.relaxed;
    const s = this.nodeSize();
    let lox = Infinity, loy = Infinity, hix = -Infinity, hiy = -Infinity;
    for (let i = 0; i < this.entries.length; i++) {
      if (p.x[i] < lox) lox = p.x[i];
      if (p.y[i] < loy) loy = p.y[i];
      if (p.x[i] > hix) hix = p.x[i];
      if (p.y[i] > hiy) hiy = p.y[i];
    }
    const w = (hix - lox) + s, h = (hiy - loy) + s;
    const k = Math.min((r.width - 8) / Math.max(1, w),
                       (r.height - 8) / Math.max(1, h), 1);
    this.view.k = k;
    this.view.x = (r.width - w * k) / 2 - (lox - s / 2) * k;
    this.view.y = (r.height - h * k) / 2 - (loy - s / 2) * k;
  }

  _positions() {
    if (this.relaxed) return this.relaxed;
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

    if (settling) {
      // Cheap while it moves: one square per node, no images.
      c.fillStyle = '#7f8c99';
      for (let i = 0; i < this.entries.length; i++) {
        c.fillRect(p.x[i] - s / 4, p.y[i] - s / 4, s / 2, s / 2);
      }
      c.restore();
      this._progress(c, r);
      return;
    }

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
