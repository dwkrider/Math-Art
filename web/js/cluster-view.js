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
const SETTLE_ITERS = 900;    // measured: overlap still resolving until ~900

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
    this.layout = new SpringLayout(n, this.edges, {
      width: r.width || 900,
      height: r.height || 600,
      // Overlap is part of the energy, so the layout solves for an
      // arrangement the tiles fit into rather than being shuffled
      // afterwards to make them fit.
      minDist: this.nodeSize() * 0.95,
    });
    this.iters = 0;
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
    // canvas were divided evenly -- the largest size that could fit, and
    // then only in a perfect grid. A clustered field is deliberately
    // uneven, so it needs slack: at 0.85 the tiles want 72% of the canvas
    // and no arrangement with visible clusters fits (441 of 473 tiles
    // overlapped). 0.60 asks for 36% and leaves 2 overlapping, while
    // keeping tiles big enough to recognise.
    return Math.max(12, Math.min(132, 0.60 * Math.sqrt(area / n)));
  }

  /**
   * One sprite sheet instead of 466 requests.
   *
   * Fetched individually the tiles are 466 requests and ~34 MB, and the
   * field visibly fills in over several seconds. tools/build_thumb_atlas.py
   * packs them at the size this view draws them, which is a couple of
   * megabytes in a single request -- the whole field appears at once.
   *
   * If the atlas is missing the view still works: it falls back to the
   * individual PNGs, so a checkout that has not run the packer yet is
   * slower rather than broken.
   */
  _loadTiles() {
    if (this._atlasTried) { this._loadLoose(); return; }
    this._atlasTried = true;
    fetch(new URL('../thumbs/surfaces-atlas.json', import.meta.url))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('no atlas'))))
      .then((meta) => new Promise((res, rej) => {
        const img = new Image();
        img.onload = () => res({ meta, img });
        img.onerror = rej;
        img.src = new URL('../thumbs/surfaces-atlas.png', import.meta.url).href;
      }))
      .then(({ meta, img }) => {
        this.atlas = { img, ...meta };
        this.dirty = true;
      })
      .catch(() => this._loadLoose());
  }

  /** The old path: one image per tile. Used only without an atlas. */
  _loadLoose() {
    for (const e of this.entries) {
      if (this.bitmaps.has(e.slug) || this.pending.has(e.slug)) continue;
      if (!this.hasMesh(e.slug)) continue;
      this.pending.add(e.slug);
      const img = new Image();
      img.decoding = 'async';
      img.onload = () => {
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
        this.dirty = true;
      }
      if (this.dirty) this._draw();
      this._raf = requestAnimationFrame(frame);
    };
    this._raf = requestAnimationFrame(frame);
  }

  /**
   * The converged positions, used as they are.
   *
   * The layout already works in canvas coordinates and is bounded to
   * them, so there is nothing to fit and nothing to rescale. Both of
   * those were tried: rescaling is invariant under the fit that followed
   * it, and separating pairs afterwards destroyed the clustering the
   * whole view is for.
   */
  _positions() {
    return { x: this.layout.x, y: this.layout.y };
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
      const x = p.x[i] - s / 2, y = p.y[i] - s / 2;
      const at = this.atlas && this.atlas.tiles[e.slug];
      const bm = this.bitmaps.get(e.slug);
      if (at) {
        c.drawImage(this.atlas.img, at[0], at[1], this.atlas.cell,
                    this.atlas.cell, x, y, s, s);
      } else if (bm) {
        c.drawImage(bm, x, y, s, s);
      }
      // No placeholder rectangle: the tiles are transparent so that they
      // do not clip one another, and a filled square would put the very
      // background back that the alpha channel exists to remove.
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
