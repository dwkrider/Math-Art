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

// The sprite sheet, fetched once for the page rather than once per view.
// Held at module scope because there is only ever one sheet, and because
// the fetch has to be startable before any ClusterView exists.
let atlasPromise = null;

/**
 * Start loading the tile sheet.
 *
 * Call this when the page loads. The clustered view is built lazily, on
 * the first switch into it, so without this the sheet is not even
 * requested until the reader asks for the view -- and then the layout
 * runs against tiles that have not arrived, which is a field of nothing
 * arranging itself. Starting at page load means the image is normally
 * there by the time it is wanted.
 *
 * Idempotent, and never rejects: a missing sheet resolves to null and
 * the view falls back to the individual PNGs.
 */
export function preloadAtlas() {
  if (atlasPromise) return atlasPromise;
  atlasPromise = fetch(new URL('../thumbs/surfaces-atlas.json', import.meta.url))
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error('no atlas'))))
    .then((meta) => new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res({ img, ...meta });
      img.onerror = rej;
      img.src = new URL('../thumbs/surfaces-atlas.png', import.meta.url).href;
    }))
    .catch(() => null);
  return atlasPromise;
}

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
    // Converged layouts, keyed by the subset they were solved for. See
    // _cacheKey and show().
    this.cache = new Map();
    this._bind();
  }

  /**
   * What a cached layout is a layout OF.
   *
   * The slugs, in order, because that is precisely the input the layout
   * solves for -- and the canvas size with them, because it is an input
   * too: it sets the tile size, and through it the collision distance
   * and the whole length scale. A layout solved for one canvas is not
   * the layout for another, so a resize has to miss.
   */
  _cacheKey(entries, w, h) {
    return Math.round(w) + 'x' + Math.round(h) + '|' +
           entries.map((e) => e.slug).join(',');
  }

  /**
   * Lay out the given subset (the catalogue's current filter).
   *
   * A subset that has been solved before is restored instead of solved
   * again. Going to another family and back was re-running the whole
   * thing -- for the full catalogue that is a 19,148-edge nearest
   * neighbour graph and 900 iterations over 112k pairs, about 1.1s of
   * work to arrive at the arrangement it had a moment ago. It is also
   * deterministic (the generator is seeded), so the recomputed layout is
   * identical to the one thrown away: the wait bought nothing at all.
   */
  show(entries) {
    this.entries = entries;
    this.selectedIdx = -1;
    const r = this.canvas.getBoundingClientRect();
    this.view = { x: 0, y: 0, k: 1 };
    this.userMoved = false;

    const key = this._cacheKey(entries, r.width || 900, r.height || 600);
    this._key = key;

    // NOTHING STARTS UNTIL THE TILES ARE HERE.
    //
    // The layout used to begin immediately and the sheet arrived
    // whenever it arrived, so the reader watched an empty field organise
    // itself and the thumbnails appeared at the end, already settled --
    // the one part worth watching, missed. Building the neighbour graph
    // is deferred with it: it is ~220ms of synchronous work on the full
    // catalogue, and running it while the image is in flight just takes
    // the main thread away from decoding.
    cancelAnimationFrame(this._raf);
    this._waiting = true;
    this._draw();
    this._loadTiles().then(() => {
      // A later show() may have superseded this one while we waited.
      if (this._key !== key) return;
      this._waiting = false;
      this._begin(entries, key, r);
    });
    return this;
  }

  /** Build or restore the layout, once the tiles are available. */
  _begin(entries, key, r) {
    const n = entries.length;
    const hit = this.cache.get(key);
    if (hit) {
      // Re-inserted so the map's insertion order stays least-recent
      // first, which is what the eviction below relies on.
      this.cache.delete(key);
      this.cache.set(key, hit);
      // Copies: the restored arrays are handed to the view, and a later
      // run must not be able to write through them into the cache.
      this.layout = { x: Float64Array.from(hit.x),
                      y: Float64Array.from(hit.y) };
      this.edges = null;
      this.iters = SETTLE_ITERS;          // already settled; do not step
      this._frame();
      this._run();
      return;
    }

    this.edges = n > 1 ? knnGraph(entries) : [];
    this.layout = new SpringLayout(n, this.edges, {
      width: r.width || 900,
      height: r.height || 600,
      // Overlap is part of the energy, so the layout solves for an
      // arrangement the tiles fit into rather than being shuffled
      // afterwards to make them fit.
      minDist: this.nodeSize() * 0.95,
    });
    this.iters = 0;
    this._run();
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
    // uneven, so it needs slack, and clusters are the dense part: mean
    // tile coverage over the full set runs 69% at a factor of 0.60 and
    // 60% at 0.52. Below that the tiles stop being recognisable faster
    // than the crowding improves.
    return Math.max(12, Math.min(132, 0.52 * Math.sqrt(area / n)));
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
    return preloadAtlas().then((atlas) => {
      if (atlas) {
        this.atlas = atlas;
        this.dirty = true;
      } else {
        // No sheet: the individual PNGs still work, and they arrive
        // progressively, so this path is not gated on them.
        this._loadLoose();
      }
    });
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
        // Re-framed every step: the free field grows as it organises, and
        // following it keeps the whole arrangement in view throughout
        // instead of letting it expand off the edges.
        this._frame();
        if (this.iters >= SETTLE_ITERS) this._store();
        this.dirty = true;
      }
      if (this.dirty) this._draw();
      this._raf = requestAnimationFrame(frame);
    };
    this._raf = requestAnimationFrame(frame);
  }

  /**
   * The converged positions, used exactly as they are.
   *
   * The layout is unbounded (see SpringLayout._one), so these are in its
   * own space and may lie anywhere. Framing is the view's job and is
   * done with the view transform in _frame, never by moving a node:
   * a post-hoc shuffle draws something other than what converged, which
   * is the thing this view had wrong twice before.
   */
  _positions() {
    return { x: this.layout.x, y: this.layout.y };
  }

  /**
   * Keep the converged layout, so returning to this subset is instant.
   *
   * Bounded, because a search box can generate a new subset per
   * keystroke and each one holds two arrays for as long as the page
   * lives. 32 is comfortably more than the 16 families plus whatever
   * searches one sitting produces, and the oldest goes first.
   */
  _store() {
    if (!this._key || this.cache.has(this._key)) return;
    this.cache.set(this._key, { x: Float64Array.from(this.layout.x),
                                y: Float64Array.from(this.layout.y) });
    while (this.cache.size > 32) {
      this.cache.delete(this.cache.keys().next().value);
    }
  }

  /**
   * Point the view at the field: centre it, and scale down if it is
   * larger than the canvas.
   *
   * This is a pure view transform -- translate and uniform scale -- so
   * it cannot change the arrangement, and because the tiles are drawn
   * inside the same transform it cannot change how much they overlap
   * either. Position and tile size scale together, so the picture is
   * the converged one, just framed.
   *
   * It never scales UP. A two-record filter could be magnified to fill
   * the canvas, but the tile size is already chosen (nodeSize) and
   * blowing two thumbnails up to 300px to fill space is not an
   * improvement on two at their proper size with room around them.
   */
  _frame() {
    if (this.userMoved || !this.entries.length) return;
    const p = this.layout;
    let lox = Infinity, loy = Infinity, hix = -Infinity, hiy = -Infinity;
    for (let i = 0; i < this.entries.length; i++) {
      if (p.x[i] < lox) lox = p.x[i];
      if (p.x[i] > hix) hix = p.x[i];
      if (p.y[i] < loy) loy = p.y[i];
      if (p.y[i] > hiy) hiy = p.y[i];
    }
    const r = this.canvas.getBoundingClientRect();
    const w = r.width || 900, h = r.height || 600;
    // Half a tile of margin, so an edge tile is framed rather than cut.
    const m = this.nodeSize() / 2 + 8;
    const k = Math.min(1, (w - 2 * m) / Math.max(1e-6, hix - lox),
                          (h - 2 * m) / Math.max(1e-6, hiy - loy));
    this.view.k = k;
    this.view.x = w / 2 - ((lox + hix) / 2) * k;
    this.view.y = h / 2 - ((loy + hiy) / 2) * k;
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
    if (this._waiting) {
      c.fillStyle = '#6b7480';
      c.font = '12px system-ui, sans-serif';
      c.fillText('loading thumbnails…', 12, r.height - 12);
      return;
    }
    if (!this.entries.length || !this.layout) return;

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
    if (!this.pos || this._waiting) return -1;
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
        this.userMoved = true;
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
      this.userMoved = true;
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
    this.cache.clear();
  }
}
