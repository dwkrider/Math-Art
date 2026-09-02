# Math Art — companion site

An interactive companion to the Math Art extension. Each module explores
one family of mathematical objects, drawn **live in the browser** from
exact geometry: nothing on these pages is a pre-rendered picture of the
thing being discussed, and no computation happens anywhere but the
reader's own machine.

One module is live — **Polyhedra**, covering all 448 solids in
[`data/polyhedra/`](../data/polyhedra/). Patterns, Knots and Fractals are
listed on the landing page as planned.

**Surfaces** is a viewer rather than a catalogue so far:
[`modules/surfaces.html`](modules/surfaces.html) demonstrates
[`js/surface-viewer.js`](js/surface-viewer.js), which is written to be
used on its own (see below).

## Running it

```sh
python web/serve.py          # http://localhost:8000/
```

There is **no build step**. The site is plain ES modules with an import
map and a vendored copy of three.js, so editing a file and reloading the
page is the entire development loop. A server is needed only because
module imports and `fetch()` do not work from `file://` URLs.

## Layout

```
web/
  index.html            landing page -- the module cards
  modules/
    polyhedra.html      the Polyhedra module
    surfaces.html       the surface viewer, demonstrated
  js/
    data.js             database access, caching and the facet queries
    tessellate.js       face cycles -> triangles (see below)
    geometry.js         record -> three.js geometry, colouring, styles
    viewer.js           the scene: camera, lights, orbit, styles
    catalog.js          faceted browsing over the whole database
    detail.js           the mathematics panel
    polyhedra.js        page entry point and hash routing
    surface-viewer.js   standalone mesh viewer -- no dependencies at all
  css/site.css
  vendor/three/         three.js r169 (MIT), vendored -- no CDN at runtime
  thumbs/               448 catalogue tiles, one per solid (Git LFS)
  serve.py              local dev server
  data/                 NOT in git; see "The database" below
```

## The database

The site reads the repository's real polyhedron database rather than a
curated export. `index.json` is 256 KB and carries enough to drive the
whole catalogue — name, families, V/E/F, symmetry — so filtering all 448
solids costs a single request; the per-solid records average 22 KB and
are fetched only when a solid is actually opened. That is why there is no
data pipeline here, and it is what stops the site drifting from the
database.

`web/js/data.js` asks for `../data/polyhedra/…` relative to itself, which
resolves to `web/data/polyhedra/`. That directory is **not** committed:

* locally, `serve.py` routes that URL straight to `data/polyhedra/`;
* in CI, `.github/workflows/pages.yml` copies the directory into place.

Either way a page requests one URL and gets one answer, in development
and in production alike, and the database is never duplicated in git.

Paths are resolved relative to the module's own URL, never from the site
root, because GitHub Pages serves the project from `/Math-Art/` — an
absolute `/data/…` would work locally and 404 in production.

## Tessellation

`js/tessellate.js` is a port of
[`tools/polyhedra_tessellate.py`](../tools/polyhedra_tessellate.py), and
has to stay a faithful one: the thumbnails are rendered by the Python
side and the live view is drawn by the JS, so if the two disagree a
catalogue tile stops matching the model it opens. They are cross-checked
over all 448 solids and agree to the triangle on counts and to 1e-9 on
surface area.

The problem it solves is that records store each face as its **true
winding cycle**, which is what lets a `{5/2}` pentagram be a single face.
Every simple way of triangulating that is wrong somewhere in this
database: a fan is wrong for the ~7,600 concave faces, and a fill keyed
off the polygon's *turning number* drops the 568 crossed quadrilaterals
of the uniform duals, whose turning number is 0 though both lobes are
plainly there. The stored metadata cannot help either — `faces_by_type`
labels a pentagram `{5}`, so self-intersection has to be found
geometrically.

## Thumbnails

The 448 catalogue tiles are rendered by
[`tools/render_polyhedra_thumbs.py`](../tools/render_polyhedra_thumbs.py):

```sh
blender --background --factory-startup --python tools/render_polyhedra_thumbs.py
```

It renders only what is missing; pass `-- --all` to redo everything, or
`-- <slug> <slug>` for particular solids. Each mesh is built from the
record's own geometry rather than by driving the add-on's operators,
which is what lets it cover the database exactly — about a third of these
solids are reached through a boolean, an integer or a repeat count rather
than an enum item, so no operator-driven pass could enumerate them. The
studio rig is imported from `docs/render_docs.py`, so the tiles match the
documentation figures and follow any change to the lighting.

## Gate

```sh
python tests/test_web.py
```

Checks that every solid has a thumbnail and every thumbnail a solid, that
local links and relative imports resolve, that the import map declares
the bare specifiers the modules use, and that nothing fetches from a
third-party host at runtime. With no bundler there is no other safety
net: a renamed file fails silently, and only for the reader who opens
that page.


## `surface-viewer.js`

A triangle-mesh viewer with **no dependencies** — not even three.js. It
is raw WebGL 2 in one file, so it runs from a plain
`<script type="module">` and also inlines cleanly into a single-page
document; the minimal-surface validation report embeds it that way, with
sixty-odd surfaces on one page and no network access at all.

It is written for **surfaces** rather than solids, which is what makes it
a separate thing from `viewer.js` next door:

- **Two-sided shading.** An open sheet has no inside to cull to, so the
  two faces are given different colours. On a minimal surface that is the
  only way to read which way the sheet folds through itself.
- **No normals.** Flat shading is recovered per-fragment from screen-space
  derivatives, so a mesh needs nothing but positions and indices — which
  halves what an embedded payload has to carry.
- **Linked viewers.** `linkViewers([a, b])` makes several viewers share
  one orientation: drag either and both turn. Two reconstructions of the
  same surface can then be compared without a difference in viewpoint
  hiding a difference in shape.
- **A context pool.** Browsers keep only around sixteen live WebGL
  contexts and silently drop the oldest, so `ViewerPool` builds viewers
  as cards scroll into view and retires them afterwards.

```js
import { SurfaceViewer, linkViewers } from './js/surface-viewer.js';

const a = new SurfaceViewer(canvasA);
const b = new SurfaceViewer(canvasB, { front: [0.55, 0.63, 0.70] });
a.setMesh({ verts, faces });        // polygonal faces are triangulated
b.setMesh({ positions, indices });  // or typed arrays
linkViewers([a, b]);
```

`setMesh` also accepts a packed payload — positions quantised to uint16
over the mesh's own bounding box — for pages that embed geometry as
base64; `decodeMesh` unpacks it. Meshes are centred and scaled to a unit
ball on load, so a single cell and a block of eight are comparable
without either being a speck.
