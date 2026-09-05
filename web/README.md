# Math Art — companion site

An interactive companion to the Math Art extension. Each module explores
one family of mathematical objects, drawn **live in the browser** from
exact geometry: nothing on these pages is a pre-rendered picture of the
thing being discussed, and no computation happens anywhere but the
reader's own machine.

Two modules are live:

* **Polyhedra** — all 471 solids in [`data/polyhedra/`](../data/polyhedra/),
  drawn live in the browser from each record's own vertices and faces.
* **Surfaces** — all 462 records in [`data/surfaces/`](../data/surfaces/),
  drawn from meshes baked out of the extension's own generators.

The difference between those two sentences is the main thing to
understand about this site, and it is explained under
"Geometry: live for solids, baked for surfaces" below.

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
    surfaces.html       the Surfaces module
  js/
    data.js             database access, caching and the facet queries
    tessellate.js       face cycles -> triangles (see below)
    geometry.js         record -> three.js geometry, colouring, styles
    viewer.js           the scene: camera, lights, orbit, styles
    catalog.js          faceted browsing over the whole database
    detail.js           the mathematics panel
    polyhedra.js        page entry point and hash routing
    surface-data.js     the same, for the surface database
    surface-detail.js   the mathematics panel for a surface
    surfaces.js         Surfaces page entry point
    surface-viewer.js   standalone mesh viewer -- no dependencies at all
  css/site.css
  vendor/three/         three.js r169 (MIT), vendored -- no CDN at runtime
  thumbs/
    polyhedra/          471 catalogue tiles, one per solid (Git LFS)
    surfaces/           one per implemented surface (Git LFS)
  surfaces/             baked surface meshes, <slug>.json (Git LFS)
  serve.py              local dev server
  data/                 NOT in git; see "The databases" below
```

## The databases

The site reads the repository's real databases rather than a
curated export. `index.json` is 256 KB and carries enough to drive the
whole catalogue — name, families, V/E/F, symmetry — so filtering all 471
solids costs a single request; the per-solid records average 22 KB and
are fetched only when a solid is actually opened. That is why there is no
data pipeline here, and it is what stops the site drifting from the
database.

`web/js/data.js` asks for `../data/polyhedra/…` relative to itself (and
`surface-data.js` for `../data/surfaces/…`), resolving under
`web/data/`. That directory is **not** committed:

* locally, `serve.py` routes those URLs straight to `data/<name>/`;
* in CI, `.github/workflows/pages.yml` copies both directories into place.

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
over all 471 solids and agree to the triangle on counts and to 1e-9 on
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

## Geometry: live for solids, baked for surfaces

The two modules get their geometry in opposite ways, and the reason is a
property of the data rather than a preference.

A polyhedron record **is** its geometry: an exact vertex table and the
face cycles. So the Polyhedra module evaluates nothing and bakes nothing
— it reads the record and builds the mesh in the browser.

A surface record deliberately carries **no mesh**. `data/surfaces/`
argues, rightly, that a mesh is a rendering at a chosen resolution and
not part of a surface's identity. But the site has to draw something, and
for much of that database the record is not sufficient to draw from:
parametric and implicit/nodal records do carry evaluable formulas, but
the 163 Weierstrass records mostly store no (g, dh) pair at all — their
own notes say the shipped mesher is authoritative and that an unverified
transcription "would silently define a different surface".

So surface geometry is baked, from the one source authoritative for all
of them: the add-on.

```sh
blender --background --python tools/surfdb_export.py
```

**Never pass `--factory-startup` to that one.** It disables extensions,
so every operator appears unregistered and the whole pass fails. (The
polyhedron renderer below is the opposite case and does want it, because
it builds meshes from data and never touches an operator.)

It writes `web/surfaces/<slug>.json` — positions quantised to uint16 over
the mesh bounding box, indices 16- or 32-bit, and no normals at all,
since the viewer recovers flat shading from screen-space derivatives —
and `web/thumbs/surfaces/<slug>.png` beside it. Two phases, which have to
stay apart: driving an operator clears the scene, camera and lights
included, so the studio is built afterwards and each thumbnail is
rendered **from the exported mesh**. A tile is therefore a picture of the
exact file the viewer loads.

61 records are implemented by no generator. Those get no mesh and no
tile, the module says so, and the gate does not ask for them.

## Thumbnails

The 471 catalogue tiles are rendered by
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

Checks that every solid has a thumbnail and every thumbnail a solid; that
every **implemented** surface has a baked mesh and a tile, and that no
artifact names a slug the database has forgotten; that local links and
relative imports resolve; that the import map declares the bare
specifiers the modules use; and that nothing fetches from a third-party
host at runtime. With no bundler there is no other safety net: a renamed
file fails silently, and only for the reader who opens that page.


## `surface-viewer.js`

A triangle-mesh viewer with **no dependencies** — not even three.js. It
is raw WebGL 2 in one file, so it runs from a plain
`<script type="module">` and also inlines cleanly into a single-page
document; the minimal-surface validation report embeds it that way, with
sixty-odd surfaces on one page and no network access at all.

It is written for **surfaces** rather than solids, which is what makes it
a separate thing from `viewer.js` next door:

- **The documentation studio's light.** The shader carries
  `docs/render_docs.py`'s four-lamp rig — each lamp's Blender vector
  resolved onto the studio camera basis, at the studio's own relative
  energies — and its White Plastic base colour. `STUDIO_VIEW` is the
  orientation that studio shoots from, so a viewer opened with
  `{ rotation: STUDIO_VIEW }` shows a surface in the same pose as its
  thumbnail and its documentation figure. `home` follows whatever
  orientation the page opened at.
- **Two-sided shading.** An open sheet has no inside to cull to, so the
  two faces are given slightly different colours — the same plastic,
  cooled and stepped down. On a minimal surface that is the only way to
  read which way the sheet folds through itself, and it is the one place
  the viewer departs from the studio, which never needed it because a
  solid is closed.
- **No normals.** Flat shading is recovered per-fragment from screen-space
  derivatives, so a mesh needs nothing but positions and indices — which
  halves what an embedded payload has to carry.
- **Linked viewers.** `linkViewers([a, b])` makes several viewers share
  one orientation: drag either and both turn. Two reconstructions of the
  same surface can then be compared without a difference in viewpoint
  hiding a difference in shape. This is a capability, not a default: the
  Surfaces module shows **one** surface at a time, because browsing is a
  different task from comparing and a split stage halves the space each
  surface gets.
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
