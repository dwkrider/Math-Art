# Math Art — companion site

An interactive companion to the Math Art extension. Each module explores
one family of mathematical objects, drawn **live in the browser** from
exact geometry: nothing on these pages is a pre-rendered picture of the
thing being discussed, and no computation happens anywhere but the
reader's own machine.

One module is live — **Polyhedra**, covering all 448 solids in
[`data/polyhedra/`](../data/polyhedra/). Surfaces, Patterns, Knots and
Fractals are listed on the landing page as planned.

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
  js/
    data.js             database access, caching and the facet queries
    tessellate.js       face cycles -> triangles (see below)
    geometry.js         record -> three.js geometry, colouring, styles
    viewer.js           the scene: camera, lights, orbit, styles
    catalog.js          faceted browsing over the whole database
    detail.js           the mathematics panel
    polyhedra.js        page entry point and hash routing
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
