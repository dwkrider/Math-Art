# Math Art — Blender Extension

A Blender **4.2+ / 5.x** extension bundling **50+ generators** for
mathematical sculpture: minimal surfaces, polyhedra (Platonic through
Archimedean to four‑dimensional polytopes), knots and links, fractals,
woven and tangled compounds, hyperbolic tilings, and much more. Each
one adds a ready‑to‑render mesh or curve, centered on the origin and
fitting a 2 m cube.

<p align="center">
  <img src="docs/images/tangle.png" width="250">
  <img src="docs/images/twisted_polyhedron.png" width="250">
  <img src="docs/images/polytope4d.png" width="250">
</p>

## Gallery & Documentation

Browse the **[generator gallery →](docs/README.md)** — a render of
every shape, each linking to a page with its configuration options, the
underlying mathematics, and full references.

## Install

1. Grab the latest **`dist/math_art-*.zip`** (or build it, below).
2. In Blender: **Edit ▸ Preferences ▸ Get Extensions**, open the **⌄**
   menu (top‑right) ▸ **Install from Disk…**, and choose the zip — or
   simply drag the zip into the Blender window.
3. Everything appears under **Add ▸ Mesh ▸ Math Art**, plus the
   *Minimal Surfaces* / *Scherk* / *Woven* N‑panel tabs for the live
   tools.

If you previously installed the individual files as legacy add‑ons,
remove those first.

### Build from source

```sh
blender --command extension build --source-dir math_art \
    --output-filepath dist/math_art-<version>.zip
```

Each module in `math_art/` also works on its own as a classic
single‑file add‑on via *Install from Disk*.

## Regenerating the renders

Every documentation image is produced by one consistent studio script
(black‑velvet backdrop, soft studio lighting, white‑plastic material on
uncolored shapes while generator colors are preserved):

```sh
blender --background --factory-startup --python docs/render_docs.py
```

Pass slugs to render a subset, e.g. `-- tangle polytope4d`.

## License

GPL‑3.0‑or‑later. Each generator page lists the mathematical sources and
attributions its implementation draws on.
