# Math Art — Blender Extension

A Blender **4.2+ / 5.x** extension bundling **150+ generators** for
mathematical sculpture. Each one adds a ready‑to‑render mesh or curve straight
to the viewport — centered on the origin, fit to a 2 m cube, and driven by a
live redo panel so every parameter is explorable. The mathematics is
implemented faithfully (and self‑tested), and every generator credits the
mathematicians and papers behind it.

<p align="center">
  <img src="docs/images/tangle.png" width="250">
  <img src="docs/images/twisted_polyhedron.png" width="250">
  <img src="docs/images/polytope4d.png" width="250">
</p>

## What's inside

Everything lives under **Add ▸ Mesh ▸ Math Art**, organized into families:

| Family | Count | What it covers |
| --- | --- | --- |
| **Surfaces** | 30 | The deepest collection here — classical and triply‑periodic **minimal surfaces** (Costa, gyroid, the Schwarz and Schoen families, Scherk towers), an **encyclopedia** of algebraic, quadric, ruled, swept and revolution surfaces (backed by a 470‑record surface database), plus canal, focal and Calabi‑Yau cross‑sections |
| **Polyhedra** | 29 | Platonic → Archimedean → Catalan solids, Johnson solids, uniform and canonical forms, stellations, compounds, zonohedra, and 4‑dimensional polytopes projected to 3‑space |
| **Patterns** | 22 | Wallpaper and frieze groups, Islamic strapwork, isohedral and k‑uniform tilings, Penrose and aperiodic (hat/spectre) tilings, and soft cells — many now mappable onto curved surfaces |
| **Knots & Curves** | 16 | Torus and tight knots, links and connected sums, Seifert surfaces, space‑filling and parametric curves |
| **Fractals** | 13 | L‑systems, diffusion‑limited aggregation, iterated function systems, and fractal polyhedra |
| **Weaves & Tangles** | 10 | Woven polyhedra and shells, Celtic interlace, polylinks and tangled compounds |
| **Odds & Ends** | 9 | Supershapes, spherical harmonics, and other one‑offs |
| **Plants & Growth** | 8 | Phyllotaxis, leaf venation, and growth‑driven forms |
| **Rollers** | 7 | Sphericons, oloids, and other constant‑width and rolling solids |
| **Styles** | 5 | Post‑process any mesh — ball‑and‑stick, face coloring, fabrication slicing, relief screens |
| **Origami** | 2 | Classical crease patterns with a real folding solver, and a crease‑pattern importer |

## Highlights

- **Minimal‑surface toolkit** — 150+ minimal surfaces, from the textbook
  catenoid and Enneper to the full triply‑periodic zoo (gyroid, Schwarz P/D,
  the Schoen family), each built from its exact Weierstrass data rather than
  contoured.
- **Calabi‑Yau cross‑section** — Hanson's parametrisation of the quintic slice,
  the surface behind nearly every "Calabi‑Yau" picture, with its torus‑knot
  boundary.
- **Symmetric Sculpture** — a Blender adaptation of George W. Hart's
  sculpture‑design program (behind pieces like *Frabjous* and *Whimsy*): draw one
  flat motif and it replicates live into a whole symmetric family of planes.
- **Origami** — nine classical crease patterns (Miura, waterbomb, Kresling,
  Ron Resch…) folded by a rigid‑panel *or* bending‑paper solver, with the whole
  fold path cached as shape keys so it animates.
- **Scherk–Collins sculptures** — the toroidal warped‑saddle towers of Brent
  Collins and Carlo Séquin.
- **Polyhedra, end to end** — every Platonic, Archimedean and Catalan solid,
  all 92 Johnson solids, the 59 stellations of the icosahedron, and regular
  compounds, on one consistent footing.
- **Knots & Seifert surfaces** — torus and tight knots with the spanning
  surfaces of their links.
- **Curved‑surface tilings** — wallpaper, Islamic and isohedral patterns wrapped
  conformally onto spheres and tori.

## Companion website

Explore the mathematics in your browser at
**[dwkrider.github.io/Math-Art](https://dwkrider.github.io/Math-Art/)** — an
interactive companion that computes **polyhedra** and **surfaces** live from
exact geometry (nothing pre‑rendered), backed by the project's 470‑record surface
database. You can spin each object, read its defining formulae, and export a
printable STL straight from the surface viewer. (More families will follow.)

## Gallery & Documentation

Browse the **[generator gallery →](docs/README.md)** — a render of every shape,
each linking to a page with a usage walkthrough, the underlying mathematics, its
configuration options, and full references.

## Install

1. Download the latest **`math_art-*.zip`** from the
   **[Releases page](https://github.com/dwkrider/Math-Art/releases/latest)**
   (or build it, below).
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

## Regenerating the docs

Every documentation image is produced by one consistent studio script
(black‑velvet backdrop, soft studio lighting, white‑plastic material on
uncolored shapes while generator colors are preserved). The figure set
is the Add menu itself, so nothing needs listing by hand:

```sh
blender --background --factory-startup --python docs/render_docs.py
```

It renders only what is missing; pass `-- --all` to re-render, slugs to
pick a subset (`-- tangle polytope4d`), or `-- variants` for the
per‑option galleries. `-- --list` shows coverage without rendering.

The rest of the pipeline:

```sh
blender --background --factory-startup --python docs/scaffold_pages.py
python docs/insert_variants.py     # galleries into the pages
python docs/build_index.py         # regenerate docs/README.md
python tests/test_docs.py          # coverage gate
```

## License

GPL‑3.0‑or‑later. Each generator page lists the mathematical sources and
attributions its implementation draws on.
