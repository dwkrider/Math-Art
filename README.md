# Scherk-Collins Sculpture Generator — Blender Add-on

A Blender re-implementation of the geometry engine of Carlo Séquin's
**Sculpture Generator I** (the 1996/97 program that designed the
Scherk-Collins saddle-chain sculptures created with Brent Collins —
*Hyperbolic Hexagon II*, *Heptoroid*, the *Minimal Trefoil*, etc.).

Only the sculpture **geometry** is reproduced. Materials, textures,
backgrounds and rendering are left to Blender.

![Hyperbolic Hexagon](renders/preset_hex.png)

## Install

1. Blender → `Edit > Preferences > Add-ons > Install from Disk…`
2. Pick `src/scherk_collins_generator.py`
3. Enable **"Scherk-Collins Sculpture Generator"**

Tested on Blender 5.1 (needs ≥ 4.2).

## Use

- `Add > Mesh > Scherk-Collins Sculpture` — choose a preset
  (Hyperbolic Hexagon, Minimal Trefoil, Monkey-Saddle Trefoil,
  Heptoroid, straight Scherk Tower) or the plain default.
- Open the **N-panel → "Scherk" tab** to edit parameters. With
  *Auto Update* on, the mesh regenerates live as you drag sliders —
  the original program's interactive workflow.
- **Load / Save** reads and writes the original program's spec-file
  format. All 20 `demo*.txt` files shipped with the original
  (`Sculpture Generator\bin\demo\`) load directly, and saved files can
  be loaded back into `generator.exe`. Display-related fields
  (textures, colors, view matrix) are ignored on load and written with
  neutral defaults on save.
- Export for 3D printing with Blender's built-in STL/3MF exporters.
  Generated solids are watertight, manifold meshes.

## Parameters (same meaning as the original sliders)

| Slider | Range | Effect |
|---|---|---|
| Branches | 1–10 | Order of the saddles (2 = classic Scherk, 3 = monkey saddle…) |
| Storeys | 1–16 | Number of hole/saddle units in the chain |
| Storey Height | 0.1–5 | Vertical stretch of each storey (1.5 ≈ natural proportions) |
| Flange Width | 0.7–5 | Truncation half-width; below ≈0.88 the holes break open |
| Thickness | 0–0.5 | Wall thickness (0 = zero-thickness surface, no rims) |
| Rim Bulge | 0–4 | Outward bulge of the rounded rim beads |
| Twist | −900–1080° | Total axial twist along the chain |
| Azimuth | −360–360° | Fixed turn of the profile about the tower axis |
| Warp | 0–1080° | Bend into an arch/toroid; 360° closes the ring, >360 wraps multiply |
| Detail | 1–16 | Tessellation density |
| Stretch X/Y/Z | 0.2–5 | Affine stretch (the "Totem" trick) |
| Overall Scale | — | Uniform output scale (Blender-side extra) |

The panel shows whether the current twist closes the ring smoothly.
A ring closes when `(twist + storeys·180/branches) mod (360/branches) = 0`
— consecutive storeys of a Scherk tower are naturally rotated by
180°/branches, and the toroidal warp must reconcile that at the seam.
(All closed-ring demo files of the original satisfy this.)

## How the geometry works

The tower template is the exact singly-periodic Scherk minimal surface
`sin z = sinh x · sinh y`, parametrized per height level `c = sin z` as

    x(σ) = asinh(√c · e^{σΛ}),  y(σ) = asinh(√c · e^{−σΛ}),
    Λ = ln(sinh W / √c),  σ ∈ [−1, 1]

which places the curve ends exactly on the truncation planes
(`W` = flange). Higher-order saddles compress the 90° wedge of this
curve into a 180°/branches wedge. Storeys stack with the natural
180°/branches alternation; then azimuth + linear twist rotate the
cross-section, and the warp bends the tower around a circle whose
radius preserves arc length. Thickness is applied as a two-sided
offset with semicircular rim tubes (elongated by rim bulge) welded on
— including correctly through Möbius (single-sided) configurations.

Differences from the original: Séquin's C code is not public, so
tessellation layout and the exact flange/bulge profiles are
re-derivations, not byte-identical copies — shapes match the published
sculptures and the demo files visually. Thickness is uniform (applied
after the affine stretch). Storey-joint creases are C1 like the
original's; add a Subdivision Surface modifier if you want extra
smoothness.

## Files

- `src/scherk_collins_generator.py` — the add-on (single file; the
  geometry core also runs standalone:
  `python src/scherk_collins_generator.py` prints a smoke test)
- `tests/test_scherk.py` — headless test:
  `blender --background --factory-startup --python tests/test_scherk.py`
  builds presets + original demo files, checks watertightness, and
  renders to `renders/`
- `renders/` — sample renders of the presets and demo files

## References

- [Scherk-Collins Sculpture Generator](https://people.eecs.berkeley.edu/~sequin/SCULPTS/scherk.html) (C. H. Séquin)
- C. H. Séquin, *Virtual Prototyping of Scherk-Collins Saddle Rings*, Leonardo 30(2), 1997
- C. H. Séquin, H. Meshkin, L. Downs, *Interactive Generation of Scherk-Collins Sculptures*, I3D '97
- C. H. Séquin, *15 Years of Scherk-Collins Saddle Chains*, UCB/EECS-2010-41
