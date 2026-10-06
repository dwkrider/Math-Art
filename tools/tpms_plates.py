"""Pack the exported TPMS surfaces into a Bambu Studio project.

    python tools/tpms_plates.py [stl-dir] [out-dir] [--nine-up]

Default: ONE project file holding every surface, nine to a plate in a
three-by-three grid -- twenty-two surfaces over three plates. With
`--nine-up`: one file per surface instead, each holding nine copies of
that one surface.

Input is whatever `tools/tpms_batch.mjs` wrote -- binary STLs, already
at their printed size.

WHY 3MF AND NOT STLS. An STL is a triangle soup with no notion of an
instance or a plate: it cannot say "these nine go together" and it
cannot say "and these nine go on the next one". A 3MF stores each mesh
once, refers to it from build items that carry placements, and -- in
Bambu's dialect -- names which plate each instance belongs to in
`Metadata/model_settings.config`.

WHERE THE PLATES ARE, AND WHY THE MARGIN IS WHAT IT IS. Bambu lays its
plates out in a grid in ONE world coordinate system, so plate 2 is not
the same coordinates as plate 1; it is a stride away. Three things
were measured rather than assumed, by handing the slicer jobs and
reading its re-exports:

  * which plate an instance lands on is decided by its POSITION, not
    by the <plate> metadata. Four cubes declared one per plate but all
    placed on plate 1 came back with all four on plate 1 and three
    empty plates. So the coordinates have to be right; the metadata
    alone will not carry it.
  * an instance belongs to a plate if it INTERSECTS that plate's box,
    not if its centre is inside. On the CLI's 200 mm bed this
    predicts, exactly, the 9 + 6 + 4 it reports for a layout built
    for a 256 mm one.
  * on that 200 mm bed, plate 1 spans 0..200 and plate 2 begins at
    240 -- a 40 mm gap -- and three plates were laid out in two
    columns, the third dropping to a second row at negative y.

What could NOT be measured is whether that 40 mm gap is a constant or
a fifth of the bed: they agree at 200 mm and the CLI will not take a
256 mm bed by any route (--load-settings refuses the profiles, which
inherit; flattening them is refused too; --datadir does not change the
fallback). So the two candidates for a 256 mm bed are a stride of 296
or of 307.2, and the layout is built to survive either: the objects
are packed to leave 12.5 mm of margin, which is more than the 11.2 mm
the two hypotheses differ by. Worst case every object is 11.2 mm off
its plate's centre and still on it.
"""
import json
import math
import os
import struct
import sys
import zipfile

import numpy as np

BED_MM = 256.0        # square bed: X1 Carbon, P1, A1
PLATE_GAP_MM = 40.0   # between plates in Bambu's world, measured
GAP_MM = 3.0          # between neighbouring objects on a plate --
                      # chosen so the margin exceeds the stride
                      # uncertainty above, not for its own sake
ACROSS = 3            # three by three

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def read_binary_stl(path):
    """An STL is a triangle soup, so the vertices come back welded.

    Welding is on the exact 32-bit values, not a tolerance: these files
    were written from an indexed mesh, so a shared corner was written
    from the same float three times and its bytes are identical. A
    tolerance would be slower and would risk pulling genuinely distinct
    corners together.
    """
    with open(path, "rb") as fh:
        head = fh.read(84)
        n = struct.unpack("<I", head[80:84])[0]
        raw = np.frombuffer(fh.read(n * 50), dtype=np.uint8)
    if len(raw) != n * 50:
        raise ValueError("%s: truncated (%d of %d triangles)"
                         % (path, len(raw) // 50, n))
    rec = raw.reshape(n, 50)
    # 12 bytes of normal, then three vertices of 12, then 2 spare
    V = rec[:, 12:48].copy().view("<f4").reshape(n * 3, 3)
    uniq, inv = np.unique(V, axis=0, return_inverse=True)
    return uniq, inv.reshape(n, 3).astype(np.int64)


def centred(verts):
    """Centre on the origin in x and y, and stand on z = 0."""
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return (verts - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]),
            hi - lo)


def plate_centres(count):
    """Where each plate sits in Bambu's one world coordinate system."""
    cols = max(1, math.ceil(math.sqrt(count)))
    stride = BED_MM + PLATE_GAP_MM
    return [(BED_MM / 2 + (k % cols) * stride,
             BED_MM / 2 - (k // cols) * stride) for k in range(count)]


def grid_spots(centre, pitch):
    """The three-by-three square, centred on its plate."""
    span = pitch * (ACROSS - 1)
    return [(centre[0] + (i % ACROSS) * pitch - span / 2,
             centre[1] + (i // ACROSS) * pitch - span / 2)
            for i in range(ACROSS * ACROSS)]


def write(path, meshes, items, plates, title):
    """meshes: [(name, verts, tris)]; items: [(mesh_index, x, y)];
    plates: [[item_index, ...]]."""
    model = ['<?xml version="1.0" encoding="UTF-8"?>\n',
             '<model unit="millimeter" xml:lang="en-US"'
             ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n',
             ' <metadata name="Application">Math Art</metadata>\n',
             ' <metadata name="Title">%s</metadata>\n' % title,
             " <resources>\n"]
    for i, (name, verts, tris) in enumerate(meshes):
        model.append('  <object id="%d" type="model" name="%s">\n   <mesh>\n'
                     "    <vertices>\n" % (i + 1, name))
        model.append("".join(
            '     <vertex x="%.4f" y="%.4f" z="%.4f"/>\n' % tuple(v)
            for v in verts))
        model.append("    </vertices>\n    <triangles>\n")
        model.append("".join(
            '     <triangle v1="%d" v2="%d" v3="%d"/>\n' % tuple(t)
            for t in tris))
        model.append("    </triangles>\n   </mesh>\n  </object>\n")
    model.append(" </resources>\n <build>\n")
    for mi, x, y in items:
        model.append('  <item objectid="%d" transform="1 0 0 0 1 0 0 0 1'
                     ' %.4f %.4f 0" printable="1"/>\n' % (mi + 1, x, y))
    model.append(" </build>\n</model>\n")

    cfg = ['<?xml version="1.0" encoding="UTF-8"?>\n<config>\n']
    for i, (name, _v, tris) in enumerate(meshes):
        cfg.append('  <object id="%d">\n'
                   '    <metadata key="name" value="%s"/>\n'
                   '    <metadata key="extruder" value="1"/>\n'
                   '    <part id="%d" subtype="normal_part">\n'
                   '      <metadata key="name" value="%s"/>\n'
                   '      <metadata key="matrix" value="1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1"/>\n'
                   '      <mesh_stat face_count="%d" edges_fixed="0"'
                   ' degenerate_facets="0" facets_removed="0"'
                   ' facets_reversed="0" backwards_edges="0"/>\n'
                   "    </part>\n  </object>\n"
                   % (i + 1, name, 1000 + i, name, len(tris)))
    seen = {}
    for p, group in enumerate(plates):
        cfg.append("  <plate>\n"
                   '    <metadata key="plater_id" value="%d"/>\n'
                   '    <metadata key="plater_name" value=""/>\n'
                   '    <metadata key="locked" value="false"/>\n' % (p + 1))
        for ii in group:
            mi = items[ii][0]
            k = seen.get(mi, 0)
            seen[mi] = k + 1
            cfg.append("    <model_instance>\n"
                       '      <metadata key="object_id" value="%d"/>\n'
                       '      <metadata key="instance_id" value="%d"/>\n'
                       "    </model_instance>\n" % (mi + 1, k))
        cfg.append("  </plate>\n")
    cfg.append("</config>\n")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", "".join(model))
        z.writestr("Metadata/model_settings.config", "".join(cfg))


def load_all(stl_dir):
    index = {}
    idx_path = os.path.join(stl_dir, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as fh:
            index = {os.path.basename(r["file"]): r["name"]
                     for r in json.load(fh)}
    out = []
    for fn in sorted(f for f in os.listdir(stl_dir) if f.endswith(".stl")):
        verts, tris = read_binary_stl(os.path.join(stl_dir, fn))
        verts, size = centred(verts)
        out.append((index.get(fn, os.path.splitext(fn)[0]), verts, tris, size,
                    os.path.splitext(fn)[0]))
        print("  read %-42s %9d tris" % (fn, len(tris)))
    return out


def combined(loaded, out_dir):
    """One project: every surface, nine to a plate."""
    per = ACROSS * ACROSS
    nplates = math.ceil(len(loaded) / per)
    pitch = max(max(s[3][0], s[3][1]) for s in loaded) + GAP_MM
    centres = plate_centres(nplates)
    meshes, items, plates = [], [], []
    for p in range(nplates):
        here = loaded[p * per:(p + 1) * per]
        spots = grid_spots(centres[p], pitch)
        group = []
        for k, (name, verts, tris, _sz, _slug) in enumerate(here):
            meshes.append((name, verts, tris))
            group.append(len(items))
            items.append((len(meshes) - 1, spots[k][0], spots[k][1]))
        plates.append(group)
    path = os.path.join(out_dir, "tpms-all-%d-plates.3mf" % nplates)
    write(path, meshes, items, plates, "Triply-periodic minimal surfaces")
    tris = sum(len(m[2]) for m in meshes)
    print("\n%s\n  %d surfaces over %d plates (%s), %s triangles, %.0f MB"
          % (path, len(meshes), nplates,
             " + ".join(str(len(g)) for g in plates), format(tris, ","),
             os.path.getsize(path) / 1e6))
    span = pitch * (ACROSS - 1) + max(max(s[3][0], s[3][1]) for s in loaded)
    margin = (BED_MM - span) / 2
    print("  each plate a %.0f mm square, %.1f mm clear of a %.0f mm bed"
          % (span, margin, BED_MM))
    # the stride is the one thing measurement could not settle, so say
    # what happens under each candidate instead of hoping
    for gap, why in ((PLATE_GAP_MM, "a constant 40 mm gap"),
                     (0.2 * BED_MM, "a gap of a fifth of the bed")):
        off = abs((BED_MM + gap) - (BED_MM + PLATE_GAP_MM))
        print("    if Bambu uses %-28s objects sit %.1f mm off centre, "
              "%.1f mm to spare" % (why + ":", off, margin - off))


def nine_up(loaded, out_dir):
    """One file per surface, nine copies of it."""
    for name, verts, tris, size, slug in loaded:
        pitch = max(size[0], size[1]) + GAP_MM
        spots = grid_spots((BED_MM / 2, BED_MM / 2), pitch)
        path = os.path.join(out_dir, slug + "-9up.3mf")
        write(path, [(name, verts, tris)],
              [(0, x, y) for x, y in spots], [list(range(len(spots)))], name)
        print("  %-44s %9d tris  %.1f MB"
              % (os.path.basename(path), len(tris),
                 os.path.getsize(path) / 1e6))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stl_dir = (args[0] if args
               else r"C:\Users\dkrid\Projects\2026_07_21_Math_Art\dev\tpms-stl")
    out_dir = (args[1] if len(args) > 1
               else r"C:\Users\dkrid\Projects\2026_07_21_Math_Art\dev\tpms-3mf")
    os.makedirs(out_dir, exist_ok=True)
    loaded = load_all(stl_dir)
    if "--nine-up" in sys.argv:
        nine_up(loaded, out_dir)
    else:
        combined(loaded, out_dir)


main()
