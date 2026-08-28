"""Render one thumbnail per solid in the polyhedron database.

These are the catalogue tiles for the companion website's Polyhedra
module.  Every image is built straight from the record's own
`geometry.vertices` / `geometry.faces` in `data/polyhedra/`, NOT by
driving the add-on's operators, and that is the whole point:

  * It covers the database exactly.  All 448 records get a tile, named
    by slug.  Driving operators instead would only reach what each one
    exposes as an enum item, which is about two thirds of the database:
    the uniform duals hang off a `dual` boolean, the star prisms off a
    (p, q) pair of integers, the geodesics off a frequency, and
    Skilling's compound families off a repeat count.  None of those
    combinations is an enum item, so none of them is enumerable that way.

  * There is no name mapping to keep in step.  Rendering per-operator
    would mean a table from each operator's enum identifiers to database
    slugs -- and some of those identifiers are positional, so the table
    would rot silently the first time an enum's order changed.

  * The tile and the website's live WebGL view read the SAME record, so
    they cannot drift.  A thumbnail that disagrees with the model beside
    it is impossible by construction rather than by discipline.

The studio rig is imported from `docs/render_docs.py` rather than
restated, so these tiles match the documentation figures exactly and
follow any future change to the lighting.

Run (renders only what is missing):

    blender --background --factory-startup --python tools/render_polyhedra_thumbs.py

Re-render everything, or just some solids (by slug):

    ... --python tools/render_polyhedra_thumbs.py -- --all
    ... --python tools/render_polyhedra_thumbs.py -- cube great-dodecicosacron

Report coverage without rendering:

    ... --python tools/render_polyhedra_thumbs.py -- --list
"""
import bpy
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "tools"))
sys.path.insert(0, os.path.join(PROJ, "docs"))

import render_docs                                            # noqa: E402
import subjects as subject_cfg                                # noqa: E402
from polyhedra_tessellate import tessellate_face              # noqa: E402

DB = os.path.join(PROJ, "data", "polyhedra")
OUT = os.path.join(PROJ, "web", "thumbs")

# Matches the documentation's variant galleries: the catalogue shows
# these small, and the site's live viewer is there for a closer look.
RES = 320
SAMPLES = 64

# The add-on's face-size palette, copied from
# regular_solids_generator.py so a tile coloured here matches the same
# solid built in Blender.
FACE_PALETTE = {3: (0.90, 0.36, 0.23), 4: (0.27, 0.52, 0.79),
                5: (0.30, 0.69, 0.42), 6: (0.95, 0.77, 0.29),
                7: (0.62, 0.40, 0.75), 8: (0.25, 0.72, 0.72),
                9: (0.91, 0.56, 0.71), 10: (0.55, 0.60, 0.29),
                12: (0.52, 0.45, 0.40)}

# compound_generator.py's component palette, for the same reason.
COMPOUND_PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                    (0.30, 0.69, 0.42), (0.95, 0.77, 0.29),
                    (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                    (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                    (0.80, 0.45, 0.30), (0.45, 0.55, 0.80)]


def load_index():
    with open(os.path.join(DB, "index.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_record(entry):
    with open(os.path.join(DB, entry["path"]), encoding="utf-8") as fh:
        return json.load(fh)


def face_components(faces):
    """Group faces into connected components by SHARED EDGE.

    Shared-vertex connectivity is the obvious choice and it is wrong:
    the components of a compound routinely share vertices.  The five
    cubes inscribed in a dodecahedron sit on its twenty vertices, three
    cubes meeting at each, so vertex connectivity reports the compound
    as a single blob and it renders in one flat colour.  Components of a
    compound never share an EDGE, which is the relation used here --
    it recovers 5 for the five cubes and 60 for the sixty dodecahedra.
    """
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for fi, f in enumerate(faces):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            e = (a, b) if a < b else (b, a)
            if e in seen:
                ra, rb = find(seen[e]), find(fi)
                if ra != rb:
                    parent[ra] = rb
            else:
                seen[e] = fi
    roots = {}
    out = []
    for fi in range(len(faces)):
        r = find(fi)
        if r not in roots:
            roots[r] = len(roots)
        out.append(roots[r])
    return out, len(roots)


_MAT_CACHE = {}


def material(name, rgb):
    if name in _MAT_CACHE:
        return _MAT_CACHE[name]
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
    _MAT_CACHE[name] = mat
    return mat


def build_object(rec):
    """Build a Blender mesh from a record's stored geometry.

    Faces go through the shared tessellator, so pentagrams, {8/3} and
    {10/3} stars and the crossed quadrilaterals of the uniform duals all
    come out as the filled figures they are drawn as, rather than as
    whatever a triangle fan would make of them.
    """
    geom = rec["geometry"]
    verts = [tuple(v) for v in geom["vertices"]]
    faces = geom["faces"]
    comp_of, ncomp = face_components(faces)
    # A compound is coloured by component, anything else by face size.
    # Colouring a compound by face size would paint all five cubes the
    # same blue and lose the very thing the picture is of.
    by_component = ncomp > 1

    tris, keys = [], []
    for fi, f in enumerate(faces):
        local = list(f)
        pts, tri, _nrm = tessellate_face([geom["vertices"][i] for i in f])
        for extra in pts[len(f):]:
            verts.append(tuple(extra))
            local.append(len(verts) - 1)
        key = comp_of[fi] if by_component else len(f)
        for a, b, c in tri:
            tris.append((local[a], local[b], local[c]))
            keys.append(key)

    mesh = bpy.data.meshes.new(rec["slug"])
    mesh.from_pydata(verts, [], tris)
    mesh.validate(verbose=False)

    order = sorted(set(keys))
    slot_of = {}
    for k in order:
        if by_component:
            mat = material("Compound %d" % (k % len(COMPOUND_PALETTE)),
                           COMPOUND_PALETTE[k % len(COMPOUND_PALETTE)])
        else:
            rgb = FACE_PALETTE.get(k, (0.84, 0.84, 0.86))
            mat = material("Face %d-gon" % k, rgb)
        slot_of[k] = len(mesh.materials)
        mesh.materials.append(mat)
    if len(mesh.polygons) == len(keys):
        mesh.polygons.foreach_set("material_index",
                                  [slot_of[k] for k in keys])
    # Polyhedra are flat by definition; smoothing would round the very
    # edges the picture is about.
    mesh.polygons.foreach_set("use_smooth", [False] * len(mesh.polygons))
    mesh.update()

    obj = bpy.data.objects.new(rec["slug"], mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def render_one(rec, path):
    render_docs.clear_sculpts()
    build_object(rec)
    render_docs.normalize_subjects(2.0)
    subject_cfg.aim_rig(False)
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = RES
    scene.cycles.samples = SAMPLES
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    missing_only = "--all" not in argv
    only = [a for a in argv if not a.startswith("--")] or None

    index = load_index()
    entries = index["entries"]
    if only:
        want = set(only)
        entries = [e for e in entries if e["slug"] in want]
        unknown = want - {e["slug"] for e in entries}
        for u in sorted(unknown):
            print("unknown slug: %s" % u)

    os.makedirs(OUT, exist_ok=True)

    if "--list" in argv:
        have = sum(1 for e in entries
                   if os.path.exists(os.path.join(OUT, e["slug"] + ".png")))
        for e in entries:
            p = os.path.join(OUT, e["slug"] + ".png")
            print(("  have     " if os.path.exists(p) else "  MISSING  ")
                  + e["slug"])
        print("\n%d of %d present" % (have, len(entries)))
        return

    render_docs.setup_studio()
    subject_cfg.capture_rig()

    done = failed = skipped = 0
    for i, e in enumerate(entries, 1):
        path = os.path.join(OUT, e["slug"] + ".png")
        if missing_only and os.path.exists(path):
            skipped += 1
            continue
        try:
            render_one(load_record(e), path)
            done += 1
            print("[%d/%d] %s" % (i, len(entries), e["slug"]))
        except Exception as exc:                              # noqa: BLE001
            failed += 1
            print("FAIL %s: %r" % (e["slug"], exc))
    print("\nrendered=%d skipped=%d failed=%d" % (done, skipped, failed))
    print("DONE")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
