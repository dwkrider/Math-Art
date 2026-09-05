"""Bake a mesh and a thumbnail for every implemented surface.

These are the artifacts the companion site's Surfaces module reads:

    web/surfaces/<slug>.json        packed triangle mesh
    web/thumbs/surfaces/<slug>.png  catalogue tile

Both come from ONE run of the surface's own operator, because running the
operator is by far the expensive part -- several of these solve a
variational problem or integrate Weierstrass data -- and doing it twice
would double a long pass for nothing.

WHY THE MESHES ARE BAKED AT ALL.  `data/surfaces/` deliberately stores no
meshes; its README argues that a mesh is a rendering at a chosen
resolution and not part of a surface's identity, which is right. But the
site has to draw something, and for a large part of the database the
record is NOT sufficient to draw from:

  * parametric (109) and implicit/nodal (187) records do carry evaluable
    formulas, so those could in principle be evaluated in the browser;
  * weierstrass (163) mostly does NOT. Those records store no (g, dh)
    pair -- their own notes say the shipped mesher is authoritative and
    that "an unverified transcription would silently define a different
    surface".

So the geometry is taken from the one place that is authoritative for all
of them: the add-on. Baking every mode the same way also means the site
cannot show a surface the extension would not produce.

NEVER RUN THIS WITH --factory-startup.  It disables extensions, so every
Math Art operator appears unregistered and the whole pass fails with
"operator is not registered". This is the trap documented at the top of
tools/surfdb_drive.py, and it applies to anything that drives operators
rather than building meshes from data.

    blender --background --python tools/surfdb_export.py
    blender --background --python tools/surfdb_export.py -- --all
    blender --background --python tools/surfdb_export.py -- catenoid gyroid
    blender --background --python tools/surfdb_export.py -- --list
"""
import base64
import json
import os
import struct
import sys

import bpy
import bmesh

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "tools"))
sys.path.insert(0, os.path.join(PROJ, "docs"))

import surfdb_drive as DRIVE                                  # noqa: E402

DB = os.path.join(PROJ, "data", "surfaces")
MESH_OUT = os.path.join(PROJ, "web", "surfaces")
THUMB_OUT = os.path.join(PROJ, "web", "thumbs", "surfaces")

# A triangle budget, not a resolution. The generators produce anything
# from a few hundred to a few hundred thousand triangles; at ~12 bytes per
# triangle packed, 40k would be half a megabyte for one surface. 20k is
# still smooth on screen at the size the module draws it, and keeps the
# whole set to roughly the size of the polyhedron thumbnails.
TRI_BUDGET = 20000

RES = 320
SAMPLES = 48


def load_index():
    with open(os.path.join(DB, "index.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_record(entry):
    with open(os.path.join(DB, entry["path"]), encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------- packing

def pack_mesh(positions, indices):
    """Pack to the payload `decodeMesh` in web/js/surface-viewer.js reads.

    Positions are quantised to uint16 across the mesh's own bounding box,
    which is invisible at 16 bits per axis and roughly halves the payload;
    indices are 16-bit where the vertex count allows. Normals are not
    stored at all -- the viewer recovers flat shading per-fragment from
    screen-space derivatives, which is what makes positions-plus-indices
    a complete mesh here.

    The JS decoder already existed; this is the encoder it was written
    against, which had never been committed.
    """
    n = len(positions) // 3
    lo = [min(positions[i::3]) for i in range(3)]
    hi = [max(positions[i::3]) for i in range(3)]
    span = [(hi[k] - lo[k]) or 1.0 for k in range(3)]
    q = bytearray()
    for i in range(n):
        for k in range(3):
            t = (positions[i * 3 + k] - lo[k]) / span[k]
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            q += struct.pack("<H", int(round(t * 65535)))
    wide = n > 65535
    fmt = "<I" if wide else "<H"
    ib = bytearray()
    for v in indices:
        ib += struct.pack(fmt, v)
    return {
        "p": base64.b64encode(bytes(q)).decode("ascii"),
        "i": base64.b64encode(bytes(ib)).decode("ascii"),
        "lo": [round(v, 6) for v in lo],
        "hi": [round(v, 6) for v in hi],
        "w": 4 if wide else 2,
        "n": n,
        "t": len(indices) // 3,
    }


def mesh_arrays(obj, budget=TRI_BUDGET):
    """Evaluated, triangulated, decimated positions + indices."""
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    ntri = len(bm.faces)
    if ntri > budget:
        # Collapse edges rather than sampling: an unsubdivide or a planar
        # decimate can punch through a thin sheet, and these surfaces are
        # sheets. Ratio is on faces, which is what the budget counts.
        bmesh.ops.dissolve_degenerate(bm, dist=1e-8, edges=bm.edges[:])
        me2 = bpy.data.meshes.new("tmp_decimate")
        bm.to_mesh(me2)
        bm.free()
        tmp = bpy.data.objects.new("tmp_decimate", me2)
        bpy.context.scene.collection.objects.link(tmp)
        mod = tmp.modifiers.new("Decimate", 'DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = float(budget) / float(ntri)
        deps = bpy.context.evaluated_depsgraph_get()
        me3 = tmp.evaluated_get(deps).to_mesh()
        bm = bmesh.new()
        bm.from_mesh(me3)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bpy.data.objects.remove(tmp, do_unlink=True)

    bm.verts.index_update()
    positions = []
    for v in bm.verts:
        co = obj.matrix_world @ v.co
        positions += [co.x, co.y, co.z]
    indices = []
    for f in bm.faces:
        indices += [v.index for v in f.verts[:3]]
    bm.free()
    try:
        ev.to_mesh_clear()
    except Exception:                                          # noqa: BLE001
        pass
    return positions, indices


# ----------------------------------------------------------------- driving

def unpack_mesh(packed):
    """Positions and triangle indices back out of a packed payload.

    Phase B renders each thumbnail from the EXPORTED mesh rather than
    from the operator's output, so the tile in the catalogue is a picture
    of the exact file the viewer will load -- quantisation, decimation
    and all. The same guarantee the polyhedron tiles have.
    """
    q = struct.unpack("<%dH" % (packed["n"] * 3),
                      base64.b64decode(packed["p"]))
    lo, hi = packed["lo"], packed["hi"]
    positions = []
    for i in range(packed["n"]):
        for k in range(3):
            positions.append(lo[k] + (hi[k] - lo[k]) * (q[i * 3 + k] / 65535.0))
    fmt = "<%d%s" % (packed["t"] * 3, "I" if packed["w"] == 4 else "H")
    indices = struct.unpack(fmt, base64.b64decode(packed["i"]))
    return positions, list(indices)


def build(rec):
    """Run the record's operator and hand back the object it made."""
    err = None
    for con in rec.get("construction") or []:
        if not con.get("implemented"):
            continue
        obj, e = DRIVE.run_op(con["operator_id"], con.get("params") or {},
                              key=con.get("key"), family=con.get("family"))
        if obj is not None:
            return obj, None
        err = e
    return None, err or "no implemented construction"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    missing_only = "--all" not in argv
    only = {a for a in argv if not a.startswith("--")} or None

    index = load_index()
    entries = [e for e in index["entries"] if e.get("implemented")]
    if only:
        entries = [e for e in entries if e["slug"] in only]

    os.makedirs(MESH_OUT, exist_ok=True)
    os.makedirs(THUMB_OUT, exist_ok=True)

    if "--list" in argv:
        have = sum(1 for e in entries
                   if os.path.exists(os.path.join(MESH_OUT, e["slug"] + ".json")))
        print("%d of %d implemented surfaces have a mesh" % (have, len(entries)))
        for e in entries:
            p = os.path.join(MESH_OUT, e["slug"] + ".json")
            print(("  have     " if os.path.exists(p) else "  MISSING  ") + e["slug"])
        return

    # PHASE A -- drive the operators.
    #
    # Kept strictly apart from rendering because run_op() calls
    # clear_scene(), which deletes every object including the studio's
    # camera and lights; a studio built first does not survive the first
    # surface, and the pass fails with "Cannot render, no camera".
    done = failed = skipped = 0
    problems = []
    todo = []
    for i, e in enumerate(entries, 1):
        slug = e["slug"]
        mpath = os.path.join(MESH_OUT, slug + ".json")
        tpath = os.path.join(THUMB_OUT, slug + ".png")
        if missing_only and os.path.exists(mpath) and os.path.exists(tpath):
            skipped += 1
            continue
        if missing_only and os.path.exists(mpath):
            todo.append((slug, tpath))          # mesh done, tile missing
            continue
        try:
            rec = load_record(e)
            obj, err = build(rec)
            if obj is None:
                raise RuntimeError(err)
            positions, indices = mesh_arrays(obj)
            if not indices:
                raise RuntimeError("operator produced an empty mesh")
            packed = pack_mesh(positions, indices)
            packed["slug"] = slug
            packed["name"] = e.get("name")
            with open(mpath, "w", encoding="utf-8") as fh:
                json.dump(packed, fh, separators=(",", ":"))
            todo.append((slug, tpath))
            done += 1
            print("[%d/%d] %-46s %6d tri" % (i, len(entries), slug,
                                             packed["t"]))
        except Exception as exc:                              # noqa: BLE001
            failed += 1
            problems.append((slug, repr(exc)[:110]))
            print("FAIL %-46s %r" % (slug, exc))
        sys.stdout.flush()

    # PHASE B -- the thumbnails, in the documentation studio.
    import render_docs                                        # noqa: E402
    import subjects as subject_cfg                            # noqa: E402
    render_docs.setup_studio()
    subject_cfg.capture_rig()
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = RES
    scene.cycles.samples = SAMPLES

    shot = 0
    for slug, tpath in todo:
        try:
            with open(os.path.join(MESH_OUT, slug + ".json"),
                      encoding="utf-8") as fh:
                packed = json.load(fh)
            positions, indices = unpack_mesh(packed)
            render_docs.clear_sculpts()
            me = bpy.data.meshes.new(slug)
            verts = [tuple(positions[i * 3:i * 3 + 3])
                     for i in range(packed["n"])]
            tris = [tuple(indices[i * 3:i * 3 + 3])
                    for i in range(packed["t"])]
            me.from_pydata(verts, [], tris)
            me.validate(verbose=False)
            # Surfaces are smooth; flat shading would show the mesh, not
            # the surface. Solids are the other way round.
            me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
            me.update()
            ob = bpy.data.objects.new(slug, me)
            bpy.context.scene.collection.objects.link(ob)
            render_docs.apply_material()
            render_docs.normalize_subjects(2.0)
            subject_cfg.aim_rig(False)
            scene.render.filepath = tpath
            bpy.ops.render.render(write_still=True)
            shot += 1
            print("shot %-46s" % slug)
        except Exception as exc:                              # noqa: BLE001
            failed += 1
            problems.append((slug, "thumbnail: " + repr(exc)[:96]))
            print("FAIL %-46s thumbnail %r" % (slug, exc))
        sys.stdout.flush()

    print("\nmeshes=%d thumbnails=%d skipped=%d failed=%d"
          % (done, shot, skipped, failed))
    for s, why in problems[:25]:
        print("   %-46s %s" % (s, why))
    print("DONE")


if __name__ == "__main__":
    main()
