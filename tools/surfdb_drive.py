# Drive every implemented generator and compare it with its record.
#
#     blender --background --python tools/surfdb_drive.py
#     blender --background --python tools/surfdb_drive.py -- --family quadric
#     blender --background --python tools/surfdb_drive.py -- --write
#
# WHY THIS EXISTS.  Before it, `construction[].implemented` was set from
# MODULE EXISTENCE -- the file is on disk, therefore the surface ships.
# That is not a check.  This turns it into "the operator was run, and the
# mesh it produced agrees with what the record claims".
#
# It is the direct analogue of data/polyhedra's `crosscheck` stage, which
# is the thing that found twelve wrong Johnson solids and two wrong
# compounds in math_art/.  A database that never drives the code it
# describes will eventually describe code that no longer exists.
#
# TWO TRAPS, both already paid for once in this repo and both documented
# in CLAUDE.md:
#
#   * NEVER pass --factory-startup.  It disables extensions, so every
#     Math Art operator appears unregistered and you conclude the whole
#     add-on is broken when it is merely switched off.
#   * "STATUS Reinstalled" is not proof the files landed.  If results look
#     wrong, check the installed tree before believing the install.
#
# This is the ONLY part of the surface-database toolchain that needs
# Blender; everything else runs headless, because math_art/minsurf/ and
# math_art/surfaces/ are bpy-free and the flat generator modules are read
# as source rather than imported.

import json
import os
import sys

try:
    import bpy
    import bmesh
except ImportError:                                   # pragma: no cover
    sys.stderr.write(
        "surfdb_drive must run inside Blender:\n"
        "    blender --background --python tools/surfdb_drive.py\n"
        "(and NOT with --factory-startup, which disables the extension)\n")
    raise SystemExit(2)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "data", "surfaces", "surfaces")

# Operators that act on selected objects; their poll() fails in an empty
# scene by design.
NEEDS_SELECTION = {
    "object.minimal_span",
    "object.fabrication_slice",
}


def argv_after_dashdash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.objects):
        for item in list(block):
            if getattr(item, "users", 0) == 0:
                block.remove(item)


def measure(obj):
    """Topological and metric invariants of the produced mesh.

    These are exactly the quantities math_art's own minimal-surface
    self-tests already gate on -- chi, components, boundary loops,
    manifoldness, orientability, one-sidedness and the bounding box -- so
    this measures the same things the code measures about itself.
    """
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        V, E, F = len(bm.verts), len(bm.edges), len(bm.faces)
        boundary = [e for e in bm.edges if len(e.link_faces) == 1]
        nonmanifold = [e for e in bm.edges if len(e.link_faces) > 2]

        # boundary loops: connected components of the boundary edge graph
        loops = 0
        seen = set()
        badj = {}
        for e in boundary:
            for v in e.verts:
                badj.setdefault(v.index, []).append(e)
        for e in boundary:
            if e.index in seen:
                continue
            loops += 1
            stack = [e]
            while stack:
                cur = stack.pop()
                if cur.index in seen:
                    continue
                seen.add(cur.index)
                for v in cur.verts:
                    for nxt in badj.get(v.index, ()):
                        if nxt.index not in seen:
                            stack.append(nxt)

        # connected components over faces
        comps = 0
        fseen = set()
        for f in bm.faces:
            if f.index in fseen:
                continue
            comps += 1
            stack = [f]
            while stack:
                cur = stack.pop()
                if cur.index in fseen:
                    continue
                fseen.add(cur.index)
                for e in cur.edges:
                    for nf in e.link_faces:
                        if nf.index not in fseen:
                            stack.append(nf)

        bb = [max(c[i] for c in obj.bound_box) - min(c[i] for c in obj.bound_box)
              for i in range(3)]
        ext = sorted(x for x in bb)
        aspect = (ext[0] / ext[2]) if ext[2] > 1e-12 else 0.0

        return {
            "vertices": V, "edges": E, "faces": F,
            "euler_characteristic": V - E + F,
            "components": comps,
            "boundary_loops": loops,
            "manifold": not nonmanifold,
            "orientable": None,          # Blender meshes carry no global sign
            "one_sided": None,
            "bounding_box": [round(v, 6) for v in bb],
            "aspect_ratio": round(aspect, 6),
        }
    finally:
        bm.free()


_ENUM_CACHE = {}
_VALUES_CACHE = {}


def enum_props(op_id):
    """Enum property identifiers on an operator, most-likely-first.

    Introspection alone is NOT enough here, and assuming it was cost a
    whole run: several of these operators build their surface list with a
    dynamic `items=` callback, and a dynamic enum reports NO
    `enum_items` until a context supplies them. So the identifiers are
    collected, ordered by how likely they are to be the selector, and
    then TRIED -- an invalid value raises and the next candidate is used.
    """
    if op_id in _ENUM_CACHE:
        return _ENUM_CACHE[op_id]
    mod, _, name = op_id.partition(".")
    fn = getattr(getattr(bpy.ops, mod), name, None)
    out = []
    if fn is not None:
        try:
            for prop in fn.get_rna_type().properties:
                if getattr(prop, "type", None) == "ENUM":
                    out.append(prop.identifier)
        except Exception:                             # noqa: BLE001
            pass
    prefer = ("surface", "preset", "kind", "mode", "shape", "family", "type")
    out.sort(key=lambda n: (prefer.index(n) if n in prefer else len(prefer), n))
    _ENUM_CACHE[op_id] = out
    return out


def enum_values(op_id, prop_name):
    """Static values of an operator enum; empty for a dynamic one."""
    ck = (op_id, prop_name)
    if ck in _VALUES_CACHE:
        return _VALUES_CACHE[ck]
    mod, _, name = op_id.partition(".")
    fn = getattr(getattr(bpy.ops, mod), name, None)
    out = []
    if fn is not None:
        try:
            for prop in fn.get_rna_type().properties:
                if prop.identifier == prop_name and                         getattr(prop, "type", None) == "ENUM":
                    out = [i.identifier for i in prop.enum_items]
        except Exception:                             # noqa: BLE001
            out = []
    _VALUES_CACHE[ck] = out
    return out


def run_op(op_id, params, key=None, family=None):
    """Invoke a `mesh.foo_add` style operator with the record's selection.

    A record's `construction[].key` is the registry row -- KLEIN, BOY,
    CATENOID -- but the property naming it differs per operator, and some
    operators need a `family` set first because the surface list is
    filtered by it. Both are tried.

    Without this, every record on a multi-surface operator drove that
    operator's DEFAULT, so Boy's surface, the Klein bottle and the genus-g
    surface were all silently measured as the same shape and the resulting
    "agreement" meant nothing.
    """
    mod, _, name = op_id.partition(".")
    fn = getattr(getattr(bpy.ops, mod), name, None)
    if fn is None:
        return None, "operator %s is not registered" % op_id

    base = dict(params or {})
    attempts = []
    if key:
        names = enum_props(op_id)
        dynamic, static = [], []
        for n in names:
            (dynamic if not enum_values(op_id, n) else static).append(n)

        # A DYNAMIC enum -- one whose items are built by a callback and so
        # report empty until something supplies them -- is the surface
        # selector, and it is usually FILTERED by a static one. On
        # mesh.periodic_minimal_add the selector is `surface` and the
        # filter is `periodicity`, whose value for a TPMS row is 'TRIPLY'
        # while the record's registry family says 'TPMS'. The two
        # vocabularies do not match and cannot be mapped by name, so every
        # filter value is tried against the selector.
        for t in dynamic:
            for f in static:
                for v in enum_values(op_id, f):
                    attempts.append(dict(base, **{f: v, t: key}))
            attempts.append(dict(base, **{t: key}))

        for n in names:
            if family and "family" in names and n != "family":
                attempts.append(dict(base, **{"family": family, n: key}))
            attempts.append(dict(base, **{n: key}))
        if not names:
            attempts.append(base)
    else:
        attempts.append(base)

    last = None
    for call in attempts:
        clear_scene()
        try:
            fn(**call)
        except (TypeError, ValueError) as exc:
            last = "%s" % exc
            continue
        except Exception as exc:                      # noqa: BLE001
            last = "%s raised: %s" % (op_id, exc)
            continue
        objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if objs:
            return objs[-1], None
        last = "%s produced no mesh" % op_id
    return None, ("could not select %r on %s (%s)"
                  % (key, op_id, last or "no enum property accepted it"))


def compare(rec, meas):
    """Record claims vs measured mesh. Returns a list of disagreements."""
    bad = []
    topo = rec.get("topology") or {}

    chi = topo.get("euler_characteristic")
    if isinstance(chi, int) and meas["boundary_loops"] == 0 \
            and meas["manifold"] and meas["components"] == 1:
        if chi != meas["euler_characteristic"]:
            bad.append("record chi = %d, mesh chi = %d"
                       % (chi, meas["euler_characteristic"]))

    # Boundary is comparable only for a COMPACT surface. A complete
    # non-compact one -- the catenoid, the helicoid, the pseudosphere --
    # is necessarily rendered as a truncated patch, so its mesh has rims
    # the mathematical surface does not. Comparing those is an
    # apples-to-oranges test that fails correct records.
    bc = topo.get("boundary_components")
    if isinstance(bc, int) and topo.get("compact") is True             and bc != meas["boundary_loops"]:
        bad.append("record boundary_components = %d, mesh has %d loops"
                   % (bc, meas["boundary_loops"]))

    # The two-sheeted hyperboloid is the one surface in the corpus whose
    # real locus is disconnected; a generator returning one sheet is
    # wrong even though it looks fine.
    if rec["slug"] == "hyperboloid-two-sheets" and meas["components"] < 2:
        bad.append("the two-sheeted hyperboloid must have 2 components, "
                   "mesh has %d" % meas["components"])

    # the repo's own collapse detector
    if meas["aspect_ratio"] < 0.02:
        bad.append("aspect ratio %.4g -- the surface came out essentially "
                   "flat, which passes every topological check"
                   % meas["aspect_ratio"])
    return bad


def main():
    args = argv_after_dashdash()
    want_family = None
    if "--family" in args:
        want_family = args[args.index("--family") + 1]
    write = "--write" in args

    records, paths = {}, {}
    for dp, _d, fs in os.walk(DB):
        for f in fs:
            if not f.endswith(".json"):
                continue
            p = os.path.join(dp, f)
            with open(p, encoding="utf-8") as fh:
                rec = json.load(fh)
            if want_family and rec["primary_family"] != want_family:
                continue
            records[rec["slug"]] = rec
            paths[rec["slug"]] = p

    driven = ok = failed = skipped = needs_ctx = 0
    problems = []
    for slug, rec in sorted(records.items()):
        entries = [c for c in rec.get("construction") or []
                   if c.get("implemented") and c.get("operator_id")]
        if not entries:
            skipped += 1
            continue
        # Operators that TRANSFORM a selection rather than add a shape
        # cannot be driven from an empty scene -- their poll() fails by
        # design. That is a context requirement, not a defect, and
        # counting it as a failure would be wrong. (tools/subjects.py
        # solves the same problem for the icon baker with a SETUP entry.)
        if all(c["operator_id"] in NEEDS_SELECTION for c in entries):
            needs_ctx += 1
            continue
        measured_any = None
        for c in entries:
            params = dict(c.get("params") or {})
            obj, err = run_op(c["operator_id"], params,
                              key=c.get("preset") or c.get("key"),
                              family=c.get("family"))
            if obj is None:
                failed += 1
                problems.append("%s [%s]: %s" % (slug, c["operator_id"], err))
                continue
            meas = measure(obj)
            driven += 1
            bad = compare(rec, meas)
            if bad:
                failed += 1
                for b in bad:
                    problems.append("%s [%s]: %s" % (slug, c["operator_id"], b))
            else:
                ok += 1
            measured_any = meas
        if write and measured_any is not None:
            recipe = rec.setdefault("mesh_recipe", {})
            recipe["measured"] = measured_any
            with open(paths[slug], "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2, ensure_ascii=False)
                fh.write("\n")

    print("\n=== surfdb drive ===")
    print("drove %d operator invocations across %d records"
          % (driven, len(records) - skipped))
    print("  %d agreed, %d disagreed or failed, %d records not implemented"
          % (ok, failed, skipped))
    if needs_ctx:
        print("  %d skipped: the operator transforms a SELECTION and cannot "
              "be driven from an empty scene" % needs_ctx)
    if problems:
        print("\nPROBLEMS (%d):" % len(problems))
        for p in problems[:60]:
            print("  -", p)
        if len(problems) > 60:
            print("  ... and %d more" % (len(problems) - 60))
    return 0


main()
