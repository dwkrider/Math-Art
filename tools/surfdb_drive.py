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
import math
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


sys.path.insert(0, HERE)
try:
    from surfdb import expr as _expr
    from surfdb import invariants as _inv
except ImportError:                                   # pragma: no cover
    _expr = _inv = None


# ---------------------------------------------------------------------------
# Independent checks: things measured from the MESH and compared against
# values that did not come from the mesh.
#
# The rest of this toolchain checks the generators against themselves --
# a stored equation is verified against the shipped implementation -- so
# it is excellent at catching transcription errors in the database and
# structurally incapable of catching a bug in the generator. These three
# are the ones that can:
#
#   Gauss-Bonnet     sum of angle defects must be 2*pi*chi on a closed
#                    mesh. Purely internal, but it fails on exactly the
#                    kind of broken topology a bad build produces.
#   symmetry         apply the record's claimed group generators to the
#                    built vertices and measure how far the set moves.
#                    A generator can produce the right topology with the
#                    wrong symmetry, and nothing else here would notice.
#   published values genus per cell, node counts, area -- compared with
#                    numbers taken from the literature, never from our
#                    own build (tools/surfdb/published.py).
# ---------------------------------------------------------------------------


def angle_defect_total(bm):
    """Sum of 2*pi minus the incident face angles, over interior vertices.

    By Gauss-Bonnet this equals 2*pi*chi for a closed surface, so it is a
    discrete total curvature that needs no smooth data.
    """
    total = 0.0
    for v in bm.verts:
        if any(len(e.link_faces) < 2 for e in v.link_edges):
            continue                                 # boundary vertex
        ang = 0.0
        for f in v.link_faces:
            vs = [lv.vert for lv in f.loops]
            try:
                k = vs.index(v)
            except ValueError:
                continue
            a = vs[k - 1].co - v.co
            b = vs[(k + 1) % len(vs)].co - v.co
            la, lb = a.length, b.length
            if la < 1e-12 or lb < 1e-12:
                continue
            c = max(-1.0, min(1.0, a.dot(b) / (la * lb)))
            ang += math.acos(c)
        total += 2.0 * math.pi - ang
    return total


def mesh_area_volume(bm):
    """(area, signed enclosed volume) by fan triangulation."""
    area = 0.0
    vol = 0.0
    for f in bm.faces:
        vs = [lv.vert.co for lv in f.loops]
        for k in range(1, len(vs) - 1):
            a, b, c = vs[0], vs[k], vs[k + 1]
            area += (b - a).cross(c - a).length * 0.5
            vol += a.dot(b.cross(c)) / 6.0
    return area, abs(vol)


def _grid_key(p, cell):
    return (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)),
            int(math.floor(p[2] / cell)))


def symmetry_residual(verts, gens, params, samples=400, seed=20260828):
    """How far the claimed symmetry moves the built vertex set.

    Each generator is applied to a sample of vertices and the image is
    matched to its nearest stored vertex through a spatial hash; the
    reported residual is the worst such distance, relative to the
    bounding-box diagonal. A correct build returns ~0.
    """
    import random
    if not verts or not gens:
        return None
    # RECENTRE FIRST, ON THE BOUNDING-BOX MIDPOINT.
    #
    # Every generator's output is fitted into a 2x2x2 box for display, and
    # that fit both scales and TRANSLATES. A point group acts about the
    # surface's own centre, so testing it on un-centred vertices rotates
    # the object about the wrong point and reports a symmetric surface as
    # asymmetric.
    #
    # The midpoint, NOT the centroid. The fit centres the BOUNDING BOX, so
    # the box midpoint is where the defining equation's origin ended up;
    # the vertex centroid is somewhere else entirely, because marching
    # tetrahedra do not distribute vertices evenly over a surface. Using
    # the centroid passed the 3-fold test (the centroid happens to lie on
    # the (1,1,1) axis) and failed the mirrors, which is exactly the
    # signature of rotating about a point displaced along the axis.
    mx = (max(v[0] for v in verts) + min(v[0] for v in verts)) / 2.0
    my = (max(v[1] for v in verts) + min(v[1] for v in verts)) / 2.0
    mz = (max(v[2] for v in verts) + min(v[2] for v in verts)) / 2.0
    verts = [(v[0] - mx, v[1] - my, v[2] - mz) for v in verts]

    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    diag = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2
                     + (max(zs) - min(zs)) ** 2)
    if diag < 1e-9:
        return None
    cell = max(diag / 60.0, 1e-6)
    buckets = {}
    for v in verts:
        buckets.setdefault(_grid_key(v, cell), []).append(v)

    rng = random.Random(seed)
    pick = verts if len(verts) <= samples else rng.sample(verts, samples)
    worst = 0.0
    for gx, gy, gz in gens:
        for v in pick:
            env = dict(params or {})
            env.update({"x": v[0], "y": v[1], "z": v[2]})
            try:
                q = (_expr.evaluate(gx, env), _expr.evaluate(gy, env),
                     _expr.evaluate(gz, env))
            except Exception:                         # noqa: BLE001
                return None
            # Search outward until a neighbour is found. A fixed 3x3x3
            # probe returns "not found" whenever the mesh is coarser than
            # one cell, and reporting that as a maximal residual turns a
            # sparse mesh into a false asymmetry -- which is exactly what
            # it did before this. Give up only after the shell exceeds a
            # quarter of the object.
            k0 = _grid_key(q, cell)
            best = None
            rad = 1
            while best is None and rad <= 15:
                for dx in range(-rad, rad + 1):
                    for dy in range(-rad, rad + 1):
                        for dz in range(-rad, rad + 1):
                            if max(abs(dx), abs(dy), abs(dz)) != rad and rad > 1:
                                continue
                            for w in buckets.get((k0[0] + dx, k0[1] + dy,
                                                  k0[2] + dz), ()):
                                d = math.sqrt((w[0] - q[0]) ** 2
                                              + (w[1] - q[1]) ** 2
                                              + (w[2] - q[2]) ** 2)
                                if best is None or d < best:
                                    best = d
                rad += 1
            if best is None:
                return None                  # cannot decide; not a failure
            worst = max(worst, best / diag)
    return worst


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

        # NON-MANIFOLD VERTICES. The usual edge test -- no edge shared by
        # more than two faces -- misses a "bowtie": two cones meeting at a
        # single point. Every edge there is perfectly manifold and the
        # vertex link is two cycles instead of one, so the surface is not
        # a manifold and Gauss-Bonnet does not hold. Detected by walking
        # the face fan around each vertex and checking it closes in ONE
        # loop.
        bowtie = 0
        for v in bm.verts:
            fs = list(v.link_faces)
            if len(fs) < 3:
                continue
            adj = {}
            for f in fs:
                es = [e for e in f.edges if v in e.verts]
                if len(es) == 2:
                    adj.setdefault(id(es[0]), []).append(f)
                    adj.setdefault(id(es[1]), []).append(f)
            seen = set()
            stack = [fs[0]]
            while stack:
                cur = stack.pop()
                if id(cur) in seen:
                    continue
                seen.add(id(cur))
                for e in cur.edges:
                    if v not in e.verts:
                        continue
                    for nf in adj.get(id(e), ()):
                        if id(nf) not in seen:
                            stack.append(nf)
            if len(seen) != len(fs):
                bowtie += 1

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

        area, vol = mesh_area_volume(bm)
        defect = angle_defect_total(bm)
        verts = [tuple(v.co) for v in bm.verts]

        return {
            "vertices": V, "edges": E, "faces": F,
            "euler_characteristic": V - E + F,
            "angle_defect_total": round(defect, 6),
            "area": round(area, 6),
            "volume": round(vol, 6),
            "_verts": verts,
            "components": comps,
            "boundary_loops": loops,
            "manifold": not nonmanifold and not bowtie,
            "nonmanifold_edges": len(nonmanifold),
            "nonmanifold_vertices": bowtie,
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


def compare(rec, meas, info=None):
    """Record claims vs measured mesh. Returns a list of disagreements.

    `info` collects observations that are NOT failures -- notably scale
    mismatches explained by the display fit.
    """
    bad = []
    if info is None:
        info = []
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

    # --- Gauss-Bonnet -----------------------------------------------------
    # On a CLOSED mesh the angle defects must sum to 2*pi*chi. This needs
    # no external data and fails on exactly the broken topology a bad
    # build produces -- a seam that did not weld, a duplicated strip.
    if meas["boundary_loops"] == 0 and meas["manifold"]             and meas["components"] == 1:
        want = 2.0 * math.pi * meas["euler_characteristic"]
        got = meas["angle_defect_total"]
        # A closed manifold satisfies this EXACTLY, so the tolerance is
        # only for accumulated floating point. Where it fails, the mesh is
        # not the closed manifold it appears to be.
        if abs(got - want) > max(2e-2, 1e-3 * abs(want)):
            bad.append("Gauss-Bonnet violated: angle defects sum to %.6g but "
                       "2*pi*chi = %.6g (chi = %d)%s"
                       % (got, want, meas["euler_characteristic"],
                          "" if not meas["nonmanifold_vertices"] else
                          " -- and the mesh has %d non-manifold (bowtie) "
                          "vertices, which the edge test does not see"
                          % meas["nonmanifold_vertices"]))

    # --- published invariants --------------------------------------------
    # These are the ones that can catch a GENERATOR bug, because the value
    # came from the literature and not from this build.
    topo = rec.get("topology") or {}
    gpc = topo.get("genus_per_cell")
    if isinstance(gpc, int) and meas["boundary_loops"] == 0             and meas["manifold"] and meas["components"] == 1:
        got_g = (2 - meas["euler_characteristic"]) / 2.0
        if abs(got_g - gpc) > 0.51:
            bad.append("published genus per cell is %d but the built cell "
                       "measures %.1f (chi = %d)"
                       % (gpc, got_g, meas["euler_characteristic"]))

    # Metrics must be compared SCALE-INVARIANTLY. Every generator's output
    # is fitted into a 2x2x2 box, so a raw area is off by the square of a
    # scale factor the record knows nothing about: the oloid and the
    # pseudosphere both read "55% out" against a published area of 4*pi,
    # and both implied the SAME ~0.667 factor -- a normalisation
    # mismatch, not a geometry error. README.md says it directly: the
    # 2x2x2 fit is a DISPLAY transform and never a storage normalisation.
    #
    # So the comparison uses the isoperimetric quotient 36*pi*V^2/A^3,
    # which is dimensionless and therefore immune to the fit. Where only
    # one of area and volume is published there is nothing scale-free to
    # form, and the implied scale factor is reported for information
    # rather than counted as a failure.
    metrics = rec.get("metrics") or {}
    pa = (metrics.get("area") or {}).get("value")
    pv = (metrics.get("volume_enclosed") or {}).get("value")
    closed = meas["boundary_loops"] == 0 and meas["manifold"]
    if pa and pv and closed and meas["area"] > 0 and meas["volume"] > 0:
        want = 36.0 * math.pi * (pv ** 2) / (pa ** 3)
        got = 36.0 * math.pi * (meas["volume"] ** 2) / (meas["area"] ** 3)
        if want > 0 and abs(got - want) / want > 0.05:
            bad.append("published isoperimetric quotient is %.5g but the "
                       "build measures %.5g -- a scale-free disagreement, so "
                       "not a normalisation artefact" % (want, got))
    elif pa and meas["area"] > 0:
        implied = math.sqrt(meas["area"] / pa)
        if abs(implied - 1.0) > 0.02:
            info.append("published area %.6g vs measured %.6g, implying a "
                        "uniform scale of %.4f -- consistent with the 2x2x2 "
                        "display fit, not compared" % (pa, meas["area"], implied))

    # --- symmetry, on the built mesh --------------------------------------
    # A generator can produce the right topology with the WRONG symmetry,
    # and nothing else in this toolchain would notice.
    sym = rec.get("symmetry") or {}
    group = sym.get("generator_set") or sym.get("schoenflies")
    if _inv is not None and group in getattr(_inv, "GENERATORS", {}):
        entry = _inv.GENERATORS[group]
        r = symmetry_residual(meas.get("_verts") or [], entry["gens"],
                              {"phi": (1 + math.sqrt(5)) / 2})
        if r is not None and r > 0.02:
            # Six of the first eight failures from this check were FRAME
            # errors, not asymmetries: Kummer is tetrahedral about its
            # tangent-plane normals rather than the coordinate diagonals,
            # and the Goursat dodecahedral rows put a 5-fold axis on z.
            # So a failure here is reported as UNRESOLVED unless the
            # record's frame has been pinned with `generator_set`, which
            # is the field that says somebody checked.
            pinned = bool((rec.get("symmetry") or {}).get("generator_set"))
            msg = ("claims symmetry %s but the built mesh moves by %.3g of "
                   "its diagonal under the group's generators" % (group, r))
            if pinned:
                bad.append(msg)
            else:
                info.append(msg + " -- UNRESOLVED: the record does not pin "
                            "its frame with `generator_set`, and a wrong "
                            "frame looks exactly like a wrong surface")
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
            recipe["measured"] = {k: v for k, v in measured_any.items()
                                  if not k.startswith("_")}
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
