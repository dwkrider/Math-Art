# Spanning Pearce's saddle polygons with surfaces.
#
# A saddle polyhedron's faces are skew circuits of the Universal Node
# net, and Pearce's own definition of the face is the MINIMAL SURFACE
# spanning that circuit -- the soap film on the wire frame.  This
# module turns each circuit into a triangulated patch, with the
# boundary shared exactly between the two faces that meet along every
# branch so the solid welds into one closed shell.
#
# THE PIPELINE, per face:
#   circuit -> subdivide each branch into `density` segments
#           -> concentric disk grid toward the centroid  (the initial
#              guess; `minsurf.plateau.build_disk_grid`)
#           -> pinned-boundary cotangent-Laplacian area minimisation
#              (Pinkall-Polthier; `minsurf.plateau.minimize_area`)
#
# WHY THE BOUNDARY IS BUILT EDGE-BY-EDGE.  Two faces meeting along a
# branch must agree vertex for vertex, or the shell has a seam.  Each
# branch is therefore subdivided ONCE, into a shared list of points,
# and both faces read that list -- forward for one, reversed for the
# other.  Resampling each face's boundary independently by arclength
# would put the samples in almost, but not exactly, the same places,
# which welds visually and leaks geometrically.
#
# STYLES.  MINIMAL relaxes to the film.  RULED skips the relaxation and
# keeps the straight-line disk grid, which is the ruled saddle Pearce's
# own plastic panels approximate -- cheap, and a useful visual check
# that the relaxation is doing something.
#
# References:
# - Peter Pearce, "Structure in Nature is a Strategy for Design", The
#   MIT Press, 1978, ch. 8 -- saddle polygons spanned by minimal
#   surfaces, and the saddle polyhedra they bound.
# - Ulrich Pinkall & Konrad Polthier, "Computing Discrete Minimal
#   Surfaces and Their Conjugates", Experimental Mathematics 2(1),
#   1993, pp. 15-36 -- the cotangent-Laplacian area minimisation used
#   here through the project's `minsurf.plateau` module.

import numpy as np

try:
    from . import pearce_net as pnet
    from .minsurf import plateau as _pl
except Exception:                       # legacy single-file / CLI use
    import pearce_net as pnet
    from minsurf import plateau as _pl


def _tris(faces):
    """Triangulate the mixed quad/tri list build_disk_grid returns."""
    out = []
    for f in faces:
        if len(f) == 3:
            out.append(tuple(f))
        else:
            a, b, c, d = f
            out.append((a, b, c))
            out.append((a, c, d))
    return out


def edge_points(verts, faces, density):
    """Shared subdivision points for every branch of the solid.

    Returns {(a, b): [interior points, a->b]} with a < b, so both faces
    on a branch read the same list and the shell welds exactly."""
    pts = {}
    for (a, b) in pnet.edge_counts(faces):
        A = np.asarray(verts[a], float)
        B = np.asarray(verts[b], float)
        pts[(a, b)] = [A + (B - A) * (i / float(density))
                       for i in range(1, density)]
    return pts


def face_boundary(cyc, verts, epts):
    """One face's boundary loop, walking its circuit branch by branch."""
    loop = []
    n = len(cyc)
    for i in range(n):
        a, b = cyc[i], cyc[(i + 1) % n]
        loop.append(np.asarray(verts[a], float))
        seg = epts[(a, b)] if a < b else list(reversed(epts[(b, a)]))
        loop.extend(seg)
    return np.asarray(loop, float)


def face_patch(boundary, rings=None, iters=30, relax=True):
    """A minimal-surface patch spanning one skew circuit.

    `boundary` is the already-subdivided loop; its points stay pinned,
    so patches sharing a branch stay welded through the solve."""
    m = len(boundary)
    if rings is None:
        rings = max(2, m // 6)
    V, quads, fixed = _pl.build_disk_grid(np.asarray(boundary, float), rings)
    T = _tris(quads)
    if relax and iters > 0:
        V = np.asarray(V, float)
        _pl.minimize_area(V, np.asarray(T, dtype=np.int64), fixed,
                          outer_iters=iters)
    return np.asarray(V, float), T, m


def solid_surface(verts, faces, density=3, iters=30, relax=True,
                  rings=None):
    """Surface a whole saddle polyhedron.

    Returns (V, T, face_id) with V welded across shared branches:
    corner and branch-subdivision points are global and shared, patch
    interiors are per-face.  face_id[i] is the index of the face
    triangle i came from, for per-face materials and creases."""
    epts = edge_points(verts, faces, density)

    # global index for every boundary point: corners first, then each
    # branch's interior points in a fixed order
    gidx = {}
    GV = []
    for i, p in enumerate(verts):
        gidx[('v', i)] = len(GV)
        GV.append(np.asarray(p, float))
    for (a, b), seg in epts.items():
        for k, p in enumerate(seg):
            gidx[('e', a, b, k)] = len(GV)
            GV.append(p)

    def boundary_ids(cyc):
        ids = []
        n = len(cyc)
        for i in range(n):
            a, b = cyc[i], cyc[(i + 1) % n]
            ids.append(gidx[('v', a)])
            if a < b:
                ids.extend(gidx[('e', a, b, k)]
                           for k in range(len(epts[(a, b)])))
            else:
                ids.extend(gidx[('e', b, a, k)]
                           for k in reversed(range(len(epts[(b, a)]))))
        return ids

    T = []
    face_id = []
    for fi, cyc in enumerate(faces):
        ids = boundary_ids(cyc)
        loop = np.asarray([GV[i] for i in ids], float)
        PV, PT, m = face_patch(loop, rings=rings, iters=iters, relax=relax)
        # map the patch's own indices to global ones
        remap = {}
        for k in range(m):
            remap[k] = ids[k]
        for k in range(m, len(PV)):
            remap[k] = len(GV)
            GV.append(PV[k])
        for t in PT:
            T.append(tuple(remap[i] for i in t))
            face_id.append(fi)
    return np.asarray(GV, float), T, face_id


def fit_unit(V, size=2.0):
    """Centre on the origin and scale to fit a `size` cube."""
    V = np.asarray(V, float)
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    c = 0.5 * (lo + hi)
    ext = float((hi - lo).max())
    s = (size / ext) if ext > 1e-12 else 1.0
    return (V - c) * s


def mesh_extent(V):
    V = np.asarray(V, float)
    return tuple(float(x) for x in (V.max(axis=0) - V.min(axis=0)))


def aspect(V):
    """min/max bounding-box extent -- the collapse gate.

    A derived solid can pass every topology and angle check and still
    be flat; this is what catches that."""
    e = mesh_extent(V)
    return min(e) / max(e) if max(e) > 1e-12 else 0.0


def triangle_areas(V, T):
    V = np.asarray(V, float)
    out = []
    for a, b, c in T:
        out.append(0.5 * float(np.linalg.norm(
            np.cross(V[b] - V[a], V[c] - V[a]))))
    return np.asarray(out)


def _selftest():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-58s %s %s" % (name, "OK" if cond else "BAD", extra))

    print("pearce_surface: spanning the saddle polygons")

    for tag, net, L, nf in (("decatrihedron", 'SRS', 10, 3),
                            ("diamond tetrahedron", 'DIAMOND', 6, 4),
                            ("bcc tetrahedron", 'BCC', 4, 4)):
        cell = pnet.find_cell(net, L, nf, n=3 if net == 'SRS' else 2)
        if cell is None:
            chk("%s found" % tag, False)
            continue
        V0, F0 = cell
        Vr, Tr, fid = solid_surface(V0, F0, density=3, relax=False)
        Vm, Tm, fid2 = solid_surface(V0, F0, density=3, iters=25,
                                     relax=True)
        chk("%s: same topology relaxed or not" % tag,
            len(Tr) == len(Tm) and fid == fid2,
            "%d tris" % len(Tm))
        # every face contributed triangles
        chk("  every face surfaced",
            len(set(fid2)) == len(F0), "%d/%d" % (len(set(fid2)), len(F0)))
        # welding: boundary points are shared, so the shell is one piece
        ar = triangle_areas(Vm, Tm)
        chk("  no degenerate triangles", float(ar.min()) > 1e-9,
            "min area %.3e" % float(ar.min()))
        # relaxation must strictly reduce area
        a0 = float(triangle_areas(Vr, Tr).sum())
        a1 = float(ar.sum())
        chk("  relaxation reduces area", a1 < a0,
            "%.6f -> %.6f" % (a0, a1))
        # the collapse gate
        Vf = fit_unit(Vm)
        chk("  fits the 2 m cube", abs(max(mesh_extent(Vf)) - 2.0) < 1e-9)
        chk("  not collapsed (aspect >= 0.2)", aspect(Vf) >= 0.2,
            "aspect %.3f" % aspect(Vf))
        # welding check: count boundary edges of the triangulation --
        # a correctly welded closed shell has none
        cnt = {}
        for a, b, c in Tm:
            for e in ((a, b), (b, c), (c, a)):
                k = (min(e), max(e))
                cnt[k] = cnt.get(k, 0) + 1
        open_edges = [e for e, v in cnt.items() if v != 2]
        chk("  closed shell (no unwelded seam)", not open_edges,
            "%d open edges" % len(open_edges))

    print("RESULT:", "OK" if ok else "BAD")
    if not ok:
        raise AssertionError("pearce_surface self-test failed")
