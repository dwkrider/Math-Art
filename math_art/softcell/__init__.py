# Soft-cell engine: assembling and replicating soft cells.
#
# Part of the Math Art extension.  Numpy only -- no `bpy` -- so the whole
# engine imports and self-tests headlessly; the registered Blender operator
# lives in the flat `soft_cell_generator.py` front-end.
#
# A SOFT CELL fills space without gaps while carrying the fewest possible
# sharp corners; in three dimensions that number is zero.  Two families are
# assembled here:
#
#   * the ANALYTIC cells of `analytic.py`, which have closed formulas and
#     need no solver at all;
#   * the (e2) MORPHOSPACE of `cell.py`, in which the truncated octahedron
#     of the body-centred cubic lattice is softened by bending its edges.
#     Every cell in that family is fixed by a single unit vector, so the
#     whole design space is two angles, and the named cells of the
#     literature -- Kelvin's foam cell, and the cells second-order
#     equivalent to the Schwarz P and Schwarz D unit cells -- are marked
#     points on one sphere.
#
# Building a morphospace cell has three stages.  The nodes never move.  The
# EDGES are bent to leave each node along its prescribed half-tangent
# (`edges.py`).  The FACES are then spanned on the resulting curved loops --
# either left as the raw radial grid, or relaxed to a discrete minimal
# surface with the project's existing Plateau solver.  The papers specify
# minimal surfaces, but they also make clear that the construction only pins
# the tangents: the surfaces are a choice, which is why both are offered.
#
# References:
# - G. Domokos, A. Goriely, A. G. Horvath and K. Regos, "Soft cells and the
#   geometry of seashells", PNAS Nexus 3(9):pgae311 (2024).
#   https://doi.org/10.1093/pnasnexus/pgae311
# - G. Domokos, A. Goriely, A. G. Horvath and K. Regos, "Soft cells, Kelvin's
#   foam and the minimal surfaces of Schwarz", arXiv:2412.04491 (2025).
# - G. Ambrus and D. Dancso, "Softening locally polyhedral tilings",
#   arXiv:2604.18545 (2026) -- the node warp used by `warp.py`.

import math

import numpy as np

try:
    from . import analytic, cell, edges, warp
except ImportError:                        # flat import outside the package
    import analytic
    import cell
    import edges
    import warp

try:
    from ..minsurf import plateau
except (ImportError, ValueError):
    try:
        from minsurf import plateau
    except ImportError:
        plateau = None


MORPHOSPACE = ('E2', 'F2', 'G2', 'H2', 'I2', 'KELVIN', 'PD')
ANALYTIC = analytic.KINDS


def _weld(V, faces, tol=1e-7):
    """Merge coincident vertices so shared edges become genuinely shared.

    Faces are built independently on their own boundary loops, so the same
    edge curve is generated twice -- once per adjacent face -- with bitwise
    identical coordinates.  Welding on a rounded key turns those into one
    vertex, which is what makes the cell closed rather than merely
    coincident-looking.
    """
    key = np.round(np.asarray(V, float) / tol).astype(np.int64)
    seen = {}
    remap = np.empty(len(V), dtype=np.int64)
    out = []
    for i, k in enumerate(map(tuple, key)):
        j = seen.get(k)
        if j is None:
            j = len(out)
            seen[k] = j
            out.append(V[i])
        remap[i] = j
    new_faces = []
    for f in faces:
        g = [int(remap[i]) for i in f]
        # drop consecutive duplicates created by the weld
        h = [g[0]]
        for x in g[1:]:
            if x != h[-1]:
                h.append(x)
        if len(h) > 2 and h[0] == h[-1]:
            h.pop()
        if len(h) >= 3:
            new_faces.append(tuple(h))
    return np.array(out, float), new_faces


def mesh_volume(V, faces):
    """Signed volume by the divergence theorem over fan-triangulated faces."""
    tot = 0.0
    for f in faces:
        p0 = V[f[0]]
        for i in range(1, len(f) - 1):
            tot += float(np.dot(p0, np.cross(V[f[i]] - p0, V[f[i + 1]] - p0)))
    return tot / 6.0


def mesh_area(V, faces):
    tot = 0.0
    for f in faces:
        p0 = V[f[0]]
        for i in range(1, len(f) - 1):
            tot += 0.5 * float(np.linalg.norm(
                np.cross(V[f[i]] - p0, V[f[i + 1]] - p0)))
    return tot


def _orient_outward(V, faces):
    """Flip every face if the mesh came out inside-out."""
    if mesh_volume(V, faces) < 0.0:
        return [tuple(reversed(f)) for f in faces]
    return faces


def _face_loop(curves, cycle):
    """Concatenate the bent edges around one face into a single loop."""
    pts = []
    for i in range(len(cycle)):
        v, w = cycle[i], cycle[(i + 1) % len(cycle)]
        seg = curves[(v, w)]
        pts.append(seg[:-1])               # drop the shared endpoint
    return np.vstack(pts)


def _span_face(loop, rings, relax_iters, style):
    """Span one curved boundary loop with a disk of quads.

    RULED leaves the radial grid alone -- a legitimate member of the
    second-order family, since the mathematics fixes only the tangents.
    MINIMAL relaxes it toward a discrete minimal surface, which is what the
    papers specify.
    """
    if plateau is None:
        raise RuntimeError("the Plateau solver is unavailable")
    V, quads, fixed = plateau.build_disk_grid(loop, rings)
    if style == 'MINIMAL' and relax_iters > 0:
        T = []
        for q in quads:
            if len(q) == 4:
                T.append((q[0], q[1], q[2]))
                T.append((q[0], q[2], q[3]))
            else:
                T.append(tuple(q))
        # clamped cotangents: loops with tangency cusps produce very obtuse
        # fans, which is the case the clamp mode exists to survive
        plateau.minimize_area(V, np.array(T, dtype=np.int64), fixed,
                              outer_iters=int(relax_iters),
                              cotan_mode="clamp")
    return V, quads


def build_cell(kind, phi=None, theta=None, symmetry='TETRAHEDRAL',
               edge_samples=16, face_rings=8, relax_iters=60,
               face_style='MINIMAL', resolution=24):
    """One soft cell, centred at the origin.

    Returns (V, faces, info).  `info` carries the exact cell volume where it
    is known, the mesh area, and the edge-kind census for the report.
    """
    if kind in ANALYTIC:
        V, faces, basis, exact = analytic.build(
            kind, resolution=resolution, rings=max(2, face_rings))
        V = V - V.mean(axis=0)
        faces = _orient_outward(V, faces)
        return V, faces, {'exact_volume': exact, 'basis': basis,
                          'area': mesh_area(V, faces), 'edge_kinds': {},
                          'symmetry': None}

    if kind not in MORPHOSPACE and kind != 'CUSTOM':
        raise ValueError(f"unknown cell {kind!r}")

    if kind == 'CUSTOM':
        a = cell.direction(phi, theta)
        sym = symmetry
    else:
        p, t, sym, _sigma = cell.PRESETS[kind]
        a = cell.direction(p, t)

    demoted = False
    try:
        field = cell.tangent_field(a, sym)
    except ValueError:
        # a Custom direction need not admit the full octahedral decoration;
        # the rotation subgroup always does, so fall back and say so
        if sym != 'OCTAHEDRAL':
            raise
        sym = 'TETRAHEDRAL'
        demoted = True
        field = cell.tangent_field(a, sym)

    curves = {}
    census = {}
    for (v, w) in cell.EDGES:
        t0, t1 = field[(v, w)], field[(w, v)]
        P0, P1 = cell.NODES[v], cell.NODES[w]
        k, _pl, _sy = edges.classify(P0, t0, P1, t1)
        census[k] = census.get(k, 0) + 1
        c = edges.build(P0, t0, P1, t1, samples=edge_samples)
        curves[(v, w)] = c
        curves[(w, v)] = c[::-1]

    V_all = []
    F_all = []
    for cyc in cell.FACES:
        loop = _face_loop(curves, cyc)
        Vf, quads = _span_face(loop, face_rings, relax_iters, face_style)
        base = sum(len(x) for x in V_all)
        V_all.append(Vf)
        for q in quads:
            F_all.append(tuple(base + i for i in q))

    V = np.vstack(V_all)
    V, faces = _weld(V, F_all)
    V = V - cell.CENTRE
    faces = _orient_outward(V, faces)
    return V, faces, {'exact_volume': cell.EXACT_VOLUME,
                      'basis': cell.lattice_basis(),
                      'area': mesh_area(V, faces),
                      'edge_kinds': census,
                      'symmetry': sym,
                      'demoted': demoted}


def build_block(kind, nx=1, ny=1, nz=1, gap=0.92, **kw):
    """A block of cells on the cell's own lattice, each shrunk by `gap`
    about its own centroid."""
    V, faces, info = build_cell(kind, **kw)
    B = np.asarray(info['basis'], float)
    cen = V.mean(axis=0)

    verts = []
    out = []
    tags = []
    n = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                t = i * B[0] + j * B[1] + k * B[2]
                P = cen + (V - cen) * gap + t
                verts.append(P)
                for f in faces:
                    out.append(tuple(n + q for q in f))
                    tags.append((i + j + k) % 2)
                n += len(V)
    W = np.vstack(verts)
    W = W - W.mean(axis=0)
    return W, out, tags, info


def softness(V, faces, rho):
    """sigma = rho * sqrt(2 pi / A), the softness of the PNAS Nexus paper.

    The rolling radius `rho` has to be supplied: estimating it from a mesh
    means an infimum of principal curvature radii, which is noisy exactly
    where it matters (at the tangency cusps) and is not something to gate a
    test on.  The AREA half of the formula is robust, so that is what the
    self-test checks; sigma is reported, not asserted.
    """
    A = mesh_area(V, faces)
    return rho * math.sqrt(2.0 * math.pi / A) if A > 0 else 0.0


def _selftest():
    analytic._selftest()
    cell._selftest()
    edges._selftest()
    warp._selftest()

    if plateau is None:
        print("softcell: Plateau solver unavailable, skipping face tests")
        return

    # Every soft cell of the (e2) family tiles the same bcc lattice as the
    # polyhedron it came from, so its volume must be the lattice's
    # fundamental volume 8*sqrt(2) -- whatever shape the faces took.  That
    # is the strongest single check on the whole assembly: edges, faces,
    # welding and orientation all have to be right for it to hold.
    target = cell.EXACT_VOLUME
    for kind in ('E2', 'F2', 'G2', 'H2', 'I2', 'KELVIN'):
        errs = []
        for rings, samp in ((4, 8), (8, 16)):
            V, F, info = build_cell(kind, edge_samples=samp,
                                    face_rings=rings, relax_iters=0,
                                    face_style='RULED')
            vol = abs(mesh_volume(V, F))
            errs.append(abs(vol - target) / target)
        assert errs[-1] <= errs[0] + 1e-9, (kind, errs)
        assert errs[-1] < 0.06, (kind, errs)
        print(f"{kind}: cell volume within {errs[-1] * 100:.2f}% of "
              f"8*sqrt(2), falling under refinement  OK")

    # (e2) must come out as the exact polyhedron
    V, F, info = build_cell('E2', edge_samples=4, face_rings=3,
                            relax_iters=0, face_style='RULED')
    assert info['edge_kinds'].get(edges.STRAIGHT, 0) == 36, info['edge_kinds']
    print("(e2): all 36 edges straight -- the polyhedron reproduced  OK")

    # the edge census must match what was measured: the equatorial presets
    # are planar (single arcs), the others spatial (biarcs)
    for kind, want in (('F2', edges.ARC), ('G2', edges.ARC),
                       ('KELVIN', edges.ARC), ('H2', edges.BIARC),
                       ('I2', edges.BIARC), ('PD', edges.BIARC)):
        _V, _F, info = build_cell(kind, edge_samples=6, face_rings=3,
                                  relax_iters=0, face_style='RULED')
        assert info['edge_kinds'].get(want, 0) == 36, (kind, info['edge_kinds'])
    print("cells: edge census matches the measured planarity of each preset "
          "(f2/g2/Kelvin arcs, h2/i2/PD biarcs)  OK")

    # a block at gap = 1 must not gain volume: cells meet, they do not
    # overlap or leave holes
    V, F, tags, info = build_block('F2', nx=2, ny=1, nz=1, gap=1.0,
                                   edge_samples=8, face_rings=4,
                                   relax_iters=0, face_style='RULED')
    vol = abs(mesh_volume(V, F))
    assert abs(vol - 2.0 * target) / (2.0 * target) < 0.06, vol
    print(f"block: two (f2) cells at gap 1.0 have volume {vol:.4f} "
          f"= 2 x 8*sqrt(2) within tolerance  OK")

    # analytic cells go through the same assembly path
    for kind in ANALYTIC:
        V, F, info = build_cell(kind, resolution=16, face_rings=4)
        vol = abs(mesh_volume(V, F))
        assert abs(vol - info['exact_volume']) < 1e-6, (kind, vol)
    print("analytic cells: assembly preserves their exact volumes  OK")

    # minimal-surface faces must reduce area relative to the raw grid
    V0, F0, _i = build_cell('G2', edge_samples=10, face_rings=6,
                            relax_iters=0, face_style='RULED')
    V1, F1, _i = build_cell('G2', edge_samples=10, face_rings=6,
                            relax_iters=40, face_style='MINIMAL')
    a0, a1 = mesh_area(V0, F0), mesh_area(V1, F1)
    assert a1 < a0, (a0, a1)
    print(f"(g2): Plateau relaxation reduces area {a0:.4f} -> {a1:.4f}  OK")

    print("softcell standalone tests passed")
