
# Geodesic Sphere / Dome Generator for Blender
#
# Geodesic spheres and domes (after Segerman, "Visualizing Mathematics
# with 3D Printing", figs 4-5 / 4-6): Class I (h,0), Class II (h,h) and
# the general chiral Class III (h,k) Goldberg-Coxeter breakdowns of the
# icosahedron, octahedron or tetrahedron projected to the sphere,
# oriented vertex-up and optionally cut to a hemisphere or 5/8 dome.
# The Class III cells are placed as a triangular lattice on each base
# face and the projected lattice points convex-hulled.  Styles: welded
# shell (optional Solidify thickness), strut and node frame, Leonardo-
# style open panels, or a panelised dome with inset gaps.
#
# The optional colourings say something different about the same solid:
# which faces are the twelve special ones, which hexagons are actually
# congruent to each other, how far each face lies from a pentagon, which
# face of the seed polyhedron it was subdivided out of, a map colouring,
# and -- for the strut frame -- the chord length classes that a dome
# builder cuts and colour-codes as A/B/C.
#
# References:
# - Geodesic domes: R. Buckminster Fuller.
# - Geodesic/Goldberg (h,k) classification: M. Goldberg, "A class of
#   multi-symmetric polyhedra", Tohoku Math. J. 43 (1937); Caspar & Klug
#   (1962) for the triangulation number T = h²+hk+k².
# - Goldberg polyhedra (the duals): Michael Goldberg (1937).
# - Henry Segerman, "Visualizing Mathematics with 3D Printing"
#   (2016), figs 4-5, 4-6.
# - Chord factors and the A/B/C strut length classes of a geodesic
#   dome: Hugh Kenner, "Geodesic Math and How to Use It", University of
#   California Press (1976).

bl_info = {
    "name": "Geodesic Sphere / Dome Generator",
    "author": "Math Art project (after Segerman figs 4-5, 4-6)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Geodesic Sphere / Dome",
    "description": "Geodesic spheres and domes, Class I, II and III "
                   "(h,k), with Goldberg duals",
    "category": "Add Mesh",
}

import math
from math import sin, cos, pi

import numpy as np

PHI = (1 + 5 ** 0.5) / 2

try:                                  # inside the math_art package
    from .polyhedra.seeds import icosa_faces as _icosa_faces
    from .polyhedra.seeds import seed_poly as _shared_seed
except ImportError:                   # flat import (test runner)
    from polyhedra.seeds import icosa_faces as _icosa_faces
    from polyhedra.seeds import seed_poly as _shared_seed


def seed_poly(kind):
    """Platonic seed normalised to unit circumradius.

    This module and two others built their seeds this way, while
    three others used the raw coordinates.  The two sets are
    EXACTLY related by that scale (verified vertex for vertex on
    all five solids, with identical face lists), so the shared
    table carries both behind its `unit` flag and this wrapper
    keeps the call sites here reading as before.
    """
    return _shared_seed(kind, unit=True)


# ---- seed polyhedra (triangular faces only) ------------------------------

def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)




def quad_seed(kind):
    """Quad-faced seeds for geodesic cubes / rhombic triacontahedra."""
    if kind == 'CUBE':
        V = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        Fq = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
              [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
    elif kind == 'RT':                   # rhombic triacontahedron = dual iD
        try:
            from .polyhedra import (canonical as _canon, conway as cw,
                                    flags as _flags)
        except ImportError:
            from polyhedra import (canonical as _canon, conway as cw,
                                   flags as _flags)
        V, F = cw.apply_conway('aD')
        V = _canon.canonicalize_best(V, F, hart_iters=400)
        Vd, Fd = _flags.dual(V, F)
        Vd = _canon.canonicalize_best(Vd, Fd, hart_iters=600)
        return [_unit(v) for v in Vd], [list(f) for f in Fd]
    else:
        raise ValueError(kind)
    return [_unit(v) for v in V], Fq


def quad_geodesic(V, Fq, freq):
    """Class-I (freq,0) quad subdivision of a quad-faced polyhedron,
    projected to the unit sphere (geodesic cube / rhombic triacontahedron).
    Shared edges dedupe exactly by rounded position."""
    if freq <= 1:
        return [_unit(v) for v in V], [list(f) for f in Fq]
    verts = {}
    faces = []

    def vid(p):
        k = (round(p[0], 8), round(p[1], 8), round(p[2], 8))
        if k not in verts:
            verts[k] = len(verts)
        return verts[k]
    for f in Fq:
        A, B, C, D = (V[i] for i in f)

        def bil(s, t):
            return _unit(tuple((1 - s) * (1 - t) * A[c] + s * (1 - t) * B[c]
                               + s * t * C[c] + (1 - s) * t * D[c]
                               for c in range(3)))
        g = {}
        for i in range(freq + 1):
            for j in range(freq + 1):
                g[(i, j)] = vid(bil(i / freq, j / freq))
        for i in range(freq):
            for j in range(freq):
                faces.append([g[(i, j)], g[(i + 1, j)],
                              g[(i + 1, j + 1)], g[(i, j + 1)]])
    inv = {v: k for k, v in verts.items()}
    return [inv[i] for i in range(len(verts))], faces




def orient_vertex_up(V):
    """Rotate the seed so its topmost vertex sits exactly on +Z."""
    apex = max(V, key=lambda v: v[2])
    v = _unit(apex)
    if v[2] > 1.0 - 1e-12:
        return list(V)
    axis = _unit((v[1], -v[0], 0.0))       # cross(v, z)
    ang = math.acos(max(-1.0, min(1.0, v[2])))
    c, s = cos(ang), sin(ang)
    out = []
    for p in V:
        d = axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2]
        cr = (axis[1] * p[2] - axis[2] * p[1],
              axis[2] * p[0] - axis[0] * p[2],
              axis[0] * p[1] - axis[1] * p[0])
        out.append(tuple(p[k] * c + cr[k] * s + axis[k] * d * (1 - c)
                         for k in range(3)))
    return out


# ---- geodesic breakdowns -------------------------------------------------

def geodesic(V, F, freq):
    """Class-I geodesic subdivision of a triangular polyhedron,
    projected to the unit sphere.  Shared vertices are deduped exactly
    (barycentric sums along a shared edge are bitwise identical from
    both sides)."""
    if freq <= 1:
        return [_unit(v) for v in V], [list(f) for f in F]
    verts = [_unit(v) for v in V]
    key = {}
    faces = []

    def vid(p):
        k = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if k not in key:
            key[k] = len(verts)
            verts.append(p)
        return key[k]

    for f in F:
        if len(f) != 3:
            raise ValueError("geodesic subdivision needs triangles")
        A, B, C = (verts[i] for i in f)
        grid = {}
        for i in range(freq + 1):
            for j in range(freq + 1 - i):
                if (i, j) == (freq, 0):
                    grid[(i, j)] = f[0]
                    continue
                if (i, j) == (0, freq):
                    grid[(i, j)] = f[1]
                    continue
                if (i, j) == (0, 0):
                    grid[(i, j)] = f[2]
                    continue
                k = freq - i - j
                p = _unit(tuple((i * A[c] + j * B[c] + k * C[c]) / freq
                                for c in range(3)))
                grid[(i, j)] = vid(p)
        for i in range(freq):
            for j in range(freq - i):
                faces.append([grid[(i, j)], grid[(i + 1, j)],
                              grid[(i, j + 1)]])
                if j < freq - i - 1:
                    faces.append([grid[(i + 1, j)], grid[(i + 1, j + 1)],
                                  grid[(i, j + 1)]])
    return verts, faces


def kis(V, F):
    """Mid-face split: each triangle becomes three triangles about its
    centroid (projected to the sphere).  Composing this with a Class-I
    subdivision at frequency f yields the Class II (f,f) breakdown."""
    verts = [_unit(v) for v in V]
    faces = []
    for f in F:
        c = _unit(tuple(sum(verts[i][k] for i in f) / 3
                        for k in range(3)))
        ci = len(verts)
        verts.append(c)
        for i in range(3):
            faces.append([f[i], f[(i + 1) % 3], ci])
    return verts, faces


def build_sphere(base='ICOSA', freq=3, cls='I'):
    """Vertex-up unit geodesic sphere: (verts, tri faces)."""
    V, F = seed_poly(base)
    V = orient_vertex_up(V)
    if cls == 'II':
        V, F = kis(V, F)
    return geodesic(V, F, freq)


def geodesic_points(base, h, k):
    """Vertices of the general geodesic [h,k] breakdown (Goldberg-Coxeter),
    projected to the unit sphere.  h=k=0 excluded; k=0 is Class I, h=k is
    Class II, otherwise the chiral Class III.  Each base triangle carries a
    triangular lattice whose (h,k) cell defines the subdivision; the convex
    hull of the returned points is the geodesic sphere (V = 10T+2,
    T = h²+hk+k², for the icosahedron)."""
    V, F = seed_poly(base)
    V = orient_vertex_up(V)
    s3 = math.sqrt(3) / 2

    def latt(m, n):
        return (m + n * 0.5, n * s3)
    O, P, Q = latt(0, 0), latt(h, k), latt(-k, h + k)

    def bary(p):
        det = ((P[1] - Q[1]) * (O[0] - Q[0]) + (Q[0] - P[0]) * (O[1] - Q[1]))
        a = ((P[1] - Q[1]) * (p[0] - Q[0]) + (Q[0] - P[0]) * (p[1] - Q[1])) / det
        b = ((Q[1] - O[1]) * (p[0] - Q[0]) + (O[0] - Q[0]) * (p[1] - Q[1])) / det
        return a, b, 1 - a - b
    lo1 = min(0, h, -k) - 1
    hi1 = max(0, h, -k) + 1
    lo2 = min(0, k, h + k) - 1
    hi2 = max(0, k, h + k) + 1
    pts = {}
    for f in F:
        A, B, C = V[f[0]], V[f[1]], V[f[2]]
        for m in range(lo1, hi1 + 1):
            for n in range(lo2, hi2 + 1):
                a, b, c = bary(latt(m, n))
                if a < -1e-9 or b < -1e-9 or c < -1e-9:
                    continue
                sp = _unit(tuple(a * A[t] + b * B[t] + c * C[t]
                                 for t in range(3)))
                pts[tuple(round(x, 7) for x in sp)] = sp
    return list(pts.values())


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def goldberg_dual(V, F):
    """Dual of a triangulated geodesic sphere: a vertex at each face
    centroid (on the sphere) and a polygon around each original vertex
    -- hexagons plus twelve pentagons, i.e. the Goldberg polyhedron."""
    Vd = [_unit(tuple(sum(V[i][k] for i in f) / len(f)
                      for k in range(3))) for f in F]
    inc = [[] for _ in range(len(V))]
    for fi, f in enumerate(F):
        for vi in f:
            inc[vi].append(fi)
    Fd = []
    for vi in range(len(V)):
        ring = inc[vi]
        if len(ring) < 3:            # boundary vertex: no closed face
            continue
        n = _unit(V[vi])
        ref = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (1.0, 0.0, 0.0)
        u = _unit(_cross(n, ref))
        w = _cross(n, u)
        ring = sorted(ring, key=lambda fi: math.atan2(_dot(Vd[fi], w),
                                                      _dot(Vd[fi], u)))
        Fd.append(ring)
    return Vd, Fd


# ---- colouring keys ------------------------------------------------------
#
# Every scheme is the same operation: one hashable key per face, handed to
# styles.face_colors.materials_for.  They are all computed on the CLOSED
# unit sphere, before the dome cut and before the radius scale -- the cut
# strips faces from the rim, which would make ordinary six-valent hubs
# look like the twelve special ones, and the tolerances below are
# calibrated at unit radius.

#: Absolute tolerance for calling two lengths or areas equal.  The
#: quantities compared live between ~1e-3 (a face area at high frequency)
#: and ~1 (a centroid radius), so one absolute tolerance covers them only
#: because the gaps between genuinely different classes are far larger
#: than the noise within one class: on the 3v icosahedron the within-class
#: spread is ~1e-16 and the closest between-class gap is ~9e-3, and those
#: gaps shrink like 1/f², so at the frequency ceiling of 16 they are still
#: several orders above this.  It must not be tightened much further:
#: `quad_geodesic` stores vertices rounded to eight decimals, so CUBE and
#: RT carry ~1e-8 of quantisation noise of their own.
CLASS_TOL = 1e-6


def face_adjacency(F):
    """{face index: [face index, ...]} for faces that share an edge."""
    owner = {}
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            owner.setdefault((min(a, b), max(a, b)), []).append(fi)
    adj = {i: [] for i in range(len(F))}
    for fs in owner.values():
        for i in fs:
            for j in fs:
                if i != j:
                    adj[i].append(j)
    return adj


def vertex_valences(V, F):
    """How many faces meet at each vertex."""
    val = [0] * len(V)
    for f in F:
        for i in f:
            val[i] += 1
    return val


def _modal(values):
    """The most common value; ties broken by taking the smallest, so the
    answer never depends on dict ordering."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(sorted(counts), key=lambda v: counts[v])


def special_faces(V, F, dual):
    """The faces that are not of the generic kind -- on an icosahedral
    Goldberg dual, exactly the twelve pentagons.

    Stated as "not the modal kind" rather than "pentagons" on purpose.
    The twelve-pentagon reading only holds for a Class I icosahedral
    dual: an octahedral seed gives four-sided specials, a tetrahedral one
    three-sided, and the Class II construction here (a kis split followed
    by a Class I subdivision) leaves ten-valent and three-valent vertices,
    so its dual has decagons and triangles and no pentagons at all.
    Keying on the modal kind gives the intended picture in every case.
    """
    if dual:
        sizes = [len(f) for f in F]
        generic = _modal(sizes)
        return {i for i, n in enumerate(sizes) if n != generic}
    val = vertex_valences(V, F)
    generic = _modal(val)
    odd = {i for i, v in enumerate(val) if v != generic}
    return {i for i, f in enumerate(F) if any(v in odd for v in f)}


def _face_signature(V, f):
    """(sides, centroid radius, area, sorted edge lengths) -- everything
    that is invariant under the symmetry group and separates faces that
    are genuinely differently shaped."""
    pts = [V[i] for i in f]
    n = len(pts)
    c = tuple(sum(p[k] for p in pts) / n for k in range(3))
    r = math.sqrt(sum(x * x for x in c))
    edges = []
    area = 0.0
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        edges.append(math.dist(p, q))
        # fan triangulation about the centroid: valid for the convex,
        # near-planar faces this generator produces
        u = tuple(p[k] - c[k] for k in range(3))
        w = tuple(q[k] - c[k] for k in range(3))
        cr = _cross(u, w)
        area += 0.5 * math.sqrt(_dot(cr, cr))
    return (n, r, area) + tuple(sorted(edges))


def congruence_classes(V, F, tol=CLASS_TOL):
    """One key per face, equal exactly when the faces are the same shape
    and the same distance out.

    This is the orbit structure of the solid under its symmetry group,
    reached without any group theory: two faces of a Goldberg polyhedron
    are in the same orbit precisely when they are congruent and equally
    far from the centre, and that is what the signature measures.  It
    answers the question the Wikipedia tables raise but never show --
    GP(2,0) has one kind of hexagon, GP(3,0) has more than one.
    """
    reps = []
    keys = []
    for f in F:
        sig = _face_signature(V, f)
        hit = None
        for ri, rsig in enumerate(reps):
            if rsig[0] != sig[0]:
                continue
            if all(abs(a - b) < tol for a, b in zip(rsig[1:], sig[1:])):
                hit = ri
                break
        if hit is None:
            hit = len(reps)
            reps.append(sig)
        keys.append(hit)
    return keys


def ring_keys(V, F, dual):
    """Face-graph distance from the nearest special face.

    Concentric bands around each of the twelve pentagons; where the bands
    from neighbouring pentagons meet is exactly where the Goldberg
    lattice's mirror lines run.
    """
    adj = face_adjacency(F)
    frontier = sorted(special_faces(V, F, dual))
    if not frontier:
        # a face-transitive solid -- GP(1,0) is the dodecahedron, whose
        # twelve faces are ALL pentagons -- has nothing to measure from,
        # so every face is equally at distance zero
        return [0] * len(F)
    dist = {i: 0 for i in frontier}
    d = 0
    while frontier:
        d += 1
        nxt = []
        for i in frontier:
            for j in adj[i]:
                if j not in dist:
                    dist[j] = d
                    nxt.append(j)
        frontier = nxt
    # a face in no ring can only happen on a disconnected mesh; give it
    # its own key rather than silently folding it into ring 0
    return [dist.get(i, -1) for i in range(len(F))]


_SEED_CENTROIDS = {}


def seed_centroids(base):
    """Unit face centroids of the seed polyhedron, exactly as oriented by
    the build path that uses it.

    The triangular seeds are vertex-up because `build_sphere` orients
    them; the quad seeds are not, because the quad path does not.  Class
    II applies `kis` after that, which adds face centres without moving
    the original faces, so the twenty icosahedral faces are still the
    right thing to colour by.  RT is cached because building it runs a
    thousand canonicalisation iterations.
    """
    if base in _SEED_CENTROIDS:
        return _SEED_CENTROIDS[base]
    if base in ('CUBE', 'RT'):
        V, F = quad_seed(base)
    else:
        V, F = seed_poly(base)
        V = orient_vertex_up(V)
    out = [_unit(tuple(sum(V[i][k] for i in f) / len(f)
                       for k in range(3))) for f in F]
    _SEED_CENTROIDS[base] = out
    return out


def seed_keys(V, F, base):
    """Which seed face each face was subdivided out of.

    On the triangulated form this is clean: every subdivided triangle
    lies strictly inside one seed face.  On the dual it cannot be, and
    that is a fact about the solid rather than a shortcoming here -- a
    dual face sits at an original VERTEX, and the vertices along a seed
    edge lie exactly on the border between two seed faces (the twelve
    corners, on five at once).  Those ties are exact by symmetry, so they
    are detected as ties and settled by taking the lowest-numbered
    claimant, which is arbitrary but identical on every machine.
    """
    cents = seed_centroids(base)
    keys = []
    for f in F:
        c = _unit(tuple(sum(V[i][k] for i in f) / len(f)
                        for k in range(3)))
        dots = [_dot(c, s) for s in cents]
        best = max(dots)
        keys.append(min(i for i, d in enumerate(dots)
                        if d > best - 1e-9))
    return keys


def map_keys(F):
    """A proper colouring of the face-adjacency graph: adjacent faces
    always differ.

    Four colours always suffice (Appel and Haken) but no heuristic is
    guaranteed to find such a colouring, and for these solids none of
    them do -- every Goldberg dual tried needs a fifth before the solver
    succeeds -- so the shared helper escalates until the colouring is
    genuinely proper rather than shipping the improper fallback.
    """
    try:
        from .styles import face_colors
    except ImportError:                    # flat import (test runner)
        from styles import face_colors
    col, _k = face_colors.proper_coloring(len(F), face_adjacency(F))
    return [col[i] for i in range(len(F))]


def length_classes(V, edges, tol=CLASS_TOL):
    """Chord length class per edge: the A/B/C strut schedule of a dome."""
    reps = []
    keys = []
    for a, b in edges:
        L = math.dist(V[a], V[b])
        hit = next((i for i, r in enumerate(reps) if abs(r - L) < tol),
                   None)
        if hit is None:
            hit = len(reps)
            reps.append(L)
        keys.append(hit)
    return keys


FACE_SCHEMES = ('SIZE', 'ORBIT', 'RINGS', 'SEED', 'MAP')


def face_keys(V, F, scheme, base='ICOSA', dual=False):
    """One key per face for the named scheme (see FACE_SCHEMES)."""
    if scheme == 'SIZE':
        if dual:
            return [len(f) for f in F]
        odd = special_faces(V, F, dual)
        return [(i in odd) for i in range(len(F))]
    if scheme == 'ORBIT':
        return congruence_classes(V, F)
    if scheme == 'RINGS':
        return ring_keys(V, F, dual)
    if scheme == 'SEED':
        return seed_keys(V, F, base)
    if scheme == 'MAP':
        return map_keys(F)
    raise ValueError(scheme)


# ---- dome cutting --------------------------------------------------------

CUT_Z = {'FULL': None, 'HEMI': 0.0, 'FIVEEIGHTHS': -0.25}


def cut_faces_keep(V, F, zmin):
    """`cut_faces`, additionally returning which source faces survived.

    Anything computed on the closed sphere -- a colouring key, say -- has
    to be filtered the same way the faces were, and the compacted face
    list on its own does not say which ones went.
    """
    keep = []
    for fi, f in enumerate(F):
        cz = sum(V[i][2] for i in f) / len(f)
        if cz >= zmin - 1e-9:
            keep.append(fi)
    used = sorted({i for fi in keep for i in F[fi]})
    remap = {o: n for n, o in enumerate(used)}
    return ([V[i] for i in used],
            [[remap[i] for i in F[fi]] for fi in keep],
            keep)


def cut_faces(V, F, zmin):
    """Keep faces whose centroid z >= zmin; compact the vertex list."""
    Vc, Fc, _keep = cut_faces_keep(V, F, zmin)
    return Vc, Fc


def boundary_loops(F):
    """Ordered rim loops (lists of vertex indices) of an open mesh;
    consecutive entries are directed boundary edges as they appear in
    their face, so faces lie to the left of the walk."""
    count = {}
    directed = {}
    for f in F:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            k = (min(a, b), max(a, b))
            count[k] = count.get(k, 0) + 1
            directed[k] = (a, b)
    nxt = {}
    for k, n in count.items():
        if n == 1:
            a, b = directed[k]
            nxt[a] = b
    loops = []
    seen = set()
    for start in list(nxt):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur = nxt[start]
        while cur != start:
            loop.append(cur)
            seen.add(cur)
            cur = nxt[cur]
        loops.append(loop)
    return loops


def add_base_ring(verts, faces, loops, rim_coords, width,
                  reuse=None):
    """Append a flat ring band widening each rim loop outward (in XY)
    by `width`.  `rim_coords` maps rim vertex index -> coordinate.  If
    `reuse` is given it maps rim vertex index -> index in `verts`
    (welded band); otherwise fresh copies of the rim are appended."""
    for loop in loops:
        inner = {}
        outer = {}
        for vi in loop:
            p = rim_coords[vi]
            if reuse is not None:
                inner[vi] = reuse[vi]
            else:
                inner[vi] = len(verts)
                verts.append(tuple(p))
            r = math.hypot(p[0], p[1]) or 1.0
            q = (p[0] * (1 + width / r), p[1] * (1 + width / r), p[2])
            outer[vi] = len(verts)
            verts.append(q)
        m = len(loop)
        for i in range(m):
            a, b = loop[i], loop[(i + 1) % m]
            faces.append([inner[b], inner[a], outer[a], outer[b]])


# ---- panel geometry ------------------------------------------------------
# The strut (edge cylinder) and node (vertex sphere) primitives that back
# the "Ball and Stick" style now live in the shared ball_and_stick module,
# so every polyhedron generator draws struts and nodes the same way.


def add_panel(verts, faces, pts, gap, thickness):
    """One triangle shrunk by `gap` about its centroid, extruded to a
    thin prism along its outward normal."""
    pts = [np.asarray(p, dtype=float) for p in pts]
    c = sum(pts) / len(pts)
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    ln = np.linalg.norm(n) or 1.0
    n = n / ln
    inner = [c + (p - c) * (1.0 - gap) for p in pts]
    lo = [q - n * (thickness / 2) for q in inner]
    hi = [q + n * (thickness / 2) for q in inner]
    base = len(verts)
    for q in lo + hi:
        verts.append(tuple(q))
    m = len(pts)
    faces.append([base + m + i for i in range(m)])            # top
    faces.append([base + i for i in range(m - 1, -1, -1)])    # bottom
    for i in range(m):
        j = (i + 1) % m
        faces.append([base + i, base + j, base + m + j, base + m + i])


# ---- Blender layer -------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_geodesic_add(bpy.types.Operator):
        """Add a geodesic sphere or dome"""
        bl_idname = "mesh.geodesic_add"
        bl_label = "Geodesic Sphere / Dome"
        bl_options = {'REGISTER', 'UNDO'}

        base: EnumProperty(
            name="Base",
            items=[('ICOSA', "Icosahedron", ""),
                   ('OCTA', "Octahedron", ""),
                   ('TETRA', "Tetrahedron", ""),
                   ('CUBE', "Cube (quad geodesic)",
                    "Square-grid subdivision -> quad-faced geodesic"),
                   ('RT', "Rhombic Triacontahedron (quad geodesic)",
                    "Rhombus-grid subdivision -> quad-faced geodesic")],
            default='ICOSA',
            description="Seed polyhedron whose faces are subdivided and "
                        "projected to the sphere")
        geo_class: EnumProperty(
            name="Class",
            items=[('I', "Class I (f,0)",
                    "Alternate breakdown: each base triangle is "
                    "subdivided into f² triangles and projected "
                    "to the sphere"),
                   ('II', "Class II (f,f)",
                    "Triacon-style breakdown: mid-face (kis) split "
                    "followed by a Class I subdivision at f, giving "
                    "the (f,f) triangulation (3f² triangles per "
                    "base face, effective frequency 2f)"),
                   ('III', "Class III (h,k)",
                    "General chiral Goldberg-Coxeter breakdown with "
                    "h = Frequency and the k below (h != k, k > 0); "
                    "convex-hulled")],
            default='I',
            description="Goldberg-Coxeter breakdown class of the subdivision")
        frequency: IntProperty(
            name="Frequency", default=3, min=1, max=16,
            description="Breakdown frequency f (= h for Class III): "
                        "Class I gives (f,0), Class II gives (f,f)")
        k: IntProperty(
            name="k (Class III)", default=1, min=1, max=15,
            description="Second Goldberg-Coxeter index for Class III "
                        "(the chiral (h,k) breakdown, h = Frequency)")
        dual: BoolProperty(
            name="Dual (Goldberg)", default=False,
            description="Output the dual polyhedron -- hexagons and "
                        "twelve pentagons (a Goldberg polyhedron / "
                        "geodesic's honeycomb) instead of triangles")
        cut: EnumProperty(
            name="Cut",
            items=[('FULL', "Full Sphere", ""),
                   ('HEMI', "Hemisphere Dome",
                    "Keep faces whose centroid lies above z = 0"),
                   ('FIVEEIGHTHS', "5/8 Dome",
                    "Keep faces whose centroid lies above "
                    "z = -0.25 R (a clean strut ring on the "
                    "3v icosahedron)")],
            default='FULL',
            description="Keep the full sphere or cut it down to a dome")
        base_ring: BoolProperty(
            name="Base Ring", default=False,
            description="Thicken the open rim into a flat ring band "
                        "(domes are left open like real domes)")
        ring_width: FloatProperty(
            name="Ring Width", default=0.1, min=0.005, max=1.0,
            description="Radial width of the base ring band")
        style: EnumProperty(
            name="Style",
            items=[('SHELL', "Shell",
                    "Smooth welded sphere surface (Solidify modifier "
                    "if thickness > 0)"),
                   ('WIRE', "Struts",
                    "Edges as a wireframe frame of struts (Wireframe "
                    "modifier)"),
                   ('STRUTS', "Ball and Stick",
                    "Edges as cylinders, vertices as small spheres"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo Style "
                    "modifier"),
                   ('PANELS', "Panels",
                    "Each triangle inset about its centroid with a "
                    "gap and slight thickness")],
            default='SHELL',
            description="How the geodesic is rendered as geometry")
        radius: FloatProperty(name="Radius", default=1.0, min=0.01,
                              max=100.0, description="Sphere radius")
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.0, max=1.0,
            description="Shell / panel thickness (0 = single surface)")
        border: FloatProperty(name="Border", default=0.06, min=0.005,
                              max=1.0,
                              description="Frame width left around each "
                                          "open panel in the Leonardo style")
        strut_radius: FloatProperty(name="Strut Radius", default=0.02,
                                    min=0.001, max=0.5,
                                    description="Radius of the edge "
                                                "cylinders in Ball and Stick")
        node_radius: FloatProperty(name="Node Radius", default=0.035,
                                   min=0.001, max=0.5,
                                   description="Radius of the vertex "
                                               "spheres in Ball and Stick")
        color_by: EnumProperty(
            name="Face Colors",
            items=[('NONE', "None", "One material for the whole solid"),
                   ('SIZE', "Face Size",
                    "Pentagons against hexagons on a Goldberg dual; on "
                    "the triangulated sphere, the triangles that meet "
                    "one of the twelve corner vertices"),
                   ('ORBIT', "Congruence Class",
                    "Faces that are the same shape and the same "
                    "distance out share a color, so the several "
                    "distinct kinds of hexagon are told apart"),
                   ('RINGS', "Distance Rings",
                    "Bands by how many faces away the nearest pentagon "
                    "is"),
                   ('SEED', "Seed Face",
                    "Which face of the base polyhedron each face was "
                    "subdivided out of"),
                   ('MAP', "Map Coloring",
                    "Fewest colors that still give neighbouring faces "
                    "different ones")],
            default='NONE',
            description="How the faces are grouped into colors")
        strut_color: EnumProperty(
            name="Strut Colors",
            items=[('NONE', "None", "One material for the whole frame"),
                   ('LENGTH', "Strut Length",
                    "Struts colored by chord length class -- the "
                    "cutting schedule a dome is built from"),
                   ('VALENCE', "Node Valence",
                    "Nodes colored by how many struts meet there"),
                   ('BOTH', "Length and Valence",
                    "Strut length classes and node valence together")],
            default='NONE',
            description="How the struts and nodes are colored in the "
                        "Ball and Stick style")
        gap: FloatProperty(
            name="Panel Gap", default=0.15, min=0.0, max=0.9,
            description="Fraction each panel is shrunk about its "
                        "centroid")

        def execute(self, context):
            if self.base in ('CUBE', 'RT'):
                V, F = quad_geodesic(*quad_seed(self.base), self.frequency)
            elif self.geo_class == 'III':
                import bmesh
                pts = geodesic_points(self.base, self.frequency, self.k)
                bm = bmesh.new()
                for p in pts:
                    bm.verts.new(p)
                bm.verts.ensure_lookup_table()
                res = bmesh.ops.convex_hull(bm, input=bm.verts)
                junk = res.get('geom_unused', []) + \
                    res.get('geom_interior', [])
                if junk:
                    bmesh.ops.delete(bm, geom=junk, context='VERTS')
                bm.verts.ensure_lookup_table()
                bm.faces.ensure_lookup_table()
                idx = {v: i for i, v in enumerate(bm.verts)}
                V = [tuple(v.co) for v in bm.verts]
                F = [[idx[v] for v in fc.verts] for fc in bm.faces]
                bm.free()
            else:
                V, F = build_sphere(self.base, self.frequency,
                                    self.geo_class)
            if self.dual:
                V, F = goldberg_dual(V, F)

            # Colouring keys come from the CLOSED sphere: a dome cut
            # takes faces away from the rim, and a rim vertex that has
            # lost half its faces would otherwise be mistaken for one of
            # the twelve special ones.  Cut the keys the same way the
            # faces are cut instead.
            face_color = (self.color_by if self.style != 'STRUTS'
                          else 'NONE')
            fkeys = (face_keys(V, F, face_color, self.base, self.dual)
                     if face_color != 'NONE' else None)

            zmin = CUT_Z[self.cut]
            if zmin is not None:
                V, F, keep = cut_faces_keep(V, F, zmin)
                if fkeys is not None:
                    fkeys = [fkeys[i] for i in keep]
            Vu = V                       # unit radius: what CLASS_TOL
            R = self.radius              # was calibrated against
            V = [(x * R, y * R, z * R) for (x, y, z) in V]
            loops = (boundary_loops(F)
                     if zmin is not None and self.base_ring else [])

            # only round struts/nodes shade smooth; the flat facets
            # of shells and panels must stay flat or the edges blur
            smooth = self.style == 'STRUTS'
            keys = [] if (fkeys is not None
                          or self.strut_color != 'NONE') else None
            if self.style in ('SHELL', 'LEONARDO', 'WIRE'):
                verts = list(V)
                faces = [list(f) for f in F]
                if keys is not None:
                    keys.extend(fkeys)
                add_base_ring(verts, faces, loops, V,
                              self.ring_width * R,
                              reuse={i: i for i in range(len(V))})
            elif self.style == 'STRUTS':
                try:
                    from .styles import ball_and_stick
                except ImportError:
                    from styles import ball_and_stick
                edges = ball_and_stick.edges_from_faces(F)
                groups = [] if keys is not None else None
                verts, faces = ball_and_stick.build_mesh(
                    V, edges, self.strut_radius * R,
                    self.node_radius * R, groups=groups)
                if keys is not None:
                    # struts by chord length, nodes by how many struts
                    # meet there -- both measured on the mesh as built,
                    # so a dome's rim hubs read as the rim hubs they are
                    lc = (length_classes(Vu, edges)
                          if self.strut_color in ('LENGTH', 'BOTH')
                          else None)
                    val = (vertex_valences(V, F)
                           if self.strut_color in ('VALENCE', 'BOTH')
                           else None)
                    for kind, i in groups:
                        if kind == 'E':
                            keys.append(('strut', lc[i]) if lc
                                        else 'strut')
                        else:
                            keys.append(('node', val[i]) if val
                                        else 'node')
                add_base_ring(verts, faces, loops, V,
                              self.ring_width * R)
            else:                                        # PANELS
                smooth = False
                verts, faces = [], []
                for fi, f in enumerate(F):
                    before = len(faces)
                    add_panel(verts, faces, [V[i] for i in f],
                              self.gap, self.thickness * R)
                    if keys is not None:
                        # one source face becomes a whole prism
                        keys.extend([fkeys[fi]] * (len(faces) - before))
                add_base_ring(verts, faces, loops, V,
                              self.ring_width * R)
            if keys is not None:
                # the rim band is appended by every branch above and is
                # not part of any scheme, so it gets its own colour
                keys.extend(['rim'] * (len(faces) - len(keys)))

            base_name = "Goldberg" if self.dual else "Geodesic"
            name = ("%s Sphere" % base_name if self.cut == 'FULL'
                    else "%s Dome" % base_name)
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            if keys:
                self._apply_colors(me, keys)
            if smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.style == 'SHELL' and self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness * R
                mod.offset = 0.0
            elif self.style == 'LEONARDO':
                try:
                    from . import leonardo_style
                except ImportError:
                    import leonardo_style
                leonardo_style.add_modifier(obj, self.border,
                                            self.thickness)
            elif self.style == 'WIRE':
                mod = obj.modifiers.new("Wireframe", 'WIREFRAME')
                mod.thickness = self.thickness * R
                mod.use_even_offset = False
            self.report({'INFO'},
                        f"V={len(me.vertices)} E={len(me.edges)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def _apply_colors(self, me, keys):
            """Attach one material per key family and index the faces.

            `me.validate()` has already run and is allowed to drop a
            degenerate polygon, which would slide every later material
            index by one; a mismatch is reported and the colouring
            skipped rather than written misaligned.
            """
            if len(keys) != len(me.polygons):
                self.report({'WARNING'},
                            "colouring skipped: %d keys for %d faces"
                            % (len(keys), len(me.polygons)))
                return
            try:
                from .styles import face_colors
            except ImportError:
                from styles import face_colors
            mats, idx = face_colors.materials_for(keys, "Geodesic")
            for m in mats:
                me.materials.append(m)
            me.polygons.foreach_set('material_index', idx)
            me.update()

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'base')
            lay.prop(self, 'geo_class')
            lay.prop(self, 'frequency')
            if self.geo_class == 'III':
                lay.prop(self, 'k')
            lay.prop(self, 'dual')
            lay.prop(self, 'cut')
            if self.cut != 'FULL':
                lay.prop(self, 'base_ring')
                if self.base_ring:
                    lay.prop(self, 'ring_width')
            lay.prop(self, 'style')
            if self.style in ('SHELL', 'PANELS', 'WIRE'):
                lay.prop(self, 'thickness')
            elif self.style == 'LEONARDO':
                lay.prop(self, 'border')
                lay.prop(self, 'thickness')
            elif self.style == 'STRUTS':
                lay.prop(self, 'strut_radius')
                lay.prop(self, 'node_radius')
            if self.style == 'PANELS':
                lay.prop(self, 'gap')
            # two properties rather than one enum with dynamic items: an
            # items callback stores the item INDEX, so switching style
            # would silently re-point an already-chosen colouring
            if self.style == 'STRUTS':
                lay.prop(self, 'strut_color')
            else:
                lay.prop(self, 'color_by')
            lay.prop(self, 'radius')

    def _menu_func(self, context):
        self.layout.operator("mesh.geodesic_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_geodesic_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_geodesic_add)


def _selftest():
    for f in range(1, 6):
        V, F = build_sphere('ICOSA', f, 'I')
        assert len(V) == 10 * f * f + 2, (f, len(V))
        assert len(F) == 20 * f * f
    for f in range(1, 5):
        V, F = build_sphere('OCTA', f, 'I')
        assert len(V) == 4 * f * f + 2, (f, len(V))
        V, F = build_sphere('ICOSA', f, 'II')
        assert len(V) == 30 * f * f + 2, (f, len(V))
    V, F = build_sphere('ICOSA', 3, 'I')
    for cut, z in (('HEMI', 0.0), ('FIVEEIGHTHS', -0.25)):
        Vc, Fc = cut_faces(V, F, z)
        loops = boundary_loops(Fc)
        zs = sorted({round(Vc[i][2], 6)
                     for lp in loops for i in lp})
        print(f"{cut}: verts={len(Vc)} faces={len(Fc)} "
              f"rim loops={len(loops)} rim z levels={zs}")

    # the cut has to be able to say WHICH faces it kept, or a colouring
    # computed on the closed sphere cannot follow it
    Vk, Fk, keep = cut_faces_keep(V, F, 0.0)
    assert len(keep) == len(Fk), (len(keep), len(Fk))
    assert cut_faces(V, F, 0.0)[1] == Fk         # same faces either way

    # ---- colouring keys --------------------------------------------------
    # Class III is deliberately absent: its faces only exist inside
    # execute(), where bmesh convex-hulls the lattice points, so there is
    # nothing to key here without Blender.

    # exactly twelve pentagons on every Class I icosahedral dual -- the
    # defining property of a Goldberg polyhedron
    for f in (1, 2, 3, 4):
        Vg, Fg = goldberg_dual(*build_sphere('ICOSA', f, 'I'))
        sizes = face_keys(Vg, Fg, 'SIZE', 'ICOSA', dual=True)
        assert sizes.count(5) == 12, (f, sorted(set(sizes)))
        rings = face_keys(Vg, Fg, 'RINGS', 'ICOSA', dual=True)
        assert -1 not in rings, f            # every face reached
        if f == 1:
            # GP(1,0) IS the dodecahedron: all twelve faces pentagons,
            # so none of them is the odd one out and there is nothing
            # for the rings to radiate from
            assert not special_faces(Vg, Fg, True)
            assert set(rings) == {0}, set(rings)
        else:
            assert len(special_faces(Vg, Fg, True)) == 12, f
            assert rings.count(0) == 12, (f, rings.count(0))
            assert max(rings) > 0, f

    # the Class II construction here is a kis split followed by a Class I
    # subdivision, which leaves ten- and three-valent vertices, so its
    # dual has decagons and triangles and NO pentagons.  The schemes key
    # on "not the modal kind" precisely so they still say something.
    Vg, Fg = goldberg_dual(*build_sphere('ICOSA', 1, 'II'))
    sizes = [len(f) for f in Fg]
    assert sorted(set(sizes)) == [3, 10], sorted(set(sizes))
    assert sizes.count(3) == 20 and sizes.count(10) == 12, sizes
    # the twenty triangles are the majority, so the twelve decagons are
    # what stands out -- still twelve faces, still the icosahedron's
    # vertices, just not pentagons
    assert len(special_faces(Vg, Fg, True)) == 12

    # congruence classes: the dodecahedron GP(1,0) is face-transitive, and
    # the chamfered dodecahedron GP(2,0) has one kind of hexagon besides
    # its pentagons.  GP(3,0) genuinely has more than one kind -- the
    # thing the Wikipedia tables never show.
    counts = []
    for f in (1, 2, 3):
        Vg, Fg = goldberg_dual(*build_sphere('ICOSA', f, 'I'))
        cls = congruence_classes(Vg, Fg)
        counts.append(len(set(cls)))
        # a class must be internally consistent: same size, same area
        for c in set(cls):
            sig = [_face_signature(Vg, Fg[i])
                   for i in range(len(Fg)) if cls[i] == c]
            assert len({s[0] for s in sig}) == 1, c
            assert max(s[2] for s in sig) - min(s[2] for s in sig) \
                < CLASS_TOL, c
    assert counts == [1, 2, 3], counts

    # map colouring: proper, and no more colours than the palette holds
    for dual in (False, True):
        Vm, Fm = build_sphere('ICOSA', 3, 'I')
        if dual:
            Vm, Fm = goldberg_dual(Vm, Fm)
        col = face_keys(Vm, Fm, 'MAP', 'ICOSA', dual=dual)
        adj = face_adjacency(Fm)
        for i, nb in adj.items():
            assert all(col[i] != col[j] for j in nb), (dual, i)
        assert len(set(col)) <= 8, (dual, len(set(col)))

    # seed faces: one group per seed face on the triangulated form, where
    # every subdivided face lies strictly inside one of them
    for base, freq, n in (('ICOSA', 3, 20), ('OCTA', 3, 8),
                          ('TETRA', 3, 4), ('CUBE', 3, 6)):
        Vs, Fs = (quad_geodesic(*quad_seed(base), freq)
                  if base in ('CUBE', 'RT')
                  else build_sphere(base, freq, 'I'))
        keys = face_keys(Vs, Fs, 'SEED', base)
        assert len(set(keys)) == n, (base, len(set(keys)))

    # strut lengths: the 3v Class I icosahedral geodesic is the classic
    # three-strut dome (the A/B/C cutting schedule)
    try:
        from .styles import ball_and_stick as _bs
    except ImportError:
        from styles import ball_and_stick as _bs
    V3, F3 = build_sphere('ICOSA', 3, 'I')
    e3 = _bs.edges_from_faces(F3)
    lc = length_classes(V3, e3)
    assert len(set(lc)) == 3, sorted(set(lc))
    assert len(vertex_valences(V3, F3)) == len(V3)
    print("geodesic_generator self-test OK")
