# Evolver-style coarse-to-fine evolution: a port of the solver path of
# Brakke's Surface Evolver, specialised to the linear-constraint Plateau
# problems our `.fe` collection poses before conjugation.
#
# Why this exists: `plateau.minimize_area_sliding` solves at FIXED
# connectivity from a fine structured grid, and on the Schoen starfish
# family that lands in the wrong discrete basin -- the sliding corner
# stalls on a ridge the fixed mesh cannot cross (measured: in-plane
# gradient ~2e-5 at the stall, no descent direction along the valley,
# while a strictly better minimum exists in the same discretization).
# Evolver escapes because its `gg` scripts interleave the descent with
# REMESHING: gradual refinement, Delaunay edge flips and tangential
# vertex averaging, so the coarse solve settles the large-scale shape
# while the mesh is still free to reorganise.  This module reproduces
# that trajectory by running the datafile's own `gg` recipe.
#
# What is ported, and from where (Surface Evolver 2.70 C source):
#   `g N`  iterate() in src/iterate.c: gradient descent on total area
#          with the default "optimizing scale" line search -- try the
#          current scale, double while the energy falls or halve until
#          it does, then quadratic interpolation through three (scale,
#          energy) samples; revert to the last known-decreasing scale
#          if the interpolated move increases energy.  The velocity is
#          the raw area gradient projected, per vertex, onto the
#          tangent space of the vertex's level-set constraints
#          (convert_forms_to_vectors(), src/iterate.c; constr_proj(),
#          src/cnstrnt.c), and every move re-projects each constrained
#          vertex onto its constraints (project_v_constr()).
#   `U`    conjugate gradient toggle: Polak-Ribiere direction update
#          (cg_calc_gamma()/cg_direction(), src/iterate.c), reset
#          whenever the mesh changes or an uphill move is caught
#          (reset_conj_grad(), src/command.c).
#   `r`    local_refine(), src/trirevis.c: subdivide every edge at its
#          midpoint, cutting each triangle into four.  Midpoints
#          inherit the split edge's constraints and fixedness
#          (edge_divide(), src/modify.c); the mid-triangle edges
#          inherit the facet's (none, here).
#   `refine edge where ...`  edge_refine(), src/trirevis.c: split the
#          matching edges only, bisecting the adjacent triangles
#          (cross_cut(), src/modify.c).
#   `u`    equiangulate(), src/trirevis.c: flip any non-fixed interior
#          edge whose opposite angles sum past pi (the Delaunay
#          criterion, written on cosines with a -0.001 guard against
#          cycling), skipping edges whose constraint set differs from
#          their facets' and flips that would duplicate an edge.
#   `V`    vertex_average(VOLKEEP) (vertex_average() /
#          find_vertex_average(), src/veravg.c): move each free vertex
#          toward the area-weighted average of its neighbours, motion
#          projected tangent to the surface (volume-preserving to
#          first order) and tangent to the vertex's constraints; a
#          constrained vertex averages only along edges that share all
#          its constraints, and a corner on two constraints therefore
#          does not move.  All averages are computed before any vertex
#          moves.
#   `hessian`  NOT ported.  In the `gg` scripts it polishes an
#          already-converged shape to the critical point (quadratic
#          convergence); the basin is decided by the refine/flip/
#          average trajectory long before the first `hessian`.  Each
#          occurrence is substituted with a block of conjugate-gradient
#          `g` steps, and the final polish belongs to the caller's own
#          solver (plateau.minimize_area_sliding), which is known to
#          stay put once inside the right basin.
#
# Only linear (plane) level-set constraints are supported; a datafile
# whose relevant constraints do not reduce to planes makes gg_presolve
# return None and the caller falls back to its fixed-grid solve.
#
# References:
# - K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#   (1992), 141-165.
# - K. A. Brakke, Surface Evolver Manual v2.70 (2013), sections on the
#   g, r, u, V and U commands; source files src/iterate.c,
#   src/cnstrnt.c, src/trirevis.c, src/veravg.c, src/modify.c.
# - U. Pinkall and K. Polthier, "Computing discrete minimal surfaces
#   and their conjugates", Experimental Mathematics 2(1) (1993).

import re

import numpy as np


# ======================================================================
# Mesh state
# ======================================================================

class _Web(object):
    """A tiny counterpart of Evolver's web: triangles + attributes.

    Vertices carry a constraint set (ids into `planes`) and a fixed
    flag; edges (keyed by their sorted vertex pair) carry a constraint
    set, a fixed flag and the datafile edge they descend from
    (Evolver's `original`).  Facets carry nothing, which is true of
    every datafile this is used on and is asserted at build time.
    """

    def __init__(self, V, tris, vcons, vfix, eattr, planes):
        self.V = np.asarray(V, dtype=float)
        self.tris = [tuple(t) for t in tris]
        self.vcons = list(vcons)        # frozenset per vertex
        self.vfix = list(vfix)          # bool per vertex
        self.eattr = dict(eattr)        # key -> (cons, fixed, orig)
        self.planes = dict(planes)      # con id -> (vec, off)
        # persistent solver state (Evolver's web.scale and cg_*)
        self.scale = 0.1                # lexinit.c: web.scale = 0.1
        self.maxscale = 1.0             # lexinit.c: web.maxscale = 1.0
        self.conj_grad = False
        self._cg_h = None
        self._cg_oldsum = 0.0
        self._cg_gamma = 0.0
        self._cg_gold = None

    # -- bookkeeping ---------------------------------------------------

    @staticmethod
    def _ekey(a, b):
        return (a, b) if a < b else (b, a)

    def edge_get(self, a, b):
        return self.eattr.get(self._ekey(a, b),
                              (frozenset(), False, 0))

    def T(self):
        return np.asarray(self.tris, dtype=np.int64)

    def area(self, V=None):
        V = self.V if V is None else V
        T = self.T()
        n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        return 0.5 * float(np.sum(np.linalg.norm(n, axis=1)))

    def edge_tris(self):
        """{edge key: [tri indices]} for the current triangle list."""
        et = {}
        for i, (a, b, c) in enumerate(self.tris):
            for p, q in ((a, b), (b, c), (c, a)):
                et.setdefault(self._ekey(p, q), []).append(i)
        return et

    def reset_cg(self):
        """reset_conj_grad(): forget history when the web changes."""
        self._cg_h = None
        self._cg_oldsum = 0.0
        self._cg_gamma = 0.0
        self._cg_gold = None

    # -- constraint projection (constr_proj / project_v_constr) --------

    def _plane_rows(self, cons):
        vecs = [self.planes[n][0] for n in sorted(cons)
                if n in self.planes]
        offs = [self.planes[n][1] for n in sorted(cons)
                if n in self.planes]
        return np.asarray(vecs, float), np.asarray(offs, float)

    def project_point(self, x, cons):
        """project_v_constr(): put a point on its planes (exact)."""
        G, off = self._plane_rows(cons)
        if not len(G):
            return x
        A = G @ G.T
        f = off - G @ x
        try:
            lam = np.linalg.solve(A, f)
        except np.linalg.LinAlgError:
            lam, _r, _rk, _s = np.linalg.lstsq(A, f, rcond=None)
        return x + G.T @ lam

    def project_tangent(self, vec, cons):
        """constr_proj(TANGPROJ): drop the normal component."""
        G, _off = self._plane_rows(cons)
        if not len(G):
            return vec
        A = G @ G.T
        f = G @ vec
        try:
            lam = np.linalg.solve(A, f)
        except np.linalg.LinAlgError:
            lam, _r, _rk, _s = np.linalg.lstsq(A, f, rcond=None)
        return vec - G.T @ lam

    def project_all(self, V):
        """Re-project every constrained vertex after a move."""
        for i, cons in enumerate(self.vcons):
            if cons and not self.vfix[i]:
                V[i] = self.project_point(V[i], cons)
        return V


# ======================================================================
# Area gradient (calc_force: the raw gradient, no preconditioning)
# ======================================================================

def _area_and_grad(V, T):
    P0, P1, P2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(P1 - P0, P2 - P0)
    nn = np.linalg.norm(n, axis=1)
    area = 0.5 * float(np.sum(nn))
    nh = n / np.maximum(nn, 1e-300)[:, None]
    g = np.zeros_like(V)
    np.add.at(g, T[:, 0], 0.5 * np.cross(nh, P2 - P1))
    np.add.at(g, T[:, 1], 0.5 * np.cross(nh, P0 - P2))
    np.add.at(g, T[:, 2], 0.5 * np.cross(nh, P1 - P0))
    return area, g


# ======================================================================
# `g N` -- iterate() with the default optimizing-scale line search
# ======================================================================

def _velocity(web):
    """Force = -grad(area), projected per vertex onto its constraints.

    Mirrors calc_all_grads() + the constraint-projection loop in
    convert_forms_to_vectors(): fixed vertices get zero, constrained
    vertices keep only the tangential part.  Returns (area, force,
    velocity); `force` is the unprojected gradient the conjugate-
    gradient bookkeeping sums against (cg_sum_calc()).
    """
    T = web.T()
    area, g = _area_and_grad(web.V, T)
    force = -g
    for i, cons in enumerate(web.vcons):
        if web.vfix[i]:
            force[i] = 0.0
    vel = force.copy()
    for i, cons in enumerate(web.vcons):
        if cons and not web.vfix[i]:
            vel[i] = web.project_tangent(vel[i], cons)
    return area, force, vel


def _cg_adjust(web, force, vel):
    """cg_calc_gamma() + cg_direction(): Polak-Ribiere update."""
    free = ~np.asarray(web.vfix, bool)
    rsum = 0.0
    if web._cg_gold is not None and web._cg_gold.shape == force.shape:
        rsum = float(np.sum(vel[free] * web._cg_gold[free]))
    web._cg_gold = force.copy()
    s = float(np.sum(vel[free] * force[free]))
    if web._cg_oldsum >= 1e-15:
        web._cg_gamma = (s - rsum) / web._cg_oldsum
    if web._cg_gamma > 10.0:
        web._cg_gamma = 0.0
    if web._cg_gamma < 0.0:        # ribiere_flag: Shewchuk guarantee
        web._cg_gamma = 0.0
    web._cg_oldsum = s
    if web._cg_h is None or web._cg_h.shape != vel.shape:
        web._cg_h = np.zeros_like(vel)
    web._cg_h = vel + web._cg_gamma * web._cg_h
    return web._cg_h.copy()


def _g_step(web):
    """One `g` iteration: iterate() in src/iterate.c, no bodies."""
    energy0, force, vel = _velocity(web)
    if web.conj_grad:
        vel = _cg_adjust(web, force, vel)

    def trial(scale):
        X = web.V + scale * vel
        X = web.project_all(X)
        return web.area(X), X

    # optimizing scale (the !web.motion_flag branch)
    if web.scale > web.maxscale:
        web.scale = web.maxscale
    tempscale = web.scale if web.scale > 0.0 else web.maxscale * 1e-6
    scale0, scale1, scale2 = 0.0, tempscale, 0.0
    energy1, _X = trial(tempscale)
    if not np.isfinite(energy1):
        energy1 = np.inf
    energy2 = 0.0
    web.scale = tempscale
    if energy1 < energy0:
        while True:
            web.scale *= 2.0
            energy2, _X = trial(web.scale)
            scale2 = web.scale
            if not np.isfinite(energy2):
                web.scale /= 2.0
                break
            if energy2 > energy1:
                web.scale = web.scale / 2.0
                break
            energy1 = energy2
            scale1 = scale2
            if not (web.scale < web.maxscale):
                break
    else:
        seekcount = 0
        while True:
            seekcount += 1
            if seekcount > 20:
                web.scale = 0.0
                break
            energy2 = energy1
            scale2 = scale1
            web.scale /= 2.0
            if web.scale < 1e-12 * web.maxscale:
                web.scale = 0.0
                break
            energy1, _X = trial(web.scale)
            if not np.isfinite(energy1):
                energy1 = np.inf
            scale1 = web.scale
            if not (energy1 > energy0):
                break
        web.scale *= 2.0

    if web.scale > web.maxscale:
        web.scale = web.maxscale
    elif web.scale > 0.0:
        # quadratic interpolation for the energy minimum
        denom = (energy0 * (scale1 - scale2) + energy1 * (scale2 - scale0)
                 + energy2 * (scale0 - scale1))
        if denom == 0.0 or not np.isfinite(denom):
            web.scale = 0.0
        else:
            web.scale = ((energy0 - energy2) * scale1 * scale1
                         + (energy1 - energy0) * scale2 * scale2
                         + (energy2 - energy1) * scale0 * scale0) \
                / 2.0 / denom
    if web.scale > web.maxscale:
        web.scale = web.maxscale
    if not np.isfinite(web.scale) or web.scale < 0.0:
        web.scale = 0.0

    e_final, X = trial(web.scale)
    if e_final > energy0:
        # go back and use scale1, known to decrease (or hold at 0)
        web.scale = scale1
        e_final, X = trial(web.scale)
        if web.conj_grad:
            web.reset_cg()
        if e_final > energy0:
            web.scale = 0.0
            return
    web.V = X


def g_iterate(web, n):
    for _ in range(int(n)):
        _g_step(web)


# ======================================================================
# `r` and `refine edge where ...`
# ======================================================================

def _split_edge(web, key, mids):
    """Midpoint vertex for edge `key`, per edge_divide()."""
    if key in mids:
        return mids[key]
    a, b = key
    cons, efix, orig = web.eattr.get(key, (frozenset(), False, 0))
    m = len(web.vcons)
    mid = 0.5 * (web.V[a] + web.V[b])
    if cons:
        mid = web.project_point(mid, cons)
    web.V = np.vstack([web.V, mid[None, :]])
    web.vcons.append(frozenset(cons))
    web.vfix.append(bool(efix))
    # the two halves keep the parent edge's attributes
    del_attr = web.eattr.pop(key, None)
    if del_attr is not None:
        web.eattr[web._ekey(a, m)] = del_attr
        web.eattr[web._ekey(m, b)] = del_attr
    mids[key] = m
    return m


def refine_all(web):
    """`r`: every edge split, every triangle cut into four."""
    old = list(web.tris)
    mids = {}
    new_tris = []
    for (a, b, c) in old:
        mab = _split_edge(web, web._ekey(a, b), mids)
        mbc = _split_edge(web, web._ekey(b, c), mids)
        mca = _split_edge(web, web._ekey(c, a), mids)
        new_tris += [(a, mab, mca), (mab, b, mbc),
                     (mca, mbc, c), (mab, mbc, mca)]
    web.tris = new_tris
    web.reset_cg()


def refine_edges(web, keys):
    """`refine edge where ...`: edge_refine() on the listed edges."""
    mids = {}
    keys = [k for k in keys]
    split = {}
    for key in keys:
        split[key] = _split_edge(web, key, mids)
    new_tris = []
    for (a, b, c) in web.tris:
        cut = [(p, q, r) for (p, q, r) in ((a, b, c), (b, c, a), (c, a, b))
               if web._ekey(p, q) in split]
        if not cut:
            new_tris.append((a, b, c))
            continue
        # bisect on each split side in turn (cross_cut per edge)
        polys = [(a, b, c)]
        for key, m in split.items():
            nxt = []
            for tri in polys:
                placed = False
                for (p, q, r) in ((tri[0], tri[1], tri[2]),
                                  (tri[1], tri[2], tri[0]),
                                  (tri[2], tri[0], tri[1])):
                    if web._ekey(p, q) == key:
                        nxt += [(p, m, r), (m, q, r)]
                        placed = True
                        break
                if not placed:
                    nxt.append(tri)
            polys = nxt
        new_tris += polys
    web.tris = new_tris
    web.reset_cg()


# ======================================================================
# `u` -- equiangulate()
# ======================================================================

def equiangulate(web):
    et = web.edge_tris()
    tris = [list(t) for t in web.tris]
    flips = 0
    for key in list(et.keys()):
        adj = et.get(key)
        if adj is None or len(adj) != 2:
            continue
        cons, efix, _orig = web.eattr.get(key, (frozenset(), False, 0))
        if efix:
            continue
        if cons:
            # equal_constr(edge, facet): facets carry no constraints
            continue
        t1, t2 = adj
        a, b = key
        tri1, tri2 = tris[t1], tris[t2]
        if a not in tri1 or b not in tri1 or a not in tri2 or b not in tri2:
            continue  # stale entry from an earlier flip
        p = [v for v in tri1 if v != a and v != b][0]
        q = [v for v in tri2 if v != a and v != b][0]
        if p == q:
            continue
        if web._ekey(p, q) in et:
            continue  # would duplicate an existing edge
        A, B, P, Q = web.V[a], web.V[b], web.V[p], web.V[q]
        la = float(np.linalg.norm(A - B))
        lb = float(np.linalg.norm(B - P))
        lc = float(np.linalg.norm(P - A))
        ld = float(np.linalg.norm(A - Q))
        le = float(np.linalg.norm(Q - B))
        if lb * lc == 0.0 or ld * le == 0.0:
            continue
        if (lb * lb + lc * lc - la * la) / lb / lc \
                + (ld * ld + le * le - la * la) / ld / le > -0.001:
            continue
        # orientation-preserving flip: in tri1, a->b is a directed
        # edge; the two new triangles are (p,q,b') per do_edgeswap().
        i1 = tri1.index(a)
        if tri1[(i1 + 1) % 3] == b:
            ta, tb = a, b
        else:
            ta, tb = b, a
        tris[t1] = [q, p, ta]
        tris[t2] = [p, q, tb]
        del et[key]
        et[web._ekey(p, q)] = [t1, t2]
        # the four side edges may have changed which of t1/t2 they touch
        for kk in (web._ekey(ta, p), web._ekey(q, ta),
                   web._ekey(p, tb), web._ekey(tb, q)):
            lst = et.get(kk, [])
            keep = [x for x in lst if x not in (t1, t2)]
            for tt in (t1, t2):
                vv = tris[tt]
                if kk[0] in vv and kk[1] in vv:
                    keep.append(tt)
            et[kk] = keep
        old_attr = web.eattr.pop(key, None)
        if old_attr is not None:
            web.eattr[web._ekey(p, q)] = old_attr
        flips += 1
    web.tris = [tuple(t) for t in tris]
    web.reset_cg()
    return flips


# ======================================================================
# `V` -- vertex_average(VOLKEEP)
# ======================================================================

def vertex_average(web):
    et = web.edge_tris()
    T = web.T()
    P0, P1, P2 = web.V[T[:, 0]], web.V[T[:, 1]], web.V[T[:, 2]]
    fn = np.cross(P1 - P0, P2 - P0)         # 2*area*normal per facet
    farea = 0.5 * np.linalg.norm(fn, axis=1)
    # incident edges per vertex
    nbrs = [[] for _ in range(len(web.vcons))]
    for key in et:
        a, b = key
        nbrs[a].append(b)
        nbrs[b].append(a)
    # area-weighted vertex normals (new_calc_vertex_normal)
    vnorm = np.zeros_like(web.V)
    np.add.at(vnorm, T[:, 0], fn)
    np.add.at(vnorm, T[:, 1], fn)
    np.add.at(vnorm, T[:, 2], fn)
    newV = web.V.copy()
    for v in range(len(web.vcons)):
        if web.vfix[v]:
            continue
        vcons = web.vcons[v]
        kept = []
        single = False
        for nb in nbrs[v]:
            key = web._ekey(v, nb)
            econs, _efix, _orig = web.eattr.get(key,
                                                (frozenset(), False, 0))
            if vcons and not (vcons <= econs):
                continue                    # constraint compatibility
            fts = et[key]
            kept.append((nb, fts))
            if len(fts) == 1:
                single = True
        if single:
            kept = [(nb, fts) for nb, fts in kept if len(fts) == 1]
        if len(kept) <= 1:
            continue
        xsum = np.zeros(3)
        total = 0.0
        for nb, fts in kept:
            w = 1.0 if single else float(sum(farea[f] for f in fts))
            if w == 0.0:
                continue
            xsum += (w * w) * (web.V[nb] - web.V[v])
            total += w * w
        if total <= 0.0:
            continue
        if vcons:
            xsum = web.project_tangent(xsum, vcons)
        # VOLKEEP: strip the normal component so volume is kept
        n = vnorm[v]
        nn = float(np.linalg.norm(n))
        if nn > 1e-300:
            nh = n / nn
            xsum = xsum - float(xsum @ nh) * nh
        else:
            xsum[:] = 0.0
        if vcons:
            xsum = web.project_tangent(xsum, vcons)
        newx = web.V[v] + 0.25 * xsum / total
        if vcons:
            newx = web.project_point(newx, vcons)
        newV[v] = newx
    web.V = newV
    web.reset_cg()


# ======================================================================
# Building the web from a datafile, and running its `gg`
# ======================================================================

def build_web(fe):
    """Evolver's load of the datafile: fan polygonal faces from their
    centroid (verified against the binary: a pentagon face becomes a
    centroid vertex plus five triangles, the spokes carrying no
    constraints and `original` 0).  Returns None where the port does
    not apply (nonlinear constraints, missing geometry).
    """
    if not fe.vertices or not fe.edges or not fe.faces:
        return None
    vids = sorted(fe.vertices)
    vrow = {vid: i for i, vid in enumerate(vids)}
    V = [np.asarray(fe.vertices[vid], float) for vid in vids]
    vcons = [frozenset(fe.vertex_constraints.get(vid, ()))
             for vid in vids]
    vfix = [vid in fe.vertex_fixed for vid in vids]
    eattr = {}
    need = set()
    for vid in vids:
        need |= set(fe.vertex_constraints.get(vid, ()))
    for eid, (a, b) in fe.edges.items():
        if a not in vrow or b not in vrow:
            return None
        key = _Web._ekey(vrow[a], vrow[b])
        cons = frozenset(fe.edge_constraints.get(eid, ()))
        need |= set(cons)
        eattr[key] = (cons, eid in fe.edge_fixed, eid)
    planes = {}
    for n in sorted(need):
        pl = fe.constraint_plane(n, fe.params)
        if pl is None:
            return None
        vec, const, name = pl
        if name is None:
            off = float(const)
        elif name in fe.params:
            off = float(fe.params[name]) + float(const)
        else:
            return None
        planes[n] = (np.asarray(vec, float), off)
    tris = []
    for face in fe.faces:
        loop = []
        ok = True
        for se in face:
            e = fe.edges.get(abs(se))
            if e is None:
                ok = False
                break
            a, b = (e if se > 0 else (e[1], e[0]))
            loop.append(vrow[a])
        if not ok or len(loop) < 3:
            return None
        if len(loop) == 3:
            tris.append(tuple(loop))
        else:
            cen = len(V)
            V.append(np.mean([V[i] for i in loop], axis=0))
            vcons.append(frozenset())
            vfix.append(False)
            for k in range(len(loop)):
                tris.append((loop[k], loop[(k + 1) % len(loop)], cen))
    web = _Web(np.asarray(V, float), tris, vcons, vfix, eattr, planes)
    # start exactly on the constraints, as Evolver does at load
    web.V = web.project_all(web.V)
    web._vrow = vrow
    return web


def _parse_gg(body):
    """The `gg` recipe as a list of ops, or None if something in it is
    beyond the port.  Ops: ('g', n), ('r',), ('u',), ('V',), ('U',),
    ('hessian',), ('refine', predicate-source).
    """
    src = re.sub(r'//[^\n]*', '', body)
    ops = []
    for stmt in src.split(';'):
        s = stmt.strip()
        if not s:
            continue
        m = re.fullmatch(r'g\s+(\d+)', s)
        if m:
            ops.append(('g', int(m.group(1))))
            continue
        if s == 'g':
            ops.append(('g', 1))
            continue
        if s in ('r', 'u', 'V', 'U', 'hessian'):
            ops.append((s,))
            continue
        m = re.fullmatch(r'refine\s+edge\s+where\s+(.+)', s, re.S)
        if m:
            ops.append(('refine', m.group(1).strip()))
            continue
        return None
    return ops


def _edge_predicate(web, cond):
    """Edge keys matching a `refine edge where ...` condition.

    Understands the little that the collection's gg recipes use:
    valence == K, on_constraint N, original == K, length > X, joined
    with `or` / `and`, plus `not no_refine` (always true here).
    Returns None on anything else.
    """
    et = web.edge_tris()

    def term_fn(term):
        term = term.strip()
        m = re.fullmatch(r'valence\s*==\s*(\d+)', term)
        if m:
            k = int(m.group(1))
            return lambda key: len(et.get(key, ())) == k
        m = re.fullmatch(r'on_constraint\s+(\d+)', term)
        if m:
            n = int(m.group(1))
            return lambda key: n in web.eattr.get(
                key, (frozenset(), False, 0))[0]
        m = re.fullmatch(r'original\s*==\s*(\d+)', term)
        if m:
            k = int(m.group(1))
            return lambda key: web.eattr.get(
                key, (frozenset(), False, 0))[2] == k
        m = re.fullmatch(r'length\s*>\s*([0-9.eE+-]+)', term)
        if m:
            x = float(m.group(1))
            return lambda key: float(np.linalg.norm(
                web.V[key[0]] - web.V[key[1]])) > x
        if term in ('not no_refine', '(not no_refine)'):
            return lambda key: True
        return None

    def or_fn(src):
        fns = []
        for part in re.split(r'\bor\b', src):
            sub = []
            for term in re.split(r'\band\b', part):
                fn = term_fn(term)
                if fn is None:
                    return None
                sub.append(fn)
            fns.append(sub)
        return lambda key: any(all(f(key) for f in sub) for sub in fns)

    fn = or_fn(cond)
    if fn is None:
        return None
    keys = set(web.eattr.keys())
    for key in et:
        keys.add(key)
    return [key for key in sorted(keys) if fn(key)]


def gg_presolve(fe, hessian_g=30, max_tris=40000):
    """Run the datafile's own `gg` evolution with the ported solver.

    Returns the evolved `_Web`, or None when the file has no `gg`, the
    recipe contains something unported, or the mesh grows past
    `max_tris` (a guard, not a tuning knob: the starfish recipes top
    out near 3000 triangles).
    """
    body = fe.command('gg')
    if body is None:
        return None
    ops = _parse_gg(body)
    if ops is None:
        return None
    web = build_web(fe)
    if web is None:
        return None
    for op in ops:
        if len(web.tris) > max_tris:
            return None
        if op[0] == 'g':
            g_iterate(web, op[1])
        elif op[0] == 'r':
            refine_all(web)
        elif op[0] == 'u':
            equiangulate(web)
        elif op[0] == 'V':
            vertex_average(web)
        elif op[0] == 'U':
            web.conj_grad = not web.conj_grad
            web.reset_cg()
        elif op[0] == 'hessian':
            # substituted: see the header note
            keep = web.conj_grad
            web.conj_grad = True
            g_iterate(web, hessian_g)
            web.conj_grad = keep
        elif op[0] == 'refine':
            keys = _edge_predicate(web, op[1])
            if keys is None:
                return None
            refine_edges(web, keys)
    if not np.all(np.isfinite(web.V)):
        return None
    return web


def gg_arc_positions(fe, hessian_g=30):
    """The solved positions of the datafile's boundary, from its `gg`.

    Returns `(arcs, corners)` or None: `arcs[eid]` is an ordered
    polyline running from the datafile edge's tail vertex to its head,
    following the evolved shape of every mesh edge descended from
    datafile edge `eid`; `corners[vid]` is the evolved position of
    datafile vertex `vid`.  These are exactly the boundary data a
    fixed-grid solve needs as a warm start to land in the same basin.
    """
    web = gg_presolve(fe, hessian_g=hessian_g)
    if web is None:
        return None
    vrow = web._vrow
    corners = {vid: np.array(web.V[row]) for vid, row in vrow.items()}
    arcs = {}
    for eid, (va, vb) in fe.edges.items():
        keys = [k for k, (_c, _f, orig) in web.eattr.items()
                if orig == eid]
        if not keys:
            continue
        # chain the descendant edges from va to vb
        adj = {}
        for a, b in keys:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        start, goal = vrow.get(va), vrow.get(vb)
        if start not in adj or goal not in adj:
            continue
        chain = [start]
        prev = None
        ok = True
        while chain[-1] != goal:
            nxt = [x for x in adj.get(chain[-1], ()) if x != prev]
            if len(nxt) != 1:
                ok = False
                break
            prev = chain[-1]
            chain.append(nxt[0])
            if len(chain) > len(keys) + 2:
                ok = False
                break
        if not ok:
            continue
        arcs[eid] = np.asarray([web.V[i] for i in chain], float)
    return arcs, corners


# ======================================================================
# self-test
# ======================================================================

def _selftest():
    rng = np.random.default_rng(7)

    # 1. area gradient matches finite differences
    V = rng.normal(size=(6, 3))
    T = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])
    a0, g = _area_and_grad(V, T)
    h = 1e-6
    for i in (1, 2, 3):
        for k in range(3):
            Vp = V.copy()
            Vp[i, k] += h
            ap, _g = _area_and_grad(Vp, T)
            num = (ap - a0) / h
            assert abs(num - g[i, k]) < 1e-5, (i, k, num, g[i, k])

    # 2. a pinned tent descends to the flat minimum under `g`
    m = 8
    th = 2.0 * np.pi * np.arange(m) / m
    rim = np.stack([np.cos(th), np.sin(th), np.zeros(m)], axis=1)
    V = np.vstack([rim, [[0.0, 0.0, 0.8]]])
    tris = [(i, (i + 1) % m, m) for i in range(m)]
    eattr = {}
    for (a, b, c) in tris:
        for p, q in ((a, b), (b, c), (c, a)):
            eattr[_Web._ekey(p, q)] = (frozenset(), False, 0)
    web = _Web(V, tris, [frozenset()] * (m + 1),
               [True] * m + [False], eattr, {})
    g_iterate(web, 25)
    assert abs(web.V[m, 2]) < 1e-3, web.V[m]
    assert web.area() < np.pi * 1.01

    # 3. a constrained vertex slides in its plane and stays on it
    web2 = _Web(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                          [1.0, 2.0, 1.0]]),
                [(0, 1, 2)],
                [frozenset(), frozenset(), frozenset({1})],
                [True, True, False],
                {_Web._ekey(0, 2): (frozenset({1}), False, 0),
                 _Web._ekey(1, 2): (frozenset({1}), False, 0),
                 _Web._ekey(0, 1): (frozenset(), True, 0)},
                {1: (np.array([0.0, 0.0, 1.0]), 1.0)})
    web2.V = web2.project_all(web2.V)
    g_iterate(web2, 40)
    assert abs(web2.V[2, 2] - 1.0) < 1e-12       # stays on z = 1
    # minimum area at fixed z: apex above the base midpoint
    assert abs(web2.V[2, 0] - 1.0) < 1e-3
    assert abs(web2.V[2, 1]) < 1e-3

    # 4. refine + equiangulate keep a coherent mesh
    web3 = _Web(V.copy(), list(tris), [frozenset()] * (m + 1),
                [True] * m + [False], dict(eattr), {})
    refine_all(web3)
    assert len(web3.tris) == 4 * m
    T3 = web3.T()
    e_cnt = {}
    for t in web3.tris:
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = _Web._ekey(p, q)
            e_cnt[k] = e_cnt.get(k, 0) + 1
    assert max(e_cnt.values()) <= 2
    nv = len(web3.vcons)
    ne = len(e_cnt)
    nf = len(web3.tris)
    assert nv - ne + nf == 1                     # disk Euler number
    equiangulate(web3)
    e_cnt2 = {}
    for t in web3.tris:
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = _Web._ekey(p, q)
            e_cnt2[k] = e_cnt2.get(k, 0) + 1
    assert max(e_cnt2.values()) <= 2
    assert len(web3.tris) == 4 * m
    vertex_average(web3)
    assert np.all(np.isfinite(web3.V))

    return "evolve OK"
