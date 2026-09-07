
# Weierstrass-Enneper / Bjorling integration engine for the Math Art
# minimal-surface catalog.
#
# Numpy-only (no bpy): the generic machinery that turns a small data
# "spec" (Gauss map g, height differential dh, a domain, a couple of
# parameter hooks) into a surface builder matching the PARAMETRIC /
# MESH_PARAM contracts.  The data tables themselves live in `zoo.py`;
# the meshing pipeline is `parametric.py`, the elliptic-function engine
# `elliptic.py`, the shared mesh utilities `domain.py`, and the
# registered Blender operators the flat `minimal_surface_toolkit.py`.
#
#   X = Re [ e^{i theta} * Int (phi1, phi2, phi3) dz ],
#   phi1 = (1/2)(1/g - g) h,  phi2 = (i/2)(1/g + g) h,  phi3 = h,
#   where dh = h(z) dz  (Weierstrass 1866, Enneper 1864).
#
# Bjorling's problem (Schwarz's formula, 1890):
#   X(w) = Re [ e^{i theta} ( c(w) - i Int_{w0}^{w} n x c' dw~ ) ]
# solved numerically by evaluating the (analytic) curve and normal at
# complex arguments and integrating column-wise from the real axis.
#
# References:
#   K. Weierstrass (1866), A. Enneper (1864); H. A. Schwarz,
#   Gesammelte Mathematische Abhandlungen (1890) for the Bjorling
#   formula; U. Dierkes, S. Hildebrandt, F. Sauvigny, "Minimal
#   Surfaces" (2010); M. Weber, "Classical Minimal Surfaces in
#   Euclidean Space by Examples" and https://minimalsurfaces.blog/
#   for the modern data-driven presentation this engine follows.

import math
import numpy as np

from .domain import (_center_fit, _circularize_outer, _largest_component,
                     _puncture_mask, _smooth_boundary, _torus_grid)
from .elliptic import _SQUARE, _Lattice

TAU = 2.0 * math.pi

# How far the radial node grading is pushed toward its pure form.
# 1.0 is the pure grading, whose spacing collapses to zero at the
# clustered end and produces sliver quads with unstable normals; 0.0
# is a plain linear grid with no clustering at all. 0.75 keeps most
# of the clustering while bounding the worst gap ratio -- see the
# blend in the disk grid builder.
GRADE_BLEND = 0.75

# Per-build boolean grid (same shape as the last disk grid) marking
# vertices that must be EXEMPT from the mesher's boundary smoothing: the
# bell-mouth neighbourhoods of masked end punctures.  Their rim vertices
# are snapped onto the exact conformal circle |z - z_k| = eps_k, so they
# are already analytically placed -- and the surrounding cells sit on a
# steep 1/d flare where even the non-shrinking Taubin pair displaces the
# loop enough to fold high-aspect neighbour quads inside out (measured:
# every smoothing-induced flipped-normal edge in the k-noid family sat on
# a mouth ring).  Reset by the mesher before each build; set by _we_disk
# when it masks declared ends.
LAST_PROTECT = None

# diagnostic hook: set to a list to capture per-mouth rim-walk data
# (grid indices, walk angles, ramp targets) from _we_disk
RIM_DEBUG = None


def _ev(x, p):
    """Evaluate a spec field: plain value or callable(p)."""
    return x(p) if callable(x) else x


# --------------------------------------------------------------------------
# Shared numeric helpers
# --------------------------------------------------------------------------

def period_integral(f, center, rx, ry, n=4096):
    """Contour integral of f around the ellipse center + rx cos t
    + i ry sin t (midpoint rule -- spectrally accurate for analytic f)."""
    t = (np.arange(n) + 0.5) * (TAU / n)
    z = center + rx * np.cos(t) + 1j * ry * np.sin(t)
    dz = (-rx * np.sin(t) + 1j * ry * np.cos(t)) * (TAU / n)
    return np.sum(f(z) * dz)


_SOLVE_CACHE = {}


def solve_scalar(residual, lo, hi, tol=1e-13, iters=200, key=None):
    """1-parameter period solver: bisection root of `residual` on
    [lo, hi] (residual must change sign). Cached by `key`."""
    if key is not None and key in _SOLVE_CACHE:
        return _SOLVE_CACHE[key]
    flo, fhi = residual(lo), residual(hi)
    if flo == 0.0:
        x = lo
    elif fhi == 0.0:
        x = hi
    else:
        if flo * fhi > 0:
            raise ValueError("solve_scalar: no sign change on bracket")
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            fm = residual(mid)
            if fm == 0.0 or (hi - lo) < tol:
                break
            if flo * fm < 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        x = 0.5 * (lo + hi)
    if key is not None:
        _SOLVE_CACHE[key] = x
    return x


def _phi_fn(spec, p, theta):
    """Callable z -> (nu, nv, 3) complex phi-stack from the spec's
    Weierstrass data ('phi' triple, or 'g' + 'dh')."""
    rot = np.exp(1j * theta)
    if 'phi' in spec:
        fn = spec['phi']

        def phi(z):
            f1, f2, f3 = fn(z, p)
            return np.stack(np.broadcast_arrays(f1, f2, f3),
                            axis=-1) * rot
        return phi
    g, dh = spec['g'], spec['dh']

    def phi(z):
        gv = g(z, p)
        hv = dh(z, p)
        f1 = 0.5 * (1.0 / gv - gv) * hv
        f2 = 0.5j * (1.0 / gv + gv) * hv
        return np.stack(np.broadcast_arrays(f1, f2, hv), axis=-1) * rot
    return phi


# --------------------------------------------------------------------------
# Domain integrators
# --------------------------------------------------------------------------

def _cluster_nodes(lo, hi, n, centers, widths, gains, wrap=False,
                   endpoint=True, fine=4096):
    """Monotone 1-D node distribution on [lo, hi] with extra density
    near each of `centers` (Cauchy bumps of half-width `widths` and
    strength `gains`), built by inverting the density CDF at uniform
    quantiles.  `wrap` treats the interval as periodic (angular grids).
    Reduces to a uniform grid when `centers` is empty."""
    t = np.linspace(lo, hi, fine + 1)
    w = np.ones_like(t)
    per = hi - lo
    for c, h, g in zip(centers, widths, gains):
        d = np.abs(t - c)
        if wrap:
            d = np.abs((t - c + 0.5 * per) % per - 0.5 * per)
        w += g / (1.0 + (d / max(h, 1e-9)) ** 2)
    cdf = np.concatenate([[0.0],
                          np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    cdf /= cdf[-1]
    q = (np.linspace(0.0, 1.0, n) if endpoint
         else (np.arange(n) + 0.5) / n)
    return np.interp(q, cdf, t)


def _end_principal_parts(phi, ends, max_order=2):
    """Laurent principal parts of the phi-stack at each declared end
    puncture: per end z_k, the coefficients C_m of C_m/(z - z_k)^m for
    m = 1..max_order, for each of the three components, measured by
    contour integration.  Catenoid ends are at most double poles (the
    default); the winding Enneper ends of the Enneper-k-noid family put
    poles of ORDER FOUR in dh, so those rows declare
    'end_pole_order': 4 -- leaving orders 3-4 unsubtracted would let
    each ray's quadrature keep a different near-pole error and tear
    the mesh into wedges, exactly the failure the subtraction exists
    to prevent.  A lower-order pole just yields ~0 for the unused
    coefficients.  Returns a list of (z_k, C (max_order, 3)) with
    C[m - 1] the order-m coefficient."""
    zs = [zc for zc, _ in ends if zc is not None]
    out = []
    for k, zc in enumerate(zs):
        others = [abs(zc - zo) for j, zo in enumerate(zs)
                  if j != k and abs(zo - zc) > 1e-12]
        rr = 0.3 * min(others) if others else 0.3 * max(abs(zc), 1.0)
        C = np.zeros((max_order, 3), dtype=complex)
        for m in range(1, max_order + 1):
            for c in range(3):
                C[m - 1, c] = period_integral(
                    lambda z, c=c, m=m: phi(z)[..., c]
                    * (z - zc) ** (m - 1),
                    zc, rr, rr) / (2j * math.pi)
        out.append((zc, C))
    return out


def _pp_sub(F, d, C):
    """F minus the principal part sum_m C[m-1]/d^m (d carries a
    trailing component axis broadcasting against C's rows)."""
    dp = d
    for m in range(1, len(C) + 1):
        F = F - C[m - 1] / dp
        dp = dp * d
    return F


def _pp_anti(d, C):
    """Elementary antiderivative of the principal part:
    C_1 Log(d) + sum_{m>=2} C_m d^(1-m)/(1-m).  Because every real
    period is closed, each residue C_1 is real, so Re(C_1 Log d) =
    C_1 ln|d| is single-valued and the immersion has no branch
    seams."""
    S = C[0] * np.log(d)
    dp = d
    for m in range(2, len(C) + 1):
        S = S + C[m - 1] / ((1 - m) * dp)
        dp = dp * d
    return S


def _end_graded_axes(ra, r1, nu, nv, off, ends):
    """Radial and angular node distributions for a disk/annulus domain
    whose declared end punctures sit on interior rings: cluster the
    radial nodes toward each ring radius and the angular nodes toward
    each end azimuth, so the catenoid bells around the punctures get
    real mesh support instead of one or two stretched quads.  Nodes that
    land exactly on a puncture centre are nudged off it."""
    rc, rw, rg, tc, tw, tg = [], [], [], [], [], []
    for zc, eps in ends:
        if zc is None:
            # axial end at z = infinity, cut by the outer rim: cluster
            # the radial nodes toward r1 (mirror of the z = 0 case)
            rc.append(r1)
            rw.append(0.4 * r1)
            rg.append(7.0)
            continue
        R = abs(zc)
        if R < ra:
            # axial end at/near z = 0, cut by the inner rim: cluster the
            # radial nodes toward ra (log-like coverage of the 1/r flare)
            rc.append(ra)
            rw.append(2.0 * max(ra, 1e-3))
            rg.append(7.0)
            continue
        if eps <= 0.0 or R > r1:
            continue
        # adaptive gain: aim for a local spacing of ~eps/4 at the
        # puncture (so its mask spans several cells and the bell mouth
        # is a polygon, not a triangle), whatever eps the row asked
        # for.  Tighter masks get stronger clustering; the total node
        # budget spent per bell stays roughly constant because
        # gain x width is then resolution-independent.
        du0 = (r1 - ra) / max(nu - 1, 1)
        dt0 = TAU / nv
        rc.append(R)
        rw.append(max(2.0 * eps, 0.01 * (r1 - ra)))
        rg.append(min(max(4.0 * du0 / max(eps, 1e-9) - 1.0, 4.0), 30.0))
        tc.append(float(np.angle(zc)) % TAU)
        tw.append(2.0 * eps / R)
        tg.append(min(max(4.0 * dt0 * R / max(eps, 1e-9) - 1.0, 4.0),
                      30.0))
    # dedupe radial ring centres (a ring of nn ends is one radius),
    # remembering the smallest mask eps on each ring
    ring_eps = {}
    for zc, eps in ends:
        if zc is None or eps <= 0.0:
            continue
        key = round(abs(zc), 9)
        ring_eps[key] = min(eps, ring_eps.get(key, eps))
    seen = {}
    for c, w, g in zip(rc, rw, rg):
        key = round(c, 9)
        seen[key] = (c, max(w, seen[key][1]) if key in seen else w, g)
    rcd = [v[0] for v in seen.values()]
    rwd = [v[1] for v in seen.values()]
    rgd = [v[2] for v in seen.values()]
    # Corrective passes: the one-shot gain guess above competes with the
    # other rings' bumps for the shared node budget, so the delivered
    # spacing at a ring can land well above the eps/4 aim -- the engine's
    # mask floor (1.5 x local spacing) then OVERRIDES the row's eps and
    # the bell mouths coarsen into sliver cells with unstable normals.
    # Measure the actual spacing at each masked ring and scale that
    # ring's gain until the local step is <= eps/4 -- the mask floor
    # (1.5 x spacing) then sits well under the declared eps, and the
    # mouth circle spans several radial cells (the bell rim is an
    # inversion image, so a mouth crossed by only 1-2 radial cells
    # yields warped quads with flipped normals) -- within a hard cap.
    u = _cluster_nodes(ra, r1, nu, rcd, rwd, rgd)
    for _ in range(4):
        du_m = np.gradient(u)
        worst = 1.0
        for k, c in enumerate(rcd):
            eps = ring_eps.get(round(c, 9), 0.0)
            if eps <= 0.0:
                continue
            i = int(np.clip(np.searchsorted(u, c), 1, len(u) - 2))
            need = float(du_m[i]) / (0.25 * eps)
            if need > 1.05 and rgd[k] < 150.0:
                rgd[k] = min(rgd[k] * min(need, 4.0), 150.0)
                worst = max(worst, need)
        if worst <= 1.05:
            break
        u = _cluster_nodes(ra, r1, nu, rcd, rwd, rgd)
    v = _cluster_nodes(off, off + TAU, nv, tc, tw, tg, wrap=True,
                       endpoint=False) % TAU
    v.sort()
    # Same corrective passes for the angular axis: deliver a local
    # angular step <= eps/(4 R) at each end azimuth, so a bell mouth is
    # crossed by several angular cells too (the flip census on the
    # symmetrized-Riemann family showed mouth cells starved in ANGLE at
    # high symmetry m -- 2m mouths share one fixed angular budget).
    tgt = {}
    for zc, eps in ends:
        if zc is None or eps <= 0.0 or abs(zc) < ra or abs(zc) > r1:
            continue
        th = float(np.angle(zc)) % TAU
        tgt[round(th, 9)] = min(0.25 * eps / abs(zc),
                                tgt.get(round(th, 9), np.inf))
    if tgt:
        for _ in range(4):
            dv_m = np.gradient(np.unwrap(v))
            worst = 1.0
            for k, c in enumerate(tc):
                t = tgt.get(round(c, 9))
                if not t or not np.isfinite(t):
                    continue
                j = int(np.argmin(np.abs((v - c + math.pi) % TAU
                                         - math.pi)))
                need = float(dv_m[j]) / t
                if need > 1.05 and tg[k] < 150.0:
                    tg[k] = min(tg[k] * min(need, 4.0), 150.0)
                    worst = max(worst, need)
            if worst <= 1.05:
                break
            v = _cluster_nodes(off, off + TAU, nv, tc, tw, tg, wrap=True,
                               endpoint=False) % TAU
            v.sort()
    # nudge any node off an exact puncture centre (radially/angularly)
    for zc, eps in ends:
        if zc is None:
            continue
        R, th = abs(zc), float(np.angle(zc)) % TAU
        j = np.abs(u - R) < 1e-9 * max(r1, 1.0)
        u[j] += 1e-4 * (r1 - ra)
        j = np.abs(v - th) < 1e-9
        v[j] += 1e-4 * TAU / nv
    return u, v


def _metric_axes(spec, p, theta, ra, r1, nu, nv, off, ends):
    """Radial and angular node distributions equidistributing the pulled
    back surface METRIC instead of a parameter-space proxy.

    A Weierstrass immersion is conformal: ds = lambda(z) |dz| with
    lambda = (|g| + 1/|g|) |dh| / 2 = sqrt(sum_k |phi_k|^2 / 2), so the
    local face AREA element is lambda^2 r dr dtheta.  Sampling that
    density on a fine polar grid (masked at the declared end punctures,
    where it diverges -- that divergence is exactly what earns the ends
    their nodes) and inverting its two marginals' CDFs puts grid lines
    where the surface actually has area: both end rings of an
    antiprismatic k-noid get equal budgets regardless of whether they
    sit at |z| = b or 1/b, which no r-space bump heuristic manages.
    The marginals are blended with a uniform floor so spacing can never
    collapse (the GRADE_BLEND lesson), and the density is ceiling-capped
    at a high percentile so a near-pole sample cannot swallow the whole
    budget.  Returns (u, v) or None when the density is unusable
    (caller falls back to the Cauchy-bump axes)."""
    NF, BLEND, CAP_PCT = 384, 0.95, 99.5
    rf = np.exp(np.linspace(math.log(max(ra, 1e-4)), math.log(r1), NF))
    tf = off + (np.arange(NF) + 0.5) * (TAU / NF)
    Rf, Tf = np.meshgrid(rf, tf, indexing='ij')
    zf = Rf * np.exp(1j * Tf)
    try:
        phi = _phi_fn(spec, p, theta)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            F = phi(zf)
            lam2 = 0.5 * np.sum(np.abs(F) ** 2, axis=-1)
    except Exception:
        return None
    valid = np.isfinite(lam2)
    for zc, eps in ends:
        if zc is None or eps <= 0.0:
            continue
        valid &= np.abs(zf - zc) > eps
    if valid.sum() < 0.5 * lam2.size:
        return None
    dens = np.where(valid, lam2 * Rf, 0.0)      # area density lambda^2 r
    vv = dens[valid & (dens > 0.0)]
    if not len(vv):
        return None
    dens = np.minimum(dens, np.percentile(vv, CAP_PCT))
    # radial marginal (area per fine annulus row), uniform-floored
    q = dens.sum(axis=1)
    if q.sum() <= 0.0 or not np.all(np.isfinite(q)):
        return None
    q = (1.0 - BLEND) * q.mean() + BLEND * q

    def _invert_r(qd):
        # hard uniform floor (independent of any corrective boosts
        # below): no region's density may fall under 30% of the average,
        # so the widest spacing stays within ~3x uniform and a boosted
        # ring band can never drain the rest of the axis
        tot = float(np.sum(0.5 * (qd[1:] + qd[:-1]) * np.diff(rf)))
        qf = np.maximum(qd, 0.3 * tot / (rf[-1] - rf[0]))
        c = np.concatenate([[0.0], np.cumsum(0.5 * (qf[1:] + qf[:-1])
                                             * np.diff(rf))])
        c /= c[-1]
        return np.interp(np.linspace(0.0, 1.0, nu), c, rf)

    u = _invert_r(q)
    # Corrective ring passes ON TOP of the metric marginal: the marginal
    # sees the high-lambda band around a ring of masked ends, but not
    # the exact eps-scale mouth structure (a 1e-3-scale neck smears to
    # nothing on the fine grid, and the percentile cap flattens it).
    # Guarantee the mouth-resolving spacing (<= eps/4 at each masked
    # ring) the flip census demands by boosting the density in a narrow
    # band at the exact ring radius until delivered spacing complies.
    rings = {}
    for zc, eps in ends:
        if zc is not None and eps > 0.0:
            key = round(abs(zc), 9)
            rings[key] = min(eps, rings.get(key, eps))
    for _ in range(5):
        du_m = np.gradient(u)
        worst = 1.0
        for c, eps in rings.items():
            i = int(np.clip(np.searchsorted(u, c), 1, len(u) - 2))
            need = float(du_m[i]) / (0.25 * eps)
            if need > 1.05:
                # smooth Cauchy bump (same shape as _cluster_nodes):
                # abrupt density steps make abrupt spacing jumps, which
                # are themselves a normal-instability source
                h = max(2.0 * eps, 0.005 * (r1 - ra))
                lvl = float(np.interp(c, rf, q))
                q = q + (min(need, 4.0) - 1.0) * lvl \
                    / (1.0 + ((rf - c) / h) ** 2)
                worst = max(worst, need)
        if worst <= 1.05:
            break
        u = _invert_r(q)
    # angular marginal (area per fine wedge), uniform-floored
    w = (dens * np.gradient(rf)[:, None]).sum(axis=0)
    w = (1.0 - BLEND) * w.mean() + BLEND * w

    def _invert_t(wd):
        # same hard uniform floor as _invert_r
        tot = float(np.sum(0.5 * (wd[1:] + wd[:-1]) * np.diff(tf)))
        wf = np.maximum(wd, 0.3 * tot / (tf[-1] - tf[0]))
        c = np.concatenate([[0.0], np.cumsum(0.5 * (wf[1:] + wf[:-1])
                                             * np.diff(tf))])
        c /= c[-1]
        vt = np.interp((np.arange(nv) + 0.5) / nv, c, tf) % TAU
        vt.sort()
        return vt

    v = _invert_t(w)
    # same corrective passes for the angular axis, per end azimuth
    tgt = {}
    for zc, eps in ends:
        if zc is None or eps <= 0.0:
            continue
        th = float(np.angle(zc)) % TAU
        tgt[round(th, 9)] = min(0.25 * eps / abs(zc),
                                tgt.get(round(th, 9), np.inf))
    for _ in range(5):
        dv_m = np.gradient(np.unwrap(v))
        worst = 1.0
        for th, t in tgt.items():
            j = int(np.argmin(np.abs((v - th + math.pi) % TAU - math.pi)))
            need = float(dv_m[j]) / t
            if need > 1.05:
                dtf = np.abs((tf - th + math.pi) % TAU - math.pi)
                h = max(4.0 * t, 0.005 * TAU)
                lvl = float(w[np.argmin(dtf)])
                w = w + (min(need, 4.0) - 1.0) * lvl \
                    / (1.0 + (dtf / h) ** 2)
                worst = max(worst, need)
        if worst <= 1.05:
            break
        v = _invert_t(w)
    # nudge any node off an exact puncture centre (radially/angularly)
    for zc, eps in ends:
        if zc is None:
            continue
        R, th = abs(zc), float(np.angle(zc)) % TAU
        j = np.abs(u - R) < 1e-9 * max(r1, 1.0)
        u[j] += 1e-4 * (r1 - ra)
        j = np.abs(v - th) < 1e-9
        v[j] += 1e-4 * TAU / nv
    if not (np.all(np.diff(u) > 0.0) and np.all(np.diff(v) > 0.0)):
        return None
    return u, v


def _we_disk(spec, p, nu, nv, theta):
    """Radial-ray integration on a disk/annulus domain
    ('disk', r_in, r_out).  Rays get a half-step angular offset so none
    lands exactly on a puncture (a la the Jorge-Meeks k-noid); an
    annulus (r_in > 0) is stitched with a base-ring arc integral, which
    closes because the engine's rows have vanishing real periods."""
    global LAST_PROTECT
    d = spec['domain']
    r0 = max(float(_ev(d[1], p)), 0.0)
    r1 = float(_ev(d[2], p))
    dth = TAU / nv
    off = 0.5 * dth if spec.get('offset_rays', True) else 0.0
    v = off + np.arange(nv) * dth
    ra = max(r0, 1e-3)
    # Declared end punctures ('ends': list of (z_k, eps_k)): the row
    # names where its catenoid ends live INSIDE the domain (typically
    # rings of double poles of dh).  They drive three things: (1) the
    # radial/angular sampling is clustered toward the rings so the bells
    # get mesh support; (2) the integrand's principal parts at the poles
    # are subtracted and re-added in closed form, so the radial
    # quadrature never steps across a double pole (which used to leave
    # per-ray period errors that tore the mesh into wedges); (3) a small
    # parameter-space disk |z - z_k| <= eps_k is masked out, cutting
    # each bell on a clean conformal circle instead of amputating it
    # with the object-space percentile clip.
    ends = _ev(spec.get('ends'), p) if spec.get('ends') else None
    # radial node distribution.  Enneper-type ends grow like a power of
    # the radius, so a linear-in-r grid starves the fast-growing rim with
    # a few huge facets; 'radial_grade' clusters nodes toward the
    # end(s) instead ('rim' -> toward r_out, 'both' -> a cosine/Chebyshev
    # grid dense at r_in and r_out for two-ended annuli).
    grade = spec.get('radial_grade') if ends is None else None
    s = np.linspace(0.0, 1.0, nu)
    if grade == 'log' and ra > 0.0:
        # log-spaced rings (equal RELATIVE steps), the natural grid for a
        # z -> 1/z symmetric annulus with power-law ends at both rims
        # (Weber's ExpNRange): the image-step growth per ring is a
        # constant factor, and the node split about |z| = 1 mirrors the
        # surface's own symmetry.  No blend needed -- the spacing ratio
        # between neighbouring rings is bounded by construction.
        s = (np.exp(np.linspace(math.log(ra), math.log(r1), nu)) - ra) \
            / (r1 - ra)
    elif grade in ('rim', 'both'):
        if grade == 'rim':
            g = 1.0 - (1.0 - s) ** 2
        else:
            g = 0.5 - 0.5 * np.cos(math.pi * s)
        # BLEND with the linear grid rather than using the graded one
        # neat.  Both curves have ZERO derivative at the clustered end,
        # so the spacing between the last two rings collapses like
        # 1/nu^2 while the angular spacing only falls like 1/nv: the
        # outermost quads become slivers whose aspect ratio grows
        # without bound as resolution rises.  The blend keeps the
        # clustering (which is real: Enneper flares like r^(2k+1), and
        # a linear grid leaves the rim a coarse polygon) but puts a
        # floor under the spacing: the derivative at the clustered end
        # is now at least 1 - GRADE_BLEND, so the gap ratio is bounded
        # by a constant instead of growing with nu.
        #
        # NB the blend is an aspect-ratio bound, nothing more.  The
        # folded-over "lip" once seen along Enneper's rim was NOT these
        # slivers' fault: it was the boundary smoother's first-order
        # shrink pulling the rim ring through its (tightly graded)
        # neighbour ring -- fixed in domain._smooth_boundary, which is
        # now a non-shrinking Taubin pair.  The tight last-ring gap
        # merely made that pull easier to overshoot.
        s = (1.0 - GRADE_BLEND) * s + GRADE_BLEND * g
    u = ra + (r1 - ra) * s
    if ends:
        # metric-equidistributed axes (see _metric_axes); the Cauchy-bump
        # grading remains for rows that opt out ('metric_axes': False --
        # geometry whose lambda concentration lives BELOW the mask scale,
        # e.g. the symmetrized finite Riemann's 1e-3 necks, needs the
        # exact-ring bump grading) and as the fallback when the density
        # is unusable
        ax = (_metric_axes(spec, p, theta, ra, r1, nu, nv, off, ends)
              if spec.get('metric_axes', True) else None)
        u, v = ax if ax is not None else _end_graded_axes(
            ra, r1, nu, nv, off, ends)
    R, TH = np.meshgrid(u, v, indexing='ij')
    z = R * np.exp(1j * TH)
    endmask = None
    if ends:
        # floor each end's mask radius at ~1.5 local grid spacings, so a
        # masked disk can never fall BETWEEN nodes: a coarse cell that
        # contains a pole with all four corners outside the mask would
        # survive as one flat quad sealing the bell mouth (a "cap").
        # With the floor, the cell containing a pole always has a corner
        # inside the mask and the mouth stays open.
        du = np.gradient(u)
        dvw = np.diff(np.concatenate([v, [v[0] + TAU]]))
        eff = []
        for zc, eps in ends:
            if zc is None:
                continue          # z = infinity marker: grading only
            if eps > 0.0:
                Rk = abs(zc)
                i = int(np.clip(np.searchsorted(u, Rk), 0, len(u) - 1))
                thk = float(np.angle(zc)) % TAU
                j = int(np.argmin(np.abs(
                    (v - thk + math.pi) % TAU - math.pi)))
                eps = max(eps, 1.5 * float(du[i]),
                          1.5 * Rk * float(dvw[j]))
            eff.append((zc, eps))
        ends = eff
        endmask = np.ones(z.shape, dtype=bool)
        for zc, eps in ends:
            if eps > 0.0:
                endmask &= np.abs(z - zc) > eps
        # bell-mouth neighbourhoods: exempt from boundary smoothing (the
        # mouth rims are snapped onto exact conformal circles below)
        prot = np.zeros(z.shape, dtype=bool)
        for zc, eps in ends:
            if eps > 0.0:
                prot |= np.abs(z - zc) <= 2.5 * eps
        LAST_PROTECT = prot
    if 'Xexact' in spec:
        # closed-form immersion (no radial quadrature): the antiderivative
        # is known analytically, so evaluate it straight on the grid --
        # avoids the near-pole capping/folding that tears wing ends.
        with np.errstate(divide='ignore', invalid='ignore'):
            xx, yy, zz = spec['Xexact'](z, p, theta)
        X = np.stack(np.broadcast_arrays(xx, yy, zz), axis=-1).astype(float)
        mask = np.isfinite(X).all(axis=-1)
        punct = (list(_ev(spec.get('mask_punctures'), p))
                 if spec.get('mask_punctures') else [])
        if ends:
            punct = punct + [(zc, e) for zc, e in ends if e > 0.0]
        if punct:
            for zc, rho in punct:
                mask &= np.abs(z - zc) > rho
            if spec.get('clip_punctures'):
                # marching-squares-style boundary CLIP: instead of dropping
                # whole grid quads that straddle a puncture circle (which
                # leaves a one-quad staircase along every wing rim), pull the
                # first ring of just-inside grid vertices radially out onto
                # the circle |z - zc| = rho and re-evaluate the immersion
                # there, so the cut lands exactly on the mask boundary and
                # the wing edge reads as a clean smooth curve.  Exactly-cut
                # rims must not be re-smoothed afterwards (see LAST_PROTECT).
                prot = np.zeros(z.shape, dtype=bool)
                for zc, rho in punct:
                    if rho > 0.0:
                        prot |= np.abs(z - zc) <= 2.5 * rho
                LAST_PROTECT = prot
                zc_a = np.array([zc for zc, _ in punct])
                rho_a = np.array([rho for _, rho in punct])
                keep = mask
                nb = np.zeros_like(keep)
                nb[:-1] |= keep[1:]
                nb[1:] |= keep[:-1]
                nb |= np.roll(keep, 1, axis=1)
                nb |= np.roll(keep, -1, axis=1)
                ring = (~keep) & nb          # inside a puncture, but adjacent
                ii, jj = np.nonzero(ring)    # to a kept vertex -> snap it out
                if len(ii):
                    zr = z[ii, jj]
                    k = np.abs(zr[:, None] - zc_a[None, :]).argmin(axis=1)
                    d = zr - zc_a[k]
                    znew = zc_a[k] + rho_a[k] * d / np.abs(d)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        nx, ny, nz = spec['Xexact'](znew, p, theta)
                    good = np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)
                    gi, gj = ii[good], jj[good]
                    X[gi, gj, 0] = nx[good]
                    X[gi, gj, 1] = ny[good]
                    X[gi, gj, 2] = nz[good]
                    mask = mask.copy()
                    mask[gi, gj] = True
        return X[..., 0], X[..., 1], X[..., 2], False, True, mask
    phi = _phi_fn(spec, p, theta)
    # principal-part subtraction at the declared end punctures: phi =
    # phi_smooth + sum_k sum_m C_km/(z-z_k)^m, m up to the row's
    # 'end_pole_order' (2 for catenoid ends, 4 for the winding Enneper
    # ends).  The smooth part is integrated numerically (it is analytic
    # across the end rings, so the per-ray trapezoid no longer
    # accumulates a different error on each side of a pole); the
    # singular part has the elementary antiderivative S(z) (_pp_anti),
    # evaluated pointwise.  This is what actually GROWS the end
    # geometry: the near-pole immersion is exact at any resolution.
    pp = (_end_principal_parts(phi, ends,
                               int(spec.get('end_pole_order', 2)))
          if ends else None)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = phi(z)
        S = None
        if pp:
            S = np.zeros(z.shape + (3,), dtype=complex)
            for zc, Ck in pp:
                dd = (z - zc)[..., None]
                F = _pp_sub(F, dd, Ck)
                S = S + _pp_anti(dd, Ck)
    ez = np.exp(1j * TH)[..., None]
    Xr = np.real(F * ez)                       # dz = e^{i th} dr
    Xr = np.where(np.isfinite(Xr), Xr, 0.0)
    # loose cap: kill only true numerical garbage from rays that graze
    # a pole, without flattening tall catenoid/planar ends.  (With the
    # principal parts subtracted the integrand is already tame near the
    # declared ends, so the cap is a no-op there.)
    cap = 400.0 * float(np.median(np.abs(Xr))) or 1.0
    Xr = np.clip(Xr, -cap, cap)
    dr = np.diff(R, axis=0)[..., None]
    X = np.concatenate(
        [np.zeros((1, nv, 3)),
         np.cumsum(0.5 * (Xr[1:] + Xr[:-1]) * dr, axis=0)], axis=0)
    if u[0] > 2e-3:
        # annulus: connect the ray base points along the inner circle
        zi = u[0] * np.exp(1j * v)
        with np.errstate(divide='ignore', invalid='ignore'):
            Fi = phi(zi)
            if pp:
                for zc, Ck in pp:
                    Fi = _pp_sub(Fi, (zi - zc)[:, None], Ck)
        arc = np.real(Fi * (1j * zi)[:, None])   # dz = i z dth
        arc = np.where(np.isfinite(arc), arc, 0.0)
        dv_arc = np.diff(v)
        A = np.zeros((nv, 3))
        A[1:] = np.cumsum(0.5 * (arc[1:] + arc[:-1])
                          * dv_arc[:, None], axis=0)
        X = X + A[None, :, :]
    if S is not None:
        # add back the exact singular antiderivative (constant offset
        # S(base) is irrelevant -- the mesher re-centres)
        Sr = np.real(S)
        X = X + np.where(np.isfinite(Sr), Sr, 0.0)
    mask = None
    punct = spec.get('mask_punctures')
    if punct or endmask is not None:
        valid = np.ones(z.shape, dtype=bool)
        if punct:
            for zc, rho in _ev(punct, p):
                valid &= np.abs(z - zc) > rho
        if endmask is not None:
            valid &= endmask
        mask = valid
    if ends and pp and mask is not None:
        # Cut each bell mouth ON the exact conformal circle.  The mesh's
        # boundary loop around a masked end consists of the VALID grid
        # vertices whose face neighbourhood (8-stencil) touches the
        # masked disk -- so move exactly those vertices radially inward
        # onto |z - z_k| = eps_k and re-evaluate the immersion there.
        # (The earlier scheme snapped the just-INSIDE ring outward
        # instead; the loop then alternated between snapped on-circle
        # vertices and untouched just-outside ones -- vertices at
        # different heights up the 1/d flare -- and read as a ragged
        # staircase crown: rim zigzag ~0.31 of an edge length, measured
        # across the whole k-noid family.  Snapping the OUTSIDE ring is
        # the single-ring formulation: every loop vertex lands on the
        # circle, and no snapped-vertex pair can degenerate a quad,
        # because no vertex changes validity.)  The displacement is
        # exact for the singular part (S is closed form) plus a short
        # trapezoid step of the smooth remainder.
        zc_a = np.array([zc for zc, e in ends if e > 0.0])
        eps_a = np.array([e for zc, e in ends if e > 0.0])
        if len(zc_a) and endmask is not None:
            inv = ~endmask                  # inside some end's mask disk
            nb = np.zeros_like(inv)
            for dj in (-1, 0, 1):           # angular axis wraps
                sh = np.roll(inv, dj, axis=1)
                nb |= sh                    # (di = 0 row)
                nb[:-1] |= sh[1:]           # di = +1
                nb[1:] |= sh[:-1]           # di = -1
            ring = mask & nb                # the future boundary loop
            ii, jj = np.nonzero(ring)
        else:
            ii = jj = np.array([], dtype=int)
        if len(ii):
            zr = z[ii, jj]
            k = np.abs(zr[:, None] - zc_a[None, :]).argmin(axis=1)
            dvec = zr - zc_a[k]
            adv = np.abs(dvec)
            # only vertices actually hugging their end's mouth (a valid
            # vertex can graze a *different* end's disk diagonally)
            sel = (adv > 1e-12) & (adv < 2.5 * eps_a[k])
            ii, jj, zr, k, dvec = ii[sel], jj[sel], zr[sel], k[sel], \
                dvec[sel]
            ang = np.angle(dvec)
            # Equalize the rim spacing per end: radial projection alone
            # keeps each vertex's grid azimuth, so the loop inherits the
            # grid's uneven angular footprint -- including staircase
            # steps where two LOOP-adjacent vertices are radially
            # aligned (identical azimuth): any per-vertex respacing that
            # sorts by angle backtracks against the actual boundary walk
            # there (rim edge-length CV ~0.3-0.5 measured).  Around a
            # pole of order M the immersion maps the circle at
            # near-uniform speed (the -C_M/((M-1) d^(M-1)) term of the
            # antiderivative dominates: d = eps e^{i phi} -> circle of
            # radius |C_M|/((M-1) eps^(M-1)) traversed at uniform
            # speed, winding M-1 times), so uniform PARAMETER angles
            # along the LOOP are uniform 3-D rim edges -- for the
            # double poles of catenoid ends and the order-four poles
            # of Enneper ends alike.  Per end: trace the hole's
            # boundary loop on the
            # grid (edges with exactly one kept quad, kept = all four
            # corners valid, angular axis wrapped) and assign a uniform
            # angular ramp in loop order -- monotone by construction,
            # no backtracking possible.  Falls back to an
            # order-preserving angle-sorted blend if the trace fails
            # (mask merged with another hole or the domain rim).
            nu_g, nv_g = z.shape
            mrn = np.roll(mask, -1, axis=1)
            cellok = (mask[:-1] & mask[1:] & mrn[:-1] & mrn[1:])

            def _kept(a, b):
                return (0 <= a < nu_g - 1) and bool(cellok[a, b % nv_g])

            pos = {(int(a), int(b)): t
                   for t, (a, b) in enumerate(zip(ii, jj))}

            def _sorted_blend(idx):
                srt = idx[np.argsort(ang[idx])]
                a_s = ang[srt]
                n_k = len(srt)
                uni = TAU * np.arange(n_k) / n_k
                dphi = float(np.angle(np.exp(1j * (a_s - uni)).mean()))
                tgt = uni + dphi
                lo = 0.5 * (a_s + np.roll(a_s, 1))
                lo[0] -= 0.5 * TAU
                hi = 0.5 * (a_s + np.roll(a_s, -1))
                hi[-1] += 0.5 * TAU
                dd = (tgt - a_s + math.pi) % TAU - math.pi
                ang[srt] = np.clip(a_s + dd, lo, hi)

            walks = []                       # per-end rim reparam data
            for ke in range(len(zc_a)):
                idx = np.nonzero(k == ke)[0]
                if len(idx) < 4:
                    continue
                # boundary edges among this end's rim vertices
                adj = {}
                okwalk = True
                for t in idx:
                    a, b = int(ii[t]), int(jj[t])
                    for (a2, b2), cells in (
                            ((a, b + 1), ((a, b), (a - 1, b))),
                            ((a + 1, b), ((a, b), (a, b - 1)))):
                        if a2 >= nu_g:      # radial axis does not wrap
                            continue
                        t2 = pos.get((a2, b2 % nv_g))
                        if t2 is None or k[t2] != ke:
                            continue
                        if _kept(*cells[0]) != _kept(*cells[1]):
                            adj.setdefault(t, []).append(t2)
                            adj.setdefault(t2, []).append(t)
                if (len(adj) != len(idx)
                        or any(len(v) != 2 for v in adj.values())):
                    okwalk = False
                if okwalk:
                    start = int(idx[0])
                    walk = [start]
                    prev, cur = None, start
                    while True:
                        nxt = [w for w in adj[cur] if w != prev]
                        if not nxt:
                            okwalk = False
                            break
                        prev, cur = cur, nxt[0]
                        if cur == start:
                            break
                        walk.append(cur)
                        if len(walk) > len(idx):
                            okwalk = False
                            break
                    okwalk = okwalk and len(walk) == len(idx)
                if okwalk:
                    aw = ang[np.array(walk)]
                    dth = (np.diff(aw) + math.pi) % TAU - math.pi
                    tot = float(dth.sum()) + float(
                        (aw[0] - aw[-1] + math.pi) % TAU - math.pi)
                    if abs(abs(tot) - TAU) > 0.1 * TAU:
                        okwalk = False       # not a simple mouth loop
                if okwalk:
                    unw = np.concatenate([[aw[0]],
                                          aw[0] + np.cumsum(dth)])
                    ramp = np.arange(len(walk)) * tot / len(walk)
                    ph = float(np.angle(
                        np.exp(1j * (unw - ramp)).mean()))
                    tgt = ramp + ph
                    wk = np.array(walk)
                    if RIM_DEBUG is not None:
                        RIM_DEBUG.append({
                            'zc': zc_a[ke], 'eps': float(eps_a[ke]),
                            'ij': (ii[wk].copy(), jj[wk].copy()),
                            'unw': unw.copy(), 'tgt': tgt.copy()})
                    ang[wk] = tgt
                    walks.append((ke, unw, tgt))
                else:
                    _sorted_blend(idx)
            znew = zc_a[k] + eps_a[k] * np.exp(1j * ang)
            # Boundary-layer accommodation.  The uniform rim can sit
            # several grid columns from the vertices' original azimuths
            # (the deviation field peaks at ~4-5 rim slots on the KNOID
            # mouths, where the grid's angular density seen from the end
            # centre varies most), and ONE row of quads cannot absorb
            # that much tangential shear without folding (26-83 flipped
            # edges measured).  So distribute it: every interior vertex
            # of the protected annulus eps < |z - z_k| < 2.4 eps gets
            # the same angular deviation, interpolated in azimuth and
            # tapered to zero at the outer edge -- the shear spreads
            # over the ~6 rings the smoothing exemption already
            # reserves for the mouth, well under one cell per ring.
            # (Bands of neighbouring ends never overlap: every 'ends'
            # row masks at most 0.35 x the half-gap between ring
            # neighbours, so 2.4 eps < 0.85 x half-gap.)
            if walks:
                rimflag = np.zeros(z.shape, dtype=bool)
                rimflag[ii, jj] = True
                dist_all = np.abs(z[..., None] - zc_a[None, None, :])
                near_all = dist_all.argmin(axis=-1)
                add = []
                for ke, unw, tgt in walks:
                    zck, epsk = zc_a[ke], float(eps_a[ke])
                    xs = unw % TAU
                    # Wrap the deviation per-vertex.  tgt - unw can
                    # carry a spurious constant +-2 pi: the phase fit
                    # ph is a CIRCULAR mean, so when the walk's start
                    # angle sits near the +-pi branch of np.angle
                    # (every end whose origin-facing side is the cut,
                    # e.g. an end on the positive real axis) the mean
                    # lands on the other side of the branch and the
                    # whole field shifts by a full turn.  On the rim
                    # itself that is a no-op (e^{2 pi i} = 1), but the
                    # tapered band multiplies the deviation by
                    # tap < 1, so an un-wrapped 2 pi swirled the
                    # band's rings most of a full turn around the end
                    # (measured: max|dv| = 6.6 on the azimuth-0 end of
                    # the Enneper-ended k-noid vs 0.33 on its other
                    # two ends).
                    dv = (tgt - unw + math.pi) % TAU - math.pi
                    o = np.argsort(xs)
                    xs, dv = xs[o], dv[o]
                    # The outer band only needs the LOW-frequency part
                    # of the deviation field: it exists to absorb the
                    # multi-slot drift one quad row cannot, while the
                    # slot-to-slot wiggle (radially aligned staircase
                    # pairs) matters only right at the rim.  Feeding
                    # the wiggle deep into the interior rings shears
                    # them against each other and shows up as
                    # cotan-|H| noise (KNOID median doubled), but
                    # dropping it at the FIRST ring misaligns that
                    # ring against the exactly-placed rim (4 flips on
                    # M3_PYR) -- so keep the raw field beside the rim
                    # and fade to the smoothed one outward.
                    kw = min(7, len(dv) | 1)
                    ker = np.ones(kw) / kw
                    dvs = np.convolve(
                        np.concatenate([dv[-(kw // 2):], dv,
                                        dv[:kw // 2]]),
                        ker, mode='valid')
                    xs = xs + np.arange(len(xs)) * 1e-9   # break ties
                    xse = np.concatenate([[xs[-1] - TAU], xs,
                                          [xs[0] + TAU]])
                    dve = np.concatenate([[dv[-1]], dv, [dv[0]]])
                    dvse = np.concatenate([[dvs[-1]], dvs, [dvs[0]]])
                    dk = dist_all[..., ke]
                    band = (mask & ~rimflag & (near_all == ke)
                            & (dk > epsk) & (dk < 2.4 * epsk))
                    band[0, :] = False       # never the domain rims
                    band[-1, :] = False
                    bi, bj = np.nonzero(band)
                    if not len(bi):
                        continue
                    zb = z[bi, bj]
                    az = np.angle(zb - zck) % TAU
                    rho = dk[bi, bj] / epsk
                    wraw = np.clip((1.5 - rho) / 0.5, 0.0, 1.0)
                    dvb = (wraw * np.interp(az, xse, dve)
                           + (1.0 - wraw) * np.interp(az, xse, dvse))
                    tap = np.clip((2.4 - rho) / 1.4, 0.0, 1.0)
                    zbn = zck + (zb - zck) * np.exp(1j * dvb * tap)
                    add.append((bi, bj, zb, zbn))
                if add:
                    ii = np.concatenate([ii] + [a[0] for a in add])
                    jj = np.concatenate([jj] + [a[1] for a in add])
                    zr = np.concatenate([zr] + [a[2] for a in add])
                    znew = np.concatenate([znew] + [a[3] for a in add])
            with np.errstate(divide='ignore', invalid='ignore'):
                Fo = phi(zr)
                Fn = phi(znew)
                dS = np.zeros((len(zr), 3), dtype=complex)
                for zc, Ck in pp:
                    do = (zr - zc)[:, None]
                    dn = (znew - zc)[:, None]
                    Fo = _pp_sub(Fo, do, Ck)
                    Fn = _pp_sub(Fn, dn, Ck)
                    dS += _pp_anti(dn, Ck) - _pp_anti(do, Ck)
            step = (znew - zr)[:, None]
            dX = np.real(dS + 0.5 * (Fo + Fn) * step)
            Xr_old = X[ii, jj, :]
            Xn = Xr_old + dX
            good = np.isfinite(Xn).all(axis=1) & np.isfinite(Xr_old).all(axis=1)
            gi, gj = ii[good], jj[good]
            X[gi, gj, :] = Xn[good]
    clip = spec.get('clip', True)
    tail = mask if mask is not None else clip
    return X[..., 0], X[..., 1], X[..., 2], False, True, tail


def _we_rect(spec, p, nu, nv, theta):
    """2-D cumulative-trapezoid integration on a rectangle
    ('rect', u0, u1, v0, v1): base row along v0, columns upward."""
    d = spec['domain']
    u0, u1 = float(_ev(d[1], p)), float(_ev(d[2], p))
    v0, v1 = float(_ev(d[3], p)), float(_ev(d[4], p))
    u = np.linspace(u0, u1, nu)
    v = np.linspace(v0, v1, nv)
    U, V = np.meshgrid(u, v, indexing='ij')
    z = U + 1j * V
    phi = _phi_fn(spec, p, theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = phi(z)
    F = np.where(np.isfinite(F), F, 0.0)
    du = (u1 - u0) / max(nu - 1, 1)
    dv = (v1 - v0) / max(nv - 1, 1)
    base = np.zeros((nu, 3), dtype=complex)
    base[1:] = np.cumsum(0.5 * (F[1:, 0, :] + F[:-1, 0, :]) * du, axis=0)
    col = np.zeros((nu, nv, 3), dtype=complex)
    col[:, 1:, :] = np.cumsum(
        0.5 * (F[:, 1:, :] + F[:, :-1, :]) * (1j * dv), axis=1)
    X = np.real(base[:, None, :] + col)
    return (X[..., 0], X[..., 1], X[..., 2], False, False,
            spec.get('clip', False))


def _we_torus(spec, p, nu, nv, theta):
    """Closed-form antiderivative on a torus C / <2w1, 2w1 tau>
    ('torus', w1, tau): spec['X'](z, p, L) -> (x, y, z) real arrays.
    Punctures are removed with the toolkit's toroidal mask.  A spec
    may unwrap one direction ('torus_wrap') and span several copies
    ('copies') -- used by singly periodic surfaces whose deck
    translation is a genuine space translation."""
    d = spec['domain']
    w1 = float(_ev(d[1], p))
    tau = _ev(d[2], p)
    L = _Lattice(w1, tau) if (w1, tau) != (0.5, 1j) else _SQUARE
    wrap_u, wrap_v = spec.get('torus_wrap', (True, True))
    m = int(_ev(spec.get('copies', 1), p))
    if wrap_u and wrap_v and m == 1:
        U, V = _torus_grid(nu, nv)
    else:
        u = (np.linspace(0.0, m, nu, endpoint=not wrap_u) if not wrap_u
             else np.linspace(0.0, m, nu, endpoint=False))
        v = np.linspace(0.0, 1.0, nv, endpoint=not wrap_v) if not wrap_v \
            else np.linspace(0.0, 1.0, nv, endpoint=False)
        U, V = np.meshgrid(u, v, indexing='ij')
    z = 2.0 * w1 * (U + tau * V)
    with np.errstate(divide='ignore', invalid='ignore'):
        x, y, zc = spec['X'](z, p, L)
    mask = _puncture_mask(U % 1.0, V % 1.0, _ev(spec['punctures'], p))
    return x, y, zc, wrap_u, wrap_v, mask


# --- half-plane patch + dihedral tiling (Costa-Hoffman-Meeks family) ------

def halfplane_patch(phi, seed, rot, n_in, n_out, nv, r_out=12.0,
                    grade=None):
    """Integrate a phi-stack over the closed upper half plane -> one
    fundamental patch (nr, nv, 3).  Grid: Chebyshev in theta (clustered
    at the ends theta = 0, pi), graded in r (clustered at the branch
    point r = 0 and the ends r = 1).  `seed(z0)` supplies the analytic
    integral of phi over the first spine cell 0 -> i r1 (the integrand
    is singular at the branch point).  `rot` is the dihedral order the
    patch will be tiled to (drives the default r-grading)."""
    nv = nv if nv % 2 else nv + 1              # odd: a column hits pi/2
    j = np.arange(nv)
    th = (math.pi / 2) * (1 - np.cos(math.pi * j / (nv - 1)))
    s = np.linspace(0.0, 1.0, n_in + 1)[1:]
    gexp = grade if grade is not None else 0.5 * (rot)
    r_in = ((1 - np.cos(math.pi * s)) / 2) ** gexp
    r_o = np.exp(np.linspace(0.0, math.log(r_out), n_out + 1))[1:]
    r = np.concatenate([[0.0], r_in, r_o])
    nr = len(r)
    R, TH = np.meshgrid(r, th, indexing='ij')
    Z = R * np.exp(1j * TH)
    Z[0, :] = 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        F = phi(Z)                             # (nr, nv, 3) complex
    jm = (nv - 1) // 2                         # spine column theta = pi/2
    X = np.zeros((nr, nv, 3), dtype=complex)
    z0 = 1j * r[1]
    sd = np.asarray(seed(z0), dtype=complex)
    # spine: cumulative along r at column jm (dz = i dr)
    dr = np.diff(r)[:, None]
    sp_inc = 0.5 * (F[1:, jm, :] + F[:-1, jm, :]) * (1j * dr)
    spine = np.concatenate([[np.zeros(3)], [sd],
                            sd + np.cumsum(sp_inc[1:], axis=0)], axis=0)
    X[:, jm, :] = spine
    # arcs: cumulative along theta per row (dz = i z dtheta), from jm out
    gfac = F * (1j * Z)[..., None]
    dth = np.diff(th)
    trap = np.zeros((nr, nv, 3), dtype=complex)
    trap[:, 1:, :] = 0.5 * (gfac[:, 1:, :] + gfac[:, :-1, :]) \
        * dth[None, :, None]
    # zero non-finite increments (masked end nodes theta = 0, pi at
    # z = +-1) so one bad cell cannot poison a whole row's cumsum
    trap = np.where(np.isfinite(trap), trap, 0.0)
    C = np.cumsum(trap, axis=1)
    X = X[:, jm, :][:, None, :] + (C - C[:, jm, :][:, None, :])
    return Z, np.real(X)


def tile_dihedral(Z, Xr, rot, punctures, radius, scale,
                  circularize=True):
    """Snap the patch's symmetry seams, tile into 2*rot rigid copies
    (rot rotations x mirror), weld, trim the ends, smooth the rims,
    center and fit the 2 m cube.  Returns (V, quads, uv) where uv is
    the per-vertex fundamental-domain parametrization (repeated per
    copy -- a tiled surface has no global conformal chart).
    Generalizes the Costa-Hoffman-Meeks assembly; assumes the standard
    hyperelliptic cut structure along [0, 1] and (-inf, -1]."""
    nr, nvp, _ = Xr.shape
    Rabs = np.abs(Z)
    ang = -math.pi / rot
    uvec = np.array([math.cos(ang), math.sin(ang)])
    # snap the two boundary columns onto their symmetry planes
    for jcol, is0 in ((0, True), (nvp - 1, False)):
        xy = Xr[:, jcol, :2]
        bank = (Rabs[:, jcol] < 1.0) if is0 else (Rabs[:, jcol] > 1.0)
        proj = (xy @ uvec)[:, None] * uvec[None, :]
        flat = np.stack([xy[:, 0], np.zeros_like(xy[:, 1])], axis=-1)
        Xr[:, jcol, :2] = np.where(bank[:, None], proj, flat)
    Xr[0, :, :] = 0.0                          # center vertex
    valid = np.ones(Z.shape, dtype=bool)
    for zc, rho in punctures:
        valid &= np.abs(Z - zc) > rho
    valid[0, :] = True
    Xr = np.where(np.isfinite(Xr), Xr, 0.0)
    Xr[~valid] = 0.0
    V0 = Xr.reshape(-1, 3)
    iu, ju = np.meshgrid(np.arange(nr), np.arange(nvp), indexing='ij')
    uv0 = np.stack([iu / max(nr - 1, 1), ju / max(nvp - 1, 1)],
                   axis=-1).reshape(-1, 2)
    vv = valid.reshape(-1)
    ii, jj = np.meshgrid(np.arange(nr - 1), np.arange(nvp - 1),
                         indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    q0 = np.stack([ii * nvp + jj, ii * nvp + jj + 1,
                   (ii + 1) * nvp + jj + 1, (ii + 1) * nvp + jj], axis=1)
    q0 = q0[np.all(vv[q0], axis=1)]
    # tile: rot rotations, each with its y-mirror -> 2*rot copies
    M = np.diag([1.0, -1.0, 1.0])
    Vparts, Fparts, base = [], [], 0
    for jrot in range(rot):
        a = TAU * jrot / rot
        Rj = np.array([[math.cos(a), -math.sin(a), 0.0],
                       [math.sin(a), math.cos(a), 0.0],
                       [0.0, 0.0, 1.0]])
        for mir in (False, True):
            T = Rj @ (M if mir else np.eye(3))
            Vparts.append(V0 @ T.T)
            qf = (q0[:, ::-1] if mir else q0) + base
            Fparts.append(qf)
            base += len(V0)
    V = np.concatenate(Vparts, axis=0)
    uvcat = np.concatenate([uv0] * (2 * rot), axis=0)
    faces = np.concatenate(Fparts, axis=0)
    # weld (quantize + unique); tight tolerance -- the seam snap already
    # makes shared vertices coincide to machine epsilon
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    keyq = np.round(V / (1e-7 * max(diag, 1.0))).astype(np.int64)
    _, inv = np.unique(keyq, axis=0, return_inverse=True)
    inv = inv.ravel()
    Vw = np.zeros((int(inv.max()) + 1, 3))
    Vw[inv] = V
    uvw = np.zeros((len(Vw), 2))
    uvw[inv] = uvcat
    faces = inv[faces]
    # collapse welded seam quads to triangles; drop true degenerates
    flist = []
    for f in faces:
        g = [int(f[0])]
        for t in range(1, 4):
            if int(f[t]) != g[-1]:
                g.append(int(f[t]))
        if len(g) >= 3 and g[0] != g[-1] and len(set(g)) == len(g):
            flist.append(tuple(g))
    # object-space radius clip to trim the (infinite) ends
    cen = np.median(Vw, axis=0)
    rad = np.linalg.norm(Vw - cen, axis=1)
    thr = float(np.percentile(rad, 93.0))
    flist = [f for f in flist if all(rad[i] <= thr for i in f)]
    used = np.unique(np.array([i for f in flist for i in f],
                              dtype=np.int64))
    remap = np.full(len(Vw), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    Vf = Vw[used]
    uvf = uvw[used]
    quads = [tuple(int(remap[i]) for i in f) for f in flist]
    # keep the main body only (the clip can shear off small islands);
    # ride uv along by stacking it into the vertex array
    Vu, quads = _largest_component(np.hstack([Vf, uvf]), quads)
    Vf, uvf = Vu[:, :3], Vu[:, 3:]
    Vf = _smooth_boundary(Vf, quads)
    if circularize:
        # taper_rings spreads each puncture mouth's snap correction over
        # the rings behind it (1/7 of it per quad row) instead of
        # parking it all in the first row -- measured on COSTA_HM, the
        # untapered snap left a 0-spread rim against a 13%-radius-spread
        # first ring, a visible corrugation along every mouth.
        # trim_sphere marks the object-space percentile trim: those
        # loops get the (tapered) radial rounding but keep the
        # surface's own z -- the planar end's cut is a genuinely wavy
        # curve, and flattening it curled the disc edge into a "brim"
        # (see _circularize_outer).
        Vf = _circularize_outer(Vf, quads, taper_rings=6,
                                trim_sphere=(cen, thr))
    Vf = _center_fit(Vf, scale, Vf)
    return Vf, quads, uvf


def _we_halfplane(spec, p, nu, nv, radius, scale, theta):
    rot = int(_ev(spec['symmetry']['rot'], p))
    phi = _phi_fn(spec, p, theta)
    n_in = max(70, int(1.4 * nu))
    n_out = max(30, int(0.6 * nu))
    r_out = float(_ev(spec['domain'][1], p)) if len(spec['domain']) > 1 \
        else 12.0
    grade = _ev(spec['r_grade'], p) if 'r_grade' in spec else None
    Z, Xr = halfplane_patch(phi, lambda z0: spec['seed'](p, z0), rot,
                            n_in, n_out, nv, r_out=r_out, grade=grade)
    punct = _ev(spec['hp_punctures'], p) if 'hp_punctures' in spec else []
    punct = [(zc, rho / max(radius / 1.2, 0.4)) for zc, rho in punct]
    return tile_dihedral(Z, Xr, rot, punct, radius, scale,
                         circularize=spec.get('circularize', True))


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def we_surface(spec, nu, nv, order, radius, scale=None, theta=0.0):
    """Build a surface from a Weierstrass spec.  Grid domains return
    the PARAMETRIC tuple (x, y, z, wrap_u, wrap_v, clip|mask); the
    halfplane domain returns a finished (V, quads, uv) mesh."""
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    if 'solve' in spec:
        p = spec['solve'](p) or p
    kind = spec['domain'][0]
    if theta and kind in ('torus', 'halfplane'):
        raise ValueError("associate family needs a simply connected "
                         "domain (disk/rect/Bjorling)")
    if kind == 'disk':
        return _we_disk(spec, p, nu, nv, theta)
    if kind == 'rect':
        return _we_rect(spec, p, nu, nv, theta)
    if kind == 'torus':
        return _we_torus(spec, p, nu, nv, theta)
    if kind == 'halfplane':
        return _we_halfplane(spec, p, nu, nv, radius, scale, theta)
    raise ValueError(f"unknown domain {kind!r}")


def _bj_curve_fns(spec, p):
    c = spec['curve']
    h = 1e-5

    def c1(w):
        return tuple((a - b) / (2 * h) for a, b in
                     zip(c(w + h, p), c(w - h, p)))

    def c2(w):
        cp, c0, cm = c(w + h, p), c(w, p), c(w - h, p)
        return tuple((a - 2 * b + d) / (h * h) for a, b, d in
                     zip(cp, c0, cm))
    return (lambda w: c(w, p)), c1, c2


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def bjorling_surface(spec, nu, nv, order, radius, theta=0.0):
    """Schwarz's solution of the Bjorling problem for an analytic seed
    curve + unit normal field: the surface strip containing the curve
    with the prescribed surface normal along it.  Integrates n x c'
    vertically from the real axis per grid column (the strip is simply
    connected, so no period machinery is needed); theta sweeps the
    associate family."""
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    t0, t1 = _ev(spec['t_range'], p)
    closed = spec.get('closed', False)
    vh = _ev(spec.get('v_half', 0.6), p) if not callable(
        spec.get('v_half', 0.6)) else spec['v_half'](p)
    nv = nv if nv % 2 else nv + 1              # middle row on the axis
    wrap_u = bool(closed) and abs(theta) < 1e-12
    u = np.linspace(t0, t1, nu, endpoint=not wrap_u)
    v = np.linspace(-vh, vh, nv)
    U, V = np.meshgrid(u, v, indexing='ij')
    W = U + 1j * V
    cfn, c1fn, c2fn = _bj_curve_fns(spec, p)
    C = cfn(W)
    Cp = c1fn(W)
    nrm = spec.get('normal')
    if nrm is None or nrm == 'frenet':
        # principal normal, analytically continued (principal-branch
        # square roots -- valid for the thin strips we build)
        Cpp = c2fn(W)
        dot_pp = sum(a * a for a in Cp)
        T = tuple(a / np.sqrt(dot_pp) for a in Cp)
        proj = sum(a * b for a, b in zip(Cpp, T))
        Nn = tuple(a - proj * b for a, b in zip(Cpp, T))
        nlen = np.sqrt(sum(a * a for a in Nn))
        N = tuple(a / nlen for a in Nn)
    else:
        N = nrm(W, p)
    Q = np.stack(np.broadcast_arrays(*_cross(N, Cp)), axis=-1)
    Q = np.where(np.isfinite(Q), Q, 0.0)
    # real-axis leg (real on the axis, so it only matters off-axis)
    jm = (nv - 1) // 2
    du = (t1 - t0) / max(len(u) - 1, 1) if not wrap_u \
        else (t1 - t0) / len(u)
    A = np.zeros((len(u), 3), dtype=complex)
    qa = Q[:, jm, :]
    A[1:] = np.cumsum(0.5 * (qa[1:] + qa[:-1]) * du, axis=0)
    # vertical legs from the axis, both directions
    dv = (2.0 * vh) / max(nv - 1, 1)
    G = np.zeros((len(u), nv, 3), dtype=complex)
    up = np.cumsum(0.5 * (Q[:, jm:-1, :] + Q[:, jm + 1:, :]) * dv,
                   axis=1)
    G[:, jm + 1:, :] = up
    dn = np.cumsum(0.5 * (Q[:, jm:0:-1, :] + Q[:, jm - 1::-1, :]) * dv,
                   axis=1)
    G[:, jm - 1::-1, :] = -dn
    Cs = np.stack(np.broadcast_arrays(*C), axis=-1)
    F = Cs - 1j * A[:, None, :] + G            # int n x c' folded in
    X = np.real(np.exp(1j * theta) * F)
    return (X[..., 0], X[..., 1], X[..., 2], wrap_u, False,
            spec.get('clip', False))


# --------------------------------------------------------------------------
# Saddle-tower stacking (singly periodic Scherk / Karcher towers)
# --------------------------------------------------------------------------

def _rot_z(ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _open_boundary(V, quads):
    """Vertex indices on open (once-used) mesh edges, plus the median
    edge length -- used to size the seam-weld tolerance."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for q in quads:
        m = len(q)
        for k in range(m):
            a, b = q[k], q[(k + 1) % m]
            cnt[(a, b) if a < b else (b, a)] += 1
    bset = {v for (a, b), c in cnt.items() if c == 1 for v in (a, b)}
    lens = [float(np.linalg.norm(V[a] - V[b])) for (a, b) in cnt]
    med = float(np.median(lens)) if lens else 1.0
    return np.array(sorted(bset), dtype=np.int64), med


def we_saddle_tower(spec, nu, nv, order, radius, scale, theta, storeys):
    """Stack `storeys` copies of one saddle unit into a genuinely periodic
    tower.

    Each fundamental domain is a single 2n-winged saddle from the exact
    log-sum immersion (_tower_X) on the punctured disk.  Consecutive
    storeys are related by the surface's deck isometry -- a *screw motion*:
    a vertical rise T = pi/(2n) composed with a rotation by pi/n about the
    axis.  The screw (not a pure translation) is the true deck map: the two
    rims of one disk unit are related by a roto-reflection, not a z-shift,
    so no pure translation registers them -- whereas rot(pi/n) permutes the
    2n vertical wing-walls onto themselves, keeping every wall one flat,
    uncrossed vertical half-plane running the whole height (its azimuth is
    independent of z), and puts consecutive storeys in disjoint z-slabs so
    the tower stays embedded.  Storey-s top boundary and storey-(s+1) bottom
    boundary are the same seam curve, so a nearest-neighbour match between
    the two storeys' boundary rings welds the joins watertight."""
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    # Karcher unequal-wing tower: the angle knob (theta) is the alpha modulus,
    # not a Bonnet rotation; fold it into the params so the end positions /
    # residues / puncture mask all pick it up, and build a single fundamental
    # domain (the unequal-wing unit has no screw deck isometry to stack -- see
    # the SADDLE_TOWER_A note in zoo.py).
    if spec.get('alpha_from_theta'):
        p = dict(p, alpha=theta)
        theta = 0.0
        storeys = 1
    n = int(p['n'])
    S = max(1, int(storeys))
    rb = spec.get('res_boost')
    if rb:
        nu = max(3, int(round(nu * rb[0])))
        nv = max(3, int(round(nv * rb[1])))
    nv = max(2 * n, int(round(nv / (2 * n))) * (2 * n))   # whole wings
    # sample one fundamental domain (exact immersion + puncture mask)
    x, y, z, _, _, mask = _we_disk(spec, p, nu, nv, theta)
    Vg = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    vm = np.asarray(mask).reshape(-1)
    gu = np.arange(nu) / max(nu - 1, 1)
    gv = np.arange(nv) / nv
    UVg = np.stack(np.meshgrid(gu, gv, indexing='ij'), axis=-1).reshape(-1, 2)
    N = nu * nv

    def vid(i, j):
        return i * nv + j
    quads0 = []
    for i in range(nu - 1):
        for j in range(nv):                       # wrap in v
            j2 = (j + 1) % nv
            f = (vid(i, j), vid(i + 1, j), vid(i + 1, j2), vid(i, j2))
            if vm[f[0]] and vm[f[1]] and vm[f[2]] and vm[f[3]]:
                quads0.append(f)
    used = (np.unique(np.array(quads0).ravel()) if quads0
            else np.array([], dtype=np.int64))
    remap = np.full(N, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    V0 = Vg[used]
    UV0 = UVg[used]
    quads0 = [tuple(int(remap[i]) for i in q) for q in quads0]
    M = len(V0)

    T = math.pi / (2 * n)                          # one storey's height
    Rm = _rot_z(math.pi / n)

    if S > 1 and M:
        # Stack S copies under the surface's deck isometry -- the screw
        # motion screw(P) = P @ Rm.T + T*zhat (rotate pi/n about the axis,
        # rise T = pi/(2n)) -- and weld the shared seam.
        #
        # Why the screw (and not a pure vertical translation): the two rims
        # of one disk unit are related by a roto-reflection, not a pure
        # z-shift (domain map z -> e^{i pi/n} z acts on the surface as
        # rot * diag(1,1,-1) with ZERO z-shift), so no pure translation
        # tiles the unit -- its rims never register.  The screw is the true
        # deck isometry: rot(pi/n) permutes the 2n vertical wing-walls onto
        # themselves so every wall stays one flat, uncrossed vertical
        # half-plane running the whole height (its azimuth is independent of
        # z), while consecutive storeys occupy disjoint z-slabs
        # [sT - T/2, sT + T/2] and so cannot self-intersect.
        #
        # Seam weld: storey-s TOP boundary and storey-(s+1) BOTTOM boundary
        # are the *same space curve* (top rim = screw(bottom rim)), only
        # sampled at slightly offset grid points, so a nearest-neighbour
        # match between the two storeys' boundary rings welds it watertight.
        # The tolerance is a fraction of the slab height T: the only other
        # inter-storey approach is a full slab away (>= T), so wing side
        # edges -- free, and in disjoint slabs -- are never falsely merged.
        bnd, med = _open_boundary(V0, quads0)
        Vparts, Rp = [], np.eye(3)
        for s in range(S):
            Vparts.append(V0 @ Rp.T + np.array([0.0, 0.0, s * T]))
            Rp = Rp @ Rm.T                        # (Rm^{s+1}).T
        Vcat = np.concatenate(Vparts, axis=0)
        UVcat = np.concatenate([UV0] * S, axis=0)
        parent = np.arange(S * M)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        tol = min(0.45 * T, max(3.0 * med, 0.15 * T))
        for s in range(S - 1):
            A = Vcat[s * M + bnd]                 # storey s boundary ring
            B = Vcat[(s + 1) * M + bnd]           # storey s+1 boundary ring
            d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
            jmin = d.argmin(axis=1)
            dmin = d[np.arange(len(bnd)), jmin]
            for a in np.nonzero(dmin < tol)[0]:
                ra = find(s * M + int(bnd[a]))
                rb = find((s + 1) * M + int(bnd[jmin[a]]))
                if ra != rb:
                    parent[ra] = rb
        roots = np.array([find(a) for a in range(S * M)])
        uniq, inv = np.unique(roots, return_inverse=True)
        nuq = len(uniq)
        Vf = np.zeros((nuq, 3))
        UVf = np.zeros((nuq, 2))
        cnt = np.zeros(nuq)
        np.add.at(Vf, inv, Vcat)
        np.add.at(UVf, inv, UVcat)
        np.add.at(cnt, inv, 1)
        Vf /= cnt[:, None]
        UVf /= cnt[:, None]
        quads = []
        for s in range(S):
            for q in quads0:
                fq = tuple(int(inv[s * M + i]) for i in q)
                if len(set(fq)) >= 3:
                    quads.append(fq)
    else:
        Vf, quads, UVf = V0, quads0, UV0

    # the wing rims are cut exactly on the puncture circles by the
    # marching-squares clip in _we_disk (clip_punctures), so they arrive
    # already clean; a short boundary relaxation just evens out the sampling.
    Vf = _smooth_boundary(Vf, quads, iters=6)
    Vf = _center_fit(Vf, scale, Vf)
    return Vf, quads, UVf


def make_entry(key, spec):
    """PARAMETRIC/MESH_PARAM builder closure for a WE spec."""
    if spec.get('mesher'):
        # bespoke finished-mesh builder (higher-genus Chen-Gackstatter,
        # periodic Callahan-Hoffman-Meeks): the spec supplies the whole
        # mesher, the engine only wires it in.  A spec that declares a
        # storeys_label is periodic -- its mesher takes the storey count.
        def build(nu, nv, order, radius, scale, theta=0.0, storeys=1):
            if spec.get('storeys_label'):
                return spec['mesher'](spec, nu, nv, order, radius, scale,
                                      theta, storeys)
            return spec['mesher'](spec, nu, nv, order, radius, scale, theta)
        build.finished_mesh = True
    elif spec.get('tower'):
        def build(nu, nv, order, radius, scale, theta=0.0, storeys=1):
            return we_saddle_tower(spec, nu, nv, order, radius, scale,
                                   theta, storeys)
        build.finished_mesh = True
    elif spec['domain'][0] == 'halfplane':
        def build(nu, nv, order, radius, scale, theta=0.0, storeys=1):
            return we_surface(spec, nu, nv, order, radius, scale, theta)
        build.finished_mesh = True
    else:
        if spec.get('associate') and spec['domain'][0] not in \
                ('disk', 'rect'):
            raise ValueError(f"{key}: associate family requires a "
                             "simply connected domain")

        def build(nu, nv, order, radius, theta=0.0):
            return we_surface(spec, nu, nv, order, radius, None, theta)
        build.finished_mesh = False
    build.spec = spec
    build.__name__ = f"we_{key.lower()}"
    return build


def make_bjorling_entry(key, spec):
    def build(nu, nv, order, radius, theta=0.0):
        return bjorling_surface(spec, nu, nv, order, radius, theta)
    build.spec = spec
    build.finished_mesh = False
    build.__name__ = f"bj_{key.lower()}"
    return build


# ==========================================================================
# Schwarz P / Gyroid / Schwarz D -- the exact genus-3 associate (Bonnet)
# family from ONE Weierstrass representation with a single angle theta
# ==========================================================================
# Unlike the nodal marching-tets TPMS (mesh.tpms_add), which have no
# associate parameter, this is the *exact* Enneper-Weierstrass immersion
#     X_theta = Re[ e^{i theta} Int (om1, om2, om3) dz ]
# on the branched sphere.  A single angle theta continuously morphs the
# whole iconic family:  theta = 0  -> Schwarz P,  theta ~ 38.0148 deg ->
# Gyroid,  theta = 90 deg -> Schwarz D.  P and D are conjugate; the gyroid
# is the embedded associate discovered by A. Schoen (1970).
#
# Weierstrass data (Schwarz P and D share ONE algebraic Gauss map; branch
# points {0, +-1, +-3, inf}), as harvested from M. Weber's repository
# (research/msblog_harvest/triply_periodic.json):
#     om1 = -1 / ( sqrt(z) sqrt(z-1) sqrt(z+3) )
#     om2 = -2 / ( sqrt(z+3) sqrt(z+1) sqrt(z-1) sqrt(z-3) )
#     om3 = -i / ( sqrt(z) sqrt(z+1) sqrt(z-3) )
# These satisfy om1^2 + om2^2 + om3^2 = 0 (conformal minimal immersion,
# verified < 1e-16 by the self-test).
#
# Fundamental domain + tiling scheme.  The upper half z-plane (all six
# branch points lie on the real axis) is one fundamental domain -- a
# hyperbolic hexagon.  It is mapped conformally to the unit w-disk by the
# Cayley transform  w = (z-i)/(z+i)  (base point z=i -> w=0), and the WE
# 1-forms are integrated radially over that disk.  Its six rim arcs (the
# images of the six real intervals between consecutive branch points) are the
# boundary edges of the fundamental surface patch, and by the Schwarz
# reflection principle each edge is a symmetry element that continues the
# surface across itself.  WHICH isometry it is depends on the edge geometry,
# and this is read off each edge directly (an SVD of the sampled edge curve),
# so no theta-tracking is needed -- every special angle lives in its own fixed
# symmetry frame:
#   * theta = 0 (Schwarz P): all six edges are PLANAR geodesics lying in the
#     cube's coordinate mirror planes, so the continuation is a plane
#     REFLECTION.  Reflecting the patch across the six coordinate planes is
#     drift-free (the maps are exact and compose to exact translations), and
#     fills one cubic cell watertight -- the classic rounded-cube "P" network.
#   * theta = 90 deg (Schwarz D, P's conjugate): the six edges are STRAIGHT
#     lines, so the continuation is a 180-degree ROTATION about the line; the
#     2-fold reassembly fills the (larger) D cell.
#   * the gyroid (theta ~ 38.0148 deg): the edges are SKEW curves (it is
#     chiral -- no straight line and no mirror plane), so only 2-fold rotations
#     apply.  Its cell over-covers under the six edge-2-folds and is not
#     reassembled watertight here; the exact fundamental piece is built instead
#     (see the pgd_build / _pgd_tile_cell docstrings).
# Every generator's linear part is snapped to the exact 48-element cube group
# O_h so the generated space group closes without composition drift; a
# breadth-first orbit of the patch, welded on the shared symmetry elements,
# fills one cubic cell, and pgd_build arrays that cell on the verified cubic
# period (a ~ 4.31 for P, 4.69 for D), which is independent of theta up to the
# associate family's overall scale.
#
# References:
#   H. A. Schwarz, "Gesammelte Mathematische Abhandlungen" (1890) -- the P
#     and D surfaces and the reflection principle;
#   A. H. Schoen, "Infinite periodic minimal surfaces without
#     self-intersections", NASA TN D-5541 (1970) -- the gyroid as the
#     theta ~ 38.0148-degree associate of P;
#   A. Weyhaupt, "Deformations of the gyroid and lidinoid minimal
#     surfaces", Pacific J. Math. 235 (2008) 137-171;
#   H. Karcher, K. Polthier, "Construction of triply periodic minimal
#     surfaces", Phil. Trans. R. Soc. Lond. A 354 (1996) 2077-2104 -- the
#     reflection-group assembly of the P and D cells;
#   M. Weber, https://minimalsurfaces.blog/ (triply periodic) -- the
#     explicit g/dh data and the "Associate rPD" notebook this follows.

import itertools as _itertools

# branch points ordered by their w-disk rim angle (the vertices of the
# fundamental hexagon); edge i runs between vertex i and vertex i+1
_PGD_ORDER = ('inf', '-3', '-1', '0', '1', '3')
_PGD_ZBR = {'inf': None, '-3': -3.0, '-1': -1.0, '0': 0.0, '1': 1.0,
            '3': 3.0}


def _pgd_wbp(key):
    """w-disk rim position of a branch point (Cayley image of z)."""
    z = _PGD_ZBR[key]
    return 1.0 + 0j if z is None else (z - 1j) / (z + 1j)


def _pgd_angs():
    return {k: math.atan2(_pgd_wbp(k).imag, _pgd_wbp(k).real) % TAU
            for k in _PGD_ORDER}


def _pgd_om(z):
    """The three Weierstrass coordinate 1-forms (om1, om2, om3)."""
    z = np.asarray(z, dtype=complex)

    def s(a):
        return np.sqrt(z - a)
    om1 = -1.0 / (s(0.0) * s(1.0) * s(-3.0))
    om2 = -2.0 / (s(-3.0) * s(-1.0) * s(1.0) * s(3.0))
    om3 = -1j / (s(0.0) * s(-1.0) * s(3.0))
    return om1, om2, om3


def _pgd_z_of_w(w):
    return 1j * (1.0 + w) / (1.0 - w)


def _pgd_dzdw(w):
    return 2j / (1.0 - w) ** 2


def _pgd_forms_w(w, theta):
    """e^{i theta} * (om1, om2, om3) * dz/dw, evaluated on the w-disk."""
    z = _pgd_z_of_w(w)
    o1, o2, o3 = _pgd_om(z)
    J = _pgd_dzdw(w)
    return np.stack([o1 * J, o2 * J, o3 * J], axis=-1) * np.exp(1j * theta)


def _pgd_good_nv(nv):
    """Round nv up to a multiple of 4 so no offset ray lands on the 0/90/
    180/270-degree branch points (a ray through a branch point integrates
    a 1/sqrt singularity and blows the patch up)."""
    nv = int(nv)
    return nv + ((-nv) % 4) if nv % 4 else nv


def pgd_patch(theta, nu=64, nv=160, rmax=1.0, mask_eps=0.05):
    """One fundamental surface patch: the WE immersion integrated radially
    over the Cayley w-disk.  Returns (V (nu*nv, 3), faces, valid-mask,
    w-grid).  Small disks of radius `mask_eps` around the six rim branch
    points are dropped so the (parametrization-)singular hexagon corners
    become small clean holes rather than spikes."""
    nv = _pgd_good_nv(nv)
    s = np.linspace(0.0, 1.0, nu)
    r = rmax * (1.0 - (1.0 - s) ** 1.7)          # graded dense toward rim
    v = (np.arange(nv) + 0.5) * (TAU / nv)       # offset rays
    R, TH = np.meshgrid(r, v, indexing='ij')
    w = R * np.exp(1j * TH)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = _pgd_forms_w(w, theta) * np.exp(1j * TH)[..., None]
    F = np.where(np.isfinite(F), F, 0.0)
    dr = np.diff(R, axis=0)[..., None]
    X = np.concatenate([np.zeros((1, nv, 3)),
                        np.cumsum(0.5 * (F[1:] + F[:-1]) * dr, axis=0)],
                       axis=0)
    X = np.real(X)
    bw = np.array([_pgd_wbp(k) / abs(_pgd_wbp(k)) for k in _PGD_ORDER])
    valid = np.ones(w.shape, dtype=bool)
    for b in bw:
        valid &= np.abs(w - b) > mask_eps
    V = X.reshape(-1, 3)
    vm = valid.reshape(-1)
    faces = []
    for i in range(nu - 1):
        for j in range(nv):
            j2 = (j + 1) % nv
            if i == 0:
                f = (j, nv + j, nv + j2)         # center fan
            else:
                f = (i * nv + j, (i + 1) * nv + j,
                     (i + 1) * nv + j2, i * nv + j2)
            if vm[f[0]] and vm[f[1]] and vm[f[2]] and (len(f) == 3
                                                       or vm[f[3]]):
                faces.append(f)
    return V, faces, valid, w


def _pgd_grid_pts(theta, nu=60, nv=160):
    """Raw immersion sampled on a fixed w-disk grid (no masking) -- a
    parametrization-stable array for the Bonnet-morph continuity check."""
    nv = _pgd_good_nv(nv)
    s = np.linspace(0.0, 1.0, nu)
    r = 0.985 * (1.0 - (1.0 - s) ** 1.7)
    v = (np.arange(nv) + 0.5) * (TAU / nv)
    R, TH = np.meshgrid(r, v, indexing='ij')
    w = R * np.exp(1j * TH)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = _pgd_forms_w(w, theta) * np.exp(1j * TH)[..., None]
    F = np.where(np.isfinite(F), F, 0.0)
    dr = np.diff(R, axis=0)[..., None]
    X = np.concatenate([np.zeros((1, nv, 3)),
                        np.cumsum(0.5 * (F[1:] + F[:-1]) * dr, axis=0)],
                       axis=0)
    return np.real(X)


def _pgd_cube_rotations():
    """The 24 proper rotations of the cube (signed permutation matrices,
    det = +1) -- the point group O shared by the whole P/Gyroid/D family."""
    mats = []
    for perm in _itertools.permutations(range(3)):
        for sg in _itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = sg[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                mats.append(M)
    return np.array(mats)


_PGD_CUBE = _pgd_cube_rotations()


def _pgd_snap_rot(M):
    """Nearest exact cube rotation to M (kills composition drift)."""
    return _PGD_CUBE[int(np.argmin(
        np.abs(_PGD_CUBE - M).reshape(24, -1).max(axis=1)))]


def _pgd_edge_curve(theta, i, m=48, rr=0.9993):
    """Sample boundary edge i of the fundamental hexagon (the rim arc
    between two consecutive branch vertices), avoiding the singular tips."""
    angs = _pgd_angs()
    a, b = _PGD_ORDER[i], _PGD_ORDER[(i + 1) % 6]
    lo = angs[a]
    span = (angs[b] - lo) % TAU
    ts = lo + np.linspace(0.06, 0.94, m) * span
    s = np.linspace(0.0, 1.0, 2600)
    rr_ = rr * (1.0 - (1.0 - s) ** 2.0)
    dr = np.diff(rr_)[:, None]
    pts = []
    for t in ts:
        w = rr_ * np.exp(1j * t)
        with np.errstate(divide='ignore', invalid='ignore'):
            F = _pgd_forms_w(w, theta) * np.exp(1j * t)
        F = np.where(np.isfinite(F), F, 0.0)
        pts.append(np.real(np.sum(0.5 * (F[1:] + F[:-1]) * dr, axis=0)))
    return np.array(pts)


def _pgd_fit_twofold(C):
    """Proper 180-degree rotation R(x) = M x + b mapping the edge curve C to
    its own reversal (the surface's 2-fold rotation about that edge)."""
    Q = C[::-1]

    def resid(d):
        d = d / np.linalg.norm(d)
        M = 2.0 * np.outer(d, d) - np.eye(3)
        b = (Q - C @ M.T).mean(axis=0)
        return (float(np.sqrt(np.mean(np.sum((C @ M.T + b - Q) ** 2, 1)))),
                M, b)
    grid = [(math.sin(t) * math.cos(p), math.sin(t) * math.sin(p),
             math.cos(t))
            for t in np.linspace(0, math.pi, 13)
            for p in np.linspace(0, TAU, 25)]
    d0 = np.array(min(grid, key=lambda g: resid(np.array(g))[0]))
    step = 0.2
    for _ in range(80):
        r0 = resid(d0)[0]
        improved = False
        for e in np.eye(3):
            for sgn in (step, -step):
                dt = d0 + sgn * e
                if resid(dt)[0] < r0:
                    d0 = dt / np.linalg.norm(dt)
                    r0 = resid(dt)[0]
                    improved = True
        if not improved:
            step *= 0.5
        if step < 1e-4:
            break
    r, M, b = resid(d0)
    return M, b, r


def pgd_gluings(theta):
    """The six edge 2-fold rotations (M, b), linear parts snapped to exact
    cube rotations so the generated space group closes without drift.

    At exactly theta = 0 (P) and theta = pi/2 (D) the boundary edges are
    straight lines, so the 2-fold *axis* fit is degenerate (any axis in the
    plane perpendicular to the line reverses the segment) and would snap to
    an arbitrary cube rotation.  The correct rotation is the continuous
    limit, so the axis is fit at an angle nudged just inside the open
    interval (0, pi/2) -- where the edge has curved enough to pin the axis
    -- while the translation is refit at the true theta so the patch and
    its lattice stay exact."""
    tf = min(max(theta, 0.035), 0.5 * math.pi - 0.035)   # axis-fit angle
    gens = []
    for i in range(6):
        Cf = _pgd_edge_curve(tf, i)
        M, _b, _r = _pgd_fit_twofold(Cf)
        Ms = _pgd_snap_rot(M)
        C = _pgd_edge_curve(theta, i)
        b = (C[::-1] - C @ Ms.T).mean(axis=0)    # refit translation at theta
        gens.append((Ms, b))
    return gens


def _pgd_compose(g, h):
    return (g[0] @ h[0], g[0] @ h[1] + g[1])


def pgd_lattice(gens, maxlen=4):
    """Pure-translation lattice vectors of the tiling: words in the six
    generators (up to `maxlen` letters) whose linear part is the identity
    cube rotation.  P and the gyroid expose the cubic period in 2-letter
    words; the conjugate D surface (whose opposite edges compose to
    near-identity) needs longer words.  A real period has norm ~ 4; the
    small (< ~0.5) near-identity words are translation-fit drift and are
    filtered out.  Returned deduplicated, shortest first."""
    allg = []
    for M, b in gens:
        allg.append((M, b))
        allg.append((M.T, -M.T @ b))
    trans = []
    frontier = [(np.eye(3), np.zeros(3))]
    for _ in range(maxlen):
        nxt = []
        for g in frontier:
            for h in allg:
                gh = _pgd_compose(g, h)
                nxt.append(gh)
                if np.abs(gh[0] - np.eye(3)).max() < 1e-6:
                    n = np.linalg.norm(gh[1])
                    if 2.5 < n < 7.0:
                        trans.append(gh[1])
        frontier = nxt
    if not trans:
        return np.zeros((0, 3))
    trans = np.array(trans)
    trans = trans[np.argsort(np.linalg.norm(trans, axis=1))]
    keep = []
    for t in trans:
        if not any(np.linalg.norm(t - k) < 0.2 for k in keep):
            keep.append(t)
    return np.array(keep)


def _pgd_frames(gens, box, cap=600):
    """Breadth-first orbit of the identity frame under the six 2-fold
    generators (+ inverses), bounded to translations within `box` of the
    origin.  Cube-snapped rotations make the dedup key exact."""
    allg = []
    for M, b in gens:
        allg.append((M, b))
        allg.append((M.T, -M.T @ b))
    I = (np.eye(3), np.zeros(3))

    def rid(M):
        return int(np.argmin(np.abs(_PGD_CUBE - M).reshape(24, -1).max(1)))

    def key(g):
        return (rid(g[0]), tuple(np.round(g[1], 1)))
    frames = [I]
    seen = {key(I)}
    queue = [I]
    while queue:
        g = queue.pop()
        for h in allg:
            g2 = _pgd_compose(g, h)
            k = key(g2)
            if k in seen:
                continue
            if np.max(np.abs(g2[1])) > box:      # cen0 ~ 0, so t is centroid
                continue
            seen.add(k)
            frames.append(g2)
            queue.append(g2)
            if len(frames) >= cap:
                return frames
    return frames


def _pgd_smooth(V, tris, iters=6, lam=0.5):
    """A few Laplacian relaxation sweeps over the triangle mesh -- evens out
    the radial-grid sampling and the small seams left where cube-snapped
    neighbour patches meet, without moving the surface off itself."""
    if not tris or iters <= 0:
        return V
    T = np.asarray(tris, dtype=np.int64)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]],
                        T[:, [1, 0]], T[:, [2, 1]], T[:, [0, 2]]], axis=0)
    n = len(V)
    deg = np.zeros(n)
    np.add.at(deg, e[:, 0], 1.0)
    deg = np.maximum(deg, 1.0)
    V = V.copy()
    for _ in range(iters):
        acc = np.zeros_like(V)
        np.add.at(acc, e[:, 0], V[e[:, 1]])
        V += lam * (acc / deg[:, None] - V)
    return V


def _pgd_weld_tris(Vc, faces_per, nframes, nV, tol):
    """Concatenate `nframes` transformed copies of one patch (already in
    Vc) and weld coincident vertices (absolute tolerance `tol`, chosen a
    little under half a grid cell so adjacent patch seams merge but the
    patch's own grid is preserved); return welded V and triangle list."""
    q = np.round(Vc / tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    Vw = np.zeros((int(inv.max()) + 1, 3))
    Vw[inv] = Vc
    tris = []
    for fr in range(nframes):
        base = fr * nV
        for f in faces_per:
            idx = [int(inv[base + i]) for i in f]
            if len(f) == 3:
                if len(set(idx)) == 3:
                    tris.append(tuple(idx))
            else:
                a, b, c, d = idx
                if len({a, b, c}) == 3:
                    tris.append((a, b, c))
                if len({a, c, d}) == 3:
                    tris.append((a, c, d))
    return Vw, tris


def _pgd_patch_tris(theta, nu, nv, mask_eps=0.045):
    """The exact fundamental surface patch as a welded triangle mesh:
    integrate over the Cayley w-disk, weld the coincident centre-fan
    vertices, triangulate, and relax the sampling.  Small clean holes are
    left at the six (parametrization-singular) hexagon corners."""
    V0, faces, valid, w = pgd_patch(theta, nu=nu, nv=nv, rmax=1.0,
                                    mask_eps=mask_eps)
    diag = float(np.linalg.norm(V0.max(0) - V0.min(0))) or 1.0
    Vw, tris = _pgd_weld_tris(V0, faces, 1, len(V0), 1e-5 * diag)
    used = np.unique(np.array(tris, dtype=np.int64).ravel())
    remap = np.full(len(Vw), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    Vf = Vw[used]
    tris = [(int(remap[x]), int(remap[y]), int(remap[z]))
            for (x, y, z) in tris]
    Vf = _pgd_smooth(Vf, tris, iters=4, lam=0.5)
    return Vf, tris


# --- watertight space-group tiling of one unit cell (P and D) --------------
# The fundamental patch tiles space by the Schwarz reflection principle applied
# to its six hexagon edges.  Which isometry continues the surface across an
# edge depends on the edge's geometry (verified per edge by an SVD of the edge
# curve, so no theta-tracking is needed -- each special angle is handled in its
# own fixed symmetry frame):
#   * a PLANAR-but-curved edge  -> mirror reflection in the plane it lies in
#     (Schwarz P at theta = 0: all six edges are planar geodesics lying in the
#     cube's coordinate mirror planes -- so P assembles by pure coordinate-plane
#     reflections, which compose EXACTLY with no drift);
#   * a STRAIGHT edge           -> 180-degree rotation about the line (Schwarz D
#     at theta = 90 deg: the six edges are straight cube-diagonal lines);
#   * a SKEW edge               -> 180-degree rotation about the fitted 2-fold
#     axis (the chiral gyroid -- no mirror planes exist, so it never reflects).
# The linear part of every generator is snapped to the exact 48-element cube
# group O_h so the generated space group closes without composition drift; the
# breadth-first orbit of the patch, welded on the shared symmetry elements,
# fills one cubic cell, which pgd_build then arrays by the (verified) cubic
# period.  See Schwarz (1890) for the reflection principle and Karcher-Polthier,
# "Construction of triply periodic minimal surfaces", Phil. Trans. R. Soc. A
# 354 (1996) 2077-2104, for the reflection-group assembly of P and D.

def _pgd_cube48():
    """The 48 signed permutation matrices -- the full cube point group O_h
    (proper rotations det +1 and improper/mirror operations det -1)."""
    mats = []
    for perm in _itertools.permutations(range(3)):
        for sg in _itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = sg[i]
            mats.append(M)
    return np.array(mats)


_PGD_CUBE48 = _pgd_cube48()


def _pgd_snap48(M):
    """Nearest exact O_h operation to M (kills composition drift)."""
    return _PGD_CUBE48[int(np.argmin(
        np.abs(_PGD_CUBE48 - M).reshape(48, -1).max(axis=1)))]


def _pgd_rid48(M):
    return int(np.argmin(np.abs(_PGD_CUBE48 - M).reshape(48, -1).max(axis=1)))


def _pgd_arclen_mid(C):
    """Arc-length midpoint of a polyline -- a fixed point of the edge's 2-fold
    (it lies on the rotation axis) / a point of the edge's mirror plane."""
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    half = s[-1] / 2.0
    j = max(0, min(int(np.searchsorted(s, half)) - 1, len(C) - 2))
    f = (half - s[j]) / max(s[j + 1] - s[j], 1e-12)
    return C[j] * (1 - f) + C[j + 1] * f


def _pgd_edge_ops(theta):
    """The six edge-continuation isometries (M, b, kind): mirror for a planar
    edge, 2-fold for a straight/skew one.  M is snapped to O_h and the offset
    b = (I - M) q is taken from the edge's arc-length midpoint q (which lies on
    the mirror plane / rotation axis), giving an accurate, mutually consistent
    generator set."""
    tf = min(max(theta, 0.05), 0.5 * math.pi - 0.05)   # 2-fold axis-fit angle
    ops = []
    for i in range(6):
        C = _pgd_edge_curve(theta, i, m=400)
        _, s, vt = np.linalg.svd(C - C.mean(0), full_matrices=False)
        sr = s / s[0]
        q = _pgd_arclen_mid(C)
        if sr[2] < 0.003 and sr[1] > 0.02:             # planar curve -> mirror
            n = vt[2]
            M = np.eye(3) - 2.0 * np.outer(n, n)
            kind = 'mirror'
        else:                                          # straight/skew -> 2fold
            Cf = _pgd_edge_curve(tf, i, m=96)
            M, _b, _r = _pgd_fit_twofold(Cf)
            kind = 'twofold'
        Ms = _pgd_snap48(M)
        ops.append((Ms, (np.eye(3) - Ms) @ q, kind))
    return ops


def _pgd_lattice_basis(lat):
    """Three shortest linearly independent lattice vectors (basis rows)."""
    order = np.argsort(np.linalg.norm(lat, axis=1))
    basis = []
    for i in order:
        v = lat[i]
        if not basis or np.linalg.matrix_rank(
                np.vstack(basis + [v]), tol=1e-2) > len(basis):
            basis.append(v)
        if len(basis) == 3:
            break
    return np.array(basis)


def _pgd_coset_reps(gens, B, tol=0.25):
    """Canonical coset translation tau_R (reduced mod the lattice L) for each
    cube-group element reached -- lets composed frames be snapped back onto the
    exact space group {tau_R + L n}, so the orbit stays drift-free and finite
    even for the 2-fold (D / gyroid) assemblies."""
    Binv = np.linalg.inv(B.T)

    def redmod(t):
        c = Binv @ t
        return B.T @ (c - np.round(c))
    tau = {_pgd_rid48(np.eye(3)): np.zeros(3)}
    queue = [(np.eye(3), np.zeros(3))]
    guard = 0
    while queue and guard < 5000:
        guard += 1
        g = queue.pop(0)
        for M, b in gens:
            M2 = g[0] @ M
            t2 = g[0] @ b + g[1]
            r = _pgd_rid48(M2)
            tr = redmod(t2)
            if r in tau and np.linalg.norm(redmod(tr - tau[r])) < tol:
                continue
            if r in tau and np.linalg.norm(tr) >= np.linalg.norm(tau[r]):
                continue
            tau[r] = tr
            queue.append((_PGD_CUBE48[r], tr))
    return tau


def _pgd_tile_cell(theta, nu, nv, mask_eps=0.02, smooth=3):
    """Assemble one watertight filled unit cell of the (periodic) surface at
    `theta` by orbiting the fundamental patch under its six edge isometries and
    welding on the shared symmetry elements.  Returns (V, tris, a) in natural
    (un-fit) coordinates with a the cubic period, or None if the angle is not
    one of the cleanly-tileable members (only P and D reassemble watertight)."""
    ops = _pgd_edge_ops(theta)
    kinds = [k for (_M, _b, k) in ops]
    # gyroid / generic angles: the 2-fold reassembly over-covers and cannot be
    # made watertight here -> caller falls back to the fundamental piece.
    is_P = all(k == 'mirror' for k in kinds)
    is_D = all(k == 'twofold' for k in kinds) and \
        abs(theta - 0.5 * math.pi) < 0.02
    if not (is_P or is_D):
        return None
    Vp, tris_p = _pgd_patch_tris(theta, nu, nv, mask_eps=mask_eps)
    diag = float(np.linalg.norm(Vp.max(0) - Vp.min(0))) or 1.0
    cen0 = Vp.mean(0)
    gens = [(M, b) for (M, b, k) in ops]
    lat = pgd_lattice(gens)
    a = float(np.min(np.linalg.norm(lat, axis=1))) if len(lat) else 4.4
    B = _pgd_lattice_basis(lat) if len(lat) >= 3 else None
    tau = _pgd_coset_reps(gens, B) if B is not None else None
    Binv = np.linalg.inv(B.T) if B is not None else None

    def snap_frame(M2, b2):
        if tau is None:
            return b2
        tr = tau.get(_pgd_rid48(M2))
        if tr is None:
            return b2
        return tr + B.T @ np.round(Binv @ (b2 - tr))

    box = 0.5 * a + 0.05 * a
    # interior sample points for the interpenetration test (2-fold orbits of the
    # over-large patch would otherwise pile copies onto the same sheet)
    rin = np.linalg.norm(Vp - cen0, axis=1)
    Vint = Vp[np.argsort(rin)[:max(24, len(Vp) // 5)]]
    ohit = 0.06 * diag
    ghash = {}

    def _gk(p):
        return (int(round(p[0] / ohit)), int(round(p[1] / ohit)),
                int(round(p[2] / ohit)))

    def add_pts(P):
        for p in P:
            ghash.setdefault(_gk(p), []).append(p)

    def overlap_frac(P):
        hit = 0
        for p in P:
            gk = _gk(p)
            f = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for q in ghash.get((gk[0] + dx, gk[1] + dy,
                                            gk[2] + dz), ()):
                            if (abs(q[0] - p[0]) < ohit
                                    and abs(q[1] - p[1]) < ohit
                                    and abs(q[2] - p[2]) < ohit):
                                f = True
                                break
                        if f:
                            break
                    if f:
                        break
            hit += f
        return hit / len(P)

    I = (np.eye(3), np.zeros(3), +1)
    frames = [I]
    add_pts(Vint)
    dedup = max(0.12 * a, 0.15)
    seen = {tuple(np.round(cen0 / dedup).astype(int))}
    queue = [I]
    while queue and len(frames) < 400:
        (M, b, o) = queue.pop(0)
        for (Mg, bg) in gens:
            M2 = Mg @ M
            b2 = snap_frame(M2, Mg @ b + bg)
            c = M2 @ cen0 + b2
            if np.max(np.abs(c)) > box:
                continue
            key = tuple(np.round(c / dedup).astype(int))
            if key in seen:
                continue
            seen.add(key)
            cand = Vint @ M2.T + b2
            if overlap_frac(cand) > 0.4:               # would interpenetrate
                continue
            o2 = o * int(round(np.linalg.det(Mg)))
            frames.append((M2, b2, o2))
            add_pts(cand)
            queue.append((M2, b2, o2))
    # assemble with consistent winding (mirror copies flip orientation)
    Vparts, Tp, base = [], [], 0
    for (M, b, o) in frames:
        Vparts.append(Vp @ M.T + b)
        for (x, y, z) in tris_p:
            Tp.append((x + base, z + base, y + base) if o < 0
                      else (x + base, y + base, z + base))
        base += len(Vp)
    Vc = np.concatenate(Vparts, axis=0)
    tol = 0.9 * diag / nu
    q = np.round(Vc / tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    Vw = np.zeros((int(inv.max()) + 1, 3))
    Vw[inv] = Vc
    tris = []
    for (x, y, z) in Tp:
        t = (int(inv[x]), int(inv[y]), int(inv[z]))
        if len(set(t)) == 3:
            tris.append(t)
    Vw = _pgd_smooth(Vw, tris, iters=smooth, lam=0.4)
    return Vw, tris, a


def pgd_build(cells, res, scale, theta):
    """Build the exact Schwarz P / Gyroid / Schwarz D associate surface at
    Bonnet angle `theta`, centered and fit to a 2 m cube (times `scale`).
    Returns (V (n,3) float, tris list).

    Two regimes, chosen by the angle:

    * P (theta ~ 0) and D (theta ~ 90 deg) are assembled into a *watertight
      filled unit cell* by the Schwarz reflection principle -- P by reflecting
      the fundamental patch across its six coordinate mirror planes, D by
      180-degree rotations about its six straight edge lines (see
      `_pgd_tile_cell`).  `cells` > 1 arrays that whole cell on the verified
      cubic period, so the recognizable P / D network fills the lattice.
      `cells` may be an int (symmetric block) or a (cx, cy, cz) triple for
      independent per-axis counts.

    * The Gyroid (theta ~ 38.0148 deg) and every generic (non-periodic)
      intermediate angle keep the exact Weierstrass *fundamental piece* -- the
      image of one translational fundamental domain -- which morphs
      continuously and correctly through the whole family.  Only P / Gyroid / D
      are truly triply periodic, and the chiral gyroid (no mirror planes) has
      no drift-free reflection assembly: its 2-fold reassembly over-covers and
      cannot be closed watertight in this scheme, so -- following the honest-
      scope rule -- the clean fundamental piece is built rather than a torn
      multi-cell approximation.  (`cells` is ignored for these angles; the
      single piece is returned.)"""
    if isinstance(cells, (tuple, list)):
        cx, cy, cz = (int(max(1, c)) for c in (list(cells) + [1, 1, 1])[:3])
    else:
        cx = cy = cz = max(1, int(cells))
    nu = max(24, int(round(res)))
    nv = _pgd_good_nv(max(120, int(round(res * 2.4))))
    tiled = _pgd_tile_cell(theta, nu, nv)
    if tiled is not None:
        Vc, tris_c, a = tiled
        if cx > 1 or cy > 1 or cz > 1:
            oxs = (np.arange(cx) - 0.5 * (cx - 1)) * a
            oys = (np.arange(cy) - 0.5 * (cy - 1)) * a
            ozs = (np.arange(cz) - 0.5 * (cz - 1)) * a
            Vparts, Tparts, base = [], [], 0
            for ox in oxs:
                for oy in oys:
                    for oz in ozs:
                        Vparts.append(Vc + np.array([ox, oy, oz]))
                        Tparts.extend((x + base, y + base, z + base)
                                      for (x, y, z) in tris_c)
                        base += len(Vc)
            V, tris = np.concatenate(Vparts, axis=0), Tparts
        else:
            V, tris = Vc, tris_c
    else:
        # gyroid / generic angle: honest single fundamental piece
        V, tris = _pgd_patch_tris(theta, nu, nv)
    lo, hi = V.min(0), V.max(0)
    cen = 0.5 * (lo + hi)
    ext = float(np.max(hi - lo)) or 1.0
    V = (V - cen) * (2.0 / ext) * float(scale)
    return V, tris


# ==========================================================================
# Higher-genus Chen-Gackstatter surfaces (genus 2, 4, 5)
# ==========================================================================
# Complete minimal surfaces of genus g with ONE Enneper-type end of winding
# 3: the higher-genus continuation of the Chen-Gackstatter torus.  They live
# on the hyperelliptic curve  y^2 = z * prod_i (z^2 - r_i^2)  with the
# unified Weierstrass data
#     g = rho * y / D(z),      dh = dz          (so x3 = Re z),
# where D collects half of the branch factors (for genus 2 the z sits in the
# DENOMINATOR of g^2, for genus 4/5 in the numerator) and the real constants
# r_i, rho solve the period problem (values below verified numerically:
# the null identity |phi1^2 + phi2^2 + phi3^2| ~ 1e-16 and all periods of
# the double cover close).
#
# Symmetry group: D2d of order 8, the SAME for every genus:
#   * z -> -z acts on the immersion as the rotoreflection S4 about the
#     vertical axis, (x1, x2, x3) -> (x2, -x1, -x3)  (g -> i g, dh -> -dh);
#   * S4^2 = C2 is the hyperelliptic sheet swap (y -> -y);
#   * z -> conj(z) and z -> -conj(z) are antiholomorphic: two vertical
#     sigma_d mirror planes x = cx, y = cy (the images of the real-axis
#     branch intervals) and two horizontal 2-fold axes at x3 = 0 (the
#     images of the imaginary axis);
#   * the S4 axis threads every branch-point image at (cx, cy, z_branch).
#
# Meshing scheme (the part that makes the assembly watertight): ONE
# 1/8 Coxeter fundamental domain -- the quarter  {Re z >= 0}  of the upper
# half plane, i.e. the half  {Im w <= 0}  of the Cayley w-disk
# (w = (z-i)/(z+i)) -- whose boundary lies ON the symmetry elements:
# the real-w diameter maps onto a horizontal 2-fold axis (a straight line
# through (cx, cy, 0) at 45 degrees to the mirror planes), and the rim
# semicircle maps into the two mirror planes, split at the branch points.
# The WE forms are integrated radially on one continuous sqrt branch, each
# boundary vertex is snapped exactly onto its symmetry element (line /
# plane / axis point), the patch is orbited under the full 8-element D2d
# group and the copies weld by coincidence -- boundary vertices only, so
# interior verts can never fuse.  Face winding flips exactly for the four
# antiholomorphic copies (mirrors and 2-fold axes), NOT by det: S4 has
# det -1 but is holomorphic (z -> -z) and keeps its winding.  A spherical
# clip about (cx, cy, 0) trims the one flaring end; with the trim radius
# past the outermost handles the result is exactly Euler characteristic
# chi = 1 - 2g (one boundary loop), edge-manifold and connected -- gated
# by the self-tests below.
#
# References:
#   C. C. Chen, F. Gackstatter, "Elliptische und hyperelliptische
#     Funktionen und vollstaendige Minimalflaechen vom Enneperschen Typ",
#     Math. Ann. 259 (1982) -- the genus-1 and genus-2 surfaces;
#   E. C. Thayer, "Higher-genus Chen-Gackstatter surfaces and the
#     Weierstrass representation for surfaces of infinite genus",
#     Experiment. Math. 4 (1995) -- the genus >= 2 family;
#   H. Karcher, "Construction of minimal surfaces", Univ. of Tokyo lecture
#     notes (1989) -- the symmetry/period method;
#   M. Weber, https://minimalsurfaces.blog/ (higher-genus Chen-Gackstatter
#     pages) -- the numerical data this implementation follows.

_CGH_DATA = {
    2: dict(roots=(1.0, 1.7126826390981942),
            Dfac=(('z',), ('sq', 1.7126826390981942)),
            rho=None),                # solved from the [0,1] period ratio
    4: dict(roots=(1.0, 1.81645934660556296, 3.11436011061010598,
                   3.77509108812262628),
            Dfac=(('sq', 1.0), ('sq', 3.11436011061010598)),
            rho=0.580558059350863508),
    5: dict(roots=(1.0, 2.19951977246661467, 3.04734348507243302,
                   4.58374227188035909, 5.28690084560405004),
            Dfac=(('sq', 1.0), ('sq', 3.04734348507243302),
                  ('sq', 5.28690084560405004)),
            rho=1.97502242055676724),
}

_CGH_RHO_CACHE = {}


def _cgh_forms(genus):
    """(Pfun, Dfun, rho, roots) for the genus: y^2 = P(z), g = rho*y/D."""
    d = _CGH_DATA[genus]
    roots = d['roots']

    def Pfun(z):
        z = np.asarray(z, dtype=complex)
        out = z.copy()
        for r in roots:
            out = out * (z ** 2 - r ** 2)
        return out

    def Dfun(z):
        z = np.asarray(z, dtype=complex)
        out = np.ones_like(z)
        for f in d['Dfac']:
            out = out * (z if f[0] == 'z' else (z ** 2 - f[1] ** 2))
        return out

    rho = d['rho']
    if rho is None:                    # genus 2: scalar period ratio on [0,1]
        if genus not in _CGH_RHO_CACHE:
            zz = np.linspace(1e-9, 1 - 1e-9, 400000)
            y = np.sqrt(Pfun(zz))
            _trapz = getattr(np, 'trapezoid', None) or np.trapz
            _CGH_RHO_CACHE[genus] = math.sqrt(
                _trapz((y / (zz ** 2 - 1.0)).real, zz)
                / _trapz(((zz ** 2 - 1.0) / y).real, zz))
        rho = _CGH_RHO_CACHE[genus]
    return Pfun, Dfun, rho, roots


def _cgh_octant(genus, nu, arcn):
    """Mesh the 1/8 fundamental domain (half of the Cayley disk,
    Im w <= 0) by radial integration from w = 0 (z = i).  Returns
    (V, faces, boundary-classification dict, per-vertex uv)."""
    Pfun, Dfun, rho, roots = _cgh_forms(genus)
    bps = [0.0] + sorted(roots)        # finite branch points, z = 0 first
    ang = [np.angle((b - 1j) / (b + 1j)) % TAU for b in bps]
    ang = [a + TAU if a < math.pi - 1e-12 else a for a in ang]
    ang.append(TAU)                    # z = inf at w = 1
    # angular grid with the exact branch angles as shared nodes
    th = []
    for k in range(len(ang) - 1):
        n_k = max(8, int(round(arcn * (ang[k + 1] - ang[k])
                               / (math.pi / 4))))
        seg = np.linspace(ang[k], ang[k + 1], n_k + 1)
        th.extend(seg[:-1] if k < len(ang) - 2 else seg)
    th = np.array(th)
    bcol = [int(np.argmin(np.abs(th - a))) for a in ang[:-1]]
    nv = len(th)
    s = np.linspace(0.0, 1.0, nu)
    r = 1.0 - (1.0 - s) ** 1.7         # radially graded toward the rim
    R, TH = np.meshgrid(r, th, indexing='ij')
    w = R * np.exp(1j * TH)
    # one continuous branch of sqrt(P(z(w))) along the radial rays
    P = Pfun(1j * (1.0 + w) / (1.0 - w))
    pang = np.unwrap(np.angle(P), axis=0)
    pang = pang - pang[:1, :] + np.unwrap(pang[0])[None, :]
    ycont = np.sqrt(np.abs(P)) * np.exp(0.5j * pang)
    z = 1j * (1.0 + w) / (1.0 - w)
    g = rho * ycont / Dfun(z)
    J = 2j / (1.0 - w) ** 2            # dz/dw
    with np.errstate(divide='ignore', invalid='ignore'):
        F = np.stack([0.5 * (1.0 / g - g) * J,
                      0.5j * (1.0 / g + g) * J,
                      np.broadcast_to(J, g.shape)], axis=-1) \
            * np.exp(1j * TH)[..., None]
    F = np.where(np.isfinite(F), F, 0.0)
    dr = np.diff(R, axis=0)[..., None]
    X = np.concatenate([np.zeros((1, nv, 3)),
                        np.cumsum(0.5 * (F[1:] + F[:-1]) * dr, axis=0)],
                       axis=0)
    X = np.real(X)

    def vid(i, j):                     # row 0 collapses to one center vert
        return 0 if i == 0 else (i - 1) * nv + j + 1

    V = np.concatenate([X[:1, 0, :], X[1:].reshape(-1, 3)], axis=0)
    UV = np.zeros((len(V), 2))
    UV[0] = (0.5, 0.0)
    UV[1:, 0] = np.tile((th - math.pi) / math.pi, nu - 1)
    UV[1:, 1] = np.repeat(r[1:], nv)
    faces = []
    for j in range(nv - 1):
        faces.append((0, vid(1, j), vid(1, j + 1)))
    for i in range(1, nu - 1):
        for j in range(nv - 1):
            faces.append((vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)))
    b = {'seam': np.array([0] + [vid(i, j) for i in range(1, nu - 1)
                                 for j in (0, nv - 1)], dtype=np.int64),
         'rim': {}, 'branch': {}, 'branch_z': {}}
    for k in range(len(ang) - 1):
        j0 = bcol[k]
        j1 = bcol[k + 1] if k + 1 < len(bcol) else nv - 1
        b['rim'][k] = np.array([vid(nu - 1, j) for j in range(j0 + 1, j1)],
                               dtype=np.int64)
    for k in range(len(bcol)):
        b['branch'][k] = vid(nu - 1, bcol[k])
        b['branch_z'][k] = bps[k]
    # rim ring triples (rim vert, its inner neighbor, next inner) -- used
    # to keep the snapped rim clear of the last interior row
    b['ring'] = np.array([[vid(nu - 1, j), vid(nu - 2, j), vid(nu - 3, j)]
                          for j in range(nv)], dtype=np.int64)
    return V, faces, b, UV


def _cgh_snap(V, b):
    """Snap every boundary vertex exactly onto its symmetry element.
    The mirror-plane offsets cx, cy are read off the rim arcs themselves
    (each arc's near-constant coordinate); returns the axis point q."""
    arc_ax, arc_val = {}, {}
    for k, idx in b['rim'].items():
        if len(idx) == 0:
            continue
        sx = float(np.median(np.abs(V[idx, 0] - np.median(V[idx, 0]))))
        sy = float(np.median(np.abs(V[idx, 1] - np.median(V[idx, 1]))))
        ax = 0 if sx < sy else 1
        arc_ax[k] = ax
        arc_val[k] = float(np.median(V[idx, ax]))
    cx = float(np.median([v for k, v in arc_val.items() if arc_ax[k] == 0]))
    cy = float(np.median([v for k, v in arc_val.items() if arc_ax[k] == 1]))
    q = np.array([cx, cy, 0.0])
    for k, idx in b['rim'].items():
        if len(idx):
            V[idx, arc_ax[k]] = q[arc_ax[k]]
    # the imaginary axis maps onto the horizontal 2-fold line through q at
    # 45 degrees to the mirror planes; project the seam onto the best of
    # the two candidate directions
    sv = b['seam']
    Pq = V[sv] - q
    best = None
    for sgn in (1.0, -1.0):
        d = np.array([1.0, sgn, 0.0]) / math.sqrt(2.0)
        t = Pq @ d
        res = float(np.median(np.linalg.norm(
            Pq - t[:, None] * d[None, :], axis=1)))
        if best is None or res < best[0]:
            best = (res, d)
    d = best[1]
    t = (V[sv] - q) @ d
    V[sv] = q + t[:, None] * d[None, :]
    # branch-point images: exactly on the S4 axis at height z_branch
    for k, vidx in b['branch'].items():
        V[vidx] = np.array([q[0], q[1], b['branch_z'][k]])
    return V, q


def _cgh_frames(q):
    """The 8 affine isometries of D2d about q.  Frames 0..3 are the
    holomorphic copies e, S4, C2, S4^3 (winding kept); frames 4..7 are
    the antiholomorphic mirror / 2-fold copies (winding reversed)."""
    S4 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    SX = np.diag([-1.0, 1.0, 1.0])
    mats = []
    M = np.eye(3)
    for _ in range(4):
        mats.append(M)
        M = S4 @ M
    mats += [SX @ Mk for Mk in mats]
    return [(Mk, q - Mk @ q) for Mk in mats]


def cg_higher_assemble(genus, nu, arcn, Rend):
    """Watertight D2d assembly: 8 snapped copies of the 1/8 domain,
    boundary-coincidence weld, spherical end trim.  Returns
    (V, faces, uv) of the largest component."""
    V0, faces0, b, UV0 = _cgh_octant(genus, nu, arcn)
    V0, q = _cgh_snap(V0.copy(), b)
    # keep the last interior row clear of the snapped rim: the plane snap
    # can land a rim vertex arbitrarily close to its inward neighbor, and
    # a later mesh-level weld (the operator's remove-doubles) would pinch
    # the sheets there.  Interior verts are free, so back the neighbor off
    # to the midpoint of its own inward edge.
    vr, v1, v2 = b['ring'][:, 0], b['ring'][:, 1], b['ring'][:, 2]
    close = np.linalg.norm(V0[v1] - V0[vr], axis=1) < 2e-4
    if np.any(close):
        V0[v1[close]] = 0.5 * (V0[vr[close]] + V0[v2[close]])
    bmask0 = np.zeros(len(V0), dtype=bool)
    bmask0[b['seam']] = True
    for idx in b['rim'].values():
        bmask0[idx] = True
    bmask0[list(b['branch'].values())] = True
    keep = np.linalg.norm(V0 - q, axis=1) <= Rend
    faces0 = [f for f in faces0 if all(keep[i] for i in f)]
    used = sorted(set(i for f in faces0 for i in f))
    rmv = {v: k for k, v in enumerate(used)}
    V0, UV0, bmask0 = V0[used], UV0[used], bmask0[used]
    faces0 = [tuple(rmv[i] for i in f) for f in faces0]
    nV = len(V0)
    frames = _cgh_frames(q)
    Vp, Fp = [], []
    for fr, (M, bb) in enumerate(frames):
        Vp.append(V0 @ M.T + bb)
        rev = fr >= 4                  # antiholomorphic copies flip winding
        for f in faces0:
            ff = tuple(int(x) + fr * nV for x in f)
            Fp.append(ff[::-1] if rev else ff)
    V = np.concatenate(Vp)
    UVall = np.tile(UV0, (len(frames), 1))
    N = len(V)
    # coincidence weld restricted to the snapped boundary verts
    tol = 1e-6
    parent = np.arange(N)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bidx = np.nonzero(np.tile(bmask0, len(frames)))[0]
    key = np.floor(V[bidx] / tol + 0.5).astype(np.int64)
    H = {}
    for t, i in enumerate(bidx):
        H.setdefault((int(key[t, 0]), int(key[t, 1]), int(key[t, 2])),
                     []).append(int(i))
    for t, i in enumerate(bidx):
        k0 = key[t]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in H.get((int(k0[0]) + dx, int(k0[1]) + dy,
                                    int(k0[2]) + dz), ()):
                        if j > i and np.linalg.norm(V[j] - V[i]) < tol:
                            ra, rb = find(int(i)), find(j)
                            if ra != rb:
                                parent[ra] = rb
    rt = np.array([find(a) for a in range(N)])
    uniq, inv = np.unique(rt, return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    UVw = np.zeros((len(uniq), 2))
    cnt = np.zeros(len(uniq))
    np.add.at(Vw, inv, V)
    np.add.at(UVw, inv, UVall)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    UVw /= cnt[:, None]
    F = []
    for f in Fp:
        gg = [int(inv[i]) for i in f]
        h = [gg[0]]
        for t in range(1, len(gg)):
            if gg[t] != h[-1]:
                h.append(gg[t])
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            F.append(tuple(h))
    # largest face-connected component (drops any stray trim islands)
    parent2 = np.arange(len(Vw))

    def find2(a):
        while parent2[a] != a:
            parent2[a] = parent2[parent2[a]]
            a = parent2[a]
        return a

    for f in F:
        for i in range(1, len(f)):
            ra, rb = find2(f[0]), find2(f[i])
            if ra != rb:
                parent2[ra] = rb
    from collections import Counter
    sizes = Counter(find2(f[0]) for f in F)
    root = sizes.most_common(1)[0][0]
    F = [f for f in F if find2(f[0]) == root]
    used = sorted(set(i for f in F for i in f))
    rmv = {v: k for k, v in enumerate(used)}
    return (Vw[used], [tuple(rmv[i] for i in f) for f in F], UVw[used])


def cg_higher_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """MESH_PARAM builder: finished (V, quads, uv), fit to the 2 m cube."""
    p = spec['p_from'](order, radius)
    genus = p['genus']
    pnu = int(np.clip(nu * 1.5, 80, 280))
    arcn = int(np.clip(nv * 0.65, 28, 110))
    R0 = {2: 5.5, 4: 6.0, 5: 8.0}[genus]
    lo, hi = {2: (3.2, 10.0), 4: (4.2, 10.0), 5: (6.2, 12.0)}[genus]
    Rend = float(np.clip(R0 * radius / 1.2, lo, hi))
    V, quads, uv = cg_higher_assemble(genus, pnu, arcn, Rend)
    V = _smooth_boundary(V, quads, iters=6)
    V = _center_fit(V, scale, V)
    return V, quads, uv


# ==========================================================================
# Translation-invariant genus-one helicoid ("helicoid with a handle")
# ==========================================================================
# The singly periodic minimal surface asymptotic to a helicoid whose
# quotient by its vertical translation is a rhombic torus minus two
# helicoidal ends -- i.e. a helicoid that carries ONE handle per period.
# It was the key existence step toward the (non-periodic) genus-one
# helicoid.
#
# Weierstrass data on the rhombic torus C/<1, tau>, written with the
# Jacobi theta function theta_11 (data as in Weber's notebook,
# harvested in research/msblog_harvest/singly_periodic.json under
# translation_invariant_helicoid_with_handle):
#
#   G(z)  = rho1 e^{i pi (b - 2z + 2 tau + b tau)}
#           theta(z + (b-2)c) theta(z - (1+b)c)
#           / ( theta(z + (b-1)c) theta(z - b c) ),      c = (1+tau)/2
#   dh    = theta(z + (b-2)c) theta(z - b c)
#           / ( theta(z + (b-1)c) theta(z - (1+b)c) ) dz / dhper
#
# Both are elliptic (fully periodic) on the torus; the four theta
# points on the diagonal are the two helicoidal ends (parameters 1-b
# and 1+b, where dh has simple poles) and the two points of vertical
# normal (parameters b and 2-b) that make the handle.  The solved
# period problem (all constants harvested verbatim, no re-solving
# here): tau = e^{i alpha0 deg}, plus rho1, b, dhper, and the domain
# constants r0, a0.  With them the lattice cycle z -> z+1 integrates
# to the exact vertical translation (0, 0, 2) and the cycle z -> z+tau
# closes to ~1e-7 (the self-test checks both).
#
# The parameter domain is the conformal half-strip R x (0, pi): w maps
# to the torus by  z = tst(tr(e^w)) (1+tau)/2, where tr is a real
# Moebius map and tst the Schwarz-Christoffel rectangle map
# F(arcsin s | 1/r0^2) / (2 K(1/r0^2)) + 1/2, evaluated for complex s
# via Carlson's R_F.  The strip covers half the torus; the immersed
# sheet contains the vertical z-axis and horizontal rulings at integer
# heights, and one fundamental cell is the sheet plus its 180-degree
# rotation about the z axis, welded along those lines; cells stack by
# (0, 0, 2).  A stack of S cells has genus exactly S (one handle per
# period; the self-test verifies chi = 2 - 2S - loops).
#
# References:
#   D. Hoffman, H. Karcher, F. Wei, "Adding handles to the helicoid",
#     Bull. Amer. Math. Soc. 29 (1993), 77-84.
#   D. Hoffman, H. Karcher, F. Wei, "The singly periodic genus-one
#     helicoid", Comment. Math. Helv. 74 (1999), 248-279.
#   D. Hoffman, M. Weber, M. Wolf, "An embedded genus-one helicoid",
#     Ann. of Math. 169 (2009), 347-448 (the non-periodic limit).
#   M. Weber, "The translation invariant helicoid with handle",
#     https://minimalsurfaces.blog/ (notebook data, 1996).
#   B. C. Carlson, "Numerical computation of real or complex elliptic
#     integrals", Numer. Algorithms 10 (1995), 13-98 (R_F).

# harvested constants (verbatim from the notebook; do not re-solve)
_G1H_ALPHA0 = 70.7083362972048057                 # degrees
_G1H_TAU = complex(np.exp(1j * np.pi * _G1H_ALPHA0 / 180.0))
_G1H_B = 0.629065098323904514
_G1H_RHO1 = 108.369522264594063 - 62.8417365006266681j
_G1H_DHPER = 0.386191090012370175 - 0.169838749468014027j
_G1H_R0 = 2.43050611112724901
_G1H_A0 = -0.409955776251214221
# x-position of the normal-symmetry point of the strip (the two sums
# solve tr(e^x) = 1 on y = 0 and tr(e^x) = -r0 on y = pi)
_G1H_SYM = -0.359811577777830482 - 0.528287934072206422
# strip x of the four rectangle corners (branch points of the domain
# map): tr(e^x) = 1, r0 on y = 0 and -1, -r0 on y = pi
_G1H_XA = -0.3598115777778303
_G1H_XB = 0.6834249528850583
_G1H_XC = -1.5715244647350952
_G1H_XD = -0.5282879340722066
_G1H_EPS = 1e-7      # inset from the strip boundary: keeps the Carlson
#                      arguments off their branch cut (negative reals)


def genus1helicoid_theta11(z, tau, nterms=30):
    """Jacobi theta_11 (odd theta), theta_11(z, tau) = 2 sum_{n>=0}
    (-1)^n q^{(n+1/2)^2} sin((2n+1) pi z), q = e^{i pi tau}.  z is
    first reduced modulo the lattice <1, tau> and the exact
    quasi-periodicity factors are applied, so the truncated series
    converges fast for any argument."""
    z = np.asarray(z, dtype=complex)
    m = np.round(z.imag / tau.imag)
    z1 = z - m * tau
    n = np.round(z1.real)
    z2 = z1 - n
    fac = ((-1.0) ** (m + n)
           * np.exp(-1j * np.pi * m * m * tau - 2j * np.pi * m * z2))
    q = np.exp(1j * np.pi * tau)
    s = np.zeros_like(z2)
    for k in range(nterms):
        s = s + ((-1.0) ** k * q ** ((k + 0.5) ** 2)
                 * np.sin((2 * k + 1) * np.pi * z2))
    return 2.0 * fac * s


def _g1h_rf(x, y, z, iters=26):
    """Carlson symmetric elliptic integral R_F for complex arguments
    off the negative real axis (duplication iteration + the standard
    5th-order tail; Carlson 1995)."""
    x = np.asarray(x, dtype=complex).copy()
    y = np.asarray(y, dtype=complex).copy()
    z = np.asarray(z, dtype=complex).copy()
    for _ in range(iters):
        sx, sy, sz = np.sqrt(x), np.sqrt(y), np.sqrt(z)
        lam = sx * sy + sy * sz + sz * sx
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)
    A = (x + y + z) / 3.0
    X = 1.0 - x / A
    Y = 1.0 - y / A
    Z = -(X + Y)
    E2 = X * Y - Z * Z
    E3 = X * Y * Z
    return (1.0 - E2 / 10.0 + E3 / 14.0 + E2 * E2 / 24.0
            - 3.0 * E2 * E3 / 44.0) / np.sqrt(A)


def _g1h_ellf(z, msq):
    """Incomplete elliptic integral F(arcsin z | m), analytically
    continued to the upper half plane: F = z R_F(1-z^2, 1-m z^2, 1)."""
    z = np.asarray(z, dtype=complex)
    return z * _g1h_rf(1.0 - z * z, 1.0 - msq * z * z, np.ones_like(z))


_G1H_M = 1.0 / (_G1H_R0 * _G1H_R0)
_G1H_QUOT = float(2.0 * _g1h_ellf(np.array(1.0 - 1e-15 + 0j),
                                  _G1H_M).real)          # 2 K(m)


def _g1h_tst(s):
    """Schwarz-Christoffel map: upper half plane -> rectangle
    [0,1] x [0, h] (the notebook's tst)."""
    return _g1h_ellf(s, _G1H_M) / _G1H_QUOT + 0.5


def _g1h_map(w):
    """Half-strip coordinate w = x + iy (0 < y < pi) -> torus coord."""
    ew = np.exp(np.asarray(w, dtype=complex))
    s = (-_G1H_A0 - _G1H_R0 * ew) / (-1.0 + _G1H_A0 * ew)
    return _g1h_tst(s) * 0.5 * (1.0 + _G1H_TAU)


def _g1h_omega(z):
    """The three Weierstrass 1-forms (om1, om2, om3) as functions of
    the torus coordinate (values w.r.t. dz), om3 normalized by the
    harvested dhper so the z -> z+1 cycle translates by (0, 0, 2)."""
    c = 0.5 * (1.0 + _G1H_TAU)
    th = genus1helicoid_theta11
    t1 = th(z + (_G1H_B - 2.0) * c, _G1H_TAU)
    t2 = th(z - (1.0 + _G1H_B) * c, _G1H_TAU)
    t3 = th(z + (_G1H_B - 1.0) * c, _G1H_TAU)
    t4 = th(z - _G1H_B * c, _G1H_TAU)
    e = np.exp(1j * np.pi * (_G1H_B - 2.0 * z + 2.0 * _G1H_TAU
                             + _G1H_B * _G1H_TAU))
    G = _G1H_RHO1 * e * t1 * t2 / (t3 * t4)
    o3 = (t1 * t4) / (t3 * t2) / _G1H_DHPER
    o1 = 0.5 * (1.0 / G - G) * o3
    o2 = 0.5j * (1.0 / G + G) * o3
    return o1, o2, o3


def _g1h_path_int(za, zb, n=20001):
    """Integral of (om1, om2, om3) along the straight segment za->zb."""
    t = np.linspace(0.0, 1.0, n)
    path = za + (zb - za) * t
    o1, o2, o3 = _g1h_omega(path)
    dz = np.diff(path)
    return np.array([np.sum(0.5 * (o[1:] + o[:-1]) * dz)
                     for o in (o1, o2, o3)])


def _g1h_graded(lo, hi, n, specials, w=0.2):
    """n samples on [lo, hi] clustered near each special value."""
    t = np.linspace(0.0, 1.0, n)
    x = lo + (hi - lo) * t
    sp = np.asarray(specials)
    for _ in range(3):
        d = np.min(np.abs(x[:, None] - sp[None, :]), axis=1)
        wgt = 1.0 / (w + d)
        cdf = np.concatenate([[0.0],
                              np.cumsum(0.5 * (wgt[1:] + wgt[:-1])
                                        * np.diff(x))])
        cdf /= cdf[-1]
        x = np.interp(t, cdf, x)
    return x


_G1H_SHEET_CACHE = {}


def genus1helicoid_sheet(r1=-2.5, nu=131, nv=53, K=10):
    """Immersed fundamental sheet over the half-strip
    [r1, SYM - r1] x [0, pi]: returns (xs, ys, X) with X (nu', nv, 3)
    real.  The x-grid is symmetric about SYM/2 and contains the four
    corner x-values exactly, so the sheet's straight boundary arcs
    (the z-axis segment and the horizontal rulings) land sample-exact
    and every weld of the assembly is vertex-to-vertex.  Cumulative
    trapezoid integration along grid lines with each interval
    subdivided K times (resolves the sqrt branch corners)."""
    ck = (round(r1, 6), nu, nv, K)
    if ck in _G1H_SHEET_CACHE:
        return _G1H_SHEET_CACHE[ck]
    x_hi = _G1H_SYM - r1
    corners = (_G1H_XA, _G1H_XB, _G1H_XC, _G1H_XD)
    spec = sorted(set(list(corners)
                      + [_G1H_SYM - c for c in corners]))
    xs = _g1h_graded(r1, x_hi, nu, spec)
    xs = np.unique(np.round(np.concatenate(
        [xs, _G1H_SYM - xs, spec, [_G1H_SYM - s for s in spec]]), 12))
    t = np.linspace(0.0, 1.0, nv)
    ys = _G1H_EPS + (np.pi - 2 * _G1H_EPS) * (0.5 - 0.5
                                              * np.cos(np.pi * t))
    nu2 = len(xs)
    j0 = nv // 2
    i0 = int(np.argmin(np.abs(xs - _G1H_SYM / 2.0)))

    def seg(wa, wb):
        tt = np.linspace(0.0, 1.0, K + 1)
        W = wa[:, None] + (wb - wa)[:, None] * tt[None, :]
        Z = _g1h_map(W)
        o1, o2, o3 = _g1h_omega(Z)
        O = np.stack([o1, o2, o3], axis=-1)
        dZ = np.diff(Z, axis=1)
        return np.sum(0.5 * (O[:, 1:] + O[:, :-1]) * dZ[..., None],
                      axis=1)

    F = np.zeros((nu2, nv, 3), complex)
    row = np.concatenate([np.zeros((1, 3), complex),
                          np.cumsum(seg(xs[:-1] + 1j * ys[j0],
                                        xs[1:] + 1j * ys[j0]), axis=0)])
    F[:, j0] = row - row[i0]
    for j in range(j0 + 1, nv):
        F[:, j] = F[:, j - 1] + seg(xs + 1j * ys[j - 1], xs + 1j * ys[j])
    for j in range(j0 - 1, -1, -1):
        F[:, j] = F[:, j + 1] - seg(xs + 1j * ys[j], xs + 1j * ys[j + 1])
    # base point: integrate from the lattice point 1 through tau/2 so
    # the surface's vertical line is exactly the z axis (the notebook's
    # w0 offset)
    C = _g1h_path_int(1.0 + 0.0j, _G1H_TAU / 2.0) \
        + _g1h_path_int(_G1H_TAU / 2.0,
                        complex(_g1h_map(xs[i0] + 1j * ys[j0])))
    out = (xs, ys, np.real(F + C[None, None, :]))
    _G1H_SHEET_CACHE[ck] = out
    return out


def _g1h_weld_pairs(V, quads, pairs):
    """Weld the given exact vertex-index pairs (union-find; merged
    positions averaged).  Returns (V', quads', vertex_map)."""
    n = len(V)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(a) for a in range(n)])
    uniq, first, inv = np.unique(roots, return_index=True,
                                 return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(Vw, inv, V)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    qw = []
    seen = set()
    for q in quads:
        f = tuple(int(inv[i]) for i in q)
        if len(set(f)) >= 3 and frozenset(f) not in seen:
            seen.add(frozenset(f))
            qw.append(f)
    return Vw, qw, first


def genus1helicoid_assemble(storeys=1, r1=-2.5, nu=131, nv=53, K=10):
    """Finished (V, quads, uv): `storeys` translational cells, each the
    strip sheet plus its 180-degree rotation about the z axis, stacked
    by (0, 0, 2) and welded along the shared straight arcs (the axis
    segments and the horizontal rulings).  Genus = storeys.

    Every weld is an exact grid-index correspondence -- the symmetric
    x-grid makes the partner of sample i the sample nu-1-i (x maps to
    SYM - x) -- so no positional tolerance is involved and dense grid
    regions can never over-merge:
      * axis, y=0 edge  (x in [XA, XB], z in [-2, -1] of the cell):
        sheet <-> rotated sheet at the SAME i (the axis is pointwise
        fixed by the 180-degree rotation);
      * axis, y=pi edge (x in [XC, XD], z in [-1, 0]): likewise;
      * in-cell ruling z = -1: E0 arc x < XA of one sheet <-> E1 arc
        x > XD of the other, i <-> nu-1-i;
      * cell-to-cell rulings: the E1 arc x < XC (z = 0) of cell k
        <-> the E0 arc x > XB (z = -2) of cell k+1's other sheet."""
    xs, ys, X = genus1helicoid_sheet(r1, nu, nv, K)
    nu2, nv2 = X.shape[:2]
    U, Vv = np.meshgrid((xs - xs[0]) / (xs[-1] - xs[0]), ys / np.pi,
                        indexing='ij')
    uv0 = np.stack([U, Vv], axis=-1).reshape(-1, 2)
    Rz = np.array([-1.0, -1.0, 1.0])
    zoff = -(storeys - 1)                      # center the stack
    sheets, flips = [], []
    for s in range(storeys):
        off = np.array([0.0, 0.0, 2.0 * s + zoff])
        sheets.append(X + off)
        flips.append(False)
        sheets.append(X * Rz + off)
        flips.append(True)
    V = np.concatenate([S.reshape(-1, 3) for S in sheets], axis=0)
    uv = np.concatenate([uv0] * len(sheets), axis=0)
    quads = []
    for k, fl in enumerate(flips):
        b = k * nu2 * nv2
        for i in range(nu2 - 1):
            for j in range(nv2 - 1):
                a = b + i * nv2 + j
                c = b + (i + 1) * nv2 + j
                q = (a, c, c + 1, a + 1)
                # the rotated sheets get reversed winding so the welded
                # surface is consistently oriented
                quads.append(q[::-1] if fl else q)

    iA = int(np.argmin(np.abs(xs - _G1H_XA)))
    iB = int(np.argmin(np.abs(xs - _G1H_XB)))
    iC = int(np.argmin(np.abs(xs - _G1H_XC)))
    iD = int(np.argmin(np.abs(xs - _G1H_XD)))

    def gid(sheet, i, j):
        return sheet * nu2 * nv2 + i * nv2 + j

    pairs = []
    for kk in range(storeys):
        p, r = 2 * kk, 2 * kk + 1
        for i in range(iA, iB + 1):            # axis segment on E0
            pairs.append((gid(p, i, 0), gid(r, i, 0)))
        for i in range(iC, iD + 1):            # axis segment on E1
            pairs.append((gid(p, i, nv2 - 1), gid(r, i, nv2 - 1)))
        for i in range(0, iA + 1):             # in-cell ruling z = -1
            pairs.append((gid(p, i, 0), gid(r, nu2 - 1 - i, nv2 - 1)))
            pairs.append((gid(r, i, 0), gid(p, nu2 - 1 - i, nv2 - 1)))
        if kk + 1 < storeys:                   # cell-to-cell rulings
            p2, r2 = 2 * (kk + 1), 2 * (kk + 1) + 1
            for i in range(0, iC + 1):
                pairs.append((gid(p, i, nv2 - 1),
                              gid(r2, nu2 - 1 - i, 0)))
                pairs.append((gid(r, i, nv2 - 1),
                              gid(p2, nu2 - 1 - i, 0)))
    Vw, qw, first = _g1h_weld_pairs(V, quads, pairs)
    return Vw, qw, uv[first]


def genus1helicoid_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """MESH_PARAM builder: finished (V, quads, uv), fit to the 2 m
    cube.  order = number of translational periods (= handles);
    radius sets how far the two helicoidal ends flare (the strip
    truncation)."""
    p = spec['p_from'](order, radius)
    storeys = p['storeys']
    r1 = -(1.7 + 0.8 * float(np.clip(radius / 1.2, 0.6, 2.0)))
    pnu = int(np.clip(nu * 1.8, 90, 240))
    pnv = int(np.clip(nv * 0.85, 36, 96))
    V, quads, uv = genus1helicoid_assemble(storeys, r1, pnu, pnv)
    V = _center_fit(V, scale, V)
    return V, quads, uv


# ==========================================================================
# Toroidal Karcher-Scherk towers (genus 1 per period)
# ==========================================================================
# Karcher-Scherk saddle towers with a VERTICAL HANDLE: singly periodic,
# genus 1 in the quotient, first mentioned in Karcher's Tokyo notes and
# presented on Weber's repository page (mirror ch157) whose notebook
# `Singly Scherk (g=1).nb` carries the data transcribed here:
#
#     on the torus C/<1, i tau1>, with s+- = (1 + tau)/2 +- (k-1)/(2k),
#     G  = theta11(z - s+) / theta11(z - s-),
#     dh = theta11(z - s-) theta11(z - s+)
#          / ( theta11(z - i a1) theta11(z - (1 + tau) + i a1) ),
#
# and per k a FindRoot-solved table of (tau1, a1) members killing the
# horizontal period, Re int_0^1 dh(tau/2 + t) dt = 0.  The tables for
# k = 3, 4, 5, 7, 8 are in the notebook; one member per k ships below
# and the record note carries which.
#
# What the deck maps do -- MEASURED, not assumed (the self-test keeps
# measuring them):
#   * z -> z + 1 closes exactly (translation 0 to quadrature): the
#     x-cycle is the HANDLE loop;
#   * z -> z + tau is a pure rotation by -2 pi / k about a vertical
#     axis (affine fit spread 6e-10), rise ZERO: the tower's k-fold
#     symmetry;
#   * the loop AROUND a dh pole (the helicoidal end at z = i a1)
#     translates by exactly (0, 0, T) with T = 2 pi |res dh| -- THIS is
#     the tower's vertical period.  For the k = 4, tau1 = 0.4 member
#     T = 1.077748, matching the independently recorded T_z ~ 1.077
#     from the earlier constant-extraction pass.  The horizontal
#     components of that loop vanish (measured < 1e-6 of T), which is
#     the geometric form of the solved period condition.
#
# MESHING.  One torus rectangle (window x in [-1/2, 1/2], y in
# [0, tau1]) is one WINDING of the tower; the two dh poles sit at
# (0, a1) and (0, tau1 - a1) inside it and are excised by a mask disk
# (the wing trim -- the notebook extends into the ends with an
# incomplete-elliptic-F chart instead; that refinement is future work
# and is what the `lx` values in the notebook size).  Integration runs
# down the x = 1/4 column and out along rows, so no path meets a pole.
# Successive windings are translated copies at (0, 0, s T) and the
# x = +1/2 edge of winding s IS the x = -1/2 edge of winding s+1 (same
# y grid, so the seam welds BY INDEX, positions averaged) -- the raw
# gap is 7e-4 at n = 240, pure quadrature, and averaging closes it.
#
# References:
# - H. Karcher, "Construction of minimal surfaces", Univ. of Tokyo
#   Surveys in Geometry (1989); Lecture Notes 12, SFB 256 Bonn --
#   the saddle towers and the Tokyo-notes lineage the page credits.
# - M. Weber, "Toroidal Karcher-Scherk Surfaces", minimalsurfaces.blog
#   (mirror ch157; notebook `Singly Scherk (g=1).nb`).
# - R. Yol, "Symmetrization of Minimal Surfaces in Three Dimensional
#   Euclidean Space", PhD thesis, Indiana University (2024), section
#   5.1.4 -- the generalized toroidal Karcher-Scherk family.

# --------------------------------------------------------------------------
# The ENDS BACKEND: torus / rectangle domains with punctures
# --------------------------------------------------------------------------
# Shared machinery for every row whose Weierstrass forms have POLES in
# the integration window -- ends of the surface.  Grew out of the
# toroidal Karcher-Scherk mesher (which now runs on it) and exists so
# that Hackman, the catenoid field, Lubeck-Batista and Scherk's fourth
# do not each reimplement the same four decisions slightly differently:
#
#   1. GRIDS cluster toward every puncture coordinate (`we_ends_grid`);
#   2. INTEGRATION runs on a spanning tree that no puncture touches:
#      one HIGHWAY row at the y farthest from every puncture, one BASE
#      COLUMN per x-strip between puncture columns, rows outward from
#      their strip's column (`we_ends_integrate`).  A path through a
#      pole is the failure mode that produced a 3.2e10-span "patch" in
#      the first Hackman attempt; the tree makes it impossible by
#      construction.
#   3. MASKS excise a disk of radius r0 around each puncture (the end
#      trim), and quad emission drops both masked cells and CUT
#      CURTAINS -- quads whose 3-D edge jump exceeds a caller-declared
#      threshold.  The curtains are not cosmetic: around an end with
#      nonzero vertical flux the immersion is multivalued, two adjacent
#      rows on opposite sides of the cut sit one winding apart, and a
#      quad bridging them renders as a vertical wall that the true
#      surface does not contain (the shipped toroidal-KS mesh carried
#      536 such edges before this backend existed).
#   4. `we_ends_loop` integrates all three forms around a puncture:
#      the end's translation vector.  For a solved singly periodic row
#      it is (0, 0, T) -- the toroidal-KS k=4 member reproduces its
#      independently recorded T = 1.077748 through exactly this call,
#      which is the backend's own self-test oracle -- and for a row
#      that claims "no period problem" it must vanish outright.

def we_ends_grid(window, punctures, n, ny=None, specials_x=(),
                 specials_y=()):
    """Graded (xs, ys) on window = (x0, x1, y0, y1), clustered toward
    every puncture coordinate and any extra specials."""
    x0, x1, y0, y1 = (float(v) for v in window)
    px = sorted({float(np.real(p)) for p in punctures} | set(specials_x))
    py = sorted({float(np.imag(p)) for p in punctures} | set(specials_y))
    xs = _tks_graded(x0, x1, int(n), px or [0.5 * (x0 + x1)])
    ys = _tks_graded(y0, y1, int(ny or max(24, int(n * 0.8))),
                     py or [0.5 * (y0 + y1)])
    return xs, ys


def we_ends_integrate(Wfn, xs, ys, punctures):
    """Cumulative integral of the three forms over the grid, on a
    spanning tree that avoids every puncture: highway row -> base
    column per strip -> rows outward within the strip.  Returns X
    (nx, ny, 3) real."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    Z = xs[:, None] + 1j * ys[None, :]
    Wg = Wfn(Z)
    nx, ny = len(xs), len(ys)
    px = sorted({float(np.real(p)) for p in punctures})
    py = [float(np.imag(p)) for p in punctures]
    # highway: the grid row farthest from every puncture y
    dy = np.min(np.abs(ys[:, None]
                       - np.asarray(py or [np.inf])[None, :]), axis=1)
    jh = int(np.argmax(dy))
    # strips between puncture x-columns; one base column per strip,
    # at the grid x farthest from the strip's walls
    walls = [xs[0] - 1.0] + px + [xs[-1] + 1.0]
    F = np.zeros(Z.shape + (3,), dtype=complex)
    dxs = np.diff(xs)
    dys = np.diff(ys)
    done_cols = []
    for w0, w1 in zip(walls[:-1], walls[1:]):
        sel = np.where((xs > w0 + 1e-12) & (xs < w1 - 1e-12))[0]
        if not len(sel):
            continue
        # distance measured against the CLAMPED walls, so a boundary
        # strip puts its base mid-strip instead of on the window edge
        # (an edge base column makes one of the two row sweeps empty)
        w0c, w1c = max(w0, xs[0]), min(w1, xs[-1])
        dwall = np.minimum(xs[sel] - w0c, w1c - xs[sel])
        ib = int(sel[int(np.argmax(dwall))])
        done_cols.append((ib, sel))
    # 1. the highway row, integrated once left-to-right from the first
    # base column (regular everywhere: jh is far from every puncture y)
    ib0 = done_cols[0][0]
    row = Wg[:, jh, :]
    Fh = np.zeros((nx, 3), dtype=complex)
    if ib0 < nx - 1:
        Fh[ib0 + 1:] = np.cumsum(0.5 * (row[ib0 + 1:] + row[ib0:-1])
                                 * dxs[ib0:, None], axis=0)
    if ib0 > 0:
        Fh[:ib0] = np.cumsum(0.5 * (row[ib0 - 1::-1] + row[ib0:0:-1])
                             * (-dxs[ib0 - 1::-1, None]),
                             axis=0)[::-1]
    # 2. per strip: base column vertically from the highway, then rows
    # outward from the base column -- never crossing a strip wall
    for ib, sel in done_cols:
        col = Wg[ib]
        Fc = np.zeros((ny, 3), dtype=complex)
        if jh < ny - 1:
            Fc[jh + 1:] = np.cumsum(0.5 * (col[jh + 1:] + col[jh:-1])
                                    * (1j * dys[jh:])[:, None], axis=0)
        if jh > 0:
            Fc[:jh] = np.cumsum(0.5 * (col[jh - 1::-1] + col[jh:0:-1])
                                * (-1j * dys[jh - 1::-1])[:, None],
                                axis=0)[::-1]
        Fc = Fc + Fh[ib][None, :]
        lo, hi = int(sel[0]), int(sel[-1])
        F[ib] = Fc
        if hi > ib:
            F[ib + 1:hi + 1] = Fc[None] + np.cumsum(
                0.5 * (Wg[ib + 1:hi + 1] + Wg[ib:hi])
                * dxs[ib:hi, None, None], axis=0)
        if lo < ib:
            F[lo:ib] = (Fc[None] + np.cumsum(
                0.5 * (Wg[ib - 1:lo - 1 if lo else None:-1]
                       + Wg[ib:lo:-1])
                * (-dxs[ib - 1:lo - 1 if lo else None:-1, None, None]),
                axis=0))[::-1]
    return np.real(F)


def we_ends_mask(xs, ys, punctures, r0):
    Z = np.asarray(xs)[:, None] + 1j * np.asarray(ys)[None, :]
    m = np.ones(Z.shape, dtype=bool)
    for p_ in punctures:
        m &= np.abs(Z - complex(p_)) > float(r0)
    return m


def we_ends_quads(X, mask, jump=None):
    """Quad list over the grid: masked cells dropped, and -- when
    `jump` is given -- any quad with a 3-D edge longer than it (the
    cut curtains; see the backend header)."""
    nx, ny = X.shape[0], X.shape[1]
    quads = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if not (mask[i, j] and mask[i + 1, j]
                    and mask[i + 1, j + 1] and mask[i, j + 1]):
                continue
            if jump is not None:
                c = (X[i, j], X[i + 1, j], X[i + 1, j + 1], X[i, j + 1])
                if max(float(np.linalg.norm(c[t] - c[(t + 1) % 4]))
                       for t in range(4)) > jump:
                    continue
            quads.append((i * ny + j, (i + 1) * ny + j,
                          (i + 1) * ny + j + 1, i * ny + j + 1))
    return quads


def we_ends_loop(Wfn, p, r=0.015, n=8001):
    """Translation of the loop around puncture p: the end's period
    vector (real part of the contour integral of all three forms)."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    zz = complex(p) + r * np.exp(1j * t)
    dzdt = 1j * r * np.exp(1j * t)
    return np.real(np.trapezoid(Wfn(zz) * dzdt[:, None], t, axis=0))


# k -> (tau1, a1, lx): one FindRoot-solved member per wing order,
# straight from the notebook's tables (lx is the notebook's own end
# extent for that k, recorded for provenance).
TOROIDAL_KS_MEMBERS = {
    3: (1.0, 0.38900635790684035, 8),
    4: (0.4, 0.13619041259273051, 12),
    5: (0.5, 0.21484200805849266, 16),
    7: (0.2, 0.12152998199565702, 20),
    8: (0.15, 0.056545236758766944, 20),
}


def _tks_forms(k, tau1, a1):
    """(G, dh, W) callables for one member, on the shipped theta."""
    th = genus1helicoid_theta11
    tau = 1j * float(tau1)
    spl = (1.0 + tau) / 2.0 + (k - 1) / (2.0 * k)
    smi = (1.0 + tau) / 2.0 - (k - 1) / (2.0 * k)

    def G(z):
        z = np.asarray(z, dtype=complex)
        return th(z - spl, tau) / th(z - smi, tau)

    def dh(z):
        z = np.asarray(z, dtype=complex)
        return (th(z - smi, tau) * th(z - spl, tau)
                / (th(z - 1j * a1, tau)
                   * th(z - (1.0 + tau) + 1j * a1, tau)))

    def W(z):
        g = G(z)
        d = dh(z)
        return np.stack([0.5 * (1.0 / g - g) * d,
                         0.5j * (1.0 / g + g) * d, d], axis=-1)
    return G, dh, W


def _tks_graded(lo, hi, n, specials):
    """n samples clustered toward each special coordinate (iterated
    density reweighting, the same scheme _g1h_graded uses)."""
    t = np.linspace(0.0, 1.0, n)
    x = lo + (hi - lo) * t
    spc = np.asarray(specials, dtype=float)
    for _ in range(4):
        d = np.min(np.abs(x[:, None] - spc[None, :]), axis=1)
        wgt = 1.0 / (0.02 + d) ** 1.2
        cdf = np.concatenate([[0.0],
                              np.cumsum(0.5 * (wgt[1:] + wgt[:-1])
                                        * np.diff(x))])
        cdf /= cdf[-1]
        x = np.interp(t, cdf, x)
    return x


def _tks_patch(k, tau1, a1, n, r0):
    """One winding: (xs, ys, X, mask), through the ends backend.  The
    punctures sit on the x = 0 line, so the tree has two strips with
    base columns near +-1/4 -- the same integration the first version
    hand-rolled, now shared."""
    _G, _dh, W = _tks_forms(k, tau1, a1)
    punct = (1j * a1, 1j * (tau1 - a1))
    xs, ys = we_ends_grid((-0.5, 0.5, 0.0, float(tau1)), punct, int(n))
    X = we_ends_integrate(W, xs, ys, punct)
    mask = we_ends_mask(xs, ys, punct, r0)
    return xs, ys, X, mask


def tks_vertical_period(k, tau1, a1, r=0.015, n=8001):
    """The translation of the loop around the z = i a1 end, by direct
    contour integration of all three forms: (0, 0, T) for a solved
    member.  The horizontal components vanishing is the geometric form
    of the period condition, and the self-test measures exactly that."""
    _G, _dh, W = _tks_forms(k, tau1, a1)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    zz = 1j * a1 + r * np.exp(1j * t)
    dzdt = 1j * r * np.exp(1j * t)
    loop = np.trapezoid(W(zz) * dzdt[:, None], t, axis=0)
    return np.real(loop)


def tks_period_residual(k, tau1, a1, n=20001):
    """Re of the horizontal dh period -- the notebook's own Adh
    condition, zero exactly at the tabulated (tau1, a1) members."""
    _G, dh, _W = _tks_forms(k, tau1, a1)
    t = np.linspace(0.0, 1.0, n)
    v = dh(1j * tau1 / 2.0 + t)
    return float(np.real(np.trapezoid(v, t)))


def toroidal_ks_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                     storeys=1):
    """MESH_PARAM builder: `storeys` windings of the tower, welded by
    index along the winding seam.  order picks the wing parameter k
    (snapped to the notebook's solved tables); radius sets the wing
    trim (larger radius = smaller mask = longer wings).  theta is
    unused: no associate family is claimed for the tower."""
    ks = sorted(TOROIDAL_KS_MEMBERS)
    k = min(ks, key=lambda kk: abs(kk - int(round(order))))
    tau1, a1, _lx = TOROIDAL_KS_MEMBERS[k]
    r0 = 1.2e-3 * (1.2 / float(np.clip(radius, 0.3, 6.0))) ** 2
    n = int(np.clip(nu * 2.2, 100, 300))
    xs, ys, X, mask = _tks_patch(k, tau1, a1, n, r0)
    T = tks_vertical_period(k, tau1, a1)
    S = max(1, int(storeys))
    nx, ny = X.shape[0], X.shape[1]
    NV = nx * ny
    Vs, Ms = [], []
    for s_ in range(S):
        Vs.append((X + s_ * np.array([0.0, 0.0, T[2]])).reshape(-1, 3))
        Ms.append(mask.reshape(-1))
    V = np.concatenate(Vs, axis=0)
    mall = np.concatenate(Ms, axis=0)

    parent = np.arange(S * NV)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for s_ in range(S - 1):
        for j in range(ny):
            a = find(s_ * NV + (nx - 1) * ny + j)
            b = find((s_ + 1) * NV + j)
            if a != b:
                parent[a] = b
    roots = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    sums = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(sums, inv, V)
    np.add.at(cnt, inv, 1.0)
    Vm = sums / cnt[:, None]
    ok_node = np.ones(len(uniq), dtype=bool)
    np.logical_and.at(ok_node, inv, mall)
    # cut curtains: around each helicoidal end the immersion is
    # multivalued (the loop translates by (0, 0, T)), so rows on
    # opposite sides of the cut line sit one winding apart and a quad
    # bridging them is a vertical wall the surface does not contain.
    # The 3-D jump filter drops them; the gap they leave IS the end
    # trim, which the incomplete-elliptic-F chart refinement would
    # fill properly (BACKLOG).
    jump = 0.55 * abs(T[2])
    quads = []
    for s_ in range(S):
        for i in range(nx - 1):
            b0 = s_ * NV + i * ny
            b1 = s_ * NV + (i + 1) * ny
            for j in range(ny - 1):
                f = (inv[b0 + j], inv[b1 + j],
                     inv[b1 + j + 1], inv[b0 + j + 1])
                if not (ok_node[f[0]] and ok_node[f[1]]
                        and ok_node[f[2]] and ok_node[f[3]]):
                    continue
                c = Vm[list(f)]
                if max(float(np.linalg.norm(c[t] - c[(t + 1) % 4]))
                       for t in range(4)) > jump:
                    continue
                quads.append(f)
    used = np.zeros(len(Vm), dtype=bool)
    for f in quads:
        for a in f:
            used[a] = True
    remap = -np.ones(len(Vm), dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    V = Vm[used]
    quads = [tuple(int(remap[a]) for a in f) for f in quads]
    V = _center_fit(V, scale, V)
    return V, quads, None


# --------------------------------------------------------------------------
# Plane with catenoids (doubly periodic, square lattice)
# --------------------------------------------------------------------------
# The simplest doubly periodic minimal surface with only catenoidal
# ends in the quotient: a plane with catenoid necks planted on a square
# lattice, the ends' limiting normals perpendicular to the periodicity
# plane, growth rate tuned by the Lopez-Ros factor rho -- and, the
# page states outright, NO period problem to solve.  Data printed on
# Weber's page (mirror ch039) and in `Doubly_Catenoid.nb`:
#
#     g = rho / sqrt(z),   dh = dz / (sqrt(z-1) sqrt(z+1))
#
# on the upper half plane, principal branches (all three square roots
# are analytic there, so no branch tracking).  The catenoidal ends are
# z = 0 and z = infinity; +-1 are integrable branch points.
#
# THE IDENTITY, chosen before the mesher was written and measured
# before it was trusted (now gated in the zoo self-test): the four
# real-axis segments must land on the four symmetry elements of the
# square cell, tied together by ONE number --
#
#     f((0,1))      the straight 2-fold axis along y,
#     f((-1,0))     the straight 2-fold axis along x,
#     f((1,inf))    planar, in the mirror  y = +dis,
#     f((-inf,-1))  planar, in the mirror  x = -dis,
#     dis = f(1)_y = -f(-1)_x            (the SQUARE lattice),
#
# all measured by independent 1-D integrals; at rho = 1 they agree to
# quadrature (~1e-7), and the lattice period is 4 dis.  Weber's own
# assembly chain (fr2..fr6) is followed literally: two mirrors, two
# half-turns (16 copies), then translations by (4 dis, 0, 0) and
# (0, 4 dis, 0).
#
# References:
# - M. Weber, "Plane with Catenoids", minimalsurfaces.blog (mirror
#   ch039; notebook `Doubly_Catenoid.nb` -- the data and the assembly).
# - H. Karcher, "Embedded minimal surfaces derived from Scherk's
#   examples", Manuscripta Math. 62 (1988) -- the Lopez-Ros/growth
#   mechanism for catenoidal necks in periodic surfaces.

def plane_catenoids_W(rho=1.0):
    def W(z):
        z = np.asarray(z, dtype=complex)
        s0 = np.sqrt(z)
        s1 = np.sqrt(z - 1.0)
        s2 = np.sqrt(z + 1.0)
        ph1 = rho / (s1 * s0 * s2)
        ph2 = (1.0 / rho) * s0 / (s1 * s2)
        dh = 1.0 / (s1 * s2)
        return np.stack([0.5 * (ph2 - ph1), 0.5j * (ph1 + ph2), dh],
                        axis=-1)
    return W


def _pwc_seg(Wf, z0, z1, se=None, m=2, n=20000):
    if se is None:
        t = (np.arange(n) + 0.5) / n
        zz = z0 + (z1 - z0) * t
        w = np.full(n, 1.0 / n)
    else:
        u = (np.arange(n) + 0.5) / n
        sv = u ** m
        w = m * u ** (m - 1) / n
        if se == 'z0':
            zz = z0 + (z1 - z0) * sv
        else:
            zz = z1 + (z0 - z1) * sv[::-1]
            w = w[::-1]
    return np.sum(Wf(zz) * (w[:, None] * (z1 - z0)), axis=0)


def plane_catenoids_frame(rho=1.0):
    """(dis, f1, fm1) for the member: the mirror offset and the two
    axis endpoints, by graded 1-D integrals from the true z = 0 (the
    m = 2 substitution regularises every sqrt endpoint)."""
    Wf = plane_catenoids_W(rho)
    up = _pwc_seg(Wf, 0.0, 0.5j, 'z0')
    f1 = np.real(up + _pwc_seg(Wf, 0.5j, 1.0 + 0.5j)
                 + _pwc_seg(Wf, 1.0 + 0.5j, 1.0, 'z1'))
    fm1 = np.real(up + _pwc_seg(Wf, 0.5j, -1.0 + 0.5j)
                  + _pwc_seg(Wf, -1.0 + 0.5j, -1.0, 'z1'))
    return float(f1[1]), f1, fm1


def _pwc_weld(V, Q, tol):
    """Plain coordinate weld (round-to-lattice buckets)."""
    key = np.round(V / max(tol, 1e-300)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    return V[first], inv[np.asarray(Q)]


def plane_catenoids_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                         cells=(1, 1)):
    """cells2d builder: Weber's 16-copy unit (two mirrors, two
    half-turns) tiled cu x cv on the measured square lattice.  order
    drives the Lopez-Ros growth factor rho = 2^(order-1); radius sets
    the domain reach (how far the catenoid necks flare)."""
    if isinstance(cells, (int, float)):
        cells = (int(cells), 1)
    cu = int(np.clip(cells[0], 1, 6))
    cv = int(np.clip(cells[1] if len(cells) > 1 else 1, 1, 6))
    rho = float(2.0 ** (int(np.clip(order, 1, 5)) - 1))
    Wf = plane_catenoids_W(rho)
    reach = 1.0 + 3.0 * float(np.clip(radius, 0.3, 6.0)) / 1.2
    punct = (0.0 + 0j, 1.0 + 0j, -1.0 + 0j)
    n = int(np.clip(nu * 1.8, 80, 240))
    eps = 1e-6
    xs, ys = we_ends_grid((-reach, reach, eps, reach), punct, n,
                          ny=max(24, int(n * 0.7)))
    X = we_ends_integrate(Wf, xs, ys, punct)
    # frame: patch coords are f - f(i eps); shift into the f(0) = 0
    # frame the identities are stated in
    delta = np.real(_pwc_seg(Wf, 0.0, 1j * eps, 'z0'))
    X = X + delta[None, None, :]
    dis, _f1, _fm1 = plane_catenoids_frame(rho)
    # snap the real-axis boundary onto its measured symmetry elements
    # (the reflected/rotated copies then share those vertices exactly
    # -- the same projection step the conjugate-Plateau route uses)
    for i, x in enumerate(xs):
        vx = X[i, 0]
        if 0.0 < x < 1.0:
            X[i, 0] = np.array([0.0, vx[1], 0.0])        # y axis
        elif -1.0 < x < 0.0:
            X[i, 0] = np.array([vx[0], 0.0, 0.0])        # x axis
        elif x >= 1.0:
            X[i, 0] = np.array([vx[0], dis, vx[2]])      # y = +dis
        else:
            X[i, 0] = np.array([-dis, vx[1], vx[2]])     # x = -dis
    mask = np.ones(X.shape[:2], dtype=bool)
    quads0 = np.asarray(we_ends_quads(X, mask))
    V0 = X.reshape(-1, 3)

    def refl(nrm, off):
        nrm = np.asarray(nrm, dtype=float)
        H = np.eye(4)
        H[:3, :3] = np.eye(3) - 2.0 * np.outer(nrm, nrm)
        H[:3, 3] = 2.0 * off * nrm
        return H

    def halfturn(axis):
        d = np.asarray(axis, dtype=float)
        d = d / np.linalg.norm(d)
        H = np.eye(4)
        H[:3, :3] = 2.0 * np.outer(d, d) - np.eye(3)
        return H
    M1 = refl((1.0, 0.0, 0.0), -dis)
    M2 = refl((0.0, 1.0, 0.0), dis)
    R1 = halfturn((0.0, 1.0, 0.0))
    R2 = halfturn((1.0, 0.0, 0.0))
    ops = [np.eye(4)]
    for H in (M1, M2, R1, R2):                # Weber's fr2..fr5 chain
        ops = ops + [H @ g_ for g_ in ops]
    a1 = np.array([4.0 * dis, 0.0, 0.0])
    a2 = np.array([0.0, 4.0 * dis, 0.0])
    Vs, Qs, base = [], [], 0
    for iu in range(cu):
        for iv in range(cv):
            off = ((iu - 0.5 * (cu - 1)) * a1
                   + (iv - 0.5 * (cv - 1)) * a2)
            for H in ops:
                Vh = V0 @ H[:3, :3].T + H[:3, 3] + off
                Vs.append(Vh)
                q = quads0 + base
                if np.linalg.det(H[:3, :3]) < 0.0:
                    q = q[:, ::-1]
                Qs.append(q)
                base += len(V0)
    V = np.concatenate(Vs, axis=0)
    Q = np.concatenate(Qs, axis=0)
    span = 4.0 * dis * max(cu, cv)
    V, Q = _pwc_weld(V, Q, 1e-6 * span)
    V = _center_fit(V, scale, V)
    return V, [tuple(int(i) for i in q) for q in Q], None


# --------------------------------------------------------------------------
# The catenoid field (doubly periodic half-catenoids)
# --------------------------------------------------------------------------
# A doubly periodic field of half-catenoids growing alternately up and
# down between two parallel planes: 3DXM's "Catenoid Field" exhibit,
# with the data the VMM/harvest state as g = bb * J_F(z),
# dh = dz / J_F(z) on a twice-punctured rectangular torus.  J_F is the
# degree-2 elliptic function with simple zeros at 0, 1/2 and simple
# poles at tau/2, 1/2 - tau/2 (BALANCED divisor: sum of zeros = sum of
# poles = 1/2; with the pole pair written as {tau/2, 1/2 + tau/2} the
# quotient gains an e^{2 pi i z} factor and is not elliptic at all --
# measured before this row was trusted: the raw quotient's phase walks
# along the real axis at exactly that rate).  VMM's conformal page
# ch187 documents J_F as the real rescaling of Jacobi sn whose branch
# values sit SYMMETRIC to the unit circle; the constant here is fixed
# numerically per member by |b1 * b2| = 1 over the two real branch
# values, which is that symmetry.
#
# The punctures are the dh poles at the J_F zeros: half-catenoid ends.
# THE ROW'S CLAIM, measured by the self-test rather than assumed: there
# is NO period problem -- the loop around each puncture translates by
# (0, 0, 0) to quadrature (the order-2 poles of om1, om2 have even
# principal parts, so no residue survives), and both deck translations
# are purely HORIZONTAL (the field is bounded between two planes).
# Measured at the tau = i member: loops (0, 0, 0); z -> z + 1 gives
# (-0.4926, 0, 0) and z -> z + tau gives (0, -0.6642, 0).
#
# References:
# - The 3DXM Consortium, "Catenoid Field", Virtual Math Museum
#   (mirror vmm/book/surface; J_F on conformal page ch187).
# - H. Karcher, "Construction of minimal surfaces", Univ. of Tokyo
#   Surveys in Geometry (1989) -- the chain/field-of-half-catenoids
#   construction the VMM notes credit.
# - M. Weber, minimalsurfaces.blog, "translation invariant plane with
#   catenoidal ends" pages -- the singly periodic siblings.

# --------------------------------------------------------------------------
# Hackman's toroidal 1-noid (singly periodic, one catenoid end per storey)
# --------------------------------------------------------------------------
# M. Hackman's thesis surfaces: toroidal 1-noids on every conformal
# type of torus, reported (with the notebook this transcribes) at
# M. Weber, minimalsurfaces.blog, "Hackman surfaces"
# (`Hackman-Surfaces.nb`, mirrored in research/msblog_harvest/).  Data,
# on the torus with half-periods {1/2, tau/2}, tau = t + 2i:
#
#     k  = 1/3,
#     G  = theta11(z + k/2) / theta11(z - k/2),
#     dh = e^{i phi(tau)} sigma(z - k/2) sigma(z + k/2) / sigma(z)^2,
#
# with t solved from the notebook's own period condition
#     period(k, tau) = Re I[1/2 -> 1/2+tau/2] - k Re I[tau/2 -> 1/2+tau/2] = 0.
#
# THE BONNET PHASE, RESOLVED BY MEASUREMENT (the batch-5 stop).  The
# notebook phases dh BEFORE the period condition, so the t-solve
# depends on phi; phi(tau) is a closed-form sigma/zeta/theta expression
# containing Conjugate[] (line ~19 of the extract), which this module
# implements VERBATIM in `hackman_phi` -- the Conjugate[]s matter
# because tau = t + 2i is a sheared torus and every factor is complex.
# The batch-5 hypothesis phi = -arg(sigma(-k/2) sigma(k/2)) was tested
# against that closed form BEFORE anything was built, and it is WRONG:
# off by exactly pi (a sign flip of e^{i phi} dh -- a silently wrong
# member that would render perfectly) plus a real t-dependent drift of
# ~6e-6.  The closed form is what ships.  On the solved member the
# phase is tiny but nonzero (phi = -5.4570e-6 at the root).
#
# Solved member (re-derived here, gated by re-solve in the zoo
# self-test): t = 0.333316172865672, tau = t + 2i.  The notebook's
# FindRoot seed window [0.15, 0.25] contains NO root -- the residual
# is -0.100..-0.046 across it -- and FindRoot walks out to this root;
# measured before it was trusted.
#
# MEASURED STRUCTURE at the member (all gated):
#   - the end loop around the lattice puncture translates by (0,0,0)
#     to 1e-15: the catenoid end has no period;
#   - deck z -> z+1 is a PURE VERTICAL translation (0, 0, h) with
#     h = 1.095693 (G has period 1, so no rotation);
#   - deck z -> z+tau is the SCREW: rise exactly h/3 = k h and normal
#     rotation e^{-2 pi i k} = -120 degrees, so the deck group in
#     SE(3) is the single screw generator (B^3 = A) -- the surface is
#     singly periodic with one catenoid end per storey, three storeys
#     per full turn.
#
# References:
# - M. Hackman, thesis (toroidal 1-noids on every conformal torus);
#   reported at M. Weber, "Hackman surfaces", minimalsurfaces.blog.
# - H. Karcher, "Construction of minimal surfaces" (1989) -- the
#   theta/sigma Weierstrass toolkit on tori this data lives in.

HACKMAN_K = 1.0 / 3.0
HACKMAN_T = 0.333316172865672


def _hk_th1(x, tau, d=0):
    """theta1(x | tau) and x-derivatives by q-series, COMPLEX nome
    (the sheared torus needs it); |q| = e^{-2 pi} so 8 terms are
    far beyond machine precision."""
    q = np.exp(1j * np.pi * tau)
    x = np.asarray(x, dtype=complex)
    out = np.zeros_like(x)
    for n_ in range(8):
        c = 2.0 * (-1.0) ** n_ * q ** ((n_ + 0.5) ** 2)
        m_ = 2 * n_ + 1
        if d == 0:
            out = out + c * np.sin(m_ * x)
        elif d == 1:
            out = out + c * m_ * np.cos(m_ * x)
        elif d == 3:
            out = out - c * m_ ** 3 * np.cos(m_ * x)
    return out


def _hk_eta1(tau):
    # zeta(1/2) for half-periods {1/2, tau/2}
    return -(np.pi ** 2 / 6.0) * _hk_th1(0.0, tau, 3)         / _hk_th1(0.0, tau, 1)


def _hk_sigma(z, tau):
    # normalized so sigma(z)/z -> 1 (checked: 5e-25 against mpmath)
    z = np.asarray(z, dtype=complex)
    return (np.exp(_hk_eta1(tau) * z * z) * _hk_th1(np.pi * z, tau)
            / (np.pi * _hk_th1(0.0, tau, 1)))


def _hk_zeta(z, tau):
    z = np.asarray(z, dtype=complex)
    return 2.0 * _hk_eta1(tau) * z + np.pi * _hk_th1(np.pi * z, tau, 1)         / _hk_th1(np.pi * z, tau)


def hackman_phi(tau):
    """The notebook's closed-form Bonnet phase, VERBATIM -- including
    every Conjugate[].  Returns the real phase; the zoo gate checks
    the imaginary part of the log is ~0 (pure phase)."""
    k = HACKMAN_K
    kp = k * np.pi / 2.0
    T1p_, T1m_ = _hk_th1(kp, tau), _hk_th1(-kp, tau)
    T1pp, T1pm = _hk_th1(kp, tau, 1), _hk_th1(-kp, tau, 1)
    sm, sp = _hk_sigma(-k / 2, tau), _hk_sigma(k / 2, tau)
    zm, zp = _hk_zeta(-k / 2, tau), _hk_zeta(k / 2, tau)
    A = -np.pi * T1p_ * T1pm + T1m_ * (np.pi * T1pp
                                       + T1p_ * (zm + zp))
    B = (np.pi * T1p_ * T1pm - np.pi * T1m_ * T1pp
         + T1m_ * T1p_ * zm + T1m_ * T1p_ * zp)
    num = 1j * np.sqrt(np.conj(sm * sp * A)) * T1p_
    den = np.conj(T1m_) * np.sqrt(sm) * np.sqrt(sp) * np.sqrt(B)
    val = -1j * np.log(-(num / den))
    return complex(val)


def hackman_W(t=HACKMAN_T):
    """(Wfn, tau, phi) for the member at tau = t + 2i."""
    k = HACKMAN_K
    tau = complex(float(t), 2.0)
    ph = float(np.real(hackman_phi(tau)))

    def W(z):
        z = np.asarray(z, dtype=complex)
        dh = (np.exp(1j * ph) * _hk_sigma(z - k / 2, tau)
              * _hk_sigma(z + k / 2, tau) / _hk_sigma(z, tau) ** 2)
        g = _hk_th1(np.pi * (z + k / 2), tau)             / _hk_th1(np.pi * (z - k / 2), tau)
        return np.stack([0.5 * (1.0 / g - g) * dh,
                         0.5j * (1.0 / g + g) * dh, dh], axis=-1)
    return W, tau, ph


def hackman_period_residual(t, n=4001):
    """The notebook's period(k, tau) at tau = t + 2i (with the phase
    evaluated AT that tau, as the notebook does)."""
    k = HACKMAN_K
    tau = complex(float(t), 2.0)
    ph = float(np.real(hackman_phi(tau)))

    def dh(z):
        return (np.exp(1j * ph) * _hk_sigma(z - k / 2, tau)
                * _hk_sigma(z + k / 2, tau) / _hk_sigma(z, tau) ** 2)
    tt = np.linspace(0.0, 1.0, int(n))

    def seg(a, b):
        z = a + (b - a) * tt
        return np.trapezoid(dh(z) * (b - a), tt)
    I1 = seg(0.5 + 0j, 0.5 + tau / 2.0)
    I2 = seg(tau / 2.0, 0.5 + tau / 2.0)
    return float(np.real(I1) - k * np.real(I2))


def hackman_deck(t=HACKMAN_T, n=20001):
    """Measured deck translations: (vA, riseB, vB_at_two_bases).
    vA is the z -> z+1 translation (pure vertical when the row is
    right); riseB the z -> z+tau vertical rise (= k * vA_z); the two
    vB screw offsets test the -2 pi k rotation (they agree only if
    the rotation used is the surface's actual one)."""
    W, tau, _ph = hackman_W(t)
    tt = np.linspace(0.0, 1.0, int(n))

    def seg(a, b):
        z = a + (b - a) * tt
        return np.real(np.trapezoid(W(z) * (b - a), tt, axis=0))
    zA = 0.31 + 0.83j
    vA = seg(zA, zA + 1.0)
    k = HACKMAN_K
    c, s_ = np.cos(2.0 * np.pi * k), np.sin(2.0 * np.pi * k)
    R = np.array([[c, s_, 0.0], [-s_, 0.0 * c + c, 0.0],
                  [0.0, 0.0, 1.0]])
    R[1, 1] = c
    vBs = []
    for z0 in (0.62 + 0.55j, 0.24 + 1.31j):
        # f in the universal-cover frame anchored at zA: integrate
        # from the common base to z0 and to z0 + tau
        f0 = seg(zA, z0)
        f1 = f0 + seg(z0, z0 + tau)
        vBs.append(f1 - R @ f0)
    return vA, float(vBs[0][2]), vBs[0], vBs[1], R


def hackman_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                 storeys=1):
    """Hackman toroidal 1-noid: the fundamental torus meshed on the
    ends backend (rectangle [0,1] x [0,2] IS a fundamental domain of
    {1, tau} -- same covolume, top edge glued with shear), storeys
    stacked by the MEASURED screw.  order = storeys; radius sets how
    far into the catenoid end the mesh reaches."""
    storeys = int(np.clip(max(int(storeys), int(order)), 1, 9))
    W, tau, _ph = hackman_W()
    punct = [0.0 + 0j, 1.0 + 0j, tau]
    r0 = float(np.clip(0.055 * (1.2 / max(float(radius), 0.3)), 0.02,
                       0.14))
    n = int(np.clip(nu * 1.6, 72, 200))
    # window inset by eps: the integrator's strips EXCLUDE columns at
    # exact wall x (the puncture columns), which would otherwise stay
    # uninitialized -- measured as a 211-unit bbox of zero-vertices
    # before this inset
    eps = 1e-4
    xs, ys = we_ends_grid((eps, 1.0 - eps, eps, 2.0 - eps), punct, n,
                          ny=max(60, int(n * 1.2)),
                          specials_x=(float(np.real(tau)),))
    X = we_ends_integrate(W, xs, ys, punct)
    mask = we_ends_mask(xs, ys, punct, r0)
    quads0 = we_ends_quads(X, mask)
    V0 = X.reshape(-1, 3)
    # prune to used vertices: masked grid points keep huge f-values
    # and would dominate the bounding box (and the centering)
    used = np.zeros(len(V0), dtype=bool)
    for q in quads0:
        for a_ in q:
            used[a_] = True
    remap = -np.ones(len(V0), dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    V0 = V0[used]
    quads0 = [tuple(int(remap[a_]) for a_ in q) for q in quads0]
    # the measured screw: rise + rotation about the vertical axis
    # through the point solving (I - R) a = vB_horizontal
    vA, riseB, vB, _vB2, R = hackman_deck()
    a_h = np.linalg.solve(np.eye(2) - R[:2, :2], vB[:2])
    axis = np.array([a_h[0], a_h[1], 0.0])
    Vs, Fs = [], []
    NV = len(V0)
    for m_ in range(storeys):
        Vm = V0.copy()
        for _ in range(m_):
            Vm = (Vm - axis) @ R.T + axis
            Vm[:, 2] += riseB
        Vs.append(Vm)
        Fs.extend(tuple(int(i) + m_ * NV for i in q) for q in quads0)
    V = np.concatenate(Vs, axis=0)
    V = _center_fit(V, scale, V)
    return V, Fs, None


# --------------------------------------------------------------------------
# The Lubeck-Batista surface (doubly periodic, genus 3)
# --------------------------------------------------------------------------
# F. Lubeck and V. Batista's doubly periodic genus-3 minimal surface:
# the authors' own paper is arXiv:0806.4313 (converted at
# research/papers/minimal-surfaces/0806.4313v1/), and the data here is
# Weber's notebook `L_beck-Batista.nb` (mirror extract in
# research/msblog_harvest/), transcribed verbatim:
#
#     dh = dz  (so X3 = Re z: the height is the torus coordinate),
#     G  = sqrt(th(z - ia) / th(z - (ia - 1/2)))
#          * th(z) / th(z - 1/2)
#          * sqrt(th(z - (ib - 1/2)) / th(z - ib)),      th = theta11,
#
# on the rectangular torus tau in i R, with (a, b) the notebook's 2-D
# FindRoot solutions of the AUTHORS' PERIOD CONDITIONS
#
#     test(a, b) = Re{ I_w1[ib -> 1/4 -> -tau/2] + I_w1[ib -> tau/2],
#                      I_w2[ib -> 1/8 -> -tau/2] }  = (0, 0).
#
# Those conditions are re-derived numerically by the zoo gate (branch-
# tracked theta logs along the notebook's own waypoint paths, with a
# u^2 endpoint substitution at the (z - ib)^{-1/2} branch point --
# uniform quadrature there leaves a ~3e-2 phantom residual that looks
# exactly like a wrong member) and they VANISH at every stored member
# (7e-6..3e-5, quadrature-limited, all seven rows of the table).
#
# MESHED TO THE NOTEBOOK'S OWN RECIPE (the earlier two-half-window
# tiling of the full torus rectangle produced a genuinely different
# figure -- the exported ground truth is two parallel sheets joined
# by a four-lobed neck cluster, not a winged strip).  Weber meshes
# ONE QUARTER of the torus, the rectangle [0, 1/4] x [-Im tau/2,
# Im tau/2]: his EllipticF/ArcSin chart of the upper half disk is
# exactly a graded parametrization of that rectangle (verified
# numerically against the chart; its rmin = 0.01 truncation circle
# maps to |z| = (2 y0 / S) rmin with S = 2 K(m)/a0, which is what
# `_lb_map_consts` reproduces without Mathematica).  On that quarter
# the POINTWISE PRINCIPAL BRANCH of the sqrt product is continuous
# (measured: neighbour steps stay O(grid) away from the three
# boundary singularities at all seven members), which is precisely
# what the notebook's NIntegrate evaluates -- so no branch tracking
# is needed for the patch, only for the 1-D period paths.
#
# The patch is bounded by symmetry lines and a mirror curve, all
# MEASURED off the computed boundary arcs rather than trusted:
#   - x = 0, y in (0, b): a straight line A (fit residual ~2e-5 of
#     length) through f(ib) -- Weber's first rotation axis;
#   - x = 0, y in (b, y0) and (-y0, a): ONE common straight line B
#     through f(ia) -- his second axis (the y in (a, 0) piece lands
#     on the ghost line L, PARALLEL to A at half the horizontal
#     period: rotating about it instead builds an overlapping ghost);
#   - x = 1/4: a planar curve at constant height X3 (std ~1e-17).
# A and B are horizontal, exactly perpendicular, and INTERSECT at
# f(ib), so the two Schwarz rotations commute and their product
# r2 r1 IS the z -> z+tau deck: a 180-degree rotation about the
# vertical line through the intersection, NOT a translation (which
# is why tolerance-welding translated copies can never close the
# tau seams; measured Procrustes fit R = diag(-1,-1,1) to 6e-5 of
# span).  Assembly = rotate the patch 180 deg about A, the pair
# about B, the four about the mirror plane, then WELD every seam by
# exact grid-index pairs: the A/B arcs within the cell, the two
# y-edges under the tau-deck (X(x, +y0) = r2 r1 X(x, -y0)), the
# mirror rows in-cell and across the vertical deck z -> z+1 =
# (0, 0, 1) (exact: X3 = Re z, quarter height exactly 1/4), and the
# L arcs across the horizontal deck T = twice the A -> L offset.
# Winding parities are solved from the seam traversals (each
# in-surface Schwarz rotation reverses the attached copy's mesh
# winding); the result is one consistently oriented manifold sheet:
# a single cell measures chi = -8 with 6 boundary loops, and the
# tiled block fits chi = -12 cu cv + 4 cv -- the bulk -12 per cell
# being two quotient copies of chi = 2 - 2g - e = -6, i.e. the
# authors' genus g = 3 quotient with e = 2 ends, MEASURED.
#
# GROUND TRUTH: registered against Weber's own PoVRay exports of the
# three members he renders (tau = 0.94i, 1.2i, 2.5i; the dummy.pov
# meshes ARE the assembled cell, tiled in his scene by 2*MESHxsize /
# 2*MESHzsize).  Two-sided mean point-to-surface distance lands at
# 0.07-0.08% of span for ALL THREE with the IDENTITY axis map, and
# the cell extent ratios match his declared MESHxsize : MESHysize :
# MESHzsize to ~0.2% -- those ratios are what the zoo gate pins.
#
# References:
# - K. Lubeck and V. Ramos Batista, "The doubly periodic Scherk-Costa
#   surfaces", arXiv:0806.4313; J. Math. Research 6 (2014) 77-90
#   (converted at research/papers/minimal-surfaces/0806.4313v1/) --
#   the authors' construction; their Section 6 reduces the period
#   problems to the two-real-component condition (9),
#   Re int_(1) (phi1, phi2) = 0, which is what the notebook's test
#   function implements and this module's gate re-derives.
# - M. Weber, "Lubeck-Batista surfaces", minimalsurfaces.blog
#   (notebook `L_beck-Batista.nb` -- the theta data and the solved
#   member table transcribed above).

# tau (imag part) -> (a, b), the notebook's solved members
# (the operator's member knob walks LB_ORDER below, which puts the
# three members Weber exports -- 0.94i, 1.2i, 2.5i -- at 1, 2, 3, so
# the DEFAULT lands on a member with a reference image)
LB_MEMBERS = (
    (0.935, -0.022620778269738837, 0.4599636671778001),
    (0.94, -0.05009519222020475, 0.4533441965300885),
    (1.0, -0.1653704140092093, 0.44637591301353885),
    (1.2, -0.3486894358553919, 0.49670000329301656),
    (1.5, -0.5434796870761005, 0.6111785456438003),
    (2.0, -0.8189334369185804, 0.8375822891619098),
    (2.5, -1.0758721591205636, 1.0807522771543987),
)

# member-knob order: Weber's three exported members first (0.94i,
# 1.2i, 2.5i), then the remaining solved table
LB_ORDER = (1, 3, 6, 0, 2, 4, 5)


def _lb_G_path(zp, a, b, tau):
    """G along a 1-D path, every theta factor's log unwrapped."""
    th = genus1helicoid_theta11
    zp = np.asarray(zp, dtype=complex)

    def L(shift, half):
        v = th(zp - shift, tau)
        lg = np.log(np.abs(v)) + 1j * np.unwrap(np.angle(v))
        return (0.5 if half else 1.0) * lg
    tot = (L(1j * a, True) - L(1j * a - 0.5, True)
           + L(0.0, False) - L(0.5, False)
           + L(1j * b - 0.5, True) - L(1j * b, True))
    return np.exp(tot)


def _lb_G_pv(Z, a, b, tau):
    """The notebook's G as the pointwise principal-branch product --
    what Mathematica's NIntegrate evaluates.  Continuous on the
    meshed quarter [0, 1/4] x [-Im tau/2, Im tau/2] (measured at all
    seven members; the sqrt cuts stay outside it)."""
    th = genus1helicoid_theta11
    return (np.sqrt(th(Z - 1j * a, tau) / th(Z - (1j * a - 0.5), tau))
            * th(Z, tau) / th(Z - 0.5, tau)
            * np.sqrt(th(Z - (1j * b - 0.5), tau) / th(Z - 1j * b, tau)))


def _lb_map_consts(y0):
    """(a0, S) of Weber's half-disk chart, Mathematica-free.

    a0 = modul(i / (4 y0)) with modul(t) = ModularLambda(2t)^(-1/4),
    so the chart modulus m = 1/a0^4 IS lambda(i / (2 y0)) =
    (theta2/theta3)^4 at the real nome q = exp(-pi / (2 y0)); S =
    2 K(m) / a0 with K from the AGM.  Only the product 2 y0 / S is
    consumed (the image of the chart's rmin truncation circle)."""
    q = math.exp(-math.pi / (2.0 * y0))
    t2 = 0.0
    t3 = 1.0
    for n_ in range(24):
        t2 += 2.0 * q ** ((n_ + 0.5) ** 2)
        t3 += 2.0 * q ** ((n_ + 1.0) ** 2)
    m = (t2 / t3) ** 4
    a0 = m ** -0.25
    x_, y_ = 1.0, math.sqrt(1.0 - m)
    for _ in range(40):
        x_, y_ = 0.5 * (x_ + y_), math.sqrt(x_ * y_)
        if abs(x_ - y_) < 1e-16:
            break
    K = math.pi / (2.0 * x_)
    return a0, 2.0 * K / a0


def _lb_fit_line(P):
    """(point, unit direction, relative residual) of the best line."""
    c = P.mean(axis=0)
    _, s, Vt = np.linalg.svd(P - c)
    return c, Vt[0], float(s[1] / (s[0] + 1e-30))


def _lb_W_from_G(G):
    return np.stack([-(G - 1.0 / G) / 2.0,
                     1j * (G + 1.0 / G) / 2.0,
                     np.ones_like(G)], axis=-1)


def lb_period_test(mi=2, n=8001):
    """The authors' period conditions at member mi, re-derived: the
    two real components that the notebook's FindRoot drives to zero,
    with the u^2 substitution at the singular ib endpoint."""
    tt, a, b = LB_MEMBERS[int(mi)]
    tau = 1j * tt

    def seg(z0, z1, sing0=False):
        u = np.linspace(0.0, 1.0, n)
        t = u * u if sing0 else u
        zp = z0 + (z1 - z0) * t
        if sing0:
            zp[0] = z0 + (z1 - z0) * 1e-14
        G = _lb_G_path(zp, a, b, tau)
        W = _lb_W_from_G(G)
        jac = ((2.0 * u if sing0 else np.ones_like(u))
               * (z1 - z0))[:, None]
        return np.trapezoid(W * jac, u, axis=0)
    p1a = seg(1j * b, 0.25, sing0=True)
    p1b = seg(0.25, -tau / 2.0)
    p2 = seg(1j * b, tau / 2.0, sing0=True)
    q1 = seg(1j * b, 0.125, sing0=True)
    q2 = seg(0.125, -tau / 2.0)
    return (float(np.real(p1a[0] + p1b[0] + p2[0])),
            float(np.real(q1[1] + q2[1])))


def lb_deck(mi=2, n=30001):
    """(P1, P2): measured deck translations z -> z+1 and z -> z+tau
    along cut-free probes."""
    tt, a, b = LB_MEMBERS[int(mi)]
    tau = 1j * tt
    t = np.linspace(0.0, 1.0, n)

    def seg(z0, z1):
        zp = z0 + (z1 - z0) * t
        W = _lb_W_from_G(_lb_G_path(zp, a, b, tau))
        return np.real(np.trapezoid(W * (z1 - z0), t, axis=0))
    P1 = seg(0.21j * tt, 1.0 + 0.21j * tt)
    P2 = seg(0.25 + 0j, 0.25 + tau)
    return P1, P2


def lb_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
            cells=(1, 1)):
    """Lubeck-Batista / doubly periodic Scherk-Costa mesh, built to
    Weber's own notebook recipe (see the block header above): the
    quarter-torus patch [0, 1/4] x [-y0, y0] on the pointwise
    principal branch, assembled by the two measured 180-degree
    symmetry-line rotations and the horizontal mirror, tiled by the
    measured deck translations.  `order` picks the member from the
    notebook's solved table; `radius` sets how far the flat sheets
    are followed toward the ends (1.2 = the notebook's own rmin =
    0.01 truncation, which is what Weber's exports use)."""
    del spec, theta
    if isinstance(cells, (int, float)):
        cells = (int(cells), 1)
    cu = int(np.clip(cells[0], 1, 4))
    cv = int(np.clip(cells[1] if len(cells) > 1 else 1, 1, 4))
    mi = LB_ORDER[int(np.clip(order - 1, 0, len(LB_ORDER) - 1))]
    tt, a, b = LB_MEMBERS[mi]
    tau = 1j * tt
    y0 = tt / 2.0
    # end truncation: Weber's rmin = 0.01 at the default radius; the
    # sheets grow logarithmically, so radius works the exponent
    rmin = float(np.clip(0.01 * (1.2 / max(float(radius), 0.2)) ** 2,
                         1e-4, 0.15))
    _a0, S_ = _lb_map_consts(y0)
    r0 = 2.0 * y0 / S_ * rmin
    n = int(np.clip(nu * 1.5, 60, 300))
    eps = 1e-9
    punct = [0j]
    xs, ys = we_ends_grid((0.0, 0.25, -y0 + eps, y0 - eps), punct, n,
                          ny=int(n * 2.2), specials_y=(a, b))
    # keep nodes off the two boundary branch points (integrable 1/sqrt
    # singularities: a node exactly on one evaluates to inf)
    for v_ in (a, b):
        jj = int(np.argmin(np.abs(ys - v_)))
        if abs(ys[jj] - v_) < 1e-9:
            ys[jj] += 3e-7

    def Wfn(Z):
        G = _lb_G_pv(Z, a, b, tau)
        return _lb_W_from_G(G)

    # the end puncture sits ON the window edge x = 0; nudge its wall
    # just outside so the boundary column is swept too
    X = we_ends_integrate(Wfn, xs, ys, [complex(-1e-9, 0.0)])
    mask = we_ends_mask(xs, ys, punct, r0)
    quads0 = we_ends_quads(X, mask)
    # the two symmetry lines and the ghost line, measured off the
    # x = 0 boundary arcs (margins keep the fits off the branch-point
    # corners at y = a, b and the end hole at y = 0)
    wid = b - a
    selA = (ys > 0.03 * wid) & (ys < b - 0.003) & mask[0]
    selB = ((ys > b + 0.003) | (ys < a - 0.003)) & mask[0]
    selL = (ys > a + 0.003) & (ys < -0.03 * wid) & mask[0]
    cA, uA, rA = _lb_fit_line(X[0][selA])
    cB, uB, rB = _lb_fit_line(X[0][selB])
    cL, uL, _rL = _lb_fit_line(X[0][selL])
    # idealize the measured elements into the EXACT symmetry
    # configuration the group structure needs: A and B horizontal,
    # exactly perpendicular, intersecting at q = f(ib); L exactly
    # parallel to A at horizontal offset d (the horizontal deck is
    # T = 2d).  With that, r2 r1 = r1 r2 = the z -> z+tau deck (a
    # 180-degree rotation about the vertical line through q -- NOT a
    # translation, which is why tolerance welding of translated
    # copies could never close these seams), and every seam of the
    # orbit is an exact index-to-index vertex pair.
    uA = uA * np.sign(uA[1] if abs(uA[1]) > abs(uA[0]) else uA[0])
    uA[2] = 0.0
    uA = uA / np.linalg.norm(uA)
    uB = uB - (uB @ uA) * uA
    uB[2] = 0.0
    uB = uB / np.linalg.norm(uB)
    h0 = 0.5 * (cA[2] + cB[2])
    M2 = np.array([[uA[0], -uB[0]], [uA[1], -uB[1]]])
    ts_ = np.linalg.solve(M2, (cB - cA)[:2])
    q_ = cA + ts_[0] * uA
    q_[2] = h0
    dL = (cL - q_) - ((cL - q_) @ uA) * uA
    dL[2] = 0.0
    Tx = 2.0 * dL                                # horizontal deck
    P1 = np.array([0.0, 0.0, 1.0])               # z -> z+1, exact
    # snap the WHOLE x = 0 boundary column onto its symmetry element
    # (full partition at the corner values a, 0, b -- the earlier
    # margin windows left unsnapped slivers), and the mirror curve
    # onto its plane, so every assembled seam welds vertex-to-vertex
    snapA = (ys > 0.0) & (ys < b) & mask[0]
    snapB = ((ys > b) | (ys < a)) & mask[0]
    snapL = (ys > a) & (ys < 0.0) & mask[0]
    for sel_, c0_, u_ in ((snapA, q_, uA), (snapB, q_, uB),
                          (snapL, q_ + dL, uA)):
        P_ = X[0][sel_]
        X[0][sel_] = c0_ + ((P_ - c0_) @ u_)[:, None] * u_[None, :]
    h = float(np.median(X[-1, :, 2]))
    X[-1, :, 2] = h
    # prune to used vertices (masked nodes keep huge near-end values)
    ny2 = len(ys)
    nx2 = len(xs)
    V0 = X.reshape(-1, 3)
    used = np.zeros(len(V0), dtype=bool)
    for q in quads0:
        for a_ in q:
            used[a_] = True
    remap = -np.ones(len(V0), dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    V0 = V0[used]
    F0 = [tuple(int(remap[a_]) for a_ in q) for q in quads0]

    def rot180(c_, u_):
        R_ = 2.0 * np.outer(u_, u_) - np.eye(3)
        return lambda P_: (P_ - c_) @ R_.T + c_

    r1 = rot180(q_, uA)
    r2 = rot180(q_, uB)
    parts = [V0, r1(V0)]
    parts = parts + [r2(P_) for P_ in parts]
    parts = parts + [P_ * np.array([1.0, 1.0, -1.0])
                     + np.array([0.0, 0.0, 2.0 * h]) for P_ in parts]
    # winding parities, SOLVED from the seam traversals (each Schwarz
    # rotation about an in-surface line reverses the mesh winding of
    # the copy it attaches, the z-mirror reverses it again, and the
    # tau-deck g preserves it): the unique consistent assignment --
    # measured, not assumed -- is e/r2r1/mr1/mr2 kept, the rest
    # reversed; the welded surface then orients consistently
    flips = (False, True, True, False, True, False, False, True)
    NV = len(V0)
    Vcell = np.concatenate(parts, axis=0)
    Fcell = []
    for k_, fl_ in enumerate(flips):
        for q in F0:
            qq = tuple(int(i) + k_ * NV for i in q)
            Fcell.append(qq[::-1] if fl_ else qq)
    # tile: u -> the horizontal deck T, v -> the vertical deck (0,0,1)
    Vs, Fs = [], []
    NC = len(Vcell)
    ci = 0
    cid = {}
    for iu in range(cu):
        for iv in range(cv):
            cid[(iu, iv)] = ci
            Vs.append(Vcell + iu * Tx[None, :] + iv * P1[None, :])
            Fs.extend(tuple(int(i) + ci * NC for i in q) for q in Fcell)
            ci += 1
    V = np.concatenate(Vs, axis=0)

    # ---- the weld table: every seam an exact index pair ----------
    # part order: 0 e, 1 r1, 2 r2, 3 r2r1, 4 m, 5 mr1, 6 mr2, 7 mr2r1
    def bid(j_):                                 # x = 0 column node
        return remap[j_]

    def mid_(j_):                                # x = 1/4 mirror row
        return remap[(nx2 - 1) * ny2 + j_]

    def yid(i_, j_):                             # y-edge node
        return remap[i_ * ny2 + j_]

    pairs = []

    def pw(cell_a, part_a, va, cell_b, part_b, vb):
        if va >= 0 and vb >= 0:
            pairs.append((cell_a * 8 * NV + part_a * NV + int(va),
                          cell_b * 8 * NV + part_b * NV + int(vb)))

    jsA = [j_ for j_ in range(ny2) if snapA[j_] and used[j_]]
    jsB = [j_ for j_ in range(ny2) if snapB[j_] and used[j_]]
    jsL = [j_ for j_ in range(ny2) if snapL[j_] and used[j_]]
    jsM = [j_ for j_ in range(ny2)
           if used[(nx2 - 1) * ny2 + j_]]
    isY = [i_ for i_ in range(nx2)
           if used[i_ * ny2] and used[i_ * ny2 + ny2 - 1]]
    for (iu, iv), c_ in cid.items():
        # in-cell: line A (fixed by r1), line B (fixed by r2)
        for pa_, pb_ in ((0, 1), (2, 3), (4, 5), (6, 7)):
            for j_ in jsA:
                pw(c_, pa_, bid(j_), c_, pb_, bid(j_))
        for pa_, pb_ in ((0, 2), (1, 3), (4, 6), (5, 7)):
            for j_ in jsB:
                pw(c_, pa_, bid(j_), c_, pb_, bid(j_))
        # in-cell: mirror rows fixed by the z-mirror
        for pa_, pb_ in ((0, 4), (3, 7)):
            for j_ in jsM:
                pw(c_, pa_, mid_(j_), c_, pb_, mid_(j_))
        # in-cell: the tau-deck g = r2 r1 pairs the two y-edges
        # (X(x, +y0) = g X(x, -y0)): top edge of h <-> bottom edge
        # of h*g, same x sample
        for pa_, pb_ in ((0, 3), (3, 0), (1, 2), (2, 1),
                         (4, 7), (7, 4), (5, 6), (6, 5)):
            for i_ in isY:
                pw(c_, pa_, yid(i_, ny2 - 1), c_, pb_, yid(i_, 0))
        # cell-to-cell, horizontal deck T (the ghost line L): the
        # rotation about L is T o r1, so h's L-arc continues into
        # part h o T o r1 of the +T cell
        if (iu + 1, iv) in cid:
            c2_ = cid[(iu + 1, iv)]
            for pa_, pb_ in ((0, 1), (2, 3), (4, 5), (6, 7)):
                for j_ in jsL:
                    pw(c_, pa_, bid(j_), c2_, pb_, bid(j_))
        # cell-to-cell, vertical deck (0,0,1): the mirror rows of
        # parts mr1, mr2 continue into parts r1, r2 one period up
        # (m o r1 = P1 o r1 o m on the mirror plane, since the
        # quarter's height is exactly 1/4)
        if (iu, iv + 1) in cid:
            c2_ = cid[(iu, iv + 1)]
            for pa_, pb_ in ((5, 1), (6, 2)):
                for j_ in jsM:
                    pw(c_, pa_, mid_(j_), c2_, pb_, mid_(j_))
    V, Fs, _first = _g1h_weld_pairs(V, Fs, pairs)
    V = _center_fit(V, scale, V)
    return V, Fs, None


# --------------------------------------------------------------------------
# Scherk's fourth surface (1835, equation 20)
# --------------------------------------------------------------------------
# The least-cited of Scherk's five 1835 surfaces: singly periodic with
# two annular and two helicoidal ends, singular at the two points where
# the horizontal symmetry curve meets the straight line the helicoidal
# ends share.  Weber's `Scherk_Surface_4.nb` (mirror ch341) recovers
# the Enneper-Weierstrass data by the Schwarz-Bjorling formula on the
# x = pi level curve of Scherk's implicit equation 20, ending in a
# fully CLOSED-FORM immersion (no integration at all):
#
#     G(z)  = i (1 + z) / sqrt(1 - z^2),
#     dh(z) = -2 sqrt(1 - z^2) / z,
#     f(z)  = Re{ 2 i Log z,
#                 2 z,
#                 -2 (sqrt(1-z^2) + Log z - Log(1 + sqrt(1-z) sqrt(1+z))) }
#
# on the upper half plane in polar coordinates.  The seed curve's
# ArcCosh(-1 + 8/y^2) is EXACTLY the closed form's -2(log r -
# log(1 + sqrt(1-r^2))) (verified algebraically and numerically), so
# the frame relation to Scherk's own (x, y, z) is x = X1 + pi,
# y = X2, z = X3.
#
# GATES (zoo self-test), the row's own claims measured:
#   1. SCHERK'S OWN EQUATION 20 -- cosh(z + sqrt(t+rho) csc(x/2) /
#      sqrt 2) = (4 sin^2(x/2) + rho)/y^2 with rho = sqrt(t^2 + y^4
#      sin^2 x), t = 4 sin^2(x/2) + y^2 cos x -- holds POINTWISE on the
#      built patch to machine precision (~1e-15, inner AND outer
#      regions; evaluated through arccosh in z-units, |x| for the odd
#      csc branch, min over the +- component);
#   2. the closed form is consistent with (G, dh): d/dz of the
#      analytic immersion equals (om1, om2, om3) to 1e-9;
#   3. the helicoidal-end winding: the loop around z = 0 advances the
#      screw axis coordinate X1 by exactly -4 pi = the assembly's
#      translation period.
#
# References:
# - H. F. Scherk, "Bemerkungen ueber die kleinste Flaeche innerhalb
#   gegebener Grenzen", J. Reine Angew. Math. 13 (1835) 185-208
#   (equation 20).
# - M. Weber, "Scherk's Fourth Surface", minimalsurfaces.blog (mirror
#   ch341; notebook `Scherk_Surface_4.nb` -- the Bjorling recovery and
#   the closed form transcribed above).

def scherk4_f(z):
    """The closed-form immersion (X1, X2, X3) at z (upper half
    plane; principal branches are correct there)."""
    z = np.asarray(z, dtype=complex)
    return np.stack(
        [np.real(2j * np.log(z)), np.real(2.0 * z),
         np.real(-2.0 * (np.sqrt(1.0 - z * z) + np.log(z)
                         - np.log(1.0 + np.sqrt(1.0 - z)
                                  * np.sqrt(1.0 + z))))], axis=-1)


def scherk4_eqn20(F):
    """Pointwise residual of Scherk's equation 20 at built points F
    (in z-units through arccosh; |x| handles the odd csc branch, and
    the minimum over the two cosh components picks the sheet)."""
    xs = np.abs(F[..., 0] + np.pi)
    ys = np.abs(F[..., 1])
    zs = F[..., 2]
    t = 4.0 * np.sin(xs / 2) ** 2 + ys ** 2 * np.cos(xs)
    rho = np.sqrt(t * t + ys ** 4 * np.sin(xs) ** 2)
    rhs = (4.0 * np.sin(xs / 2) ** 2 + rho)         / np.maximum(ys ** 2, 1e-12)
    core = (np.sqrt(np.maximum(t + rho, 0.0))
            / np.maximum(np.sin(xs / 2), 1e-12) / np.sqrt(2.0))
    ac = np.arccosh(np.maximum(rhs, 1.0))
    return np.minimum(np.abs(zs + core - ac), np.abs(zs + core + ac))


def scherk4_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                 storeys=1):
    """Scherk's fourth surface: closed-form patch on a polar domain,
    assembled by the notebook's own chain -- reflect in the plane
    x = 0, half-turn about the y-axis, translate by the 4 pi period.
    order = periods; radius sets how far the four ends are followed."""
    periods = int(np.clip(max(int(storeys), int(order)), 1, 5))
    reach = float(np.clip(radius, 0.3, 6.0)) / 1.2
    rmin = float(np.clip(0.10 / reach, 0.008, 0.3))
    # balance the annular reach against the helicoidal depth (the
    # notebook solves the same balance with FindRoot): X3 at the
    # helicoidal trim is ~ -2 log rmin, X2 at the annular trim 2 rmax
    rmax = max(2.0, abs(np.log(rmin)))
    n = int(np.clip(nu, 40, 140))
    eps = 0.02
    rr = np.exp(np.linspace(np.log(rmin), np.log(rmax), 2 * n))
    th_ = np.linspace(eps, np.pi - eps, n)
    Z = rr[:, None] * np.exp(1j * th_[None, :])
    X = scherk4_f(Z)
    nx, ny = X.shape[0], X.shape[1]
    quads0 = [(i * ny + j, (i + 1) * ny + j,
               (i + 1) * ny + j + 1, i * ny + j + 1)
              for i in range(nx - 1) for j in range(ny - 1)]
    V0 = X.reshape(-1, 3)
    # assembly: fr2 = reflect in plane x = 0; fr3 = half-turn about
    # the y-axis; fr4 = +- period translations
    npatch = len(V0)
    fpatch = [tuple(q) for q in quads0]
    # fr2: reflect in the plane x = 0
    blk = np.concatenate([V0, V0 * np.array([-1.0, 1.0, 1.0])], axis=0)
    fblk = fpatch + [tuple(a_ + npatch for a_ in q) for q in fpatch]
    # fr3: half-turn about the y-axis
    nblk = len(blk)
    cell = np.concatenate([blk, blk * np.array([-1.0, 1.0, -1.0])],
                          axis=0)
    fcell = fblk + [tuple(a_ + nblk for a_ in q) for q in fblk]
    # fr4: the 4 pi translations
    ncell = len(cell)
    per = np.array([4.0 * np.pi, 0.0, 0.0])
    Vs, Fs = [], []
    for m_ in range(periods):
        Vs.append(cell + (m_ - (periods - 1) / 2.0) * per[None, :])
        Fs.extend(tuple(a_ + m_ * ncell for a_ in q) for q in fcell)
    V = np.concatenate(Vs, axis=0)
    V = _center_fit(V, scale, V)
    return V, Fs, None


def _cf_member(t=1.0):
    """(JF, W(bb)) callables for the tau = i t member, with the
    normalising constant solved per member: constant phase on the real
    axis folded out, real scale fixed by |b1 b2| = 1 over the two real
    branch values (J_F's unit-circle symmetry)."""
    th = genus1helicoid_theta11
    tau = 1j * float(t)

    def J0(z):
        z = np.asarray(z, dtype=complex)
        return (th(z, tau) * th(z - 0.5, tau)
                / (th(z - tau / 2.0, tau)
                   * th(z - 0.5 + tau / 2.0, tau)))
    ph = float(np.angle(complex(J0(np.array(0.13)))))
    xs = np.linspace(0.01, 0.49, 2001)
    b1 = float(np.max(np.abs(np.real(J0(xs) * np.exp(-1j * ph)))))
    xs2 = np.linspace(-0.49, -0.01, 2001)
    b2 = float(np.max(np.abs(np.real(J0(xs2) * np.exp(-1j * ph)))))
    C = np.exp(-1j * ph) / math.sqrt(b1 * b2)

    def JF(z):
        return C * J0(z)
    return JF


def catenoid_field_W(bb=1.0, t=1.0):
    JF = _cf_member(t)

    def W(z):
        j = JF(np.asarray(z, dtype=complex))
        g = bb * j
        dh = 1.0 / j
        return np.stack([0.5 * (1.0 / g - g) * dh,
                         0.5j * (1.0 / g + g) * dh, dh], axis=-1)
    return W


def four_noid_sym2_params(lam):
    """Karcher's tau and rho for the 4-noid, in closed form.

    Both are printed outright in Weber's `4-Noid_sym_2.nb`; the
    notebook has no FindRoot anywhere, because this family's period
    problem is solved in closed form rather than numerically.  The
    nested radical needs lambda > 1 -- sqrt(lambda - 1) is a factor
    of the denominator -- which is the family's real parameter range,
    not a numerical guard.
    """
    L = float(lam)
    big = (1.0 + 2.0 * L ** 4 - 30.0 * L ** 8 + 2.0 * L ** 12 + L ** 16
           + (1.0 + L ** 4) * math.sqrt(
               4.0 + 45.0 * L ** 4 + 96.0 * L ** 8 - 146.0 * L ** 12
               + 96.0 * L ** 16 + 45.0 * L ** 20 + 4.0 * L ** 24))
    tau = math.sqrt(big) / (math.sqrt(L - 1.0) * L * math.sqrt(1.0 + L)
                            * math.sqrt(1.0 + L * L)
                            * math.sqrt(3.0 + 22.0 * L ** 4 + 3.0 * L ** 8))
    rho = math.sqrt(1.0 + 5.0 * L ** 4) / (
        L * math.sqrt(L ** 8 - 6.0 * L ** 2 * tau ** 2
                      + 2.0 * L ** 6 * tau ** 2 + tau ** 4
                      + L ** 4 * (5.0 - 3.0 * tau ** 4)))
    return tau, rho


def four_noid_sym2_patch(lam, mu, nx, ny, rmin=-3.0, rmax=3.0, eps=1e-4):
    """One quarter of the 4-noid, integrated from its Weierstrass data.

    G(z) = rho z (z - tau)(z + tau),
    dh   = z (z - tau)(z + tau) / (z^2 - lam^2)^2 / (z^2 + 1/lam^2)^2,

    on the notebook's strip: u = x + iy with y in (0, pi), and
    z = sqrt((e^u + lam^2 mu) / (mu - e^u lam^2)).  The notebook also
    prints a closed-form immersion; this integrates the data instead,
    which is the same surface and lets the end periods be MEASURED
    rather than trusted.

    The strip covers exactly the open first quadrant of the z-plane
    (the Moebius map takes the upper half E-plane to the upper half
    z^2-plane, and the principal sqrt halves that), so the strip's
    two long edges are NOT one symmetry curve each -- both are MIXED:
    the y -> 0 edge is z real in (lam, inf) for x < log(mu/lam^2) and
    z imaginary beyond it, the y -> pi edge is z real in (0, lam) for
    x < log(lam^2 mu) and z imaginary beyond that.  On the real axis
    g and dh are both real, so d(X2) = Re(phi2 du) = 0 and the curve
    lies in a plane X2 = const; on the imaginary axis g is imaginary
    and dh real, so d(X1) = 0 and the curve lies in X1 = const.  Four
    planar arcs, but only TWO planes -- the surface's two orthogonal
    mirror planes, one per z-axis.

    The two split columns are BRANCH POINTS of the strip chart, not
    just awkward spots.  Where mu - e^u lam^2 vanishes (y = 0,
    x = log(mu/lam^2)) z runs to infinity while the metric runs to
    zero; the surface point is regular -- it is one of the two points
    where the mirror planes' common axis pierces the surface -- but
    w = 1/z ~ sqrt(u - u0) there, so dX/du diverges like
    (u - u0)^(-1/2).  Same at (log(lam^2 mu), pi), where z = 0.  A
    quadrature path that runs a row past one of those spikes at
    distance eps picks up an O(1) kick that contaminates everything
    downstream on the row -- which is exactly what displaced the two
    imaginary-axis arcs into two different, both-wrong planes and
    broke the assembly.  So the integration runs a single spine along
    the middle of the strip (far from both branch points) and then
    up/down each column, with the y-grid graded quadratically toward
    the edges so the inverse-sqrt endpoint behaviour is resolved; the
    leftover corner error is confined to the two axis columns, where
    the assembly snaps the seam onto the planes anyway.
    """
    tau, rho = four_noid_sym2_params(lam)
    t0 = math.log(mu / (lam * lam))
    t1 = math.log(lam * lam * mu)
    # the notebook's three-piece x-range, so grid lines land ON the
    # two special columns instead of straddling them
    n1 = max(4, int(nx * 0.25))
    n2 = max(6, int(nx * 0.45))
    n3 = max(4, nx - n1 - n2)
    x = np.concatenate([np.linspace(rmin, t0, n1, endpoint=False),
                        np.linspace(t0, t1, n2, endpoint=False),
                        np.linspace(t1, rmax, n3)])
    # smoothstep grading: node spacing ~ s near both strip edges keeps
    # the branch columns' (u - u0)^(-1/2) integrand tame under the
    # trapezoid rule
    s = np.linspace(0.0, 1.0, ny)
    y = eps + (math.pi - 2.0 * eps) * s * s * (3.0 - 2.0 * s)
    U = x[:, None] + 1j * y[None, :]
    E = np.exp(U)
    D = mu - E * lam * lam
    Z = np.sqrt((E + lam * lam * mu) / D)
    # dz/du in closed form: differentiating z^2 = N/D gives
    # d(z^2)/du = e^u mu (1 + lam^4) / D^2, since N'D - N D' collapses
    # to e^u mu (1 + lam^4).  Doing this numerically instead leaves a
    # residual that does NOT converge under refinement.
    dZ = E * mu * (1.0 + lam ** 4) / (2.0 * Z * D * D)
    g = rho * Z * (Z - tau) * (Z + tau)
    dh = (Z * (Z - tau) * (Z + tau)
          / ((Z ** 2 - lam ** 2) ** 2 * (Z ** 2 + 1.0 / lam ** 2) ** 2))
    W = np.stack([0.5 * (1.0 / g - g) * dh,
                  0.5j * (1.0 / g + g) * dh,
                  dh], axis=-1) * dZ[..., None]
    jm = ny // 2
    spine = np.zeros((len(x), 3), dtype=complex)
    spine[1:] = np.cumsum(0.5 * (W[1:, jm] + W[:-1, jm])
                          * np.diff(x)[:, None], axis=0)
    acc = np.zeros(W.shape, dtype=complex)
    dy = np.diff(y)
    Wi = W * 1j                          # du = i dy along a column
    acc[:, jm + 1:] = np.cumsum(0.5 * (Wi[:, jm + 1:] + Wi[:, jm:-1])
                                * dy[jm:][None, :, None], axis=1)
    rev = Wi[:, jm::-1]
    acc[:, jm - 1::-1] = -np.cumsum(0.5 * (rev[:, 1:] + rev[:, :-1])
                                    * dy[:jm][::-1][None, :, None],
                                    axis=1)
    return np.real(spine[:, None, :] + acc), x, y


def four_noid_sym2_end_periods(lam, r=1e-3, n=4000):
    """Re(period) around each of the four ends z = +-lam, +-i/lam.

    The claim that makes this family exist -- Karcher solves its
    period problem in closed form -- reduced to four numbers.
    """
    tau, rho = four_noid_sym2_params(lam)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    out = []
    for pole in (lam + 0j, -lam + 0j, 1j / lam, -1j / lam):
        z = pole + r * np.exp(1j * th)
        dz = 1j * r * np.exp(1j * th) * (2.0 * np.pi / n)
        g = rho * z * (z - tau) * (z + tau)
        dh = (z * (z - tau) * (z + tau)
              / ((z ** 2 - lam ** 2) ** 2 * (z ** 2 + 1.0 / lam ** 2) ** 2))
        P = np.stack([0.5 * (1.0 / g - g) * dh,
                      0.5j * (1.0 / g + g) * dh, dh], axis=-1)
        out.append(np.real((P * dz[:, None]).sum(axis=0)))
    return np.array(out)


def four_noid_sym2_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                        storeys=1):
    """Karcher's 4-noid with two orthogonal symmetry planes.

    A quarter patch, then the notebook's MeshReflect assembly --
    reflect in x = 0, then in y = 0 -- but only after the patch has
    been moved so its own mirror planes ARE the coordinate planes.
    The numerical integration starts from an arbitrary base point, so
    the two planes come out at x = c2 and y = c1 rather than through
    the origin (the notebook's closed-form immersion carries the
    constant of integration that puts them there; the integral does
    not).  Both constants are measured off the boundary arcs -- see
    `four_noid_sym2_patch` for why each strip edge is two arcs in two
    different planes, split at log(mu/lam^2) and log(lam^2 mu) -- and
    the arcs are then snapped exactly onto their planes so the mirror
    seams weld pointwise.  `order` picks the member along lambda (end
    position), `radius` the growth mu.

    Verified against Weber's own PoVRay exports of this family (the
    `dummy.pov` mesh files ARE the assembled surface -- the scene
    renders the one included piece with no further reflections): a
    point-cloud registration of our mesh onto his lands at ~0.2% of
    the bounding span (mean two-sided nearest-neighbour distance) for
    BOTH members he renders, and every rigid-motion-invariant end
    statistic agrees -- all pairwise angles between the four end axes
    to 0.2 degrees, end-circle radii to 4e-4.  The `_selftest` gate
    checks those invariants plus the topology (one sheet, chi = -2,
    exactly four boundary loops, oriented): four loose discs, a
    mis-welded seam, or a wrong member all break it.

    Sliders: `radius` IS mu (the notebook's second parameter -- it
    sets where each end pair is truncated, so it trades the size of
    the two wide ends against the two narrow funnels), so the
    defaults (order 1, radius 1.2) land exactly on the lambda = 1.2,
    mu = 1.2 member pictured on the minimalsurfaces.blog page.
    `storeys` (End Reach) at its default keeps the notebook's own
    window x in [-3, 3]; larger values push the truncation further
    out the logarithmically-growing catenoid ends, which plumps the
    figure toward isotropy -- the surface is the same, the window is
    not.
    """
    del spec, theta
    # lambda > 1 is required by sqrt(lambda - 1); very close to 1 the
    # nested radical blows tau up and the ends degenerate, so the
    # slider runs over the range the notebook actually renders.
    lam = float(np.clip(1.2 + 0.15 * (max(int(order), 1) - 1), 1.2, 2.6))
    mu = float(np.clip(radius, 0.4, 12.0))
    nx = int(np.clip(nu * 2, 60, 400))
    ny = int(np.clip(nv, 30, 200))
    # reach 1 is the notebook's own window x in [-3, 3] (also what
    # Weber's PoVRay exports truncate at, which the selftest's
    # shape comparison relies on); the mapping keeps reach 1 up
    # through the operator's default storeys, so the out-of-the-box
    # figure is Weber's, and deeper reach is opt-in
    reach = float(np.clip(0.4 + 0.2 * max(int(storeys), 1), 1.0, 4.0))
    X, x, y = four_noid_sym2_patch(lam, mu, nx, ny,
                                   rmin=-reach * 3.0, rmax=reach * 3.0)
    # locate the two mirror planes from the four boundary arcs: real-
    # axis arcs (x < t0 on the y=0 edge, x < t1 on the y=pi edge) lie
    # in the plane X2 = c1, imaginary-axis arcs in X1 = c2
    t0 = math.log(mu / (lam * lam))
    t1 = math.log(lam * lam * mu)
    lo0, hi0 = x < t0 - 1e-12, x > t0 + 1e-12
    lo1, hi1 = x < t1 - 1e-12, x > t1 + 1e-12
    c1 = float(np.median(np.concatenate([X[lo0, 0, 1], X[lo1, -1, 1]])))
    c2 = float(np.median(np.concatenate([X[hi0, 0, 0], X[hi1, -1, 0]])))
    X = X - np.array([c2, c1, 0.0])
    # snap each arc exactly onto its plane (a sub-milliunit move: the
    # arcs sit eps inside the strip plus quadrature residual) so the
    # mirror copies coincide pointwise along the seams; the two axis
    # columns are on BOTH planes
    X[lo0, 0, 1] = 0.0
    X[hi0, 0, 0] = 0.0
    X[~lo0 & ~hi0, 0, 0:2] = 0.0
    X[lo1, -1, 1] = 0.0
    X[hi1, -1, 0] = 0.0
    X[~lo1 & ~hi1, -1, 0:2] = 0.0
    nxp, nyp = X.shape[0], X.shape[1]
    quads = [(i * nyp + j, (i + 1) * nyp + j,
              (i + 1) * nyp + j + 1, i * nyp + j + 1)
             for i in range(nxp - 1) for j in range(nyp - 1)]
    V0 = X.reshape(-1, 3)
    n0 = len(V0)
    # each single reflection reverses orientation, so the mirrored
    # copy's quads flip their winding (and the doubly-mirrored copy
    # flips twice, back to the original) -- the welded sheet then
    # carries ONE consistent orientation, which the selftest gates
    blk = np.concatenate([V0, V0 * np.array([-1.0, 1.0, 1.0])], axis=0)
    fblk = quads + [tuple(a + n0 for a in q)[::-1] for q in quads]
    n1 = len(blk)
    full = np.concatenate([blk, blk * np.array([1.0, -1.0, 1.0])], axis=0)
    ffull = fblk + [tuple(a + n1 for a in q)[::-1] for q in fblk]
    # WELD the two mirror seams.  Concatenating the reflected copies
    # leaves four loose quarter-patches that sit in the right places
    # and look, from far enough away, like a 4-noid with four ends --
    # they even pass an end-counting test.  They are four discs.
    # The surface is one sheet: weld, then check chi.
    from .plateau import _weld_points
    span = float(np.max(full.max(axis=0) - full.min(axis=0)))
    full, ffull = _weld_points(full, ffull, 1e-7 * max(span, 1e-12))
    # stand it up: in integration coordinates the mirror planes are
    # x = 0 and y = 0 and the two WIDE ends' axes hug +-y, so the raw
    # assembly lies on its side.  Rotate -90 deg about x (y -> z, in
    # Blender's z-up frame) so the wide ends face up/down the way
    # Weber renders the family in his y-up PoVRay scenes.
    full = full[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
    # centre and fit the way every other zoo row does -- the raw
    # integration comes out tens of units across and offset
    return _center_fit(full, scale, full), ffull


def _fournoid_end_stats(V, F):
    """Boundary-loop statistics: (mean radius, centre, outward axis)
    per loop, sorted largest-radius first.  Test-only: these are the
    rigid-motion-invariant end statistics the 4-noid selftest holds
    against Weber's own PoVRay exports -- the angles between end axes
    and the wide/narrow radius ratio survive any translation,
    rotation or uniform scale, so they pin the SHAPE where a
    bounding box or an Euler characteristic cannot."""
    V = np.asarray(V)
    ec = {}
    for f in F:
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            e = (a, b) if a < b else (b, a)
            ec[e] = ec.get(e, 0) + 1
    nbr = {}
    for (a, b), c in ec.items():
        if c == 1:
            nbr.setdefault(a, []).append(b)
            nbr.setdefault(b, []).append(a)
    eseen, out = set(), []
    for a0 in nbr:
        loop, cur = [a0], a0
        while True:
            nxt = None
            for c in nbr[cur]:
                e = (cur, c) if cur < c else (c, cur)
                if e not in eseen:
                    nxt = c
                    eseen.add(e)
                    break
            if nxt is None or nxt == a0:
                break
            loop.append(nxt)
            cur = nxt
        if len(loop) < 8:
            continue
        P = V[np.array(loop)]
        cen = P.mean(axis=0)
        Q = P - cen
        ax = np.linalg.svd(Q)[2][2]
        if float(np.dot(ax, cen)) < 0.0:
            ax = -ax
        out.append((float(np.linalg.norm(Q, axis=1).mean()), cen, ax))
    out.sort(key=lambda t: -t[0])
    return out


def catenoid_field_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                        cells=(1, 1)):
    """cells2d builder: cu x cv lattice cells of the field, seams
    index-welded (the deck maps are exact horizontal translations, so
    opposite window edges carry identical grids one lattice vector
    apart).  order drives the growth factor bb; radius the end trim
    (bigger radius = wider flare)."""
    if isinstance(cells, (int, float)):
        cells = (int(cells), 1)
    cu = int(np.clip(cells[0], 1, 8))
    cv = int(np.clip(cells[1] if len(cells) > 1 else 1, 1, 8))
    t = 1.0
    bb = 0.6 + 0.4 * float(np.clip(order, 1, 6))
    W = catenoid_field_W(bb, t)
    punct = (0.0 + 0.0j, 0.5 + 0.0j)
    r0 = 2.0e-3 * (1.2 / float(np.clip(radius, 0.3, 6.0))) ** 2
    n = int(np.clip(nu * 2.0, 90, 260))
    xs, ys = we_ends_grid((-0.25, 0.75, -0.5 * t, 0.5 * t), punct, n)
    X = we_ends_integrate(W, xs, ys, punct)
    mask = we_ends_mask(xs, ys, punct, r0)
    # lattice vectors, measured off the immersion the way the deck
    # identities are gated: straight probes away from the punctures
    tt = np.linspace(0.0, 1.0, 8001)
    z1 = (-0.2 + 0.25j * t) + tt * 1.0
    P1 = np.real(np.trapezoid(W(z1), tt, axis=0))
    z2 = (0.25 - 0.5j * t) + tt * (1j * t)
    P2 = np.real(np.trapezoid(W(z2) * (1j * t), tt, axis=0))
    nx, ny = X.shape[0], X.shape[1]
    NV = nx * ny
    Vs, Ms = [], []
    for iu in range(cu):
        for iv in range(cv):
            Vs.append((X + iu * P1 + iv * P2).reshape(-1, 3))
            Ms.append(mask.reshape(-1))
    V = np.concatenate(Vs, axis=0)
    mall = np.concatenate(Ms, axis=0)
    parent = np.arange(cu * cv * NV)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def cell(iu, iv):
        return (iu * cv + iv) * NV
    for iu in range(cu):
        for iv in range(cv):
            if iu + 1 < cu:      # right edge == next cell's left edge
                for j in range(ny):
                    a = find(cell(iu, iv) + (nx - 1) * ny + j)
                    b = find(cell(iu + 1, iv) + j)
                    if a != b:
                        parent[a] = b
            if iv + 1 < cv:      # top edge == next cell's bottom edge
                for i in range(nx):
                    a = find(cell(iu, iv) + i * ny + (ny - 1))
                    b = find(cell(iu, iv + 1) + i * ny)
                    if a != b:
                        parent[a] = b
    roots = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    sums = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(sums, inv, V)
    np.add.at(cnt, inv, 1.0)
    Vm = sums / cnt[:, None]
    ok_node = np.ones(len(uniq), dtype=bool)
    np.logical_and.at(ok_node, inv, mall)
    quads = []
    for c0 in range(cu * cv):
        for i in range(nx - 1):
            b0 = c0 * NV + i * ny
            b1 = c0 * NV + (i + 1) * ny
            for j in range(ny - 1):
                f = (inv[b0 + j], inv[b1 + j],
                     inv[b1 + j + 1], inv[b0 + j + 1])
                if (ok_node[f[0]] and ok_node[f[1]]
                        and ok_node[f[2]] and ok_node[f[3]]):
                    quads.append(f)
    used = np.zeros(len(Vm), dtype=bool)
    for f in quads:
        for a in f:
            used[a] = True
    remap = -np.ones(len(Vm), dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    V = Vm[used]
    quads = [tuple(int(remap[a]) for a in f) for f in quads]
    V = _center_fit(V, scale, V)
    return V, quads, None


# ==========================================================================
# Symmetrized Chen-Gackstatter towers (k-fold symmetry, D_kd assembly)
# ==========================================================================
# The k-fold-symmetric continuations of the Chen-Gackstatter surface: for
# each symmetry order k these are complete minimal surfaces with ONE
# Enneper-type end and dihedral-antiprismatic symmetry D_kd (order 4k),
# living on the k-fold cyclic cover of the sphere branched over the real
# branch values {0, +-r_1, ..., +-r_m, inf}.  Unified Weierstrass data
# (e = (k-1)/k throughout, dh = dz so x3 = Re z):
#   * genus k-1 tower:    g = rho z^(-e) (1 - z^2)^e
#     with the CLOSED-FORM Lopez-Ros constant (a Gamma-quotient; equal to
#     Weber's  1/Sqrt[4^e G(3/2-e/2) G(1+e/2) / (G(1-e/2) G((3+e)/2))]):
#       rho^2 = Int_0^1 |g/rho|^-1 dz / Int_0^1 |g/rho| dz
#             = G((1+e)/2) G(1-e) G((3+e)/2)
#               / ( G((3-e)/2) G((1-e)/2) G(1+e) );
#     k = 2 is the classical Chen-Gackstatter torus, k = 4 gives GENUS 3
#     (the genus the D2d normalization of cg_higher_* cannot reach).
#   * genus 2(k-1) tower: g = rho z^e (1-z^2)^(-e) (1-(z/a)^2)^e
#   * genus 3(k-1) tower: g = rho z^e (1-z^2)^(-e) (1-(z/a)^2)^e
#                             (1-(z/b)^2)^(-e)
#     where a (resp. a, b) solve the 1-D (2-D) period problem -- numeric
#     values from Weber's notebooks (see _SYMMCG_G2N / _SYMMCG_G3K); rho
#     then follows from the same per-segment period ratio, and the
#     surviving per-segment consistency (every real segment must yield
#     the SAME rho) is the self-test's validation of those constants.
#
# Symmetry / meshing scheme (generalizes the D2d = D_2d octant assembly
# of cg_higher_* from order 8 to order 4k): the QUARTER z-plane
# {Re z >= 0, Im z >= 0} is one 1/(4k) fundamental piece of the surface.
#   * the real intervals between consecutive branch values map to planar
#     geodesics in VERTICAL mirror planes through the axis; crossing the
#     branch value r_j rotates the plane by -pi b_j (the branch of
#     (1-(z/r_j)^2)^(b_j) continued from Im z > 0), so segment s lies in
#     the plane with polar angle psi_s = -pi sum_{j<s} b_j;
#   * the imaginary axis maps to a horizontal 2-fold axis in x3 = 0 at
#     polar angle pi (1 + a0) / 2  (a0 = the z-exponent of g);
#   * every branch value r_j (and z = 0) maps ONTO the vertical axis at
#     height r_j -- the branch images (0, 0, r_j) thread the axis;
#   * the group is generated by the rotoreflection
#     S = Rz(pi (1 + a0)) diag(1, 1, -1)  with  S^2 = Rz(2 pi / k) (the
#     deck rotation g -> e^{2 pi i / k} g) and the mirror
#     M = diag(1, -1, 1) (z -> conj z): 2k holomorphic frames S^j keep
#     the face winding, 2k antiholomorphic frames S^j M reverse it.  For
#     k = 2 (a0 = -1/2) S is exactly the S4 rotoreflection of the D2d
#     octant assembler.
# The WE forms are integrated along radial rays with one continuous
# branch of g (per-factor angle unwrapping; the theta = 0 ray uses the
# exact piecewise-constant branch with exact offsets to the branch
# values, so the algebraic singularities of |g|^{+-1} at 0 and r_j are
# integrated on strongly graded subgrids in the substitution variable,
# where the integrand is smooth).  Boundary vertices are snapped exactly
# onto their symmetry elements, the piece is trimmed by a sphere about
# the origin past the outermost branch images, orbited under all 4k
# frames and welded by boundary coincidence -- the result is exactly
# Euler characteristic chi = 1 - 2 genus (one boundary loop), gated by
# the self-tests below.
#
# References:
#   C. C. Chen, F. Gackstatter, "Elliptische und hyperelliptische
#     Funktionen und vollstaendige Minimalflaechen vom Enneperschen
#     Typ", Math. Ann. 259 (1982) -- the k = 2 surface;
#   H. Karcher, "Construction of minimal surfaces", Univ. of Tokyo
#     lecture notes (1989) -- the symmetrization method and the k-fold
#     towers;
#   E. C. Thayer, "Higher-genus Chen-Gackstatter surfaces and the
#     Weierstrass representation for surfaces of infinite genus",
#     Experiment. Math. 4 (1995);
#   M. Weber, https://minimalsurfaces.blog/ (repository,
#     "Symmetrized Chen-Gackstatter" pages) -- the data tables and the
#     closed-form rho this implementation follows.

# Period solutions k -> (a, rho): branch value a and the Lopez-Ros
# constant rho in THIS normalization (g2n factor (1-(z/a)^2)^e; Weber's
# notebooks use (a^2-z^2)^e, i.e. rho_nb = rho / a^{2e}).  Seeded from
# the values harvested off minimalsurfaces.blog and re-solved here to
# full double precision (secant on the per-segment period-ratio
# compatibility; the seeds agree to their published ~1e-4 precision,
# and symmcg_rho_solve in the self-test re-derives rho from scratch).
_SYMMCG_G2N = {3: (1.7168384042293399, 1.422067467716003),
               4: (1.7196230274467104, 1.4905575097998258),
               7: (1.7240865072638545, 1.5865591121925635),
               12: (1.727085736152026, 1.644449176251574)}
# k -> (a, b, rho): the 2-D period problem (2-D Newton on the three-way
# per-segment compatibility, seeded from Weber's (a, b) values); rho has
# no closed form -- the stored value is the common per-segment ratio
_SYMMCG_G3K = {2: (2.328309968067604, 3.1051967096086575,
                   1.228240310564583),
               3: (2.3294796517339766, 3.11368463663408,
                   1.3120297016852307),
               4: (2.3303049378429166, 3.1194657181873082,
                   1.3548397561984624),
               5: (2.330902276725775, 3.123554235422156,
                   1.380698302281986),
               7: (2.3316997357092446, 3.1288985424668323,
                   1.4103169367659762)}

_SYMMCG_RHO_CACHE = {}


def _symmcg_data(tower, k):
    """(a0, [(root, b), ...], rho) with g = rho z^a0 prod (1-(z/root)^2)^b;
    rho None means 'solve numerically' (see symmcg_rho_solve)."""
    e = (k - 1.0) / k
    G = math.gamma
    if tower == 'gn':
        rho = math.sqrt(G((1 + e) / 2) * G(1 - e) * G((3 + e) / 2)
                        / (G((3 - e) / 2) * G((1 - e) / 2) * G(1 + e)))
        return -e, [(1.0, e)], rho
    if tower == 'g2n':
        a, rho = _SYMMCG_G2N[k]
        return e, [(1.0, -e), (a, e)], rho
    if tower == 'g3k':
        a, b, rho = _SYMMCG_G3K[k]
        return e, [(1.0, -e), (a, e), (b, -e)], rho
    raise KeyError(f"symmcg tower {tower!r}")


def _symmcg_abs_logG(a0, factors, r, dL, dR, segA, segB):
    """log |g / rho| on a real segment; dL / dR are EXACT offsets to the
    segment endpoints (no cancellation at the branch values)."""
    with np.errstate(divide='ignore'):
        out = a0 * np.log(np.abs(r))
        for (rt, b) in factors:
            if abs(rt - segA) < 1e-12:
                t = np.abs(dL) / rt
            elif abs(rt - segB) < 1e-12:
                t = np.abs(dR) / rt
            else:
                t = np.abs(1.0 - r / rt)
            out = out + b * (np.log(t) + np.log1p(r / rt))
    return out


def _symmcg_seg_int(logf, A, B, eL, eR, n=3000):
    """Integral of exp(logf(r, dL, dR)) over [A, B] with known algebraic
    endpoint exponents |eL|, |eR| (0 = regular end): midpoint split, each
    half integrated in the graded variable u (r = end +- D u^p with
    p = 2/(1-e), where the integrand is smooth) by trapezoid."""
    total = 0.0
    M = 0.5 * (A + B)
    for lo, hi, ee, from_left in ((A, M, eL, True), (M, B, eR, False)):
        D = hi - lo
        p = 2.0 / max(1e-9, 1.0 - min(ee, 0.95)) if ee > 0 else 1.0
        u = np.linspace(0.0, 1.0, n + 1)
        off = D * u ** p
        if from_left:
            r = lo + off
            dL, dR = off + (lo - A), (B - lo) - off
        else:
            r = hi - off
            dL, dR = (hi - A) - off, off + (B - hi)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            f = np.exp(logf(r, dL, dR)) * D * p * u ** (p - 1.0)
        f = np.where(np.isfinite(f), f, 0.0)
        total += float(np.trapezoid(f, u) if hasattr(np, 'trapezoid')
                       else np.trapz(f, u))
    return total


def symmcg_rho_solve(tower, k):
    """Per-segment rho estimates: on every real segment between
    consecutive branch values the period condition reads
    rho^2 = Int |g/rho|^-1 / Int |g/rho|; all segments must agree (that
    IS the period problem -- the spread validates the (a, b) data).
    Returns (rho_from_segment_0, [estimates...])."""
    key = (tower, k)
    if key in _SYMMCG_RHO_CACHE:
        return _SYMMCG_RHO_CACHE[key]
    a0, factors, _ = _symmcg_data(tower, k)
    bounds = [0.0] + [rt for rt, _ in factors]
    est = []
    for s in range(len(bounds) - 1):
        A, B = bounds[s], bounds[s + 1]
        eL = abs(a0) if s == 0 else abs(factors[s - 1][1])
        eR = abs(factors[s][1])
        Ig = _symmcg_seg_int(
            lambda r, dL, dR: _symmcg_abs_logG(a0, factors, r, dL, dR,
                                               A, B), A, B, eL, eR)
        Iq = _symmcg_seg_int(
            lambda r, dL, dR: -_symmcg_abs_logG(a0, factors, r, dL, dR,
                                                A, B), A, B, eL, eR)
        est.append(math.sqrt(Iq / Ig))
    _SYMMCG_RHO_CACHE[key] = (est[0], est)
    return _SYMMCG_RHO_CACHE[key]


def _symmcg_radial_grids(roots, rmax, nu, e_by_end, msub=8, msing=100,
                         pmesh=1.7):
    """Coarse mesh nodes + fine integration nodes with per-interval
    trapezoid weights: inside the graded blocks adjacent to an algebraic
    singularity the weights are the u-substitution Jacobian weights (the
    integrand is smooth in u), elsewhere plain dr/2.  Returns
    (r_coarse, r_fine, coarse_idx, sing_id, sing_off, w_lo, w_hi); the
    ray integral is cumsum(w_lo f[:-1] + w_hi f[1:])."""
    bounds = [0.0] + list(roots) + [rmax]
    nseg = len(bounds) - 1
    lens = np.diff(np.array(bounds))
    w = np.sqrt(lens)
    counts = np.maximum(10, np.round(nu * w / w.sum()).astype(int))
    r_c, r_f, sid, soff = [0.0], [0.0], [-1], [0.0]
    cidx = [0]
    w_lo, w_hi = [], []

    def root_id(x):
        for i, rt in enumerate(roots):
            if abs(x - rt) < 1e-12:
                return i
        return -1

    for s in range(nseg):
        A, B = bounds[s], bounds[s + 1]
        n = int(counts[s])
        last = s == nseg - 1
        t = np.linspace(0.0, 1.0, n + 1)[1:]
        if last:
            tt = t ** pmesh              # dense only at the branch end
        else:                            # dense toward both branch ends
            tt = np.where(t < 0.5, 0.5 * (2 * t) ** pmesh,
                          1.0 - 0.5 * (2 * (1 - t)) ** pmesh)
        nodes = A + (B - A) * tt
        nodes[-1] = B
        prev = A
        for ci in range(n):
            hi = nodes[ci]
            singL = (ci == 0 and A in e_by_end)
            singR = (ci == n - 1 and not last and B in e_by_end)
            if singL:                    # graded from the left end A
                p = 2.0 / max(1e-9, 1.0 - min(e_by_end[A], 0.95))
                ms = int(msing * max(1.0, p / 8.0))   # denser as e -> 1
                D = hi - A
                u = np.linspace(0.0, 1.0, ms + 1)
                off = D * u ** p
                J = D * p * u ** (p - 1.0)
                du = 1.0 / ms
                rf = (A + off)[1:]
                rf[-1] = hi
                sid_i = np.full(len(rf), root_id(A))
                soff_i = off[1:]
                w_lo.extend((0.5 * du * J[:-1]).tolist())
                w_hi.extend((0.5 * du * J[1:]).tolist())
            elif singR:                  # graded toward the right end B
                p = 2.0 / max(1e-9, 1.0 - min(e_by_end[B], 0.95))
                ms = int(msing * max(1.0, p / 8.0))
                D = B - prev
                v = np.linspace(0.0, 1.0, ms + 1)
                offB = D * (1.0 - v) ** p
                J = D * p * (1.0 - v) ** (p - 1.0)
                dv = 1.0 / ms
                rf = (B - offB)[1:]
                rf[-1] = B
                sid_i = np.full(len(rf), root_id(B))
                soff_i = -offB[1:]
                w_lo.extend((0.5 * dv * J[:-1]).tolist())
                w_hi.extend((0.5 * dv * J[1:]).tolist())
            else:
                # inner cell: if its segment half ends at an algebraic
                # singularity, place the subnodes log-uniform in the
                # distance d to that end and integrate in the log
                # variable (f d varies like d^{1-e} there -- nearly
                # flat), else plain uniform trapezoid
                t = np.linspace(0.0, 1.0, msub + 1)
                left_half = (prev - A) <= (B - hi)
                if left_half and A in e_by_end and prev > A:
                    dlo, dhi = prev - A, hi - A
                    d = dlo * (dhi / dlo) ** t
                    rf_full = A + d
                    J = d * math.log(dhi / dlo)
                elif (not left_half and not last and B in e_by_end
                      and hi < B):
                    dtop, dbot = B - prev, B - hi
                    d = dtop * (dbot / dtop) ** t
                    rf_full = B - d
                    J = d * abs(math.log(dbot / dtop))
                else:
                    rf_full = prev + (hi - prev) * t
                    J = None
                rf = rf_full[1:].copy()
                rf[-1] = hi
                sid_i = np.full(len(rf), -1)
                soff_i = np.zeros(len(rf))
                if J is None:
                    h = 0.5 * (hi - prev) / msub
                    w_lo.extend([h] * msub)
                    w_hi.extend([h] * msub)
                else:
                    dt = 1.0 / msub
                    w_lo.extend((0.5 * dt * J[:-1]).tolist())
                    w_hi.extend((0.5 * dt * J[1:]).tolist())
            r_f.extend(rf.tolist())
            sid.extend(sid_i.tolist())
            soff.extend(soff_i.tolist())
            cidx.append(len(r_f) - 1)
            r_c.append(hi)
            prev = hi
    return (np.array(r_c), np.array(r_f), np.array(cidx),
            np.array(sid, dtype=int), np.array(soff),
            np.array(w_lo), np.array(w_hi))


def _symmcg_phi_ray(a0, factors, rho, r_f, sid, soff, theta):
    """(nf, 3) complex integrand phi e^{i theta} along one radial ray
    with a continuous branch of g (per-factor angle unwrap; the theta = 0
    ray uses the exact piecewise branch: angle 0 before each branch
    value, -pi after -- the limit from Im z > 0 -- and exact offsets for
    |1 - r/root|).  Non-finite entries (the branch nodes) are zeroed;
    their quadrature weight is negligible on the graded grid."""
    nf = len(r_f)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        lmag = math.log(rho) + a0 * np.log(r_f)
        ang = a0 * theta * np.ones(nf)
        if theta == 0.0:
            for fi, (rt, b) in enumerate(factors):
                t = np.abs(1.0 - r_f / rt)
                m = sid == fi
                t[m] = np.abs(soff[m]) / rt
                lmag = lmag + b * (np.log(t) + np.log1p(r_f / rt))
                # which side of the root: for the graded-block nodes the
                # stored offset sign is EXACT (their r may round onto the
                # root itself -- r = rt + 1e-40 == rt in doubles -- and a
                # plain r > rt test would put them on the wrong branch)
                side = r_f > rt
                side[m] = soff[m] > 0
                ang = ang + b * np.where(side, -math.pi, 0.0)
        else:
            z = r_f * np.exp(1j * theta)
            for (rt, b) in factors:
                P = 1.0 - (z / rt) ** 2
                lmag = lmag + b * np.log(np.abs(P))
                ang = ang + b * np.unwrap(np.angle(P))
        g = np.exp(lmag + 1j * ang)
        ph = np.stack([0.5 * (1.0 / g - g),
                       0.5j * (1.0 / g + g),
                       np.ones_like(g)], axis=-1) * np.exp(1j * theta)
    return np.where(np.isfinite(ph), ph, 0.0)


def _symmcg_probe_rmax(a0, factors, rho, target):
    """Parameter radius at which the image reaches `target` (probed on
    the mid ray theta = pi/4), so the sphere trim never runs dry."""
    r = np.linspace(0.0, 80.0, 200001)
    ph = _symmcg_phi_ray(a0, factors, rho, r, np.full(len(r), -1),
                         np.zeros(len(r)), 0.25 * math.pi)
    dr = np.diff(r)[:, None]
    F = np.concatenate([np.zeros((1, 3), complex),
                        np.cumsum(0.5 * (ph[1:] + ph[:-1]) * dr, axis=0)],
                       axis=0)
    rad = np.linalg.norm(np.real(F), axis=1)
    ok = np.nonzero(rad >= target)[0]
    return float(r[ok[0]]) if len(ok) else 80.0


def _symmcg_rend0(tower, k):
    """Default sphere-trim radius (raw units): past the outermost branch
    images (heights 1 / a / b) AND past the handle lobes (which widen
    with k -- at gn k = 8 they reach r ~ 2.5), with room for the flare."""
    return {'gn': 2.6, 'g2n': 3.9, 'g3k': 5.6}[tower]


def _symmcg_piece(tower, k, nu, nv, Rend):
    """Mesh the 1/(4k) fundamental piece (quarter plane, radial rays).
    Returns (V, faces, boundary-classification, per-vertex uv, diag)."""
    a0, factors, rho = _symmcg_data(tower, k)
    if rho is None:
        rho = symmcg_rho_solve(tower, k)[0]
    roots = [rt for rt, _ in factors]
    e_by_end = {0.0: abs(a0)}
    for rt, b in factors:
        e_by_end[rt] = abs(b)
    rmax = _symmcg_probe_rmax(a0, factors, rho, 1.45 * Rend)
    r_c, r_f, cidx, sid, soff, w_lo, w_hi = _symmcg_radial_grids(
        roots, rmax, nu, e_by_end)
    th = np.linspace(0.0, 0.5 * math.pi, nv)
    F = np.zeros((len(r_c), nv, 3))
    for j, t in enumerate(th):
        ph = _symmcg_phi_ray(a0, factors, rho, r_f, sid, soff, float(t))
        contrib = w_lo[:, None] * ph[:-1] + w_hi[:, None] * ph[1:]
        Ff = np.concatenate([np.zeros((1, 3), complex),
                             np.cumsum(contrib, axis=0)], axis=0)
        F[:, j, :] = np.real(Ff[cidx])
    ncr = len(r_c)

    def vid(i, j):                       # row 0 collapses to the center
        return 0 if i == 0 else (i - 1) * nv + j + 1

    V = np.concatenate([F[:1, 0, :], F[1:].reshape(-1, 3)], axis=0)
    UV = np.zeros((len(V), 2))
    UV[1:, 0] = np.tile(th / (0.5 * math.pi), ncr - 1)
    UV[1:, 1] = np.repeat(r_c[1:] / r_c[-1], nv)
    faces = []
    for j in range(nv - 1):
        faces.append((0, vid(1, j), vid(1, j + 1)))
    for i in range(1, ncr - 1):
        for j in range(nv - 1):
            faces.append((vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)))
    root_ci = [int(np.argmin(np.abs(r_c - rt))) for rt in roots]
    b = {'seg': {}, 'branch': {}, 'branch_z': {}, 'center': 0}
    lastci = 0
    for s in range(len(roots) + 1):
        hi_ci = root_ci[s] if s < len(roots) else ncr - 1
        stop = hi_ci + (1 if s == len(roots) else 0)
        b['seg'][s] = np.array([vid(i, 0)
                                for i in range(lastci + 1, stop)],
                               dtype=np.int64)
        if s < len(roots):
            b['branch'][s] = vid(hi_ci, 0)
            b['branch_z'][s] = roots[s]
        lastci = hi_ci
    b['seam'] = np.array([vid(i, nv - 1) for i in range(1, ncr)],
                         dtype=np.int64)
    psis = [0.0]                         # mirror-plane angle per segment
    for (rt, bb) in factors:
        psis.append(psis[-1] - math.pi * bb)
    b['psi'] = psis
    b['psi_seam'] = 0.5 * math.pi * (1.0 + a0)
    return V, faces, b, UV, {'rmax': rmax, 'rho': rho, 'a0': a0}


def _symmcg_snap(V, b):
    """Snap every boundary vertex exactly onto its symmetry element.
    Returns the PRE-snap deviations -- the numeric period-closure
    residuals (they must be small; the snap only removes quadrature
    noise, it cannot fix wrong period data)."""
    res = {}
    for s, idx in b['seg'].items():
        if len(idx) == 0:
            continue
        psi = b['psi'][s]
        u = np.array([math.cos(psi), math.sin(psi)])
        xy = V[idx][:, :2]
        t = xy @ u
        perp = xy - t[:, None] * u[None, :]
        res[f'seg{s}'] = float(np.max(np.linalg.norm(perp, axis=1)))
        V[idx, 0] = t * u[0]
        V[idx, 1] = t * u[1]
    for s, vi in b['branch'].items():
        res[f'branch{s}'] = float(np.hypot(V[vi, 0], V[vi, 1]))
        V[vi] = (0.0, 0.0, b['branch_z'][s])
    u = np.array([math.cos(b['psi_seam']), math.sin(b['psi_seam'])])
    idx = b['seam']
    xy = V[idx][:, :2]
    t = xy @ u
    perp = xy - t[:, None] * u[None, :]
    res['seam'] = float(np.max(np.hypot(np.linalg.norm(perp, axis=1),
                                        V[idx, 2])))
    V[idx, 0] = t * u[0]
    V[idx, 1] = t * u[1]
    V[idx, 2] = 0.0
    V[b['center']] = (0.0, 0.0, 0.0)
    return V, res


def _symmcg_frames(k, a0):
    """The 4k isometries of D_kd: 2k holomorphic frames S^j (winding
    kept -- S is a rotoreflection but holomorphic, like S4 in the D2d
    assembler) and 2k antiholomorphic frames S^j M (winding reversed)."""
    ang = math.pi * (1.0 + a0)
    c, s = math.cos(ang), math.sin(ang)
    S = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, -1.0]])
    My = np.diag([1.0, -1.0, 1.0])
    holo = []
    M = np.eye(3)
    for _ in range(2 * k):
        holo.append(M)
        M = S @ M
    return holo + [Mk @ My for Mk in holo]


def symmcg_assemble(tower, k, nu, nv, Rend=None):
    """Watertight D_kd assembly: snap, sphere-trim, orbit the 1/(4k)
    piece under all 4k frames, weld by boundary coincidence.  Returns
    (V, faces, uv, diag) of the largest component; diag carries the
    period residuals."""
    if Rend is None:
        Rend = _symmcg_rend0(tower, k)
    V0, faces0, b, UV0, diag = _symmcg_piece(tower, k, nu, nv, Rend)
    V0, res = _symmcg_snap(V0.copy(), b)
    diag['res'] = res
    bmask = np.zeros(len(V0), dtype=bool)
    for idx in b['seg'].values():
        bmask[idx] = True
    bmask[b['seam']] = True
    bmask[list(b['branch'].values())] = True
    bmask[b['center']] = True
    keep = np.linalg.norm(V0, axis=1) <= Rend
    faces0 = [f for f in faces0 if all(keep[i] for i in f)]
    used = sorted(set(i for f in faces0 for i in f))
    rmv = {v: i for i, v in enumerate(used)}
    V0, UV0, bmask = V0[used], UV0[used], bmask[used]
    faces0 = [tuple(rmv[i] for i in f) for f in faces0]
    nV = len(V0)
    frames = _symmcg_frames(k, diag['a0'])
    Vp, Fp = [], []
    for fr, M in enumerate(frames):
        Vp.append(V0 @ M.T)
        rev = fr >= 2 * k                # antiholomorphic copies flip
        for f in faces0:
            ff = tuple(int(x) + fr * nV for x in f)
            Fp.append(ff[::-1] if rev else ff)
    V = np.concatenate(Vp)
    UVall = np.tile(UV0, (len(frames), 1))
    N = len(V)
    tol = 1e-6                           # coincidence weld on boundary
    parent = np.arange(N)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bidx = np.nonzero(np.tile(bmask, len(frames)))[0]
    key = np.floor(V[bidx] / tol + 0.5).astype(np.int64)
    H = {}
    for t, i in enumerate(bidx):
        H.setdefault((int(key[t, 0]), int(key[t, 1]), int(key[t, 2])),
                     []).append(int(i))
    for t, i in enumerate(bidx):
        k0 = key[t]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in H.get((int(k0[0]) + dx, int(k0[1]) + dy,
                                    int(k0[2]) + dz), ()):
                        if j > i and np.linalg.norm(V[j] - V[i]) < tol:
                            ra, rb = find(int(i)), find(j)
                            if ra != rb:
                                parent[ra] = rb
    rts = np.array([find(a) for a in range(N)])
    uniq, inv = np.unique(rts, return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    UVw = np.zeros((len(uniq), 2))
    cnt = np.zeros(len(uniq))
    np.add.at(Vw, inv, V)
    np.add.at(UVw, inv, UVall)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    UVw /= cnt[:, None]
    F = []
    for f in Fp:
        gg = [int(inv[i]) for i in f]
        h = [gg[0]]
        for t in range(1, len(gg)):
            if gg[t] != h[-1]:
                h.append(gg[t])
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            F.append(tuple(h))
    parent2 = np.arange(len(Vw))         # largest face-connected comp

    def find2(a):
        while parent2[a] != a:
            parent2[a] = parent2[parent2[a]]
            a = parent2[a]
        return a

    for f in F:
        for i in range(1, len(f)):
            ra, rb = find2(f[0]), find2(f[i])
            if ra != rb:
                parent2[ra] = rb
    from collections import Counter
    sizes = Counter(find2(f[0]) for f in F)
    root = sizes.most_common(1)[0][0]
    F = [f for f in F if find2(f[0]) == root]
    used = sorted(set(i for f in F for i in f))
    rmv = {v: i for i, v in enumerate(used)}
    return (Vw[used], [tuple(rmv[i] for i in f) for f in F],
            UVw[used], diag)


def symmcg_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """MESH_PARAM builder: finished (V, quads, uv) fit to the 2 m cube.
    The sphere trim must clear the handle lobes or the largest-component
    filter would keep only the disconnected end flare, so the build
    verifies the exact Euler characteristic and widens the trim until
    the full tower survives (chi = 1 - 2 genus gates it honestly)."""
    p = spec['p_from'](order, radius)
    tower, k = p['tower'], p['k']
    genus = {'gn': k - 1, 'g2n': 2 * (k - 1), 'g3k': 3 * (k - 1)}[tower]
    pnu = int(np.clip(nu * 1.4, 70, 200))
    pnv = int(np.clip(nv * 0.45, 16, 40))
    R0 = _symmcg_rend0(tower, k)
    Rend = float(np.clip(R0 * radius / 1.2, 0.7 * R0, 2.2 * R0))
    for _ in range(4):
        V, quads, uv, _diag = symmcg_assemble(tower, k, pnu, pnv, Rend)
        edges = set()
        for f in quads:
            m = len(f)
            for t in range(m):
                a, b = f[t], f[(t + 1) % m]
                edges.add((a, b) if a < b else (b, a))
        if len(V) - len(edges) + len(quads) == 1 - 2 * genus:
            break
        Rend *= 1.22                   # trim cut a handle: widen and retry
    V = _smooth_boundary(V, quads, iters=6)
    V = _center_fit(V, scale, V)
    return V, quads, uv


# ==========================================================================
# Callahan-Hoffman-Meeks singly periodic surface (k = 1) -- chm_* helpers
# ==========================================================================
# The singly periodic analog of the Costa surface: an embedded minimal
# surface invariant under a vertical translation, with TWO horizontal
# planar ends per translational period (so infinitely many ends in all).
# The quotient by the translation has genus 2k + 1 = 3 (the CHM theorem;
# this is the k = 1 member, Weber's CHM-(1,1) notebook), so one period
# meshes at exactly chi = -6 with the two end punctures -- the mesh gate
# below MEASURES that.  (The harvest metadata's "genus 2 / chi = -4"
# annotation is off by one handle; the built surface, the notebook and
# the theorem all agree on genus 3.)  The surface is also invariant
# under the half-period screw T_{2d} rot90 and the vertical glide
# T_{2d} swap(x, y); neither is a pure translation, so the primitive
# translation is the full 4d (verified numerically: swap(x, y) alone is
# NOT a symmetry -- probe points map ~0.1 off the surface).
#
# Weierstrass data (M. Weber's CHM-(1,1) notebook; branch values on the
# real axis at 0, +-1, +-a):
#     G    = sqrt(x) (x^2-a^2)^{3/4} / (rho (x^2-1)^{1/4})
#     phi1 = 1 / (sqrt(x) (x^2-1)^{1/4} (x^2-a^2)^{5/4})
#     phi2 = sqrt(x) (x^2-a^2)^{1/4} / (x^2-1)^{3/4}
#     om1  = (rho phi1 - phi2/rho) / 2,   om2 = i (rho phi1 + phi2/rho) / 2
#     om3  = 1 / (sqrt(x^2-1) sqrt(x^2-a^2))
# with the two solved period constants (harvested verbatim from the
# notebook's FindRoot; rho independently re-verified here to 1e-12 from
# its closure condition rho^2 = int_0^1|phi2| / int_0^1|phi1|):
#     a   = 1.31922381870184635
#     rho = 1.22370848596689185
# om1^2 + om2^2 + om3^2 = 0 holds exactly (checked in the self-test).
#
# Fundamental patch.  Following the notebook's substitution, the patch is
# integrated on the strip chart  w = u + i t,  z(w) = sqrt(a^2 + e^w)
# (so z^2 - a^2 = e^w exactly), u in [umin, umax], t in [0, pi], with
# umax = log(a sqrt(a^2-1)).  On it every principal fractional power is
# single-valued (z stays in the first quadrant, z^2-1 and z^2-a^2 in the
# closed upper half plane), so no branch cuts are crossed.  The patch
# boundary consists of symmetry curves only:
#   t = 0            planar geodesic in the vertical mirror plane y = 0
#   t = pi, u > xa   planar geodesic in the vertical mirror plane x = 0
#   t = pi, u < xa   STRAIGHT line, direction (1,1,0), at height -d
#                    (xa = log(a^2-1); z=1 maps to the axis point (0,0,-d))
#   u = umax         planar geodesic in the horizontal mirror plane z = 0
#   u = umin         the end trim: u -> -inf is the planar end at z = a
#                    (asymptotic height exactly -d; the flare radius grows
#                    as ~0.87 e^{-u/4}, so umin sizes the trimmed end disk)
# with d = 0.7288482 (the measured half-spacing; the real-axis elliptic
# integral int_0^1 dx/sqrt((1-x^2)(a^2-x^2)) equals 2d to 7 digits).
#
# Assembly.  16 isometries tile ONE translational period from the patch:
#     {E, sigma_h} x {E, Mx} x {E, My} x {E, R11}
# where sigma_h reflects across the horizontal plane z = 0, Mx / My across
# the vertical planes x = 0 / y = 0, and R11 is the 180-degree rotation
# about the straight line {x = y, z = -d}.  The period translation is
# (0, 0, 4d); `storeys` stacked periods weld vertex-exactly because the
# chunk's top boundary sigma_h Mx^b My^c R11 (arc) and the next chunk's
# bottom boundary T_{4d} Mx^b My^c R11 (arc) evaluate to bitwise-identical
# coordinates.  Boundary rows are snapped exactly onto their symmetry
# elements before tiling, so every seam welds by float equality.  (The
# harvested constants close the period problem to ~1.2e-4 absolute -- the
# residual of the notebook's FindRoot tolerance; the snap absorbs it.)
#
# References:
#   M. J. Callahan, D. Hoffman, W. H. Meeks III, "Embedded minimal
#     surfaces with an infinite number of ends", Invent. Math. 96 (1989)
#     459-505 -- the CHM_k family (this is k = 2);
#   D. Hoffman, W. H. Meeks III, "Minimal surfaces based on the catenoid",
#     Amer. Math. Monthly 97 (1990) -- exposition;
#   M. Weber, https://minimalsurfaces.blog/ (Callahan-Hoffman-Meeks
#     surfaces) and the CHM-(1,1) notebook -- the g / dh data, the strip
#     substitution and the solved constants
#     (research/msblog_harvest/singly_periodic.json).
# --------------------------------------------------------------------------

_CHMP_A = 1.31922381870184635        # branch point a (harvested)
_CHMP_RHO = 1.22370848596689185      # balance constant rho (harvested)
_CHMP_XA = math.log(_CHMP_A * _CHMP_A - 1.0)     # z = 1 at (xa, pi)
_CHMP_X0 = math.log(_CHMP_A * _CHMP_A)           # z = 0 at (x0, pi)
_CHMP_UMAX = 0.5 * (_CHMP_XA + _CHMP_X0)         # the arc edge


def chm_periodic_forms(w):
    """(om1, om2, om3) pulled back to the strip chart (times dz/dw =
    e^w / 2z).  With z^2 - a^2 = e^w the fractional powers of z^2 - a^2
    become exact exponentials, so the planar end u -> -inf is free of
    branch issues; the only strip singularity is the integrable
    (z^2-1)^{-3/4} corner at (xa, pi), the branch point z = 1."""
    a, rho = _CHMP_A, _CHMP_RHO
    ew = np.exp(w)
    z = np.sqrt(a * a + ew)                    # first quadrant (principal)
    zm1 = a * a - 1.0 + ew                     # z^2 - 1, closed UHP
    q1 = np.exp(-0.25 * w) / (2.0 * z ** 1.5 * zm1 ** 0.25)
    q2 = np.exp(1.25 * w) / (2.0 * np.sqrt(z) * zm1 ** 0.75)
    om3 = np.exp(0.5 * w) / (2.0 * z * np.sqrt(zm1))
    om1 = 0.5 * (rho * q1 - q2 / rho)
    om2 = 0.5j * (rho * q1 + q2 / rho)
    return om1, om2, om3


def _chmp_cluster(x0, h0, hmin=1e-10, ratio=0.2):
    """Geometric offsets h0*ratio^k down to hmin (largest first)."""
    offs = []
    h = h0
    while h > hmin:
        offs.append(h)
        h *= ratio
    return np.array(offs)


def _chmp_ugrid(nu, umin):
    """Ascending u grid: flare-uniform toward the end (uniform in the
    end's conformal radius e^{-u/4}), a uniform mid band, geometric
    clusters into the singular corner u = xa from BOTH sides (the
    (z^2-1)^{-3/4} branch point demands geometrically graded cells for
    the compound Gauss-Legendre row integrals to converge), and a
    uniform tail to the arc at umax.  xa is an exact grid node."""
    xa, umax = _CHMP_XA, _CHMP_UMAX
    nE = max(14, int(0.40 * nu))
    nM = max(6, int(0.16 * nu))
    nR = max(6, int(0.18 * nu))
    uf = -4.0 * np.log(np.linspace(math.exp(-0.25 * umin),
                                   math.exp(-0.25 * (xa - 0.8)), nE))
    um = np.linspace(xa - 0.8, xa - 0.16, nM + 1)[1:]
    cl = _chmp_cluster(xa, 0.16 * 0.2)
    left = xa - cl                              # ascending toward xa
    right = xa + cl[::-1]                       # ascending away from xa
    h0r = min(0.16, 0.45 * (umax - xa))
    ur = np.linspace(xa + h0r, umax, nR)
    u = np.concatenate([uf, um, left, [xa], right,
                        np.array([xa + h0r * 0.5]), ur])
    u = np.unique(u)
    return u[np.concatenate([[True], np.diff(u) > 1e-13])]


def _chmp_tgrid(nt):
    """t grid on [0, pi]: Chebyshev-clustered at both edges plus a
    geometric cluster into t = pi (the corner rows)."""
    base = 0.5 * math.pi * (1.0 - np.cos(np.pi * np.linspace(
        0.0, 1.0, max(24, nt))))
    extra = math.pi - _chmp_cluster(0.0, 0.08, ratio=0.25)
    t = np.unique(np.concatenate([base, extra]))
    return t[np.concatenate([[True], np.diff(t) > 1e-13])]


_CHMP_GL = np.polynomial.legendre.leggauss(8)


def chm_periodic_patch(u, t):
    """Integrate the immersion over the strip grid.  Compound 8-point
    Gauss-Legendre per grid cell (nodes are strictly interior, so the
    singular corner node (xa, pi) is never evaluated): down/up the arc
    column u = umax from the base (umax, ~pi/2), then leftward along
    every row.  Returns X real (nu, nt, 3)."""
    xg, wg = _CHMP_GL
    nu, nt = len(u), len(t)
    du = np.diff(u)
    umid = 0.5 * (u[1:] + u[:-1])
    W = (umid[:, None, None] + 0.5 * du[:, None, None] * xg[None, None, :]
         + 1j * t[None, :, None])
    o1, o2, o3 = chm_periodic_forms(W)
    incU = np.stack(
        [np.sum(o1 * wg, axis=-1), np.sum(o2 * wg, axis=-1),
         np.sum(o3 * wg, axis=-1)], axis=-1) * (0.5 * du)[:, None, None]
    dt = np.diff(t)
    tmid = 0.5 * (t[1:] + t[:-1])
    Wc = u[-1] + 1j * (tmid[:, None] + 0.5 * dt[:, None] * xg[None, :])
    c1, c2, c3 = chm_periodic_forms(Wc)
    incC = np.stack(
        [np.sum(c1 * wg, axis=-1), np.sum(c2 * wg, axis=-1),
         np.sum(c3 * wg, axis=-1)], axis=-1) * (0.5j * dt)[:, None]
    Fc = np.zeros((nt, 3), complex)
    Fc[1:] = np.cumsum(incC, axis=0)
    Fc -= Fc[int(np.argmin(np.abs(t - 0.5 * math.pi)))]
    # F(i, j) = Fc(j) - sum_{cells k >= i} incU(k, j)
    S = np.zeros((nu, nt, 3), complex)
    S[:-1] = np.cumsum(incU[::-1], axis=0)[::-1]
    return np.real(Fc[None, :, :] - S)


def chm_periodic_snap(X, u):
    """Normalize and snap the patch boundary exactly onto its symmetry
    elements: y = 0 mirror (t = 0 row), x = 0 mirror (t = pi row past
    xa), z = 0 horizontal mirror (arc column), the straight line
    {x = y, z = -d} (t = pi row before xa) and the axis point (0,0,-d)
    at the z = 1 corner.  Returns (X, iL, d)."""
    iL = int(np.searchsorted(u, _CHMP_XA))
    X = X.copy()
    X[..., 1] -= np.median(X[:, 0, 1])            # y = 0 mirror
    X[..., 0] -= np.median(X[iL:, -1, 0])         # x = 0 mirror
    X[..., 2] -= np.median(X[-1, :, 2])           # z = 0 mirror (arc)
    X[:, 0, 1] = 0.0
    X[iL:, -1, 0] = 0.0
    X[-1, :, 2] = 0.0
    zL = float(np.median(X[:iL + 1, -1, 2]))
    d = -zL
    X[:iL + 1, -1, 2] = zL                        # the straight line...
    m = 0.5 * (X[:iL + 1, -1, 0] + X[:iL + 1, -1, 1])
    X[:iL + 1, -1, 0] = m                         # ...x = y exactly
    X[:iL + 1, -1, 1] = m
    X[iL, -1, :] = (0.0, 0.0, zL)                 # z = 1 -> axis point
    return X, iL, d


def _chmp_frames(d, periods):
    """The 16 * periods assembly isometries (M, tvec, parity), indexed
    (((sh*2 + bx)*2 + cy)*2 + e)*periods + k:  v -> sigma_h^sh Mx^bx
    My^cy R11^e v + (0, 0, 4dk).  parity drives face winding."""
    R11 = (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0]]), np.array([0.0, 0.0, -2.0 * d]))
    out = []
    for sh in (0, 1):
        for bx in (0, 1):
            for cy in (0, 1):
                D = np.diag([-1.0 if bx else 1.0, -1.0 if cy else 1.0,
                             -1.0 if sh else 1.0])
                for e in (0, 1):
                    if e:
                        M = D @ R11[0]
                        tv = D @ R11[1]
                    else:
                        M = D.copy()
                        tv = np.zeros(3)
                    # every generator is a Schwarz continuation (z -> zbar
                    # on the parameter domain, anti-conformal), so each --
                    # including the PROPER rotation R11 -- reverses the
                    # surface orientation: winding flips with the total
                    # generator count, not with det M.
                    par = (-1.0) ** (sh + bx + cy + e)
                    for k in range(periods):
                        out.append((M, tv + np.array(
                            [0.0, 0.0, 4.0 * d * k]), par))
    return out


def chm_periodic_seams(iL, nu, nt, periods):
    """The COMBINATORIAL weld: every seam pairs frame (sh, bx, cy, e, k)
    with one partner frame along one boundary index set, with identical
    within-patch indices (the gluing isometry fixes the seam pointwise).
    Derived from the generator algebra (R11 My = Mx R11, R11 Mx = My R11,
    R11 sigma_h = T_{-4d} sigma_h R11):
      t = 0 row   (y = 0 mirror):   toggle cy if e = 0 else bx
      x row       (x = 0 mirror):   toggle bx if e = 0 else cy
      L1 row      (2-fold line):    toggle e
      arc column  (z = 0 mirror):   toggle sh; same k if e = 0, else the
                  NEIGHBOR chunk k-1 (sh = 0) / k+1 (sh = 1) -- the
                  period interface (open at the stack's outer cuts).
    Returns [(patch-index array, frame A, frame B), ...]."""
    S = periods

    def fidx(sh, bx, cy, e, k):
        return (((sh * 2 + bx) * 2 + cy) * 2 + e) * S + k

    t0 = np.arange(nu) * nt
    xr = np.arange(iL, nu) * nt + (nt - 1)
    l1 = np.arange(0, iL + 1) * nt + (nt - 1)
    ac = (nu - 1) * nt + np.arange(nt)
    out = []
    for sh in (0, 1):
        for bx in (0, 1):
            for cy in (0, 1):
                for e in (0, 1):
                    for k in range(S):
                        me = fidx(sh, bx, cy, e, k)
                        out.append((t0, me, fidx(sh, bx, 1 - cy, e, k)
                                    if e == 0 else
                                    fidx(sh, 1 - bx, cy, e, k)))
                        out.append((xr, me, fidx(sh, 1 - bx, cy, e, k)
                                    if e == 0 else
                                    fidx(sh, bx, 1 - cy, e, k)))
                        out.append((l1, me, fidx(sh, bx, cy, 1 - e, k)))
                        k2 = k if e == 0 else (k - 1 if sh == 0 else k + 1)
                        if 0 <= k2 < S:
                            out.append((ac, me,
                                        fidx(1 - sh, bx, cy, e, k2)))
    return out


def chm_periodic_assemble(u, t, X, iL, d, periods):
    """Tile the snapped patch under the 16 * periods isometries and weld
    every seam COMBINATORIALLY (chm_periodic_seams knows each edge's
    partner frame exactly, so no floating-point coincidence matching is
    involved).  Returns (V, faces, uv) of the largest component -- one
    translational period per storey, the 2*periods planar-end rims and
    the 2 outer horizontal cuts left as clean open boundaries."""
    nu, nt = len(u), len(t)
    V0 = X.reshape(-1, 3)
    UV0 = np.stack(np.meshgrid(
        (u - u[0]) / (u[-1] - u[0]), t / math.pi, indexing='ij'),
        axis=-1).reshape(-1, 2)
    q0 = []
    for i in range(nu - 1):
        for j in range(nt - 1):
            q0.append((i * nt + j, i * nt + j + 1,
                       (i + 1) * nt + j + 1, (i + 1) * nt + j))
    frames = _chmp_frames(d, periods)
    nV = len(V0)
    Vp, Fp = [], []
    for fr, (M, tv, par) in enumerate(frames):
        Vp.append(V0 @ M.T + tv)
        off = fr * nV
        if par < 0:
            Fp.extend(tuple(off + i for i in f[::-1]) for f in q0)
        else:
            Fp.extend(tuple(off + i for i in f) for f in q0)
    V = np.concatenate(Vp, axis=0)
    UVall = np.tile(UV0, (len(frames), 1))
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for idx, fA, fB in chm_periodic_seams(iL, nu, nt, periods):
        if fA >= fB:                   # each seam appears from both sides
            continue
        for i in idx:
            ra, rb = find(fA * nV + int(i)), find(fB * nV + int(i))
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(a) for a in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    UVw = np.zeros((len(uniq), 2))
    cnt = np.zeros(len(uniq))
    np.add.at(Vw, inv, V)
    np.add.at(UVw, inv, UVall)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    UVw /= cnt[:, None]
    F = []
    for f in Fp:
        g = [int(inv[i]) for i in f]
        h = [g[0]]
        for s in range(1, len(g)):
            if g[s] != h[-1]:
                h.append(g[s])
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            F.append(tuple(h))
    Vu, F = _largest_component(np.hstack([Vw, UVw]), F)
    return Vu[:, :3], F, Vu[:, 3:]


def chm_periodic_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                      storeys=1):
    """MESH_PARAM builder: finished (V, faces, uv) of `storeys` stacked
    translational periods, fit to the 2 m cube.  radius (via p_from's
    umin) sizes the trimmed planar-end disks; order and theta unused."""
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    umin = float(p.get('umin', -5.0))
    S = int(np.clip(storeys, 1, 6))
    ug = _chmp_ugrid(int(np.clip(nu, 24, 220)), umin)
    tg = _chmp_tgrid(int(np.clip(int(0.85 * nv), 20, 170)))
    X = chm_periodic_patch(ug, tg)
    X, iL, d = chm_periodic_snap(X, ug)
    V, F, uv = chm_periodic_assemble(ug, tg, X, iL, d, S)
    V = _center_fit(V, scale, V)
    return V, F, uv


# ==========================================================================
# Catenoid-Enneper (genus 2/3/4) and Costa-Wohlgemuth / Wohlgemuth
# higher-genus surfaces  (cwce_* block)
# ==========================================================================
# Two families of complete finite-total-curvature minimal surfaces on
# hyperelliptic / cyclic-cover curves, meshed watertight from ONE
# fundamental piece orbited under the full point group (the same
# snap-onto-symmetry-element-then-weld scheme as the cg_higher_* and
# symmcg_* assemblers above).
#
# CATENOID-ENNEPER, genus g = 2, 3, 4 (one catenoid + one Enneper end):
#   on the hyperelliptic curve y^2 = z (z - v_1) ... (z - v_2g) with
#     phi1 = G dh = rho z^{-1/2} prod (z - v_i)^{+-1/2},
#     phi2 = dh/G = (1/rho) z^{-3/2} prod (z - v_i)^{-+1/2} (z+1)^2,
#     dh   = (1 + z)/z dz
#   (x = Re Int (phi2 - phi1)/2, y = Re Int i(phi2 + phi1)/2,
#   x3 = Re Int dh).  The catenoid end is the branch point z = 0 (dh has
#   residue 2 on the double cover), the Enneper end is z = infinity; dh
#   vanishes at the regular point z = -1 (the neck).  The 2g + 1 real
#   constants (rho, v_2..v_2g; v_1 normalized to 1) solve the 2g-period
#   problem of the notebook: Re Int_{v_s}^{v_s+1} omega = 0 alternating
#   omega_2 / omega_1 across consecutive branch values (the planar
#   symmetry arcs on either side must lie in the SAME mirror plane) plus
#   one condition tying the negative real axis into the x-mirror.  All
#   constants in _CWCE_CE were verified/re-solved here (residuals
#   ~5e-9, quadrature-stable; g4's were solved from the notebook's
#   Newton seeds -- the notebook stores none).  Symmetry: two orthogonal vertical mirror planes intersecting
#   in a vertical axis that threads every branch-point image; group
#   {1, Mx, My, Rz(pi)} of order 4 (Rz(pi) = the hyperelliptic sheet
#   swap).  Fundamental piece: the upper-half-plane annulus
#   rmin <= |z| <= rmax; the positive real segments map to planar
#   geodesics alternating between the two mirror planes, the negative
#   real axis into the x-mirror, and the two rims are the catenoid /
#   Enneper end trims (chi = 2 - 2g - 2 exactly, gated below).
#
# COSTA-WOHLGEMUTH, genus 2(k-1), and WOHLGEMUTH SECOND SURFACE,
# genus 3(k-1)  (4 ends: 2 catenoidal + 2 planar):
#   on the k-fold cyclic cover of the sphere branched over
#   {+-1, +-a, +-b} (resp. {+-1, +-a, +-b, +-c}), with
#     phi1 = (z-1)^{-1+1/k} (z+1)^{1-1/k} (z-a)^{1+1/k} (z+a)^{-1-1/k}
#            (z-b)^{-1-1/k} (z+b)^{-1+1/k}   [second surface: extra
#            (z-c)^{1-1/k} (z+c)^{-1+1/k} and the a/b roles reshuffled
#            as in _cwce_cw_data],
#     phi2 = the exponent-negated mirror partner (phi1 phi2 = dh^2),
#     dh   = dz / ((z-b)(z+b)).
#   Catenoidal ends at z = +-b (dh poles, totally ramified), planar ends
#   at z = +-a, finite branch images (handles) at z = +-1 (and +-c).
#   k = 2 gives Wohlgemuth's genus-2 (Costa-Wohlgemuth) and genus-3
#   surfaces -- the first complete embedded minimal surfaces with four
#   ends.  Period problem: (a, b) [resp. (a, b, c)] solve the conditions
#   that the (a,b) symmetry arc's vertical plane and the (0,a) arc's
#   plane pass through the axis (plus, for the second surface, the z=c
#   branch image landing on the axis); the constants in _CWCE_CW /
#   _CWCE_W2 are the notebook literals re-verified here to ~1e-13 for
#   every k.  Symmetry: prismatic D_kh of order 4k -- k-fold vertical
#   axis through the +-1 (and +-c) branch images, k vertical mirror
#   planes at angles j pi/k, one HORIZONTAL mirror plane at the height
#   h0 of the z = 0 image (the image of the imaginary axis is a planar
#   geodesic in that plane).  Fundamental piece: the conformal strip
#   chart z = sqrt(moeb(e^w)), w in [xa, xb] x [0, pi] with
#   moeb(u) = (rho_m b^2 u + a^2)/(rho_m u + 1), rho_m = a^2/b^2 (the
#   notebook's chart of the quarter plane): the planar end opens at the
#   left strip end, the catenoidal end at the right, and the marked
#   points z = 0, 1, c, infinity sit at exact grid columns on the top
#   edge where the integrand's algebraic singularities get graded
#   quadrature blocks with exact node offsets.  4k copies weld to
#   chi = 2 - 2 genus - 4 exactly (four open end rims), gated below.
#
# The Wohlgemuth second-surface family for k > 2 is NOT shipped: the
# blog notebook's k = 3 / k = 4 constants do not satisfy the closure
# conditions (residuals ~3e-2 under the same quadrature that confirms
# every other member at ~1e-13), and re-solving from them stalls --
# see BACKLOG.md.
#
# References:
#   C. J. Costa, "Example of a complete minimal immersion in R^3 of
#     genus one and three embedded ends", Bol. Soc. Bras. Mat. 15
#     (1984) 47-54 -- the ancestor construction;
#   M. Wohlgemuth, "Higher genus minimal surfaces by growing handles
#     out of a catenoid", Manuscripta Math. 70 (1991) 397-428, Bonn
#     dissertation (1993), and "Minimal surfaces of higher genus with
#     finite total curvature", Arch. Rational Mech. Anal. 137 (1997)
#     1-25 -- the genus-2 and genus-3 four-ended embedded surfaces;
#   C. C. Chen, F. Gackstatter, "Elliptische und hyperelliptische
#     Funktionen und vollstaendige Minimalflaechen vom Enneperschen
#     Typ", Math. Ann. 259 (1982) -- the Enneper-end heritage of the
#     catenoid-Enneper family;
#   H. Karcher, "Construction of minimal surfaces", Univ. of Tokyo
#     lecture notes (1989) -- the symmetry / period method;
#   M. Weber, https://minimalsurfaces.blog/ -- repository pages
#     "Costa-Wohlgemuth surfaces", "Wohlgemuth's second surface"
#     (higher-symmetry notebook by Ramazan Yol) and "Catenoid-Enneper
#     surfaces of genus g", whose Mathematica notebooks supplied the
#     Weierstrass data, period conditions and constants followed here.

# Solved period constants.  CE: genus -> (rho, (v1..v2g)) with v1 = 1;
# g2/g3 re-verified notebook literals, g4 solved here from the
# notebook's Newton seeds (residuals ~1e-13, re-checked by the
# self-test through cwce_ce_residual).
_CWCE_CE = {
    2: (2.441868735003271,
        (1.0, 1.5899997953673903, 4.2856159413955295,
         5.235200094406395)),
    3: (2.709609741854014,
        (1.0, 1.619390029065051, 3.9554260271562294, 4.993548107702124,
         7.829353305141648, 8.864263970017321)),
    4: (2.9455445424953384,
        (1.0, 1.6412114092203107, 3.8014391806809074, 4.89812108352522,
         7.320461951519022, 8.47252262184776, 11.478960661548184,
         12.542173771958389)),
}
# Costa-Wohlgemuth: k -> (a, b), genus 2(k-1) -- notebook literals
# re-verified/refined to ~1e-13 for every k
_CWCE_CW = {
    2: (0.5936584192549862, 0.8240789231207696),
    3: (0.6342850763231503, 0.8412721089595467),
    4: (0.674940760134315, 0.8583521760688522),
    5: (0.7084980441058044, 0.8722674706663541),
    7: (0.7583644125949724, 0.8928687061979),
}
# Wohlgemuth second surface: k -> (a, b, c), genus 3(k-1); only the
# k = 2 (genus 3) period problem closes -- see the header note
_CWCE_W2 = {
    2: (0.6518984510521686, 0.9141432157734375, 3.0832637626461463),
}

_CWCE_GL_CACHE = {}


def _cwce_gl(n):
    if n not in _CWCE_GL_CACHE:
        _CWCE_GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _CWCE_GL_CACHE[n]


def cwce_path_int(fn, waypoints, e_end=(0.0, 0.0), n=400):
    """Gauss-Legendre integral of fn(z) dz along the polyline
    `waypoints`; e_end = (e_first, e_last) are algebraic singularity
    exponents |z - z_end|^-e at the path ends (graded u^p
    substitution there makes the transformed integrand smooth)."""
    x, w = _cwce_gl(n)
    u = 0.5 * (x + 1.0)
    wu = 0.5 * w
    total = 0.0 + 0.0j
    m = len(waypoints) - 1
    for i in range(m):
        A, B = complex(waypoints[i]), complex(waypoints[i + 1])
        eL = e_end[0] if i == 0 else 0.0
        eR = e_end[1] if i == m - 1 else 0.0
        if eL > 0.0 and eR > 0.0:
            Mid = 0.5 * (A + B)
            total += cwce_path_int(fn, [A, Mid], (eL, 0.0), n)
            total += cwce_path_int(fn, [Mid, B], (0.0, eR), n)
            continue
        if eR > 0.0:
            A, B = B, A
            eL, sgn = eR, -1.0
        else:
            sgn = 1.0
        if eL > 0.0:
            p = 2.0 / max(1e-9, 1.0 - min(eL, 0.95)) + 1.0
            t = u ** p
            J = p * u ** (p - 1.0)
        else:
            t, J = u, np.ones_like(u)
        z = A + t * (B - A)
        f = fn(z) * (B - A) * J
        f = np.where(np.isfinite(f), f, 0.0)
        total += sgn * np.sum(f * wu)
    return total


def _cwce_make_phi(factors, pref=1.0):
    """z -> pref * prod (z - v)^p on Im z >= 0 (principal branches --
    continuous there, matching continuation from the upper halfplane)."""
    def phi(z):
        z = np.asarray(z, dtype=complex)
        out = np.full(z.shape, pref, dtype=complex)
        for (v, p) in factors:
            out = out * np.exp(p * np.log(z - v))
        return out
    return phi


def cwce_ce_phis(genus):
    """(fac1, fac2, rho, roots) for Catenoid-Enneper genus g."""
    rho, roots = _CWCE_CE[genus]
    f1 = [(0.0, -0.5)]
    f2 = [(0.0, -1.5)]
    for i, v in enumerate(roots):
        s = 0.5 if i % 2 == 0 else -0.5
        f1.append((v, s))
        f2.append((v, -s))
    return f1, f2, rho, list(roots)


def cwce_ce_residual(genus, n=400):
    """The notebook's 2g period conditions at the stored constants
    (all must vanish -- the self-test's honesty gate on _CWCE_CE)."""
    f1, f2, rho, v = cwce_ce_phis(genus)
    phi1 = _cwce_make_phi(f1, rho)
    phi2r = _cwce_make_phi(f2, 1.0 / rho)

    def om1(z):
        return 0.5 * (phi2r(z) * (1.0 + z) ** 2 - phi1(z))

    def om2(z):
        return 0.5j * (phi2r(z) * (1.0 + z) ** 2 + phi1(z))

    res = []
    for s in range(2 * genus - 1):
        A, B = v[s], v[s + 1]
        om = om2 if s % 2 == 0 else om1
        res.append(np.real(cwce_path_int(om, [A, 1j, B], (0.5, 0.5), n)))
    res.append(np.real(cwce_path_int(
        om1, [-0.5, 1j, 0.5 * (v[0] + v[1])], (0.0, 0.0), n)))
    return np.array(res)


def cwce_cw_residual(fam, k, n=400):
    """Period-condition residuals at the stored (a, b[, c]) constants:
    the (a,b) arc's mirror plane and the (0,a) arc's plane must pass
    through the vertical axis (and z = c must land on it)."""
    a, b, c, exps, marks = _cwce_cw_data(fam, k)
    f1 = [(p, q1) for p, (q1, q2) in exps.items()]
    f2 = [(p, q2) for p, (q1, q2) in exps.items()]
    phi1 = _cwce_make_phi(f1)
    phi2 = _cwce_make_phi(f2)

    def om1(z):
        return 0.5 * (phi2(z) - phi1(z))

    def om2(z):
        return 0.5j * (phi1(z) + phi2(z))

    e1 = 1.0 - 1.0 / k
    up = 0.35j
    m = 0.5 * (a + b)
    if fam == 'cw':
        r1 = np.real(cwce_path_int(om2, [1.0, 1.0 + up, m + up, m],
                                   (e1, 0.0), n))
        I1 = np.real(cwce_path_int(om1, [1.0, 1.0 + up, up, 0.0],
                                   (e1, 0.0), n))
        I2 = np.real(cwce_path_int(om2, [1.0, 1.0 + up, up, 0.0],
                                   (e1, 0.0), n))
        al = math.pi / k
        return np.array([r1, -I1 * math.sin(al) + I2 * math.cos(al)])
    r1 = np.real(cwce_path_int(om2, [0.0, 0.3 + up, 1.0 + up, 1.0],
                               (0.0, e1), n))
    r2 = np.real(cwce_path_int(om2, [0.0, 0.3 + up, c + up, c],
                               (0.0, e1), n))
    I1 = np.real(cwce_path_int(om1, [1.0, 1.0 + up, m + up, m],
                               (e1, 0.0), n))
    I2 = np.real(cwce_path_int(om2, [1.0, 1.0 + up, m + up, m],
                               (e1, 0.0), n))
    bet = 0.5 * math.pi - math.pi / k
    return np.array([r1, r2, I1 * math.cos(bet) + I2 * math.sin(bet)])


def cwce_grids(bounds, sing, nu, msub=6, msing=80, pmesh=1.7):
    """Coarse mesh nodes + fine trapezoid nodes/weights on
    [bounds[0], bounds[-1]]; interior bounds are exact coarse nodes,
    entries of `sing` (node -> exponent e of an |x - node|^-e
    integrable singularity) get graded u^p quadrature blocks with
    EXACT node offsets (sid/soff).  Returns (x_c, x_f, cidx, sid,
    soff, w_lo, w_hi); a cumulative line integral is
    cumsum(w_lo f[:-1] + w_hi f[1:]) sampled at cidx."""
    nseg = len(bounds) - 1
    lens = np.diff(np.array(bounds, dtype=float))
    wseg = np.sqrt(lens)
    counts = np.maximum(6, np.round(nu * wseg / wseg.sum()).astype(int))
    keys = sorted(sing)

    def sing_id(x):
        for i, s in enumerate(keys):
            if abs(x - s) < 1e-12:
                return i
        return -1

    x_c, x_f = [bounds[0]], [bounds[0]]
    sid, soff = [-1], [0.0]
    cidx = [0]
    w_lo, w_hi = [], []
    for s in range(nseg):
        A, B = bounds[s], bounds[s + 1]
        n = int(counts[s])
        t = np.linspace(0.0, 1.0, n + 1)[1:]
        tt = np.where(t < 0.5, 0.5 * (2 * t) ** pmesh,
                      1.0 - 0.5 * (2 * (1 - t)) ** pmesh)
        nodes = A + (B - A) * tt
        nodes[-1] = B
        prev = A
        for ci in range(n):
            hi = nodes[ci]
            singL = ci == 0 and A in sing
            singR = ci == n - 1 and B in sing
            if singL:
                p = 2.0 / max(1e-9, 1.0 - min(sing[A], 0.95)) + 1.0
                D = hi - A
                u = np.linspace(0.0, 1.0, msing + 1)
                off = D * u ** p
                J = D * p * u ** (p - 1.0)
                du = 1.0 / msing
                xf = (A + off)[1:]
                xf[-1] = hi
                sid_i = np.full(len(xf), sing_id(A))
                soff_i = off[1:]
                w_lo.extend((0.5 * du * J[:-1]).tolist())
                w_hi.extend((0.5 * du * J[1:]).tolist())
            elif singR:
                p = 2.0 / max(1e-9, 1.0 - min(sing[B], 0.95)) + 1.0
                D = B - prev
                v = np.linspace(0.0, 1.0, msing + 1)
                offB = D * (1.0 - v) ** p
                J = D * p * (1.0 - v) ** (p - 1.0)
                dv = 1.0 / msing
                xf = (B - offB)[1:]
                xf[-1] = B
                sid_i = np.full(len(xf), sing_id(B))
                soff_i = -offB[1:]
                w_lo.extend((0.5 * dv * J[:-1]).tolist())
                w_hi.extend((0.5 * dv * J[1:]).tolist())
            else:
                tloc = np.linspace(0.0, 1.0, msub + 1)
                left_half = (prev - A) <= (B - hi)
                if left_half and A in sing and prev > A:
                    dlo, dhi = prev - A, hi - A
                    d = dlo * (dhi / dlo) ** tloc
                    xf_full = A + d
                    J = d * math.log(dhi / dlo)
                elif not left_half and B in sing and hi < B:
                    dtop, dbot = B - prev, B - hi
                    d = dtop * (dbot / dtop) ** tloc
                    xf_full = B - d
                    J = d * abs(math.log(dbot / dtop))
                else:
                    xf_full = prev + (hi - prev) * tloc
                    J = None
                xf = xf_full[1:].copy()
                xf[-1] = hi
                sid_i = np.full(len(xf), -1)
                soff_i = np.zeros(len(xf))
                if J is None:
                    h = 0.5 * (hi - prev) / msub
                    w_lo.extend([h] * msub)
                    w_hi.extend([h] * msub)
                else:
                    dt = 1.0 / msub
                    w_lo.extend((0.5 * dt * J[:-1]).tolist())
                    w_hi.extend((0.5 * dt * J[1:]).tolist())
            x_f.extend(xf.tolist())
            sid.extend(sid_i.tolist())
            soff.extend(soff_i.tolist())
            cidx.append(len(x_f) - 1)
            x_c.append(hi)
            prev = hi
    return (np.array(x_c), np.array(x_f), np.array(cidx),
            np.array(sid, dtype=int), np.array(soff),
            np.array(w_lo), np.array(w_hi))


def cwce_weld_frames(V0, faces0, UV0, bmask, mats, revs, tol=1e-6):
    """Orbit the snapped piece under the affine frames [(M, t)], weld
    boundary vertices by coincidence (frames with revs[i] True reverse
    the face winding), drop degenerate faces, and keep the largest
    face-connected component.  Returns (V, faces, uv)."""
    nV = len(V0)
    Vp, Fp = [], []
    for fr, (M, t) in enumerate(mats):
        Vp.append(V0 @ np.asarray(M).T + t)
        for f in faces0:
            ff = tuple(int(x) + fr * nV for x in f)
            Fp.append(ff[::-1] if revs[fr] else ff)
    V = np.concatenate(Vp)
    UVall = np.tile(UV0, (len(mats), 1))
    N = len(V)
    parent = np.arange(N)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bidx = np.nonzero(np.tile(bmask, len(mats)))[0]
    key = np.floor(V[bidx] / tol + 0.5).astype(np.int64)
    H = {}
    for t2, i in enumerate(bidx):
        H.setdefault((int(key[t2, 0]), int(key[t2, 1]),
                      int(key[t2, 2])), []).append(int(i))
    for t2, i in enumerate(bidx):
        k0 = key[t2]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in H.get((int(k0[0]) + dx, int(k0[1]) + dy,
                                    int(k0[2]) + dz), ()):
                        if j > i and np.linalg.norm(V[j] - V[i]) < tol:
                            ra, rb = find(int(i)), find(j)
                            if ra != rb:
                                parent[ra] = rb
    rts = np.array([find(a) for a in range(N)])
    uniq, inv = np.unique(rts, return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    UVw = np.zeros((len(uniq), 2))
    cnt = np.zeros(len(uniq))
    np.add.at(Vw, inv, V)
    np.add.at(UVw, inv, UVall)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    UVw /= cnt[:, None]
    F = []
    for f in Fp:
        gg = [int(inv[i]) for i in f]
        h = [gg[0]]
        for t2 in range(1, len(gg)):
            if gg[t2] != h[-1]:
                h.append(gg[t2])
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            F.append(tuple(h))
    parent2 = np.arange(len(Vw))

    def find2(a):
        while parent2[a] != a:
            parent2[a] = parent2[parent2[a]]
            a = parent2[a]
        return a

    for f in F:
        for i in range(1, len(f)):
            ra, rb = find2(f[0]), find2(f[i])
            if ra != rb:
                parent2[ra] = rb
    from collections import Counter
    sizes = Counter(find2(f[0]) for f in F)
    root = sizes.most_common(1)[0][0]
    F = [f for f in F if find2(f[0]) == root]
    used = sorted(set(i for f in F for i in f))
    rmv = {v: i for i, v in enumerate(used)}
    return Vw[used], [tuple(rmv[i] for i in f) for f in F], UVw[used]


# ---- Catenoid-Enneper piece / assembly -----------------------------------

def cwce_ce_omega(genus, r, theta, sid, soff, roots):
    """(n, 3) complex (om1, om2, om3) at z = r e^{i theta} on one
    radial ray; sid/soff give exact offsets to the branch values on
    the theta = 0 ray (its algebraic singularities are integrated in
    graded variables where the integrand is smooth)."""
    f1, f2, rho, _ = cwce_ce_phis(genus)
    n = len(r)
    z = r * np.exp(1j * theta)
    out = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for fac, pref in ((f1, rho), (f2, 1.0 / rho)):
            lmag = np.full(n, math.log(pref))
            ang = np.zeros(n)
            for (v, p) in fac:
                if theta == 0.0:
                    t = np.abs(r - v)
                    if v in roots:
                        m = sid == roots.index(v)
                        t[m] = np.abs(soff[m])
                        left = np.array(r < v)
                        left[m] = soff[m] < 0
                    else:
                        left = r < v
                    lmag = lmag + p * np.log(t)
                    ang = ang + p * np.where(left, math.pi, 0.0)
                else:
                    d = z - v
                    lmag = lmag + p * np.log(np.abs(d))
                    ang = ang + p * np.angle(d)
            out.append(np.exp(lmag + 1j * ang))
        phi1, phi2 = out
        phi2 = phi2 * (1.0 + z) ** 2
        phi3 = (1.0 + z) / z
        om = np.stack([0.5 * (phi2 - phi1), 0.5j * (phi2 + phi1),
                       phi3], axis=-1)
    return np.where(np.isfinite(om), om, 0.0)


def cwce_ce_piece(genus, nu, nv, rmin, rmax):
    """Mesh the upper-half-plane annulus fundamental piece: stitch arc
    at a mid radius plus radial-ray integration in both directions.
    Returns (V, faces, boundary classification, per-vertex uv)."""
    f1, f2, rho, roots = cwce_ce_phis(genus)
    bounds = [rmin] + roots + [rmax]
    sing = {v: 0.5 for v in roots}
    r_c, r_f, cidx, sid, soff, w_lo, w_hi = cwce_grids(bounds, sing, nu)
    th = np.linspace(0.0, math.pi, nv)
    rB = math.sqrt(roots[0] * roots[-1])
    ib = int(np.argmin(np.abs(r_c - rB)))
    marc = 12
    thf = np.linspace(0.0, math.pi, (nv - 1) * marc + 1)
    zf = float(r_c[ib]) * np.exp(1j * thf)
    lm1 = np.full(len(thf), math.log(rho))
    an1 = np.zeros(len(thf))
    lm2 = np.full(len(thf), -math.log(rho))
    an2 = np.zeros(len(thf))
    for (v, p) in f1:
        d = zf - v
        lm1 += p * np.log(np.abs(d))
        an1 += p * np.angle(d)
    for (v, p) in f2:
        d = zf - v
        lm2 += p * np.log(np.abs(d))
        an2 += p * np.angle(d)
    phi1 = np.exp(lm1 + 1j * an1)
    phi2 = np.exp(lm2 + 1j * an2) * (1.0 + zf) ** 2
    phi3 = (1.0 + zf) / zf
    oma = np.stack([0.5 * (phi2 - phi1), 0.5j * (phi2 + phi1), phi3],
                   axis=-1)
    dzf = np.diff(zf)[:, None]
    Aarc = np.concatenate(
        [np.zeros((1, 3), complex),
         np.cumsum(0.5 * (oma[1:] + oma[:-1]) * dzf, axis=0)],
        axis=0)[::marc]
    ncr = len(r_c)
    X = np.zeros((ncr, nv, 3))
    for j, t in enumerate(th):
        om = cwce_ce_omega(genus, r_f, float(t), sid, soff, roots)
        contrib = (w_lo[:, None] * om[:-1] + w_hi[:, None] * om[1:]) \
            * np.exp(1j * t)
        F = np.concatenate([np.zeros((1, 3), complex),
                            np.cumsum(contrib, axis=0)], axis=0)
        F = F - F[cidx[ib]]
        X[:, j, :] = np.real(Aarc[j][None, :] + F[cidx])
    V = X.reshape(-1, 3)

    def vid(i, j):
        return i * nv + j

    UV = np.zeros((len(V), 2))
    UV[:, 0] = np.tile(th / math.pi, ncr)
    UV[:, 1] = np.repeat(np.linspace(0.0, 1.0, ncr), nv)
    faces = []
    for i in range(ncr - 1):
        for j in range(nv - 1):
            faces.append((vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)))
    root_ci = [int(np.argmin(np.abs(r_c - v))) for v in roots]
    b = {'seg': {}, 'branch': {}}
    lastci = 0
    for s in range(len(roots) + 1):
        hi_ci = root_ci[s] if s < len(roots) else ncr - 1
        i0 = lastci if s == 0 else lastci + 1
        stop = hi_ci + (1 if s == len(roots) else 0)
        b['seg'][s] = np.array([vid(i, 0) for i in range(i0, stop)],
                               dtype=np.int64)
        if s < len(roots):
            b['branch'][s] = vid(hi_ci, 0)
        lastci = hi_ci
    b['neg'] = np.array([vid(i, nv - 1) for i in range(ncr)],
                        dtype=np.int64)
    return V, faces, b, UV


def cwce_ce_snap(V, b):
    """Snap the boundary onto the exact symmetry elements: even real
    segments into the y = cy mirror, odd segments and the negative
    axis into x = cx, branch images onto the axis (cx, cy).  Returns
    (V, (cx, cy), pre-snap residuals -- the numeric period closure)."""
    ys, xs = [], []
    for s, idx in b['seg'].items():
        if len(idx):
            (ys if s % 2 == 0 else xs).append(
                V[idx, 1] if s % 2 == 0 else V[idx, 0])
    xs.append(V[b['neg'], 0])
    cy = float(np.median(np.concatenate(ys)))
    cx = float(np.median(np.concatenate(xs)))
    res = {}
    for s, idx in b['seg'].items():
        if len(idx) == 0:
            continue
        ax = 1 if s % 2 == 0 else 0
        cv = cy if s % 2 == 0 else cx
        res[f'seg{s}'] = float(np.max(np.abs(V[idx, ax] - cv)))
        V[idx, ax] = cv
    res['neg'] = float(np.max(np.abs(V[b['neg'], 0] - cx)))
    V[b['neg'], 0] = cx
    for s, vi in b['branch'].items():
        res[f'branch{s}'] = float(np.hypot(V[vi, 0] - cx,
                                           V[vi, 1] - cy))
        V[vi, 0], V[vi, 1] = cx, cy
    return V, (cx, cy), res


def cwce_ce_assemble(genus, nu, nv, rmin=None, rmax=None):
    """Watertight order-4 assembly of the Catenoid-Enneper surface:
    4 snapped copies of the half-plane piece welded along the mirror
    curves; the two rims stay open (catenoid / Enneper end trims).
    Returns (V, faces, uv, diag)."""
    rho, roots = _CWCE_CE[genus]
    if rmin is None:
        rmin = roots[0] / {2: 10.0, 3: 40.0, 4: 60.0}[genus]
    if rmax is None:
        rmax = {2: 1.5, 3: 1.12, 4: 1.08}[genus] * roots[-1]
    V0, faces0, b, UV0 = cwce_ce_piece(genus, nu, nv, rmin, rmax)
    V0, (cx, cy), res = cwce_ce_snap(V0.copy(), b)
    bmask = np.zeros(len(V0), dtype=bool)
    for idx in b['seg'].values():
        bmask[idx] = True
    bmask[list(b['branch'].values())] = True
    bmask[b['neg']] = True
    I3 = np.eye(3)
    My = np.diag([1.0, -1.0, 1.0])
    Mx = np.diag([-1.0, 1.0, 1.0])
    R2 = np.diag([-1.0, -1.0, 1.0])
    mats = [(I3, np.zeros(3)),
            (R2, np.array([2 * cx, 2 * cy, 0.0])),
            (My, np.array([0.0, 2 * cy, 0.0])),
            (Mx, np.array([2 * cx, 0.0, 0.0]))]
    revs = [False, False, True, True]
    V, F, UV = cwce_weld_frames(V0, faces0, UV0, bmask, mats, revs)
    return V, F, UV, {'res': res}


def cwce_ce_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """MESH_PARAM builder: finished (V, quads, uv) fit to the 2 m cube.
    The radius slider scales both end trims (how far the catenoid
    funnel and the Enneper flare reach)."""
    p = spec['p_from'](order, radius)
    genus = p['genus']
    rho, roots = _CWCE_CE[genus]
    t = float(np.clip(radius / 1.2, 0.55, 1.8))
    rmin = roots[0] / ({2: 10.0, 3: 40.0, 4: 60.0}[genus] ** t)
    rmax = roots[-1] * (1.0 + ({2: 0.5, 3: 0.12, 4: 0.08}[genus]) * t)
    pnu = int(np.clip(nu * 1.6, 90, 260))
    pnv = int(np.clip(nv * 0.55, 21, 49))
    V, quads, uv, _diag = cwce_ce_assemble(genus, pnu, pnv, rmin, rmax)
    V = _smooth_boundary(V, quads, iters=6)
    V = _center_fit(V, scale, V)
    return V, quads, uv


# ---- Costa-Wohlgemuth / Wohlgemuth piece / assembly ----------------------

def _cwce_expm1c(w):
    """exp(w) - 1 for complex w, accurate near w = 0."""
    w = np.asarray(w, dtype=complex)
    small = np.abs(w) < 1e-5
    out = np.exp(w) - 1.0
    ws = w[small]
    out[small] = ws * (1.0 + ws * (0.5 + ws / 6.0))
    return out


def _cwce_cw_data(fam, k):
    """(a, b, c, exps, marks): exps maps branch value p -> exponents
    (in phi1, in phi2) of the factor (z - p); marks are the strip
    x-positions of the top-edge marked points z = 0, infinity, 1, c."""
    if fam == 'cw':
        a, b = _CWCE_CW[k]
        c = None
    else:
        a, b, c = _CWCE_W2[k]
    e1 = 1.0 - 1.0 / k
    ea = 1.0 + 1.0 / k
    exps = {1.0: (-e1, e1), -1.0: (e1, -e1),
            a: (ea, -ea), -a: (-ea, ea),
            b: (-1.0 - 1.0 / k, -1.0 + 1.0 / k),
            -b: (-1.0 + 1.0 / k, -1.0 - 1.0 / k)}
    if fam == 'w2':
        exps[c] = (e1, -e1)
        exps[-c] = (-e1, e1)
    marks = {'x0': 0.0,
             'xinf': math.log(b * b / (a * a)),
             'x1': math.log(b * b * (1 - a * a)
                            / (a * a * (1 - b * b)))}
    if fam == 'w2':
        rho = a * a / (b * b)
        marks['xc'] = math.log((c * c - a * a)
                               / (rho * (c * c - b * b)))
    return a, b, c, exps, marks


def cwce_cw_omega_row(fam, k, dat, x_f, y, sid, soff, skeys):
    """(nf, 3) complex (om1, om2, om3) * dz/dw along the strip row
    w = x + iy under the chart z = sqrt(moeb(e^w)).  Every pair
    difference z^2 - p^2 is evaluated through a cancellation-free
    Moebius identity (exact expm1 forms), so the rows next to the
    marked points z = 0, 1, c, infinity stay at full precision;
    sid/soff carry exact offsets to the singular marks."""
    a, b, c, exps, marks = dat
    rho = a * a / (b * b)
    w = x_f + 1j * y
    with np.errstate(divide='ignore', invalid='ignore',
                     over='ignore'):
        D = -_cwce_expm1c(w + math.log(rho) - 1j * math.pi)
        Nz = -rho * b * b * _cwce_expm1c(w - 1j * math.pi)
        z = np.sqrt(Nz / D)
        flip = (z.imag < 0) | ((z.imag == 0) & (z.real < 0))
        z = np.where(flip, -z, z)
        if y in (0.0, math.pi):
            z = np.where(np.abs(z.imag) < 1e-30 * np.abs(z.real),
                         z.real + 0j, z)
        u = np.exp(w)
        x1 = marks['x1']
        d1 = x_f - x1
        if 'x1' in skeys:
            m = sid == skeys.index('x1')
            d1 = d1.copy()
            d1[m] = soff[m]
        pairs = {1.0: rho * (b * b - 1.0) * (-math.exp(x1))
                 * _cwce_expm1c(d1 + 1j * (y - math.pi)) / D,
                 a: rho * u * (b * b - a * a) / D,
                 b: (a * a - b * b) / D}
        if fam == 'w2':
            xc = marks['xc']
            dc = x_f - xc
            if 'xc' in skeys:
                m = sid == skeys.index('xc')
                dc = dc.copy()
                dc[m] = soff[m]
            pairs[c] = rho * (b * b - c * c) * (-math.exp(xc)) \
                * _cwce_expm1c(dc + 1j * (y - math.pi)) / D
        lm1 = np.zeros(len(x_f))
        an1 = np.zeros(len(x_f))
        lm2 = np.zeros(len(x_f))
        an2 = np.zeros(len(x_f))
        for p, (q1, q2) in exps.items():
            fac = pairs[p] / (z + p) if p > 0 else z - p
            if y in (0.0, math.pi):
                fac = np.where(np.abs(fac.imag)
                               < 1e-25 * np.abs(fac.real),
                               fac.real + 0j, fac)
            L = np.log(np.abs(fac))
            A = np.angle(fac)
            lm1 = lm1 + q1 * L
            an1 = an1 + q1 * A
            lm2 = lm2 + q2 * L
            an2 = an2 + q2 * A
        phi1 = np.exp(lm1 + 1j * an1)
        phi2 = np.exp(lm2 + 1j * an2)
        dh = 1.0 / pairs[b]
        dzdw = rho * (b * b - a * a) / (D * D) * u / (2.0 * z)
        om = np.stack([0.5 * (phi2 - phi1), 0.5j * (phi1 + phi2), dh],
                      axis=-1) * dzdw[:, None]
    return np.where(np.isfinite(om), om, 0.0)


def cwce_cw_piece(fam, k, nu, nv, xa, xb):
    """Mesh the strip fundamental piece [xa, xb] x [0, pi]: row-wise
    integration rebased at a generic column plus one column stitch.
    Returns (V, faces, boundary classification, per-vertex uv)."""
    dat = _cwce_cw_data(fam, k)
    marks = dat[4]
    e1 = 1.0 - 1.0 / k
    mk = sorted(marks.items(), key=lambda t: t[1])
    bounds = [xa] + [v for _, v in mk] + [xb]
    sing = {v: (e1 if name in ('x1', 'xc') else 0.5)
            for name, v in mk}
    x_c, x_f, cidx, sid_x, soff, w_lo, w_hi = cwce_grids(
        bounds, sing, nu, msub=6, msing=80)
    skeys = [name for name, v in mk]
    ys = np.linspace(0.0, math.pi, nv)
    ncx = len(x_c)
    icB = int(np.argmin(np.abs(x_c - 0.5 * (marks['x1'] + xb))))
    marc = 10
    yf = np.linspace(0.0, math.pi, (nv - 1) * marc + 1)
    omc = np.stack([cwce_cw_omega_row(
        fam, k, dat, np.array([x_c[icB]]), float(yf[t]),
        np.array([-1]), np.array([0.0]), skeys)[0]
        for t in range(len(yf))])
    dyf = np.diff(yf)[:, None]
    Acol = np.concatenate(
        [np.zeros((1, 3), complex),
         np.cumsum(0.5 * (omc[1:] + omc[:-1]) * (1j * dyf), axis=0)],
        axis=0)[::marc]
    X = np.zeros((ncx, nv, 3))
    for j, y in enumerate(ys):
        om = cwce_cw_omega_row(fam, k, dat, x_f, float(y), sid_x,
                               soff, skeys)
        contrib = w_lo[:, None] * om[:-1] + w_hi[:, None] * om[1:]
        F = np.concatenate([np.zeros((1, 3), complex),
                            np.cumsum(contrib, axis=0)], axis=0)
        F = F - F[cidx[icB]]
        X[:, j, :] = np.real(Acol[j][None, :] + F[cidx])
    V = X.reshape(-1, 3)

    def vid(i, j):
        return i * nv + j

    UV = np.zeros((len(V), 2))
    UV[:, 0] = np.tile(ys / math.pi, ncx)
    UV[:, 1] = np.repeat(np.linspace(0.0, 1.0, ncx), nv)
    faces = []
    for i in range(ncx - 1):
        for j in range(nv - 1):
            faces.append((vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)))
    mark_ci = {name: int(np.argmin(np.abs(x_c - v)))
               for name, v in marks.items()}
    b_cls = {'bottom': np.array([vid(i, 0) for i in range(ncx)],
                                dtype=np.int64)}
    edges = [0] + [mark_ci[name] for name in skeys] + [ncx - 1]
    segs = []
    for s in range(len(edges) - 1):
        i0, i1 = edges[s], edges[s + 1]
        first = i0 if s == 0 else i0 + 1
        last = i1 + (1 if s == len(edges) - 2 else 0)
        segs.append(np.array([vid(i, nv - 1)
                              for i in range(first, last)],
                             dtype=np.int64))
    b_cls['top_segs'] = segs
    b_cls['top_names'] = skeys
    b_cls['mark_vid'] = {name: vid(ci, nv - 1)
                         for name, ci in mark_ci.items()}
    return V, faces, b_cls, UV


def cwce_cw_snap(fam, k, V, b_cls):
    """Center on the z = 1 branch image (the origin of the D_kh
    frame), then snap every boundary curve onto its exact symmetry
    element: vertical mirror planes through the axis at the family's
    angles (0 / pi/k, measured off the Weierstrass phases), the
    horizontal mirror plane at the z = 0 image height h0, and the
    branch images onto the axis.  Returns (V, h0, residuals)."""
    res = {}
    V = V - V[b_cls['mark_vid']['x1']]
    if fam == 'cw':
        seg_angle = [math.pi / k, None, 0.0, math.pi / k]
        bottom_angle = 0.0
        ang0 = math.pi / k
    else:
        seg_angle = [0.0, None, 0.0, math.pi / k, 0.0]
        bottom_angle = math.pi / k
        ang0 = 0.0
    h0 = float(V[b_cls['mark_vid']['x0'], 2])
    for s, idx in enumerate(b_cls['top_segs']):
        if len(idx) == 0:
            continue
        ang = seg_angle[s]
        if ang is None:                    # horizontal-mirror curve
            res[f'top{s}'] = float(np.max(np.abs(V[idx, 2] - h0)))
            V[idx, 2] = h0
        else:
            uvec = np.array([math.cos(ang), math.sin(ang)])
            xy = V[idx][:, :2]
            t = xy @ uvec
            perp = xy - t[:, None] * uvec[None, :]
            res[f'top{s}'] = float(np.max(np.linalg.norm(perp,
                                                        axis=1)))
            V[idx, 0] = t * uvec[0]
            V[idx, 1] = t * uvec[1]
    idx = b_cls['bottom']
    uvec = np.array([math.cos(bottom_angle), math.sin(bottom_angle)])
    xy = V[idx][:, :2]
    t = xy @ uvec
    perp = xy - t[:, None] * uvec[None, :]
    res['bottom'] = float(np.max(np.linalg.norm(perp, axis=1)))
    V[idx, 0] = t * uvec[0]
    V[idx, 1] = t * uvec[1]
    mv = b_cls['mark_vid']
    res['b_x1'] = float(np.hypot(V[mv['x1'], 0], V[mv['x1'], 1]))
    V[mv['x1'], 0] = V[mv['x1'], 1] = 0.0
    if 'xc' in mv:
        res['b_xc'] = float(np.hypot(V[mv['xc'], 0], V[mv['xc'], 1]))
        V[mv['xc'], 0] = V[mv['xc'], 1] = 0.0
    for name, ang in (('x0', ang0), ('xinf', 0.0)):
        vi = mv[name]
        uvec = np.array([math.cos(ang), math.sin(ang)])
        t = float(V[vi, :2] @ uvec)
        res[f'b_{name}'] = max(
            float(np.linalg.norm(V[vi, :2] - t * uvec)),
            abs(float(V[vi, 2] - h0)))
        V[vi, 0], V[vi, 1] = t * uvec[0], t * uvec[1]
        V[vi, 2] = h0
    return V, h0, res


def cwce_cw_assemble(fam, k, nu, nv, xa=None, xb=None):
    """Watertight D_kh assembly (4k frames: k rotations x vertical
    mirror x horizontal mirror at h0); the four end rims stay open.
    Returns (V, faces, uv, diag)."""
    dat = _cwce_cw_data(fam, k)
    marks = dat[4]
    if xa is None:
        xa = -1.4
    if xb is None:
        xb = marks['x1'] + 2.6
    V0, faces0, b_cls, UV0 = cwce_cw_piece(fam, k, nu, nv, xa, xb)
    V0, h0, res = cwce_cw_snap(fam, k, V0.copy(), b_cls)
    bmask = np.zeros(len(V0), dtype=bool)
    bmask[b_cls['bottom']] = True
    for idx in b_cls['top_segs']:
        bmask[idx] = True
    bmask[list(b_cls['mark_vid'].values())] = True
    mats, revs = [], []
    Mzh = np.diag([1.0, 1.0, -1.0])
    My = np.diag([1.0, -1.0, 1.0])
    for j in range(k):
        cj, sj = math.cos(TAU * j / k), math.sin(TAU * j / k)
        R = np.array([[cj, -sj, 0.0], [sj, cj, 0.0],
                      [0.0, 0.0, 1.0]])
        for m in (0, 1):
            for h in (0, 1):
                M = R @ (My if m else np.eye(3)) \
                    @ (Mzh if h else np.eye(3))
                mats.append((M, np.array([0.0, 0.0, 2.0 * h0 * h])))
                revs.append((m + h) % 2 == 1)
    V, F, UV = cwce_weld_frames(V0, faces0, UV0, bmask, mats, revs)
    return V, F, UV, {'res': res, 'h0': h0}


def cwce_cw_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """MESH_PARAM builder: finished (V, quads, uv) fit to the 2 m
    cube.  The radius slider slides the strip trims (planar-end reach
    on the left, catenoid depth on the right)."""
    p = spec['p_from'](order, radius)
    fam, k = p['fam'], p['k']
    marks = _cwce_cw_data(fam, k)[4]
    t = float(np.clip(radius / 1.2, 0.7, 1.5))
    xa = -1.4 * t
    xb = marks['x1'] + 2.6 * t
    pnu = int(np.clip(nu * 1.8, 110, 300))
    pnv = int(np.clip(nv * 0.42, 17, 37))
    V, quads, uv, _diag = cwce_cw_assemble(fam, k, pnu, pnv, xa, xb)
    V = _smooth_boundary(V, quads, iters=6)
    V = _center_fit(V, scale, V)
    return V, quads, uv


# ==========================================================================
# Doubly periodic hyperelliptic tiler -- Karcher-Meeks-Rosenberg + Wei
# ==========================================================================
# A reusable engine for the classical four-ended doubly periodic minimal
# surfaces whose Weierstrass data lives on a hyperelliptic curve with
# reciprocal-paired REAL branch points and dh = dz/z (the "KMR form"):
#
#     g(z) = prod_i (z - p_i)^(e_i/2),   e_i = +-1,  p_i real,
#     dh   = dz/z,
#
# plus the KMR-3 member, whose Gauss map is the Mobius map
# (z + eps)/(z - eps) with eps = e^{i Phi} and whose dh is the
# holomorphic differential dz/w of the elliptic curve
# w^2 = z^4 - 2 cos(2 Alpha) z^2 + 1 (branch points on the unit circle).
#
# Structure exploited (and verified by the self-tests):
#   * In exponential coordinates z = e^w the quotient surface is an
#     annulus; the engine integrates ONE conformal patch over the strip
#     u in [log rmin, 0], v in [0, pi] (the upper-half-plane half of the
#     annulus) with a graded, branch-aligned grid and a refined
#     trapezoid rule anchored on the smooth v = pi/2 midline.
#   * The patch boundary v = 0 / v = pi (the real z axis) decomposes at
#     the branch points into planar symmetry arcs lying in VERTICAL
#     MIRROR PLANES x1 = const / x2 = const -- alternating with the
#     parity of the branch points passed -- or (KMR-3) into straight
#     lines on the surface.  Each arc is snapped exactly onto its plane
#     or line; the mismatch of wall constants that must coincide IS the
#     period problem, measured and gated as a residual (for Wei's
#     surface it reproduces the notebook's FindRoot condition
#     Im int_a^b (G + 1/G) dz/z = 0).
#   * The whole doubly periodic surface is the orbit of that one patch
#     under the group generated by the boundary isometries -- diagonal
#     sign maps (reflections in the coordinate mirror planes, 180-degree
#     rotations about straight lines in the surface) -- together with
#     the two lattice translations they compose to:
#         T1 = (2 dx, 0, 0),  T2 = (0, 2 pi, 0)      (KMR-1/2, Wei), or
#         T2 = (0, 2 c2, 2 c3)                       (KMR-3, tilted).
#     Copies weld seam-exactly because shared arcs are snapped onto the
#     fixed sets of the gluing isometries (no loose float matching).
#
# Every gluing map is a Schwarz reflection (z -> zbar on the parameter
# domain, anti-conformal), so every generator -- including the proper
# 180-degree rotations -- reverses the surface orientation; face
# windings flip with the generator parity, keeping the welded mesh
# consistently oriented.
#
# References:
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988) 83-114 (the doubly
#     periodic Scherk deformations; toroidal saddle towers);
#   W. H. Meeks III, H. Rosenberg, "The global theory of doubly
#     periodic minimal surfaces", Invent. Math. 97 (1989) 351-379;
#   J. Perez, M. M. Rodriguez, M. Traizet, "The classification of
#     doubly periodic minimal tori with parallel ends", J. Differential
#     Geom. 69 (2005) 523-577 (the standard examples are named "KMR
#     surfaces" here);
#   F. Wei, "Some existence and uniqueness theorems for doubly periodic
#     minimal surfaces", Invent. Math. 109 (1992) 113-136 (the genus-2
#     surface: a KMR/Scherk surface with an added handle);
#   M. Weber, https://minimalsurfaces.blog/ repository, doubly periodic
#     section (KMR-2, KMR-3 and "Doubly Wei (g=2)" notebooks -- the
#     Weierstrass data, solved period constants and reflection
#     assemblies this engine follows).

# Wei genus-2 family: (b, a) pairs with a solved from b by the period
# condition Im int_a^b (G + 1/G) dz/z = 0 (the notebook's FindRoot
# values, verified against the same integral by the self-test below).
DPERIODIC_WEI_SAMPLES = (
    (0.30, 0.23640853975826828),
    (0.35, 0.18932453473239355),
    (0.40, 0.14531351679837468),
    (0.50, 0.07082564803712697),
    (0.60, 0.022673092012506006),
    (0.65, 0.009769033395931668),
    (0.70, 0.0030808434547059727),
)


def _dperiodic_graded(a, b, n):
    """n+1 cosine-graded nodes on [a, b], clustered at both endpoints
    (the branch points sit at interval ends, where the integrand has
    its inverse-square-root singularities)."""
    t = np.linspace(0.0, 1.0, max(2, int(n)) + 1)
    return a + (b - a) * (0.5 - 0.5 * np.cos(math.pi * t))


def _dperiodic_refine(x, m):
    """Insert m-1 intermediate nodes per interval (keeps the original
    nodes bit-exact at indices ::m, so a fine integration grid can be
    subsampled back onto the mesh grid)."""
    x = np.asarray(x, dtype=float)
    if m <= 1:
        return x
    seg = [x[:-1] + (x[1:] - x[:-1]) * (k / m) for k in range(m)]
    return np.append(np.stack(seg, axis=1).reshape(-1), x[-1])


def _dperiodic_ugrid(splits, total):
    """Concatenated per-interval graded u grid through the exact split
    values (branch-point logs).  Returns (u, split_indices)."""
    splits = list(splits)
    L = np.diff(np.asarray(splits))
    w = np.maximum(L, 1e-9) ** 0.6
    counts = [max(8, int(round(total * wi / w.sum()))) for wi in w]
    u = [np.array([splits[0]])]
    idx = [0]
    for k, n in enumerate(counts):
        seg = _dperiodic_graded(splits[k], splits[k + 1], n)
        u.append(seg[1:])
        idx.append(idx[-1] + n)
    return np.concatenate(u), idx


def _dperiodic_vgrid(nv):
    """Cosine-graded v grid on [0, pi] (dense at both boundary lines),
    even interval count so v = pi/2 is an exact node."""
    n = max(16, int(nv))
    n += n % 2
    t = np.linspace(0.0, 1.0, n + 1)
    return math.pi * (0.5 - 0.5 * np.cos(math.pi * t))


def _dperiodic_integrate(F, ug, vg):
    """Cumulative-trapezoid antiderivative of the (nu, nv, 3) integrand
    F (in w = u + iv coordinates, dz/dw folded in) on the nonuniform
    grid: base row along the smooth v = pi/2 midline, columns up and
    down from it.  Returns Re of the antiderivative (nu, nv, 3)."""
    F = np.where(np.isfinite(F), F, 0.0)
    jm = int(np.argmin(np.abs(vg - 0.5 * math.pi)))
    du = np.diff(ug)[:, None]
    dv = np.diff(vg)
    base = np.zeros((len(ug), 3), dtype=complex)
    row = F[:, jm, :]
    base[1:] = np.cumsum(0.5 * (row[1:] + row[:-1]) * du, axis=0)
    col = np.zeros(F.shape, dtype=complex)
    up = np.cumsum(0.5 * (F[:, jm:-1, :] + F[:, jm + 1:, :])
                   * (1j * dv[jm:])[None, :, None], axis=1)
    col[:, jm + 1:, :] = up
    Frev = F[:, jm::-1, :]
    dvrev = dv[:jm][::-1]
    dn = np.cumsum(0.5 * (Frev[:, :-1, :] + Frev[:, 1:, :])
                   * (-1j * dvrev)[None, :, None], axis=1)
    col[:, jm - 1::-1, :] = dn
    return np.real(base[:, None, :] + col)


def _dperiodic_cont_sqrt(Q, jm):
    """Continuous branch of sqrt(Q) on a grid over a simply connected
    domain with no interior zeros: principal sqrt with the sign
    propagated by continuity from the (smooth) row jm outward."""
    s = np.sqrt(Q)
    t = s.copy()
    r = s[:, jm]
    step = np.where(np.abs(r[1:] - r[:-1]) > np.abs(r[1:] + r[:-1]),
                    -1.0, 1.0)
    sgn = np.concatenate([[1.0], np.cumprod(step)])
    t[:, jm] = r * sgn
    for rng in (range(jm + 1, Q.shape[1]), range(jm - 1, -1, -1)):
        prev = jm
        for j in rng:
            sj = s[:, j]
            tp = t[:, prev]
            fj = np.where(np.abs(sj - tp) > np.abs(sj + tp), -1.0, 1.0)
            t[:, j] = sj * fj
            prev = j
    return t


def _dperiodic_sqrtprod(z, factors):
    """g(z) = prod (z - p)^(e/2), principal branches per factor (single
    valued and holomorphic on the upper half plane for real p)."""
    g = np.ones_like(z)
    for p, e in factors:
        s = np.sqrt(z - p)
        g = g * s if e > 0 else g / s
    return g


def _dperiodic_F_exp(factors):
    """Integrand closure for the exp-coordinate KMR form: z = e^w,
    dh = dz/z; returns (omega1, omega2, omega3) * dz/dw on the w grid.
    With dh = dz/z the third component is exactly 1, so x3 = Re w = u
    exactly -- the horizontal-mirror edge u = 0 needs no snapping."""
    def F(W, jm):
        z = np.exp(W)
        with np.errstate(divide='ignore', invalid='ignore'):
            g = _dperiodic_sqrtprod(z, factors)
            f1 = 0.5 * (1.0 / g - g)
            f2 = 0.5j * (1.0 / g + g)
        one = np.ones_like(f1)
        return np.stack(np.broadcast_arrays(f1, f2, one), axis=-1)
    return F


def _dperiodic_F_kmr3(alpha, phi):
    """Integrand closure for KMR-3: parameter x = e^w on the half
    annulus, z = e^{i phi} (i - x)/(x + i) (which maps the real x axis
    onto the unit circle where dh's four branch points +-e^{+-i alpha}
    live), G = (z + eps)/(z - eps) with eps = e^{i phi}, and
    dh = i dz / sqrt(z^4 - 2 cos(2 alpha) z^2 + 1) taken with a
    continuity-tracked branch."""
    eps = np.exp(1j * phi)
    c2a = math.cos(2.0 * alpha)

    def F(W, jm):
        x = np.exp(W)
        z = eps * (1j - x) / (x + 1j)
        dzdw = eps * (-2j) / (x + 1j) ** 2 * x       # dz/dx * dx/dw
        Q = z ** 4 - 2.0 * c2a * z ** 2 + 1.0
        w = _dperiodic_cont_sqrt(Q, jm)
        with np.errstate(divide='ignore', invalid='ignore'):
            h = 1j / w * dzdw                        # dh * dz/dw ... = h dw
            g = (z + eps) / (z - eps)
            f1 = 0.5 * (1.0 / g - g) * h
            f2 = 0.5j * (1.0 / g + g) * h
        return np.stack(np.broadcast_arrays(f1, f2, h), axis=-1)
    return F


def _dperiodic_arcs_exp(factors, rmin):
    """Boundary-arc tables for the exp-coordinate KMR form.  The v = 0
    edge is the positive real axis, the v = pi edge the negative one;
    crossing the real axis at x flips the sign of every factor with
    p > x, so an arc is a planar symmetry curve in a vertical plane
    x2 = const when that count is EVEN (g real there) and x1 = const
    when it is ODD (g imaginary).  Returns (usplits, arcs) with arcs =
    [(edge, k0, k1, axis, cluster)], edge 0/1 = v = 0/pi, k0..k1 the
    split-interval index range, axis the constant coordinate, cluster
    '0' (wall through the origin) or 'far' (the wall whose doubled
    offset is the lattice vector)."""
    pos = sorted(p for p, e in factors if 0 < p < 1 and p > rmin)
    neg = sorted(-p for p, e in factors if -1 < p < 0 and -p > rmin)
    usplits = sorted({math.log(rmin), 0.0}
                     | {math.log(p) for p in pos}
                     | {math.log(t) for t in neg})
    kof = {round(s, 12): k for k, s in enumerate(usplits)}
    arcs = []
    stops_pos = [rmin] + pos + [1.0]
    for lo, hi in zip(stops_pos[:-1], stops_pos[1:]):
        x = math.sqrt(lo * hi)
        cnt = sum(1 for p, e in factors if p > x)
        axis = 1 if cnt % 2 == 0 else 0
        arcs.append((0, kof[round(math.log(lo), 12)],
                     kof[round(math.log(hi), 12)], axis, '0'))
    stops_neg = [rmin] + neg + [1.0]
    for lo, hi in zip(stops_neg[:-1], stops_neg[1:]):
        t = math.sqrt(lo * hi)
        cnt = sum(1 for p, e in factors if p > -t)
        axis = 1 if cnt % 2 == 0 else 0
        arcs.append((1, kof[round(math.log(lo), 12)],
                     kof[round(math.log(hi), 12)], axis,
                     '0' if axis == 1 else 'far'))
    return usplits, arcs


def _dperiodic_arc_slice(X, edge, i0, i1):
    """View of the arc's vertex rows: edge 0 -> v = 0 row, 1 -> v = pi."""
    j = 0 if edge == 0 else X.shape[1] - 1
    return X[i0:i1 + 1, j, :]


def _dperiodic_arc_med(X, edge, i0, i1, axis):
    """Median of one coordinate along an arc, endpoints excluded (the
    corner samples sit next to the branch singularities)."""
    seg = _dperiodic_arc_slice(X, edge, i0, i1)
    core = seg[1:-1] if len(seg) > 4 else seg
    return float(np.median(core[:, axis]))


def dperiodic_patch(key, p, nu, nv, refine=3):
    """Integrate, wall-align and snap ONE conformal patch of the doubly
    periodic surface `key` ('kmr2' | 'wei' | 'kmr3').  Returns a dict:
    X (nu, nv, 3) vertex grid, ug/vg parameter grids, ops (list of
    (sign3, parity) diagonal isometries whose orbit of the patch is one
    translational fundamental cell), T1/T2 lattice vectors, res
    (residual diagnostics -- the period problem's closure errors), and
    arc bookkeeping."""
    if key == 'wei':
        b, a = p['b'], p['a']
        r = a / b
        rmin = min(p.get('rmin', 0.05), 0.45 * min(a, r))
        factors = ((a, -1), (b, 1), (1.0 / b, -1), (1.0 / a, 1),
                   (-r, 1), (-1.0 / r, -1))
        usplits, arcs = _dperiodic_arcs_exp(factors, rmin)
        Ffn = _dperiodic_F_exp(factors)
    elif key == 'kmr2':
        a = p['a']
        rmin = min(p.get('rmin', 0.05), 0.45 * a)
        factors = ((a, -1), (1.0 / a, 1), (-a, 1), (-1.0 / a, -1))
        usplits, arcs = _dperiodic_arcs_exp(factors, rmin)
        Ffn = _dperiodic_F_exp(factors)
    elif key == 'kmr3':
        alpha = p.get('alpha', 0.5 * math.pi - 0.125 * math.pi)
        phi = p.get('phi', 0.25 * math.pi)
        xmin = p.get('xmin', 0.05)
        t1 = math.tan(0.5 * (alpha - phi))          # v=0 junctions
        t2 = math.tan(0.5 * (alpha + phi))          # v=pi junctions: t2, 1/t1
        xmin = min(xmin, 0.45 * t1)
        usplits = sorted({math.log(xmin), math.log(t1), -math.log(t2),
                          math.log(t2), -math.log(t1), -math.log(xmin)})
        kof = {round(s, 12): k for k, s in enumerate(usplits)}
        # (edge, k0, k1, kind, cluster): planes are x1 = const walls,
        # 'line' arcs are straight lines parallel to x1 (x2, x3 const)
        arcs = [
            (0, kof[round(math.log(xmin), 12)],
             kof[round(math.log(t1), 12)], 0, 'far'),
            (0, kof[round(math.log(t1), 12)],
             kof[round(-math.log(t2), 12)], 'line', '0'),
            (0, kof[round(-math.log(t2), 12)],
             kof[round(-math.log(xmin), 12)], 0, '0'),
            (1, kof[round(math.log(xmin), 12)],
             kof[round(math.log(t2), 12)], 0, '0'),
            (1, kof[round(math.log(t2), 12)],
             kof[round(-math.log(t1), 12)], 'line', 'far'),
            (1, kof[round(-math.log(t1), 12)],
             kof[round(-math.log(xmin), 12)], 0, 'far'),
        ]
        Ffn = _dperiodic_F_kmr3(alpha, phi)
    else:
        raise ValueError(f"dperiodic_patch: unknown key {key!r}")

    ug, uidx = _dperiodic_ugrid(usplits, max(24, int(nu)))
    vg = _dperiodic_vgrid(max(24, int(nv)))
    m = max(1, int(refine))
    uf = _dperiodic_refine(ug, m)
    vf = _dperiodic_refine(vg, m)
    jmf = int(np.argmin(np.abs(vf - 0.5 * math.pi)))
    W = uf[:, None] + 1j * vf[None, :]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        F = Ffn(W, jmf)
    Xf = _dperiodic_integrate(F, uf, vf)
    X = Xf[::m, ::m, :].copy()
    kidx = {k: uidx[k] for k in range(len(usplits))}
    arcsg = [(e, kidx[k0], kidx[k1], ax, cl) for (e, k0, k1, ax, cl)
             in arcs]

    res = {}
    if key in ('kmr2', 'wei'):
        # exact height: with dh = dz/z, x3 = u identically (measured
        # residual is pure quadrature error -- gate it, then use u)
        res['x3=u'] = float(np.max(np.abs(
            X[..., 2] - (X[0, 0, 2] - ug[0] + ug[:, None]))))
        X[..., 2] = ug[:, None]
        # align: v=0 x1 wall and (first) v=0 x2 wall through the origin
        s1 = [_dperiodic_arc_med(X, e, i0, i1, 0)
              for (e, i0, i1, ax, cl) in arcsg if ax == 0 and cl == '0']
        s2 = [_dperiodic_arc_med(X, e, i0, i1, 1)
              for (e, i0, i1, ax, cl) in arcsg if ax == 1 and e == 0]
        X[..., 0] -= s1[0]
        X[..., 1] -= s2[0]
        # wall constants + period residuals: every x2 wall must sit at
        # a multiple of pi (0 on v=0 -- Wei's FindRoot condition -- and
        # +-pi on v=pi); the 'far' x1 walls must agree on one offset dx
        far = []
        for (e, i0, i1, ax, cl) in arcsg:
            if ax == 'line':
                continue
            med = _dperiodic_arc_med(X, e, i0, i1, ax)
            if ax == 1:
                tgt = math.pi * round(med / math.pi)
                res[f'x2wall@{e}:{i0}'] = abs(med - tgt)
            elif cl == 'far':
                far.append(med)
        dx = float(np.mean(far))
        if len(far) > 1:
            res['x1walls'] = float(np.ptp(far))
        # snap every arc exactly onto its wall plane
        vpi_sign = 0.0
        for (e, i0, i1, ax, cl) in arcsg:
            seg = _dperiodic_arc_slice(X, e, i0, i1)
            if ax == 1:
                med = float(np.median(seg[1:-1, 1] if len(seg) > 4
                                      else seg[:, 1]))
                tgt = math.pi * round(med / math.pi)
                seg[:, 1] = tgt
                if e == 1:
                    vpi_sign = math.copysign(1.0, tgt if tgt else 1.0)
            else:
                seg[:, 0] = 0.0 if cl == '0' else dx
        ops = [((sx, sy, sz), sx * sy * sz)
               for sx in (1.0, -1.0) for sy in (1.0, -1.0)
               for sz in (1.0, -1.0)]
        T1 = np.array([2.0 * dx, 0.0, 0.0])
        T2 = np.array([0.0, 2.0 * math.pi * (vpi_sign or 1.0), 0.0])
    else:                                            # kmr3
        # classify residuals BEFORE alignment: constancy of the declared
        # coordinates along each arc (plane arcs: x1; line arcs: x2, x3)
        for (e, i0, i1, ax, cl) in arcsg:
            seg = _dperiodic_arc_slice(X, e, i0, i1)
            core = seg[1:-1] if len(seg) > 4 else seg
            if ax == 'line':
                res[f'line@{e}:{i0}'] = float(
                    max(np.ptp(core[:, 1]), np.ptp(core[:, 2])))
            else:
                res[f'plane@{e}:{i0}'] = float(np.ptp(core[:, 0]))
        # align: v=0 '0' plane wall -> x1 = 0; v=0 line -> x2 = x3 = 0
        for (e, i0, i1, ax, cl) in arcsg:
            if e == 0 and ax == 0 and cl == '0':
                X[..., 0] -= _dperiodic_arc_med(X, e, i0, i1, 0)
        for (e, i0, i1, ax, cl) in arcsg:
            if e == 0 and ax == 'line':
                X[..., 1] -= _dperiodic_arc_med(X, e, i0, i1, 1)
                X[..., 2] -= _dperiodic_arc_med(X, e, i0, i1, 2)
        # walls: dx from the v=0 'far' plane; the v=pi planes must land
        # on the SAME two walls (their offsets are the period residuals)
        dx = [_dperiodic_arc_med(X, e, i0, i1, 0)
              for (e, i0, i1, ax, cl) in arcsg
              if e == 0 and ax == 0 and cl == 'far'][0]
        c23 = [( _dperiodic_arc_med(X, e, i0, i1, 1),
                 _dperiodic_arc_med(X, e, i0, i1, 2))
               for (e, i0, i1, ax, cl) in arcsg
               if e == 1 and ax == 'line'][0]
        for (e, i0, i1, ax, cl) in arcsg:
            if e == 1 and ax == 0:
                med = _dperiodic_arc_med(X, e, i0, i1, 0)
                tgt = 0.0 if cl == '0' else dx
                res[f'x1wall@{e}:{i0}'] = abs(med - tgt)
        # snap
        for (e, i0, i1, ax, cl) in arcsg:
            seg = _dperiodic_arc_slice(X, e, i0, i1)
            if ax == 'line':
                seg[:, 1], seg[:, 2] = ((0.0, 0.0) if cl == '0'
                                        else (c23[0], c23[1]))
            else:
                seg[:, 0] = 0.0 if cl == '0' else dx
        # orbit: mirror x1 = 0 and the 180-degree rotation about the
        # x1 axis (both Schwarz continuations -> parity -1 each)
        ops = [((1.0, 1.0, 1.0), 1.0), ((-1.0, 1.0, 1.0), -1.0),
               ((1.0, -1.0, -1.0), -1.0), ((-1.0, -1.0, -1.0), 1.0)]
        T1 = np.array([2.0 * dx, 0.0, 0.0])
        T2 = np.array([0.0, 2.0 * c23[0], 2.0 * c23[1]])
    return {'X': X, 'ug': ug, 'vg': vg, 'arcs': arcsg, 'ops': ops,
            'T1': T1, 'T2': T2, 'res': res, 'key': key,
            'ends_edge': 'both' if key == 'kmr3' else 'low'}


def _dperiodic_weld(V, quads, UV, decimals=6):
    """Weld coincident vertices by exact rounded-coordinate keys (the
    assembly snaps every shared arc onto the fixed set of its gluing
    isometry, so seam partners agree to machine precision -- this is a
    hash join on those exact positions, not a loose tolerance match)."""
    key = np.round(V * 10.0 ** decimals).astype(np.int64)
    view = np.ascontiguousarray(key).view(
        np.dtype((np.void, key.dtype.itemsize * key.shape[1])))
    _, first, inv = np.unique(view.ravel(), return_index=True,
                              return_inverse=True)
    inv = inv.ravel()
    Vw = V[first]
    UVw = UV[first] if UV is not None else None
    out = []
    for f in inv[np.asarray(quads, dtype=np.int64)]:
        h = [int(f[0])]
        for s in f[1:]:
            if int(s) != h[-1]:
                h.append(int(s))
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            out.append(tuple(h))
    return Vw, out, UVw


def dperiodic_assemble(P, cells=(1, 1)):
    """Tile the patch orbit over cells[0] x cells[1] lattice cells at
    the TRUE period vectors and weld all seams.  Returns (V, quads,
    per-vertex UV)."""
    X = P['X']
    nu, nv = X.shape[:2]
    Vp = X.reshape(-1, 3)
    gu = np.arange(nu) / max(nu - 1, 1)
    gv = np.arange(nv) / max(nv - 1, 1)
    UVp = np.stack(np.meshgrid(gu, gv, indexing='ij'),
                   axis=-1).reshape(-1, 2)
    ii, jj = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1),
                         indexing='ij')
    b = (ii * nv + jj).ravel()
    q0 = np.stack([b, b + nv, b + nv + 1, b + 1], axis=1)
    cu, cv = int(max(1, cells[0])), int(max(1, cells[1]))
    Vs, Qs, UVs = [], [], []
    off0 = 0
    for icell in range(cu):
        for jcell in range(cv):
            t = ((icell - 0.5 * (cu - 1)) * P['T1']
                 + (jcell - 0.5 * (cv - 1)) * P['T2'])
            for sgn, parity in P['ops']:
                Vs.append(Vp * np.asarray(sgn)[None, :] + t[None, :])
                UVs.append(UVp)
                Qs.append((q0 if parity > 0 else q0[:, ::-1]) + off0)
                off0 += len(Vp)
    V = np.concatenate(Vs, axis=0)
    UV = np.concatenate(UVs, axis=0)
    Q = np.concatenate(Qs, axis=0)
    return _dperiodic_weld(V, Q, UV)


def dperiodic_quotient(P):
    """Topology of the translational quotient: assemble ONE cell, then
    identify its opposite lattice walls (vertices matched under +-T1,
    +-T2 by the same exact-coordinate hash).  Returns (chi, boundary
    loop count, nonmanifold edge count, oriented, components) --
    ends stay as boundary rims, so chi must equal 2 - 2 genus - ends."""
    V, quads, _ = dperiodic_assemble(P, (1, 1))
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    key = np.round(V * 1e6).astype(np.int64)
    hkey = {tuple(k): i for i, k in enumerate(key)}
    for T in (P['T1'], P['T2'], -P['T1'], -P['T2']):
        kt = np.round((V + T[None, :]) * 1e6).astype(np.int64)
        for i, k in enumerate(kt):
            j = hkey.get(tuple(k))
            if j is not None:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    root = np.array([find(i) for i in range(len(V))])
    ec, dc = {}, {}
    for f in quads:
        mlen = len(f)
        for k in range(mlen):
            a2, b2 = int(root[f[k]]), int(root[f[(k + 1) % mlen]])
            e = (a2, b2) if a2 < b2 else (b2, a2)
            ec[e] = ec.get(e, 0) + 1
            dc[(a2, b2)] = dc.get((a2, b2), 0) + 1
    nvq = len({int(r) for f in quads for r in root[list(f)]})
    chi = nvq - len(ec) + len(quads)
    nonman = sum(1 for c in ec.values() if c > 2)
    orient = all(c == 1 for c in dc.values())
    bed = [e for e, c in ec.items() if c == 1]
    par = {}

    def bfind(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a2, b2 in bed:
        ra, rb = bfind(a2), bfind(b2)
        if ra != rb:
            par[ra] = rb
    loops = len({bfind(a2) for a2, b2 in bed})
    parc = {}

    def cfind(x):
        parc.setdefault(x, x)
        while parc[x] != x:
            parc[x] = parc[parc[x]]
            x = parc[x]
        return x

    for f in quads:
        for i in range(1, len(f)):
            ra, rb = cfind(int(root[f[0]])), cfind(int(root[f[i]]))
            if ra != rb:
                parc[ra] = rb
    ncomp = len({cfind(int(root[i2])) for f in quads for i2 in f})
    return chi, loops, nonman, orient, ncomp


_DPERIODIC_CACHE = {}


def dperiodic_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                   cells=(1, 1)):
    """Finished-mesh builder for the doubly periodic rows (toolkit
    MESH_PARAM / cells2d contract): patch -> orbit -> cells[0] x
    cells[1] lattice tiling -> largest component -> 2 m fit.  theta is
    unused (no associate family on the tiled surfaces)."""
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    key = spec['dp_key']
    if isinstance(cells, (int, float)):
        cells = (int(cells), 1)
    cu = int(np.clip(cells[0], 1, 8))
    cv = int(np.clip(cells[1] if len(cells) > 1 else 1, 1, 8))
    ck = (key, tuple(sorted(p.items())), int(nu), int(nv))
    P = _DPERIODIC_CACHE.get(ck)
    if P is None:
        P = dperiodic_patch(key, p, int(np.clip(nu, 24, 220)),
                            int(np.clip(nv, 24, 220)))
        if len(_DPERIODIC_CACHE) > 12:
            _DPERIODIC_CACHE.clear()
        _DPERIODIC_CACHE[ck] = P
    V, quads, UV = dperiodic_assemble(P, (cu, cv))
    Vu, quads = _largest_component(np.hstack([V, UV]), quads)
    V, UV = Vu[:, :3], Vu[:, 3:]
    V = _center_fit(V, scale, V)
    return V, quads, UV


def dperiodic_wei_residual(b, a, n=320001):
    """Wei's period condition, exactly as the notebook's FindRoot test:
    Im int_a^b (G + 1/G) dz/z along the real segment [a, b] (the cut),
    with the same principal-branch factors as the surface.  The cosine
    substitution regularizes the endpoint inverse-square-root
    singularities; the remaining O(h^2) trapezoid error is Richardson-
    extrapolated away (verified: the harvested constants then close to
    ~1e-6, and the h^2 tail alone was the earlier 1e-3 'residual')."""
    r = a / b
    factors = ((a, -1), (b, 1), (1.0 / b, -1), (1.0 / a, 1),
               (-r, 1), (-1.0 / r, -1))
    sa, sb = math.log(a), math.log(b)

    def quad(m):
        # integrate in s = log x (dz/z = ds exactly, so the 1/z pole
        # near the small branch point a never amplifies the error),
        # cosine-graded against the endpoint branch singularities
        t = np.linspace(0.0, 1.0, m)
        s = sa + (sb - sa) * (0.5 - 0.5 * np.cos(math.pi * t))
        x = np.exp(s) + 0j
        with np.errstate(divide='ignore', invalid='ignore'):
            g = _dperiodic_sqrtprod(x, factors)
            f = g + 1.0 / g
        f = np.where(np.isfinite(f), f, 0.0)
        ds = np.diff(s)
        return float(np.imag(np.sum(0.5 * (f[1:] + f[:-1]) * ds)))

    r2 = quad(n)
    r1 = quad(n // 2 + 1)
    return (4.0 * r2 - r1) / 3.0


# ==========================================================================
# Doubly periodic long tail -- Karcher-Scherk, RTW, Wei tower, Connor
# ==========================================================================
# The remaining classical/experimental four-or-more-ended doubly periodic
# minimal surfaces whose Weierstrass data is hyperelliptic with REAL
# branch points and dh = dz/z, generalizing the KMR/Wei tiler above to
# the three assembly styles the source notebooks use:
#
#   * 'half8' -- reciprocal-paired branch points with opposite square-
#     root exponents, so |g| = 1 on the unit circle: the u = 0 edge of
#     the exp-strip is a horizontal planar symmetry curve and the patch
#     orbit is the 8 diagonal sign maps (exactly the engine above).
#     Members: F. Wei's higher-genus (1,3)/(1,4)/(2,3)/(1,6) towers and
#     the Rossman-Thayer-Wohlgemuth M1+ / M1+- surfaces.
#   * 'ks'    -- the Karcher-Scherk surfaces with handles: branch
#     points at +-1 put arg g = const on the unit circle, which is a
#     horizontal STRAIGHT LINE (direction (1,1,0)) in the surface (the
#     classical doubly periodic Scherk contains these diagonal lines at
#     mid-level).  Orbit = closure of {180-degree rotation about that
#     line, mirrors x1 = 0 and x2 = 0} (8 ops); lattice fixed at
#     (2 pi, 0, 0) x (0, 2 pi, 0) by the dh = dz/z normalization, so
#     BOTH x1 and x2 walls must land on multiples of pi -- that is this
#     family's period problem, solved in the notebooks by FindRoot on
#     segment integrals of omega1/omega2 between branch points.
#   * 'full4' -- no circle symmetry at all (Connor's experimental
#     surfaces, branch sets not reciprocal-invariant in pairs with
#     mixed exponents): integrate the WHOLE annulus rlo < |z| < rhi
#     (upper half), orbit = {id, mirror x1, mirror x2, both} only,
#     ends truncated at both edges.
#
# Every member ships with the literal solved period constants recovered
# from the repository notebooks (FindRoot outputs / family tables); the
# self-tests re-measure the wall residuals, the 2-D lattice, and the
# quotient topology chi = 2 - 2 genus - ends per fundamental domain.
#
# References:
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988) 83-114 (Scherk surfaces
#     with handles; the genus 2..4 doubly periodic tower);
#   F. Wei, "Some existence and uniqueness theorems for doubly periodic
#     minimal surfaces", Invent. Math. 109 (1992) 113-136, and the
#     higher-genus (m,n) Wei family after M. Weber's repository;
#   W. Rossman, E. C. Thayer, M. Wohlgemuth, "Embedded, doubly periodic
#     minimal surfaces", Experiment. Math. 9 (2000) 197-219 (the M1+
#     and M1+- families);
#   P. Connor, "A note on special polynomials and minimal surfaces"
#     (experimental doubly periodic surfaces of genus 3, 2018-2019;
#     numerically established -- see also P. Connor, M. Weber, "The
#     construction of doubly periodic minimal surfaces via balance
#     equations", Amer. J. Math. 134 (2012) 1275-1301);
#   M. Weber, https://minimalsurfaces.blog/ repository, doubly periodic
#     section (Karcher-Scherk g=2/3, exotic g=3, RTW, higher-genus Wei
#     and Connor pages 78-85 notebooks: Weierstrass data, solved
#     constants and reflection assemblies followed here).

def _dptail_recip(vals):
    """[(p, e)] -> reciprocal-completed factor list [(p, e), (1/p, -e)]."""
    out = []
    for p, e in vals:
        out.append((p, e))
        out.append((1.0 / p, -e))
    return tuple(out)


def _dptail_members():
    """The harvested member catalog: key -> dict(factors, style, genus,
    label).  Constants are the notebooks' solved period parameters."""
    M = {}

    # --- Karcher-Scherk with handles (style 'ks') ----------------------
    a, b = 0.1253991455944982, 0.25068716043696654
    M['ksg2'] = {
        'style': 'ks', 'genus': 2,
        'label': "Karcher-Scherk genus 2",
        'factors': ((-1.0, 1), (-1.0 / a, 1), (-a, 1),
                    (1.0, -1), (-1.0 / b, -1), (-b, -1))}
    a, b, c = (0.06861386286581515, 0.11033686517910986,
               0.4744615458765442)
    M['ksg3'] = {
        'style': 'ks', 'genus': 3,
        'label': "Karcher-Scherk genus 3",
        'factors': ((-1.0 / a, 1), (-a, 1), (-1.0 / c, 1), (-c, 1),
                    (1.0, -1), (-1.0, -1), (-1.0 / b, -1), (-b, -1))}
    a, b, c = (0.0029790137955450864, 0.21361044126340875,
               0.0038598089630300695)
    M['ksg3x'] = {
        'style': 'ks', 'genus': 3,
        'label': "Karcher-Scherk genus 3 (exotic)",
        'factors': ((1.0, 1), (-1.0, 1), (-1.0 / a, 1), (-a, 1),
                    (c, -1), (1.0 / c, -1), (-1.0 / b, -1), (-b, -1))}

    # --- Rossman-Thayer-Wohlgemuth (style 'half8') ---------------------
    a = 0.26                       # M1+: b solved from a (self-test)
    b = DPTAIL_RTWMP_B
    M['rtwmp'] = {
        'style': 'half8', 'genus': 2,
        'label': "RTW M1+ (genus 2)",
        'factors': ((1.0 / a, 1), (1.0 / b, 1), (-a * b, 1),
                    (a, -1), (b, -1), (-1.0 / (a * b), -1))}
    for i, (b, c, d) in enumerate((
            (0.7348870529374415, -0.6951258227510726, -0.03),
            (0.6855592258248246, -0.5572870898881896, -0.1))):
        M[f'rtwm1pm_{i}'] = {
            'style': 'half8', 'genus': 3,
            'label': f"RTW M1+- (genus 3, d={d})",
            'factors': _dptail_recip(((b, 1), (c, 1), (d, 1),
                                      (1.0 / (b * c * d), 1)))}

    # --- Wei higher-genus towers (style 'half8') -----------------------
    # (1,3): (c; a, b) rows, r = a c / b
    for i, (c, a, b) in enumerate((
            (0.1, 0.008055067127584473, 0.028060852552439276),
            (0.2, 0.016531255568679768, 0.054948614240354035),
            (0.5, 0.04830868160434303, 0.11731752208030022),
            (0.8, 0.09242500104957037, 0.13493687204523117))):
        r = a * c / b
        M[f'wei13_{i}'] = {
            'style': 'half8', 'genus': 3,
            'label': f"Wei (1,3) genus 3, c={c}",
            'factors': _dptail_recip(((a, -1), (b, 1), (c, -1),
                                      (-r, 1)))}
    # (1,4): (d; a, b, c) rows, r = a c / (b d)
    for i, (d, a, b, c) in enumerate((
            (0.5, 0.06504976295629919, 0.07123688388229928,
             0.451792255249194),
            (0.7, 0.008088938096668071, 0.01971002640051247,
             0.07514495599787499))):
        r = a * c / (b * d)
        M[f'wei14_{i}'] = {
            'style': 'half8', 'genus': 4,
            'label': f"Wei (1,4) genus 4, d={d}",
            'factors': _dptail_recip(((a, -1), (b, 1), (c, -1),
                                      (d, 1), (-r, 1)))}
    # (2,3): a fixed 1e-4; (b, c, d) solved, r = a c / (b d)
    a, b, c, d = (1e-4, 0.04459799657015391, 0.5189369915935217,
                  0.00011126886913811259)
    r = a * c / (b * d)
    M['wei23'] = {
        'style': 'half8', 'genus': 4,
        'label': "Wei (2,3) genus 4",
        'factors': _dptail_recip(((1.0 / a, 1), (b, 1), (1.0 / c, 1),
                                  (-d, 1), (-r, 1)))}
    # (1,6): d fixed 0.02; (a, b, c, e, h) solved, r = a c e / (b d h)
    a, b, c = (0.0004555021778972459, 0.0008261557506571832,
               0.0033952038946453533)
    d, e, h = 0.009809755189394816, 0.02, 0.8260536205512837
    r = a * c * e / (b * d * h)
    M['wei16'] = {
        'style': 'half8', 'genus': 6,
        'label': "Wei (1,6) genus 6",
        'factors': _dptail_recip(((1.0 / a, 1), (b, 1), (1.0 / c, 1),
                                  (d, 1), (1.0 / e, 1), (h, 1),
                                  (-r, 1)))}

    # --- Connor experimental surfaces (style 'full4') ------------------
    # (a, c, d) Newton-refined on the notebook's three conditions at
    # fixed b, e (the published continuation tuple mixes solve stages
    # and misses closure by ~1e-2; refined residual < 1e-12)
    a, b, c = 0.014359861472204319, 0.8490182629508395, 2.39270674308568
    d, e = 31.950515854168334, -0.01
    f = a * b * d / (c * e)
    M['conn_asym'] = {
        'style': 'full4', 'genus': 2,
        'label': "Connor asymmetric (genus 2)",
        'factors': ((a, 1), (b, 1), (d, 1),
                    (c, -1), (e, -1), (f, -1))}
    a, b, c, d = (0.015, 8.45438228656637, 30.30978961482249,
                  81.61817271788935)
    e, h, i2 = (-0.026253034486005255, -0.49160372140098835,
                -1.2857374677056)
    j = b * d * e * h / (a * c * i2)
    M['conn78'] = {
        'style': 'full4', 'genus': 3,
        'label': "Connor page 78 (genus 3)",
        'factors': ((b, 1), (d, 1), (e, 1), (h, 1),
                    (a, -1), (c, -1), (i2, -1), (j, -1))}
    a, b, c, d = (0.00572883889017319, 0.05805739403536119,
                  0.221960415838744, 306.0)
    M['conn80'] = {
        'style': 'full4', 'genus': 3,
        'label': "Connor page 80 (genus 3)",
        'factors': ((-1.0 / a, 1), (a, 1), (-1.0 / b, 1), (b, 1),
                    (-1.0 / c, -1), (c, -1), (-1.0 / d, -1), (d, -1))}
    a, b, c, d = (0.01, 0.0624509711969649, 0.28200180374974865,
                  -0.005219595739646119)
    M['conn82'] = {
        'style': 'full4', 'genus': 3,
        'label': "Connor page 82 (genus 3)",
        'factors': ((a, 1), (b, 1), (1.0 / a, 1), (1.0 / b, 1),
                    (c, -1), (d, -1), (1.0 / c, -1), (1.0 / d, -1))}
    a, b, c, d = (0.0026378211134974183, 0.490458496891309, 1.0,
                  5478.680983554296)
    e, h, i2 = (16486.601223173104, 162703.99452674662,
                -0.0018594407009749243)
    j = c * d * h * i2 / (a * b * e)
    M['conn84'] = {
        'style': 'full4', 'genus': 3,
        'label': "Connor page 84 (genus 3)",
        'factors': ((c, 1), (d, 1), (h, 1), (i2, 1),
                    (a, -1), (b, -1), (e, -1), (j, -1))}
    a, b, c, d = (0.00006188567963069017, 0.0055,
                  0.012221191957186576, 10628.77254193301)
    M['conn85'] = {
        'style': 'full4', 'genus': 3,
        'label': "Connor page 85 (genus 3)",
        'factors': ((a, -1), (b, 1), (c, -1), (d, 1),
                    (-1.0 / a, -1), (-1.0 / b, 1), (-1.0 / c, -1),
                    (-1.0 / d, 1))}
    return M


def dptail_rtwmp_residual(b, a=0.26, n=420):
    """RTW M1+ period condition, as its notebook's SolvePeriod: the
    real part of int i (G + 1/G) dz/z from -a b through i to b must
    vanish; G has reciprocal-paired branch points driven by (a, b)."""
    factors = ((1.0 / a, 1), (1.0 / b, 1), (-a * b, 1),
               (a, -1), (b, -1), (-1.0 / (a * b), -1))

    def fn(z):
        g = _dperiodic_sqrtprod(np.asarray(z, dtype=complex), factors)
        return 1j * (g + 1.0 / g) / z
    return float(np.real(cwce_path_int(fn, [-a * b, 1j, b],
                                       e_end=(0.5, 0.5), n=n)))


# b for RTW M1+ at a = 0.26, solved from dptail_rtwmp_residual by
# bisection at import-verification time (value checked in the
# self-test against the same residual; the notebook only published the
# solve call, not the solved value).
DPTAIL_RTWMP_B = 0.2759530541264884

DPTAIL_MEMBERS = None            # built lazily (needs DPTAIL_RTWMP_B)


def dptail_member(key):
    global DPTAIL_MEMBERS
    if DPTAIL_MEMBERS is None:
        DPTAIL_MEMBERS = _dptail_members()
    return DPTAIL_MEMBERS[key]


def _dptail_arcs(factors, ulo, uhi):
    """Boundary-arc tables for an exp-strip u in [ulo, uhi] whose real
    z-axis edges decompose at the branch-point logs.  Same parity rule
    as _dperiodic_arcs_exp, but over an arbitrary radial window (the
    full annulus of the 'full4' style, or the [rmin, 1] half of the
    'half8'/'ks' styles).  Returns (usplits, arcs) with arcs entries
    (edge, k0, k1, axis) -- axis 0 = x1 wall, 1 = x2 wall."""
    rlo, rhi = math.exp(ulo), math.exp(uhi)
    pos = sorted(p for p, e in factors if rlo < p < rhi)
    neg = sorted(-p for p, e in factors if rlo < -p < rhi)
    usplits = sorted({ulo, uhi}
                     | {math.log(p) for p in pos}
                     | {math.log(t) for t in neg})
    kof = {round(s, 12): k for k, s in enumerate(usplits)}
    arcs = []
    for lo, hi in zip([rlo] + pos, pos + [rhi]):
        x = math.sqrt(lo * hi)
        cnt = sum(1 for p, e in factors if p > x)
        arcs.append((0, kof[round(math.log(lo), 12)],
                     kof[round(math.log(hi), 12)],
                     1 if cnt % 2 == 0 else 0))
    for lo, hi in zip([rlo] + neg, neg + [rhi]):
        t = math.sqrt(lo * hi)
        cnt = sum(1 for p, e in factors if p > -t)
        arcs.append((1, kof[round(math.log(lo), 12)],
                     kof[round(math.log(hi), 12)],
                     1 if cnt % 2 == 0 else 0))
    return usplits, arcs


def dptail_patch(key, p, nu, nv, refine=3):
    """Integrate, wall-align and snap ONE conformal patch of the
    doubly periodic tail member `key`.  Returns the same dict contract
    as dperiodic_patch, with 'ops' generalized to (M 3x3, t 3, parity)
    isometries consumed by dptail_assemble/dptail_quotient."""
    mem = dptail_member(key)
    factors, style = mem['factors'], mem['style']
    mags = sorted(abs(q) for q, e in factors)
    if style == 'full4':
        f = float(np.clip(p.get('trim', 8.0), 1.5, 200.0))
        ulo = math.log(mags[0] / f)
        uhi = math.log(mags[-1] * f)
    else:
        rmin = min(p.get('rmin', 0.05), 0.45 * mags[0])
        ulo, uhi = math.log(rmin), 0.0
    usplits, arcs = _dptail_arcs(factors, ulo, uhi)
    Ffn = _dperiodic_F_exp(factors)

    ug, uidx = _dperiodic_ugrid(usplits, max(24, int(nu)))
    vg = _dperiodic_vgrid(max(24, int(nv)))
    m = max(1, int(refine))
    uf = _dperiodic_refine(ug, m)
    vf = _dperiodic_refine(vg, m)
    jmf = int(np.argmin(np.abs(vf - 0.5 * math.pi)))
    W = uf[:, None] + 1j * vf[None, :]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        F = Ffn(W, jmf)
    Xf = _dperiodic_integrate(F, uf, vf)
    X = Xf[::m, ::m, :].copy()
    kidx = {k: uidx[k] for k in range(len(usplits))}
    arcsg = [(e, kidx[k0], kidx[k1], ax) for (e, k0, k1, ax) in arcs]

    res = {}
    # exact height: with dh = dz/z, x3 = u identically
    res['x3=u'] = float(np.max(np.abs(
        X[..., 2] - (X[0, 0, 2] - ug[0] + ug[:, None]))))
    X[..., 2] = ug[:, None]

    def med(e, i0, i1, ax):
        return _dperiodic_arc_med(X, e, i0, i1, ax)

    # base alignment: first x1 wall and first x2 wall -> 0
    w1 = [(e, i0, i1) for (e, i0, i1, ax) in arcsg if ax == 0]
    w2 = [(e, i0, i1) for (e, i0, i1, ax) in arcsg if ax == 1]
    if w1:
        X[..., 0] -= med(*w1[0], 0)
    if w2:
        X[..., 1] -= med(*w2[0], 1)

    # wall targets: x2 walls sit on multiples of pi (dh = dz/z fixes
    # the y-period at 2 pi); x1 walls sit on multiples of pi for 'ks'
    # (square 2 pi lattice) or on {0, +-dx} otherwise, dx measured
    m2 = [med(e, i0, i1, 1) for (e, i0, i1) in w2]
    for (e, i0, i1), v in zip(w2, m2):
        res[f'x2wall@{e}:{i0}'] = abs(v - math.pi * round(v / math.pi))
    m1 = [med(e, i0, i1, 0) for (e, i0, i1) in w1]
    if style == 'ks':
        dx = math.pi
        for (e, i0, i1), v in zip(w1, m1):
            res[f'x1wall@{e}:{i0}'] = abs(v - math.pi
                                          * round(v / math.pi))
    else:
        dx = max(abs(v) for v in m1) if m1 else math.pi
        if dx < 1e-9:
            dx = math.pi
        for (e, i0, i1), v in zip(w1, m1):
            res[f'x1wall@{e}:{i0}'] = abs(v - dx * round(v / dx))

    # snap every wall arc exactly onto its target plane
    for (e, i0, i1, ax) in arcsg:
        seg = _dperiodic_arc_slice(X, e, i0, i1)
        core = seg[1:-1] if len(seg) > 4 else seg
        v = float(np.median(core[:, ax]))
        step = math.pi if (ax == 1 or style == 'ks') else dx
        seg[:, ax] = step * round(v / step)

    # ops + lattice per style
    if style == 'half8':
        ops = [(np.diag([sx, sy, sz]), np.zeros(3), sx * sy * sz)
               for sx in (1.0, -1.0) for sy in (1.0, -1.0)
               for sz in (1.0, -1.0)]
        T1 = np.array([2.0 * dx, 0.0, 0.0])
        T2 = np.array([0.0, 2.0 * math.pi, 0.0])
    elif style == 'full4':
        ops = [(np.diag([sx, sy, 1.0]), np.zeros(3), sx * sy)
               for sx in (1.0, -1.0) for sy in (1.0, -1.0)]
        T1 = np.array([2.0 * dx, 0.0, 0.0])
        T2 = np.array([0.0, 2.0 * math.pi, 0.0])
    else:                                            # ks
        # u = 0 edge is a straight horizontal line along (1, 1, 0) or
        # (1, -1, 0): snap the constant diagonal combination
        line = X[-1, :, :]
        d1 = float(np.ptp(line[1:-1, 0] - line[1:-1, 1]))
        d2 = float(np.ptp(line[1:-1, 0] + line[1:-1, 1]))
        sgn = 1.0 if d1 <= d2 else -1.0            # x1 - sgn*x2 const
        res['line'] = min(d1, d2)
        delta = float(np.median(line[1:-1, 0] - sgn * line[1:-1, 1]))
        # the line offset must land on the pi wall lattice for the
        # reflection group to close -- its deviation IS a period
        # residual; snap it exactly
        tgt = math.pi * round(delta / math.pi)
        res['line_off'] = abs(delta - tgt)
        delta = tgt
        s = line[:, 0] + sgn * line[:, 1]
        line[:, 0] = 0.5 * (s + delta)
        line[:, 1] = sgn * 0.5 * (s - delta)
        line[:, 2] = 0.0
        # the u = 0 corners sit exactly ON the circle branch points
        # z = +-1, where the quadrature is unreliable: replace them by
        # the exact intersection of the line with the adjacent
        # (already snapped) wall plane
        nlast = X.shape[0] - 1
        for (e, i0, i1, ax) in arcsg:
            if i1 != nlast:
                continue
            j = 0 if e == 0 else X.shape[1] - 1
            wall = X[i1 - 1, j, ax] if i1 > i0 else X[i0, j, ax]
            wall = math.pi * round(float(wall) / math.pi)
            if ax == 0:
                X[nlast, j, 0] = wall
                X[nlast, j, 1] = sgn * (wall - delta)
            else:
                X[nlast, j, 1] = wall
                X[nlast, j, 0] = delta + sgn * wall
            X[nlast, j, 2] = 0.0
        RL = np.array([[0.0, sgn, 0.0], [sgn, 0.0, 0.0],
                       [0.0, 0.0, -1.0]])
        tL = np.array([delta, -sgn * delta, 0.0])
        gens = [(np.diag([-1.0, 1.0, 1.0]), np.zeros(3), -1.0),
                (np.diag([1.0, -1.0, 1.0]), np.zeros(3), -1.0),
                (RL, tL, -1.0)]
        T1 = np.array([2.0 * math.pi, 0.0, 0.0])
        T2 = np.array([0.0, 2.0 * math.pi, 0.0])
        ops = _dptail_close_ops(gens, T1, T2)
    return {'X': X, 'ug': ug, 'vg': vg, 'arcs': arcsg, 'ops': ops,
            'T1': T1, 'T2': T2, 'res': res, 'key': key,
            'style': style}


def _dptail_close_ops(gens, T1, T2, cap=40):
    """Close the generator list into a finite op set modulo the
    lattice {T1, T2} (translations reduced componentwise into the
    centered fundamental cell).  Ops are (M, t, parity)."""
    L = np.stack([T1[:2], T2[:2]])

    def reduce_t(t):
        # canonical representative in the HALF-OPEN centered cell:
        # components landing at +half-period (within tolerance) wrap
        # to -half-period so +-pi never split into distinct keys
        t = t.copy()
        c = np.linalg.solve(L.T, t[:2])
        c -= np.round(c)
        c[c > 0.5 - 1e-7] -= 1.0
        t[:2] = L.T @ c
        return t

    def key_of(M, t):
        return (tuple(np.round(M, 6).ravel()),
                tuple(np.round(t, 6)))

    ops = {key_of(np.eye(3), np.zeros(3)):
           (np.eye(3), np.zeros(3), 1.0)}
    frontier = list(ops.values())
    while frontier and len(ops) < cap:
        nxt = []
        for (M, t, par) in frontier:
            for (Mg, tg, pg) in gens:
                M2 = Mg @ M
                t2 = reduce_t(Mg @ t + tg)
                k = key_of(M2, t2)
                if k not in ops:
                    ops[k] = (M2, t2, par * pg)
                    nxt.append(ops[k])
        frontier = nxt
    return list(ops.values())


def dptail_assemble(P, cells=(1, 1)):
    """Tile the patch orbit over cells[0] x cells[1] lattice cells at
    the true period vectors and weld all seams (general (M, t, parity)
    ops).  Returns (V, quads, per-vertex UV)."""
    X = P['X']
    nu, nv = X.shape[:2]
    Vp = X.reshape(-1, 3)
    gu = np.arange(nu) / max(nu - 1, 1)
    gv = np.arange(nv) / max(nv - 1, 1)
    UVp = np.stack(np.meshgrid(gu, gv, indexing='ij'),
                   axis=-1).reshape(-1, 2)
    ii, jj = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1),
                         indexing='ij')
    b = (ii * nv + jj).ravel()
    q0 = np.stack([b, b + nv, b + nv + 1, b + 1], axis=1)
    cu, cv = int(max(1, cells[0])), int(max(1, cells[1]))
    Vs, Qs, UVs = [], [], []
    off = 0
    for ic in range(cu):
        for jc in range(cv):
            t0 = ((ic - 0.5 * (cu - 1)) * P['T1']
                  + (jc - 0.5 * (cv - 1)) * P['T2'])
            for (M, t, parity) in P['ops']:
                Vs.append(Vp @ M.T + (t + t0)[None, :])
                UVs.append(UVp)
                Qs.append((q0 if parity > 0 else q0[:, ::-1]) + off)
                off += len(Vp)
    V = np.concatenate(Vs, axis=0)
    UV = np.concatenate(UVs, axis=0)
    Q = np.concatenate(Qs, axis=0)
    return _dperiodic_weld(V, Q, UV)


def dptail_quotient(P):
    """Topology of the translational quotient (same contract as
    dperiodic_quotient): chi, boundary loops, nonmanifold edge count,
    oriented, components -- ends stay as boundary rims."""
    V, quads, _ = dptail_assemble(P, (1, 1))
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    key = np.round(V * 1e6).astype(np.int64)
    hkey = {tuple(k): i for i, k in enumerate(key)}
    # identify under ALL 8 neighbor translations: general (M, t) ops
    # can place seam partners a composite T1 +- T2 step apart
    shifts = [i1 * P['T1'] + i2 * P['T2']
              for i1 in (-1, 0, 1) for i2 in (-1, 0, 1)
              if (i1, i2) != (0, 0)]
    for T in shifts:
        kt = np.round((V + T[None, :]) * 1e6).astype(np.int64)
        for i, k in enumerate(kt):
            j = hkey.get(tuple(k))
            if j is not None:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    root = np.array([find(i) for i in range(len(V))])
    ec, dc = {}, {}
    for f in quads:
        mlen = len(f)
        for k in range(mlen):
            a2, b2 = int(root[f[k]]), int(root[f[(k + 1) % mlen]])
            e = (a2, b2) if a2 < b2 else (b2, a2)
            ec[e] = ec.get(e, 0) + 1
            dc[(a2, b2)] = dc.get((a2, b2), 0) + 1
    nvq = len({int(r) for f in quads for r in root[list(f)]})
    chi = nvq - len(ec) + len(quads)
    nonman = sum(1 for c in ec.values() if c > 2)
    orient = all(c == 1 for c in dc.values())
    bed = [e for e, c in ec.items() if c == 1]
    par = {}

    def bfind(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a2, b2 in bed:
        ra, rb = bfind(a2), bfind(b2)
        if ra != rb:
            par[ra] = rb
    loops = len({bfind(a2) for a2, b2 in bed})
    parc = {}

    def cfind(x):
        parc.setdefault(x, x)
        while parc[x] != x:
            parc[x] = parc[parc[x]]
            x = parc[x]
        return x

    for f in quads:
        for i2 in range(1, len(f)):
            ra, rb = cfind(int(root[f[0]])), cfind(int(root[f[i2]]))
            if ra != rb:
                parc[ra] = rb
    ncomp = len({cfind(int(root[i2])) for f in quads for i2 in f})
    return chi, loops, nonman, orient, ncomp


_DPTAIL_CACHE = {}


def dptail_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                cells=(1, 1)):
    """Finished-mesh builder for the doubly periodic tail rows
    (toolkit MESH_PARAM / cells2d contract)."""
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    key = p.pop('dpt_key')
    mem = dptail_member(key)
    nsp = max(6, len(mem['factors']))
    nuw = int(np.clip(nu * nsp / 6.0, 32, 320))
    if isinstance(cells, (int, float)):
        cells = (int(cells), 1)
    cu = int(np.clip(cells[0], 1, 8))
    cv = int(np.clip(cells[1] if len(cells) > 1 else 1, 1, 8))
    ck = (key, tuple(sorted(p.items())), int(nuw), int(nv))
    P = _DPTAIL_CACHE.get(ck)
    if P is None:
        P = dptail_patch(key, p, nuw, int(np.clip(nv, 24, 220)))
        if len(_DPTAIL_CACHE) > 10:
            _DPTAIL_CACHE.clear()
        _DPTAIL_CACHE[ck] = P
    V, quads, UV = dptail_assemble(P, (cu, cv))
    Vu, quads = _largest_component(np.hstack([V, UV]), quads)
    V, UV = Vu[:, :3], Vu[:, 3:]
    V = _center_fit(V, scale, V)
    return V, quads, UV


# ==========================================================================
# Singly periodic long tail (sptail_* block): six-ended Scherk towers,
# alternating fence of half-catenoids, fence of catenoids, helicoidal
# Karcher-Scherk (twisted saddle tower), translation-invariant Enneper
# surfaces
# ==========================================================================
# A family of Karcher-style singly periodic minimal surfaces, all meshed
# by the same scheme: integrate the Weierstrass 1-forms over ONE
# fundamental patch (compound Gauss-Legendre per grid cell, so singular
# corner nodes are never evaluated), snap the patch boundary exactly onto
# its symmetry elements (straight lines / mirror planes measured from the
# integrand's real/imaginary structure on the domain boundary), orbit the
# patch under the surface's isometry group plus `storeys` translation /
# screw copies, and weld seam-exactly.  Fractional powers are always
# taken in the LIMIT FROM THE DOMAIN INTERIOR (boundary points get a tiny
# upper-half-plane nudge), so no branch cut is ever crossed -- evaluating
# exactly ON a cut is where numpy's principal branch silently picks the
# wrong side.
#
# Period problems.  Every shipped member closes its period problem with a
# machine-checkable residual:
#   * six-ended Scherk (g = 0): rho and the branch point a are CLOSED
#     FORMS in the end-angle phi (rho = sqrt((1-cos phi)/(1+cos phi)),
#     a = 4 rho^4/(1+rho^2)^2); the residue of om2 at z = 0 gives the
#     translation dy = -(pi/4)(1+rho^2)/rho, verified against the
#     integrated patch.
#   * alternating fence of half-catenoids (g = 1): rho = 1/sqrt(a)
#     closes the horizontal period -- the patch's two vertical-mirror
#     segments land in the SAME plane (measured ~1e-15).
#   * helicoidal Karcher-Scherk: one real parameter R solved here by
#     bisection on the GEOMETRIC closure residual (the k-fold axis point
#     of the w = 0 line pair must coincide with that of the w = infinity
#     pair); the vertical rise then reproduces pi R^2/(1+R^4) to 1e-14.
#   * fence of catenoids: rho = sqrt(a) is the Lopez-Ros normalization
#     that makes the horizontal arc |z| = sqrt(a) a genuine planar
#     geodesic: |G| = 1 identically on the arc (an exact algebraic
#     identity at rho = sqrt(a), checked in the tests), so the
#     sigma_h reflection is a true symmetry; a is a free neck modulus.
#     MEASURED topology: quotient chi = -2 with 2 ends -- genus 1 per
#     period (a chain of catenoids joined neck to neck; the harvest's
#     "genus 0" annotation is off by the handle, as its metadata has
#     been before).
#   * periodic Enneper / translation-invariant Enneper with three
#     annular ends: g = z resp. g = (1/sqrt2)(1-z^2)/z with dh = dz --
#     closed-form antiderivatives, no constants at all.
#
# Topology is MEASURED, not assumed: the self-tests check the Euler
# characteristic of the welded stacks (chi(S+1) - chi(S) must equal the
# quotient chi = 2 - 2 genus - #ends) and/or of the translation-wrapped
# quotient, plus edge-manifoldness, orientability and connectivity.
# Note on the three-annular-end Enneper: its harvest metadata says
# genus 1 ("translation invariant torus"), but the surface built from
# the notebook's own rational data g = (1/sqrt2)(1-z^2)/z, dh = dz is
# parametrized by the four-punctured SPHERE (both g and dh are rational
# in z, so the quotient is genus 0 with 1 Enneper + 3 annular ends,
# chi = -2) -- which is exactly what the measured Euler characteristic
# gives.  This limit member is shipped with its measured genus.
#
# References:
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988) 83-114 -- saddle towers,
#     their helicoidal (screw-motion) deformations, and the fence of
#     catenoids;
#   H. F. Scherk (1835) and A. Enneper (1864) -- the classical data the
#     towers and the periodic Enneper interpolate;
#   M. Weber, https://minimalsurfaces.blog/ -- the harvested notebooks
#     (6-ended Scherk g=0; Alternating Fence of Half-Catenoids, 2024;
#     Helicoidal Karcher-Scherk; Fence of Catenoids; Periodic Enneper;
#     Enneper-Scherk; Translation-Invariant Torus with One Enneper and
#     Three Annular Ends, notebook by Ramazan Yol, 2024) --
#     research/msblog_harvest/singly_periodic.json.
# --------------------------------------------------------------------------

_SPTAIL_GL = np.polynomial.legendre.leggauss(8)


def sptail_cluster(h0, hmin=2e-7, ratio=0.3):
    """Geometric offsets h0 * ratio^k down to hmin (largest first)."""
    offs = []
    h = h0
    while h > hmin:
        offs.append(h)
        h *= ratio
    return np.array(offs)


def sptail_nudge(z):
    """Upper-half-plane limit for boundary evaluation: points exactly on
    the real axis get a tiny positive imaginary part, so every principal
    fractional power / log is continuous with the domain interior."""
    return np.where(np.imag(z) == 0.0, z + 1e-14j * (1.0 + np.abs(z)), z)


def sptail_polar_patch(om, r, t):
    """Integrate the 1-form triple om(z) over the polar grid (r_j, t_m)
    in the upper half plane.  Spine: the column nearest t = pi/2,
    cumulative in r; rows: arcs at fixed r outward from the spine.
    Compound GL-8 per cell -- nodes are strictly interior, so singular
    boundary nodes (branch points on the real axis) are never hit."""
    xg, wg = _SPTAIL_GL
    nr, nt = len(r), len(t)
    jm = int(np.argmin(np.abs(t - 0.5 * math.pi)))
    dr = np.diff(r)
    rmid = 0.5 * (r[1:] + r[:-1])
    zs = (rmid[:, None] + 0.5 * dr[:, None] * xg[None, :]) \
        * np.exp(1j * t[jm])
    o = om(zs)
    incS = np.stack([np.sum(c * wg, axis=-1) for c in o], axis=-1) \
        * (0.5 * np.exp(1j * t[jm]) * dr)[:, None]
    S = np.zeros((nr, 3), complex)
    S[1:] = np.cumsum(incS, axis=0)
    dt = np.diff(t)
    tmid = 0.5 * (t[1:] + t[:-1])
    Z = r[:, None, None] * np.exp(
        1j * (tmid[None, :, None] + 0.5 * dt[None, :, None] * xg))
    o = om(Z)
    incA = np.stack(
        [np.sum(c * (1j * Z) * wg, axis=-1) for c in o], axis=-1) \
        * (0.5 * dt)[None, :, None]
    C = np.zeros((nr, nt, 3), complex)
    C[:, 1:, :] = np.cumsum(incA, axis=1)
    C = C - C[:, jm, :][:, None, :]
    return np.real(S[:, None, :] + C)


def sptail_rect_patch(om, x, y):
    """Same scheme on a rectangular strip grid (x_i, y_j): spine down
    the column nearest x = 0 (dz = i dy), rows along x."""
    xg, wg = _SPTAIL_GL
    nx, ny = len(x), len(y)
    jm = int(np.argmin(np.abs(x)))
    dy = np.diff(y)
    ymid = 0.5 * (y[1:] + y[:-1])
    zs = x[jm] + 1j * (ymid[:, None] + 0.5 * dy[:, None] * xg[None, :])
    o = om(zs)
    incS = np.stack([np.sum(c * wg, axis=-1) for c in o], axis=-1) \
        * (0.5j * dy)[:, None]
    S = np.zeros((ny, 3), complex)
    S[1:] = np.cumsum(incS, axis=0)
    dx = np.diff(x)
    xmid = 0.5 * (x[1:] + x[:-1])
    Z = (xmid[:, None, None] + 0.5 * dx[:, None, None] * xg[None, None, :]) \
        + 1j * y[None, :, None]
    o = om(Z)
    incR = np.stack([np.sum(c * wg, axis=-1) for c in o], axis=-1) \
        * (0.5 * dx)[:, None, None]
    C = np.zeros((nx, ny, 3), complex)
    C[1:] = np.cumsum(incR, axis=0)
    C = C - C[jm][None, :, :]
    return np.real(S[None, :, :] + C)


def sptail_grid_quads(nr, nt, valid=None):
    """Grid quads (i, j) -> (i, j+1) -> ...; drop any touching an
    invalid (masked) vertex."""
    q = []
    for i in range(nr - 1):
        for j in range(nt - 1):
            f = (i * nt + j, i * nt + j + 1,
                 (i + 1) * nt + j + 1, (i + 1) * nt + j)
            if valid is None or (valid[f[0]] and valid[f[1]]
                                 and valid[f[2]] and valid[f[3]]):
                q.append(f)
    return q


def sptail_orbit_weld(V0, UV0, quads0, frames, tol):
    """Tile V0 under `frames` [(M, tvec, parity), ...] and weld
    coincident vertices (two offset quantization passes, so a pair
    straddling a rounding boundary is still caught).  Carries the
    per-vertex uv chart; drops vertices no face uses.  Seam vertices are
    bitwise-exact after the callers' boundary snapping, so the weld
    tolerance only needs to absorb float rounding."""
    Vp, Fp = [], []
    n0 = len(V0)
    for i, (M, tv, par) in enumerate(frames):
        Vp.append(V0 @ M.T + tv)
        off = i * n0
        if par < 0:
            Fp.extend(tuple(off + k for k in f[::-1]) for f in quads0)
        else:
            Fp.extend(tuple(off + k for k in f) for f in quads0)
    V = np.concatenate(Vp, axis=0)
    UV = np.tile(UV0, (len(frames), 1))
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    for shift in (0.0, 0.5):
        key = np.round(V / tol + shift).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        first = {}
        for i in range(len(V)):
            b = int(inv[i])
            if b in first:
                ra, rb = find(i), find(first[b])
                if ra != rb:
                    parent[ra] = rb
            else:
                first[b] = i
    roots = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    nw = len(uniq)
    Vw = np.zeros((nw, 3))
    UVw = np.zeros((nw, 2))
    cnt = np.zeros(nw)
    np.add.at(Vw, inv, V)
    np.add.at(UVw, inv, UV)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    UVw /= cnt[:, None]
    F = []
    for f in Fp:
        g = [int(inv[i]) for i in f]
        h = [g[0]]
        for s in range(1, len(g)):
            if g[s] != h[-1]:
                h.append(g[s])
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            F.append(tuple(h))
    used = sorted({i for f in F for i in f})
    remap = np.full(nw, -1, dtype=np.int64)
    remap[np.array(used, dtype=np.int64)] = np.arange(len(used))
    return (Vw[np.array(used, dtype=np.int64)],
            [tuple(int(remap[i]) for i in f) for f in F],
            UVw[np.array(used, dtype=np.int64)])


def sptail_topology(V, F):
    """(chi, nonmanifold_edges, oriented, boundary_loops, components)."""
    ec, dc = {}, {}
    for f in F:
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            e = (a, b) if a < b else (b, a)
            ec[e] = ec.get(e, 0) + 1
            dc[(a, b)] = dc.get((a, b), 0) + 1
    chi = len(V) - len(ec) + len(F)
    nonman = sum(1 for c in ec.values() if c > 2)
    orient = all(c == 1 for c in dc.values())
    bed = [e for e, c in ec.items() if c == 1]
    par = {}

    def bf(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in bed:
        ra, rb = bf(a), bf(b)
        if ra != rb:
            par[ra] = rb
    loops = len({bf(a) for a, b in bed})
    parc = list(range(len(V)))

    def cf(a):
        while parc[a] != a:
            parc[a] = parc[parc[a]]
            a = parc[a]
        return a
    for f in F:
        for i in range(1, len(f)):
            ra, rb = cf(f[0]), cf(f[i])
            if ra != rb:
                parc[ra] = rb
    ncomp = len({cf(f[0]) for f in F})
    return chi, nonman, orient, loops, ncomp


def sptail_quotient(V, F, Tvec, span, kmax=5):
    """Weld open-boundary vertices under v -> v + k Tvec (k = 1..kmax):
    the translation-wrapped quotient of a ONE-period stack.  Test-only
    (measures the true per-period chi and end-rim count)."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for f in F:
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            cnt[(a, b) if a < b else (b, a)] += 1
    bnd = sorted({v for e, c in cnt.items() if c == 1 for v in e})
    B = V[bnd]
    tol = 1e-6 * span
    key = {tuple(np.round(B[i] / tol).astype(np.int64)): bnd[i]
           for i in range(len(bnd))}
    parent = list(range(len(V)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for k in range(1, kmax + 1):
        for i in range(len(bnd)):
            j = key.get(tuple(np.round(
                (B[i] + k * Tvec) / tol).astype(np.int64)))
            if j is not None:
                ra, rb = find(bnd[i]), find(j)
                if ra != rb:
                    parent[ra] = rb
    roots = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    Vq = V[uniq]
    Fq = []
    for f in F:
        g = [int(inv[i]) for i in f]
        h = [g[0]]
        for s in g[1:]:
            if s != h[-1]:
                h.append(s)
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            Fq.append(tuple(h))
    return Vq, Fq


def _sptail_grid_uv(nr, nt):
    gu = np.arange(nr) / max(nr - 1, 1)
    gv = np.arange(nt) / max(nt - 1, 1)
    return np.stack(np.meshgrid(gu, gv, indexing='ij'),
                    axis=-1).reshape(-1, 2)


# ---- six-ended Scherk tower (genus 0, 6 ends per period) -----------------

def sptail_six_om(phid):
    """Closed-form data for end angle phid (degrees, 0 < phid < 90):
    rho, a, and the 1-form triple."""
    ph = math.radians(phid)
    rho = math.sqrt((1.0 - math.cos(ph)) / (1.0 + math.cos(ph)))
    a = 4.0 * rho ** 4 / (1.0 + rho ** 2) ** 2

    def om(z):
        z = sptail_nudge(z)
        s_za = np.sqrt(z - a)
        s_z1 = np.sqrt(z - 1.0)
        p1 = rho / z / s_za * s_z1
        p2 = 1.0 / rho / s_za / s_z1
        return (-0.5 * (p1 - p2), 0.5j * (p1 + p2),
                1.0 / (np.sqrt(z) * s_za))
    return rho, a, om


def sptail_six_build(phid=30.0, nu=48, nt=32, storeys=1, rmin=1e-4,
                     rmax=30.0):
    """One 8-frame period block ({E, R_a} x {E, Mx} x {E, My}) per
    storey, translation (0, 4 dy, 0).  Returns (V, F, uv, diag)."""
    rho, a, om = sptail_six_om(phid)
    rE = np.exp(np.linspace(math.log(rmin), math.log(a * 0.8),
                            max(8, int(0.22 * nu))))
    cl = sptail_cluster(0.15 * a, hmin=2e-6 * a)
    rC = np.linspace(a * 1.4, 0.9, max(8, int(0.25 * nu)))
    cl1 = sptail_cluster(0.08, hmin=2e-6)
    rO = np.exp(np.linspace(0.0, math.log(rmax),
                            max(8, int(0.30 * nu))))[1:]
    r = np.unique(np.concatenate(
        [rE, a - cl, [a], a + cl[::-1], rC, 1.0 - cl1, [1.0],
         1.0 + cl1[::-1], rO]))
    r = r[np.concatenate([[True], np.diff(r) > 1e-13])]
    base = 0.5 * math.pi * (1.0 - np.cos(np.pi * np.linspace(
        0, 1, 2 * (nt // 2) + 1)))
    ex = sptail_cluster(0.06, hmin=1e-7, ratio=0.25)
    t = np.unique(np.concatenate([base, ex, math.pi - ex]))
    t = t[np.concatenate([[True], np.diff(t) > 1e-13])]
    X = sptail_polar_patch(om, r, t)
    i_a = int(np.searchsorted(r, a))
    i_1 = int(np.searchsorted(r, 1.0))
    diag = {
        'line_dy': float(np.ptp(X[:i_a + 1, 0, 1])),
        'line_dz': float(np.ptp(X[:i_a + 1, 0, 2])),
        'mx_dx': float(np.ptp(X[i_a:i_1 + 1, 0, 0])),
        'my_dy': float(np.ptp(X[i_1:, 0, 1])),
        'pi_dy': float(np.ptp(X[:, -1, 1]))}
    X = X.copy()
    X[..., 0] -= np.median(X[i_a:i_1 + 1, 0, 0])
    X[..., 1] -= np.median(X[i_1:, 0, 1])
    X[..., 2] -= np.median(X[:i_a + 1, 0, 2])
    ya = float(np.median(X[:i_a + 1, 0, 1]))
    ypi = float(np.median(X[:, -1, 1]))
    dy_theory = -(math.pi / 4) * (1.0 + rho * rho) / rho
    diag['dy_vs_residue'] = abs(ya - dy_theory)
    diag['pi_vs_2dy'] = abs(ypi - 2.0 * ya)
    X[:i_a + 1, 0, 1] = ya
    X[:i_a + 1, 0, 2] = 0.0
    X[i_a:i_1 + 1, 0, 0] = 0.0
    X[i_1:, 0, 1] = 0.0
    X[:, -1, 1] = 2.0 * ya
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(r), len(t))
    T = np.array([0.0, 4.0 * ya, 0.0])
    frames = []
    for e in (0, 1):
        for bx in (0, 1):
            for cy in (0, 1):
                M = np.eye(3)
                tv = np.zeros(3)
                if e:                      # R_a: (x, 2 ya - y, -z)
                    M = M @ np.diag([1.0, -1.0, -1.0])
                    tv = np.array([0.0, 2.0 * ya, 0.0])
                if bx:
                    M = np.diag([-1.0, 1.0, 1.0]) @ M
                    tv = np.array([-1.0, 1.0, 1.0]) * tv
                if cy:
                    M = np.diag([1.0, -1.0, 1.0]) @ M
                    tv = np.array([1.0, -1.0, 1.0]) * tv
                par = (-1.0) ** (e + bx + cy)
                for s in range(storeys):
                    frames.append((M, tv + s * T, par))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(t)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def sptail_six_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                    storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sptail_six_build(
        p['phid'], nu=int(np.clip(int(0.75 * nu), 24, 120)),
        nt=int(np.clip(int(0.55 * nv), 20, 80)), storeys=S,
        rmax=p['rmax'], rmin=1.0 / p['rmax'])
    V = _center_fit(V, scale, V)
    return V, F, uv


# ---- alternating fence of half-catenoids (genus 1, 2 ends) ---------------

def sptail_fencealt_om(a):
    rho = 1.0 / math.sqrt(a)

    def om(z):
        z = sptail_nudge(z)
        p1 = rho * (z + 1.0 / a) ** -0.5 * z ** -1.5 * (z - a) ** 0.5
        p2 = (1.0 / rho) * (z + 1.0 / a) ** 0.5 * z ** -0.5 \
            * (z - a) ** -0.5
        return (0.5 * (p2 - p1), 0.5j * (p2 + p1), 1.0 / z)
    return om


def sptail_fencealt_build(a=2.0, nu=48, nt=32, storeys=1, rmin=0.05):
    """{E, Mx} x {E, My} per storey, translation (0, 2 dy, 0).  The
    period problem is closed by rho = 1/sqrt(a): the two vertical-mirror
    segments (theta = 0 past a, theta = pi past -1/a) land in the SAME
    plane -- measured in diag['period_x']."""
    om = sptail_fencealt_om(a)
    rmax = 1.0 / rmin
    b1, b2 = 1.0 / a, a
    cl1 = sptail_cluster(0.10 * b1, hmin=1e-6 * b1)
    cl2 = sptail_cluster(0.10 * (b2 - b1), hmin=1e-6)
    n3 = max(8, int(0.30 * nu))
    rA = np.exp(np.linspace(math.log(rmin), math.log(b1 * 0.85), n3))
    rB = np.exp(np.linspace(math.log(b1 * 1.15), math.log(b2 * 0.85),
                            n3))
    rC = np.exp(np.linspace(math.log(b2 * 1.15), math.log(rmax), n3))
    r = np.unique(np.concatenate(
        [rA, b1 - cl1, [b1], b1 + cl1[::-1], rB, b2 - cl2, [b2],
         b2 + cl2[::-1], rC]))
    r = r[np.concatenate([[True], np.diff(r) > 1e-13])]
    base = 0.5 * math.pi * (1.0 - np.cos(np.pi * np.linspace(
        0, 1, 2 * (nt // 2) + 1)))
    ex = sptail_cluster(0.06, hmin=1e-7, ratio=0.25)
    t = np.unique(np.concatenate([base, ex, math.pi - ex]))
    t = t[np.concatenate([[True], np.diff(t) > 1e-13])]
    X = sptail_polar_patch(om, r, t)
    i1 = int(np.searchsorted(r, b1))
    i2 = int(np.searchsorted(r, b2))
    xa = float(np.median(X[:i2 + 1, 0, 0]))
    x2 = float(np.median(X[i1:, -1, 0]))
    diag = {
        'xa_ptp': float(np.ptp(X[:i2 + 1, 0, 0])),
        'y0_ptp': float(np.ptp(X[i2:, 0, 1])),
        'y1_ptp': float(np.ptp(X[:i1 + 1, -1, 1])),
        'x2_ptp': float(np.ptp(X[i1:, -1, 0])),
        'period_x': abs(xa - x2)}
    X = X.copy()
    X[..., 0] -= xa
    X[..., 1] -= np.median(X[i2:, 0, 1])
    dy = float(np.median(X[:i1 + 1, -1, 1]))
    X[:i2 + 1, 0, 0] = 0.0
    X[i2:, 0, 1] = 0.0
    X[:i1 + 1, -1, 1] = dy
    X[i1:, -1, 0] = 0.0
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(r), len(t))
    T = np.array([0.0, 2.0 * dy, 0.0])
    frames = []
    for bx in (0, 1):
        for cy in (0, 1):
            M = np.diag([-1.0 if bx else 1.0, -1.0 if cy else 1.0, 1.0])
            par = (-1.0) ** (bx + cy)
            for s in range(storeys):
                frames.append((M, s * T, par))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(t)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def sptail_fencealt_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                         storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sptail_fencealt_build(
        p['a'], nu=int(np.clip(int(0.8 * nu), 24, 130)),
        nt=int(np.clip(int(0.55 * nv), 20, 80)), storeys=S,
        rmin=p['rmin'])
    V = _center_fit(V, scale, V)
    return V, F, uv


# ---- fence of catenoids (measured genus 1, 2 ends per period) ------------

def sptail_fencecat_om(a):
    rho = math.sqrt(a)

    def om(z):
        z = sptail_nudge(z)
        p1 = z ** -0.5 * (z - 1.0) ** -0.5 * (z - a) ** 0.5
        p2 = z ** -1.5 * (z - 1.0) ** 0.5 * (z - a) ** -0.5
        return (-0.5 * (rho * p1 - p2 / rho),
                0.5j * (rho * p1 + p2 / rho), 1.0 / z)
    return om


def sptail_fencecat_build(a=0.2, nu=48, nt=32, storeys=1, r1=6.0):
    """Patch: upper half annulus a/r1 <= |z| <= sqrt(a); boundary =
    y-mirror (0, a), TWO x-mirrors ((a, sqrt a) and (-sqrt a, 0)) whose
    plane gap is half the fence translation, the horizontal mirror arc
    |z| = sqrt(a) (height log(a)/2), and the catenoid end trim at a/r1.
    Frames {E, MxB} x {E, My} x {E, sigma_h} per storey, horizontal
    translation T = (2(xB - xA), 0, 0)."""
    om = sptail_fencecat_om(a)
    sa = math.sqrt(a)
    rlo = a / r1
    cl = sptail_cluster(0.15 * a, hmin=2e-6 * a)
    rA = np.exp(np.linspace(math.log(rlo), math.log(a * 0.85),
                            max(8, int(0.45 * nu))))
    rB = np.linspace(a * 1.15, sa, max(8, int(0.35 * nu)))
    r = np.unique(np.concatenate(
        [rA, a - cl, [a], a + cl[::-1], rB, [sa]]))
    r = r[np.concatenate([[True], np.diff(r) > 1e-13])]
    r = r[r <= sa + 1e-12]
    base = 0.5 * math.pi * (1.0 - np.cos(np.pi * np.linspace(
        0, 1, 2 * (nt // 2) + 1)))
    ex = sptail_cluster(0.06, hmin=1e-7, ratio=0.25)
    t = np.unique(np.concatenate([base, ex, math.pi - ex]))
    t = t[np.concatenate([[True], np.diff(t) > 1e-13])]
    X = sptail_polar_patch(om, r, t)
    i_a = int(np.searchsorted(r, a))
    xA = float(np.median(X[i_a:, 0, 0]))       # (a, sqrt a) mirror
    xB = float(np.median(X[:, -1, 0]))         # (-sqrt a, 0) mirror
    diag = {
        'y_ptp': float(np.ptp(X[:i_a + 1, 0, 1])),
        'xA_ptp': float(np.ptp(X[i_a:, 0, 0])),
        'xB_ptp': float(np.ptp(X[:, -1, 0])),
        'arc_z_ptp': float(np.ptp(X[-1, :, 2]))}
    X = X.copy()
    X[..., 0] -= xA
    X[..., 1] -= np.median(X[:i_a + 1, 0, 1])
    X[..., 2] -= np.median(X[-1, :, 2])
    xB = float(np.median(X[:, -1, 0]))
    X[i_a:, 0, 0] = 0.0
    X[:i_a + 1, 0, 1] = 0.0
    X[:, -1, 0] = xB
    X[-1, :, 2] = 0.0
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(r), len(t))
    T = np.array([2.0 * xB, 0.0, 0.0])
    diag['T'] = T
    frames = []
    for bx in (0, 1):
        for cy in (0, 1):
            for sh in (0, 1):
                M = np.diag([-1.0 if bx else 1.0,
                             -1.0 if cy else 1.0,
                             -1.0 if sh else 1.0])
                tv = np.array([2.0 * xB if bx else 0.0, 0.0, 0.0])
                par = (-1.0) ** (bx + cy + sh)
                for s in range(storeys):
                    frames.append((M, tv + s * T, par))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(t)),
                                 q0, frames, 1e-9 * span)
    diag['span'] = span
    return V, F, uv, diag


def sptail_fencecat_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                         storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 8))
    V, F, uv, _ = sptail_fencecat_build(
        p['a'], nu=int(np.clip(int(0.8 * nu), 24, 130)),
        nt=int(np.clip(int(0.6 * nv), 20, 90)), storeys=S, r1=p['r1'])
    V = _center_fit(V, scale, V)
    return V, F, uv


# ---- helicoidal Karcher-Scherk (twisted saddle tower) --------------------

def sptail_hks_om(k, a, R):
    def om(w):
        p1 = 0.5 * R ** 2 * (R ** 2 - w) ** (-1 + a) \
            * w ** (1 - 1.0 / k) * (1 + R ** 2 * w) ** (-1 - a)
        p2 = -0.5 * R ** 2 * (R ** 2 - w) ** (-1 - a) \
            * w ** (-1 + 1.0 / k) * (1 + R ** 2 * w) ** (-1 + a)
        dh = 0.5j / (w * (1.0 / R ** 2 - R ** 2 - 1.0 / w + w))
        return (-0.5 * (p1 - p2), 0.5j * (p2 + p1), dh)
    return om


def _sptail_hks_patch(k, a, R, nu, nt, xmax):
    om = sptail_hks_om(k, a, R)
    v1 = -math.log(R * R)

    def omz(zeta):
        e = np.exp(zeta)
        w = (e + R ** 2) / (1.0 - e * R ** 2)
        w = sptail_nudge(w)
        dw = e * (1.0 + R ** 4) / (1.0 - e * R ** 2) ** 2
        return tuple(c * dw for c in om(w))

    cl = sptail_cluster(0.25, hmin=3e-6, ratio=0.35)
    nseg = max(6, int(nu / 4))
    xs = [np.linspace(-xmax, -v1 - 0.3, nseg), -v1 - cl, [-v1],
          -v1 + cl[::-1], np.linspace(-v1 + 0.3, -0.02,
                                      max(4, nseg // 2)), [0.0],
          np.linspace(0.02, v1 - 0.3, max(4, nseg // 2)),
          v1 - cl, [v1], v1 + cl[::-1], np.linspace(v1 + 0.3, xmax,
                                                    nseg)]
    x = np.unique(np.concatenate([np.atleast_1d(np.asarray(s, float))
                                  for s in xs]))
    x = x[np.concatenate([[True], np.diff(x) > 1e-12])]
    base = 0.5 * math.pi * (1.0 - np.cos(np.pi * np.linspace(
        0, 1, 2 * (nt // 2) + 1)))
    ex = sptail_cluster(0.05, hmin=1e-6, ratio=0.3)
    y = np.unique(np.concatenate([base, ex, math.pi - ex]))
    y = y[np.concatenate([[True], np.diff(y) > 1e-12])]
    X = sptail_rect_patch(omz, x, y)
    iv1 = int(np.searchsorted(x, -v1))
    ip1 = int(np.searchsorted(x, v1))
    return X, x, y, iv1, ip1


def _sptail_line_fit(S):
    """2-D line through the xy-projection of a snapped boundary row
    (drop the sixth nearest the singular corner)."""
    n = len(S)
    cut = max(2, n // 6)
    P = S[:, :2]
    p = np.mean(P, axis=0)
    _, _, vt = np.linalg.svd(P - p, full_matrices=False)
    return p, vt[0]


def _sptail_isect(p1, u1, p2, u2):
    A = np.array([u1, -u2]).T
    tt = np.linalg.solve(A, p2 - p1)
    return p1 + tt[0] * u1


def sptail_hks_axis_residual(k, a, R, nu=32, nt=24, xmax=4.0):
    """Geometric period residual: the k-fold axis point fixed by the two
    w = 0 boundary lines minus the one fixed by the two w = infinity
    lines (y component; the residual is 1-D)."""
    X, x, y, iv1, ip1 = _sptail_hks_patch(k, a, R, nu, nt, xmax)

    def li(S, from_end):
        n = len(S)
        cut = max(2, n // 6)
        return _sptail_line_fit(S[:-cut] if from_end else S[cut:])
    q1 = _sptail_isect(*li(X[:iv1 + 1, -1, :], True),
                       *li(X[iv1:, -1, :], False))
    q2 = _sptail_isect(*li(X[:ip1 + 1, 0, :], True),
                       *li(X[ip1:, 0, :], False))
    return q2 - q1


_SPTAIL_HKS_R = {}


def sptail_hks_solveR(k, a):
    """Bisection on the geometric closure residual; cached."""
    key = (int(k), round(float(a), 8))
    if key in _SPTAIL_HKS_R:
        return _SPTAIL_HKS_R[key]
    lo, hi = 0.45, 0.985
    flo = sptail_hks_axis_residual(k, a, lo)[1]
    fhi = sptail_hks_axis_residual(k, a, hi)[1]
    if flo * fhi > 0:
        raise ValueError(f"hks solve: no bracket for k={k} a={a}")
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        fm = sptail_hks_axis_residual(k, a, mid)[1]
        if flo * fm < 0:
            hi = mid
        else:
            lo, flo = mid, fm
        if hi - lo < 1e-12:
            break
    R = 0.5 * (lo + hi)
    _SPTAIL_HKS_R[key] = R
    return R


def sptail_hks_build(k=4, a=0.1, nu=48, nt=36, storeys=1, xmax=4.0,
                     R=None):
    """{E, R_A} x k rotations x storey screws (twist -2 a pi, rise
    trans per period).  Returns (V, F, uv, diag)."""
    if R is None:
        R = sptail_hks_solveR(k, a)
    X, x, y, iv1, ip1 = _sptail_hks_patch(k, a, R, nu, nt, xmax)

    def li(S, from_end):
        n = len(S)
        cut = max(2, n // 6)
        return _sptail_line_fit(S[:-cut] if from_end else S[cut:])
    pA, uA = li(X[:iv1 + 1, -1, :], True)
    q1 = _sptail_isect(pA, uA, *li(X[iv1:, -1, :], False))
    q2 = _sptail_isect(*li(X[:ip1 + 1, 0, :], True),
                       *li(X[ip1:, 0, :], False))
    diag = {'axis_misalign': float(np.linalg.norm(q1 - q2)), 'R': R}
    zA = float(X[0, -1, 2])
    X = X.copy()
    X[..., 0] -= q1[0]
    X[..., 1] -= q1[1]
    X[..., 2] -= zA
    alA = math.atan2(uA[1], uA[0])
    ca, sa_ = math.cos(-alA), math.sin(-alA)
    X[..., :2] = X[..., :2] @ np.array([[ca, -sa_], [sa_, ca]]).T

    def snap_line(rows, beta, zval):
        u = np.array([math.cos(beta), math.sin(beta)])
        d = rows[:, :2] @ u
        rows[:, 0] = d * u[0]
        rows[:, 1] = d * u[1]
        rows[:, 2] = zval
    h = float(np.median(X[:ip1 + 1, 0, 2]))
    snap_line(X[:iv1 + 1, -1, :], 0.0, 0.0)
    snap_line(X[iv1:, -1, :], -math.pi / k, 0.0)
    snap_line(X[:ip1 + 1, 0, :], -a * math.pi, h)
    snap_line(X[ip1:, 0, :], -math.pi / k - a * math.pi, h)
    trans = 2.0 * h
    diag['trans'] = trans
    diag['trans_vs_theory'] = abs(abs(trans)
                                  - math.pi * R * R / (1.0 + R ** 4))
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(x), len(y))
    RA = np.diag([1.0, -1.0, -1.0])
    frames = []
    for j in range(storeys):
        for i in range(k):
            phi = -2.0 * a * math.pi * j - TAU * i / k
            c, s = math.cos(phi), math.sin(phi)
            Mz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            for e in (0, 1):
                M = Mz @ (RA if e else np.eye(3))
                frames.append((M, np.array([0.0, 0.0, j * trans]),
                               (-1.0) ** e))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(x), len(y)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = np.array([0.0, 0.0, trans])
    diag['span'] = span
    return V, F, uv, diag


def sptail_hks_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                    storeys=1):
    """The associate-angle knob is the TWIST modulus (like the alpha
    saddle tower): twist fraction f = theta/(pi/2) in [0, 1] maps to
    the exponent a = (0.12 + 0.78 f)/k; theta = 0 keeps a gentle
    default twist."""
    p = spec['p_from'](order, radius)
    k = p['k']
    f = float(np.clip(theta / (0.5 * math.pi), 0.0, 1.0))
    a = (0.12 + 0.78 * f) / k
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sptail_hks_build(
        k, a, nu=int(np.clip(int(0.7 * nu), 24, 110)),
        nt=int(np.clip(int(0.55 * nv), 18, 80)), storeys=S,
        xmax=p['xmax'])
    V = _center_fit(V, scale, V)
    return V, F, uv


# ---- translation-invariant Enneper with three annular ends ---------------

_SPTAIL_SQ2 = math.sqrt(2.0)
_SPTAIL_E3A_Y1 = math.pi / (2.0 * _SPTAIL_SQ2)


def sptail_e3a_f(z):
    """Closed-form immersion: g = (1/sqrt2)(1-z^2)/z, dh = dz."""
    z = z + 1e-13j * (1.0 + np.abs(z))     # interior branch limit
    l1 = np.log(z)
    l2 = np.log(1.0 - z * z)
    x = np.real((z * z - 2.0 * l1 - 2.0 * l2) / (4.0 * _SPTAIL_SQ2))
    y = np.imag(z * z - 2.0 * l1 + 2.0 * l2) / (4.0 * _SPTAIL_SQ2)
    return np.stack(np.broadcast_arrays(x, y, np.real(z)), axis=-1)


def sptail_e3a_build(nu=56, nt=44, storeys=1, rmin=0.03, Rt=2.2,
                     eps=0.10):
    """Quarter-disk chart; {E, R_L} x {E, My} per storey, translation
    (0, -pi/sqrt2, 0).  R_L is the 2-fold rotation about the straight
    line {y = -Y1/2, z = 0} (the imaginary-axis image); the mirrors
    y = 0 / y = -Y1 sit on the real segments (0,1) / (1, Rt).  Genus 0
    with 1 Enneper + 3 annular ends per period (measured; see the
    block header note on the harvest's genus-1 annotation)."""
    Y1 = _SPTAIL_E3A_Y1
    cl = sptail_cluster(0.15, hmin=2e-3, ratio=0.45)
    r1 = np.exp(np.linspace(math.log(rmin), math.log(1.0 - cl[0] - 0.05),
                            max(8, int(0.4 * nu))))
    r2 = np.exp(np.linspace(math.log(1.0 + cl[0] + 0.05), math.log(Rt),
                            max(8, int(0.35 * nu))))
    r = np.unique(np.concatenate([r1, 1.0 - cl, [1.0], 1.0 + cl[::-1],
                                  r2]))
    th = 0.5 * math.pi * (0.5 - 0.5 * np.cos(np.pi * np.linspace(
        0, 1, nt)))
    R_, T_ = np.meshgrid(r, th, indexing='ij')
    Z = R_ * np.exp(1j * T_)
    with np.errstate(divide='ignore', invalid='ignore'):
        X = sptail_e3a_f(Z)
    mask = (np.abs(Z - 1.0) > eps) & np.isfinite(X).all(axis=-1)
    i1 = int(np.searchsorted(r, 1.0))
    diag = {
        'y01_ptp': float(np.ptp(X[:i1, 0, 1][mask[:i1, 0]])),
        'y1R_vs_const': abs(float(np.median(
            X[i1:, 0, 1][mask[i1:, 0]])) + Y1),
        'line_y_vs_const': abs(float(np.median(X[:, -1, 1])) + 0.5 * Y1),
        'line_z_ptp': float(np.ptp(X[:, -1, 2]))}
    X[:, -1, 1] = -0.5 * Y1
    X[:, -1, 2] = 0.0
    X[:i1, 0, 1] = 0.0
    X[i1:, 0, 1] = -Y1
    X[~mask] = 0.0
    V0 = X.reshape(-1, 3)
    vm = mask.reshape(-1)
    q0 = sptail_grid_quads(len(r), len(th), vm)
    T = np.array([0.0, -2.0 * Y1, 0.0])
    frames = []
    for e in (0, 1):
        for b in (0, 1):
            M = np.eye(3)
            tv = np.zeros(3)
            if e:                          # R_L: (x, -Y1 - y, -z)
                M = M @ np.diag([1.0, -1.0, -1.0])
                tv = np.array([0.0, -Y1, 0.0])
            if b:                          # My (y = 0)
                M = np.diag([1.0, -1.0, 1.0]) @ M
                tv = np.array([1.0, -1.0, 1.0]) * tv
            par = (-1.0) ** (e + b)
            for s in range(storeys):
                frames.append((M, tv + s * T, par))
    vu = V0[vm]
    span = float(np.linalg.norm(vu.max(0) - vu.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(th)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def sptail_e3a_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                    storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sptail_e3a_build(
        nu=int(np.clip(nu, 30, 140)), nt=int(np.clip(int(0.8 * nv),
                                                     24, 110)),
        storeys=S, Rt=p['Rt'])
    V = _smooth_boundary(V, F, iters=4)
    V = _center_fit(V, scale, V)
    return V, F, uv


# ---- periodic (translation-invariant) Enneper ----------------------------

def sptail_penneper_X(u, v):
    """Closed-form immersion on the log chart z = e^(u+iv):
    X1 = u/2 - e^(2u) cos(2v)/4, X2 = -v/2 - e^(2u) sin(2v)/4,
    X3 = e^u cos v.  One turn v -> v + 2 pi is the exact translation
    (0, -pi, 0)."""
    e2 = np.exp(2.0 * u)
    return np.stack(np.broadcast_arrays(
        0.5 * u - 0.25 * e2 * np.cos(2.0 * v),
        -0.5 * v - 0.25 * e2 * np.sin(2.0 * v),
        np.exp(u) * np.cos(v)), axis=-1)


def sptail_penneper_build(nu=64, nt=48, storeys=2, rmin=0.05,
                          rmax=1.3):
    """`storeys` full turns of the universal cover -- the mesh is one
    continuous grid (each turn joins the next exactly)."""
    S = max(1, int(storeys))
    u = np.linspace(math.log(rmin), math.log(rmax), nu)
    ntt = max(24, nt)                      # columns per turn (exact)
    nv = S * ntt + 1
    v = np.linspace(-S * math.pi, S * math.pi, nv)
    U, V_ = np.meshgrid(u, v, indexing='ij')
    X = sptail_penneper_X(U, V_)
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(nu, nv)
    uv = _sptail_grid_uv(nu, nv)
    return V0, q0, uv, {'T': np.array([0.0, -math.pi, 0.0]),
                        'ncol_per_turn': int(round(
                            TAU / (v[1] - v[0])))}


def sptail_penneper_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                         storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 5))
    V, F, uv, _ = sptail_penneper_build(
        nu=int(np.clip(nu, 24, 160)),
        nt=int(np.clip(int(0.9 * nv), 24, 100)),
        storeys=S, rmax=p['rmax'])
    V = _center_fit(V, scale, V)
    return V, F, uv


# ==========================================================================
# SYMM/NONORIENT TAIL -- one-sided (non-orientable) quotient meshers and
# the antiprismatic k-noid period solver
# ==========================================================================
# Engine for the catalog rows appended at the end of zoo:
#
#   * symtail_crosscap_mesh -- generic one-sided quotient of a disk /
#     annulus Weierstrass domain under the free antipodal involution
#     z -> -1/conj(z).  The fundamental domain is the unit disk (or its
#     exterior); on the identification rim |z| = 1 the involution acts as
#     v -> v + pi, so welding each rim vertex to its antipode turns the
#     grid into a GENUINELY one-sided mesh (a cross-cap): Henneberg's
#     surface (Mobius strip = projective plane minus the end) and
#     Kusner's projective planes with p planar ends.  The rim identity
#     X(-1/conj z) = X(z) is verified numerically before welding and the
#     build refuses to produce a fake quotient if it fails.
#   * symtail_lopez_klein_mesh -- F. J. Lopez's one-ended minimal Klein
#     bottle (total curvature -8 pi), assembled from ONE conformal patch
#     (the upper-half annulus 1 <= |x| <= rmax of the orientation double
#     cover) and its orbit under the two straight lines contained in the
#     surface (the x- and y-axis 180-degree rotations); the Klein deck
#     identification glues the |x| = 1 rim of the patch to the rim of
#     its x-axis-rotated copy.
#   * symtail_antiprism_constants -- numeric Lopez-Ros period solve for
#     the antiprismatic k-noid family (branch constant a and scale rho
#     per (nn, b)), generalizing the single harvested nn = 5 member.
#
# References:
#   L. Henneberg, "Ueber solche Minimalflaechen, welche eine vorge-
#     schriebene ebene Curve zur geodaetischen Linie haben" (Diss. ETH
#     Zurich, 1875) -- the classical one-sided minimal surface;
#     U. Dierkes, S. Hildebrandt, F. Sauvigny, "Minimal Surfaces"
#     (2010), sec. 3.5, for the modern Weierstrass form g = z,
#     dh = 2 z (1 - z^-4) dz and the antipodal identification.
#   R. Kusner, "Conformal geometry and complete minimal surfaces",
#     Bull. Amer. Math. Soc. 17 (1987) 291-295 -- the dihedrally
#     symmetric projective planes with p planar ends; data
#     G = rho z^(p-1)(z^p - sqrt(2p-1))/(sqrt(2p-1) z^p + 1) after
#     M. Weber, minimalsurfaces.blog (Kusner notebook).
#   F. J. Lopez, "A complete minimal Klein bottle in R^3", Duke Math.
#     J. 71 (1993) 23-30; Weierstrass data (branch value
#     a = 2.5447026679682394, G = sqrt(a)(x+1) sqrt(x) sqrt(x-1/a)
#     sqrt(x+a) / ((x-1)(x+a)), dh = i sqrt(a)(x^2-1)/x^2 dx) after
#     M. Weber, minimalsurfaces.blog (KleinBottle notebook).
#   L. P. Jorge, W. H. Meeks III, Topology 22 (1983) (k-noids);
#     H. Karcher, "Construction of minimal surfaces" (1989)
#     (symmetrization); antiprismatic data after M. Weber,
#     minimalsurfaces.blog (Antiprismatic k-Noids notebook).


def symtail_edge_stats(V, faces):
    """Edge-use map -> (chi, nonman, boundary loop count, one_sided).
    one_sided is measured by orientation propagation across interior
    edges: a surface is non-orientable iff assigning consistent face
    orientations meets a contradiction."""
    from collections import defaultdict
    ec = defaultdict(int)
    e2f = defaultdict(list)
    for fi, f in enumerate(faces):
        m = len(f)
        for t in range(m):
            a, b = f[t], f[(t + 1) % m]
            e = (a, b) if a < b else (b, a)
            ec[e] += 1
            e2f[e].append((fi, a < b))
    chi = len(V) - len(ec) + len(faces)
    nonman = sum(1 for c in ec.values() if c > 2)
    bed = [e for e, c in ec.items() if c == 1]
    par = {}

    def bfind(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in bed:
        ra, rb = bfind(a), bfind(b)
        if ra != rb:
            par[ra] = rb
    nloops = len({bfind(a) for a, b in bed})
    sign = {}
    one_sided = False
    for f0 in range(len(faces)):
        if f0 in sign:
            continue
        sign[f0] = 1
        stack = [f0]
        while stack:
            fi = stack.pop()
            f = faces[fi]
            m = len(f)
            for t in range(m):
                a, b = f[t], f[(t + 1) % m]
                e = (a, b) if a < b else (b, a)
                pair = e2f[e]
                if len(pair) != 2:
                    continue
                for fj, fwd in pair:
                    if fj == fi:
                        continue
                    s = sign[fi] if (fwd != (a < b)) else -sign[fi]
                    if fj in sign:
                        if sign[fj] != s:
                            one_sided = True
                    else:
                        sign[fj] = s
                        stack.append(fj)
    return chi, nonman, nloops, one_sided


def _symtail_disk_grid(spec, p, nu, nv, theta):
    """Pole-avoiding Weierstrass integration on the unit disk for the
    crosscap rows: the row's ends form a ring of double poles strictly
    inside the disk, and radial rays that graze a pole cannot carry an
    accurate value to the rim -- so nothing is ever integrated through
    the ring.  Three anchored-clean stages instead:
      1. every ray integrates outward from the shared center up to the
         ring (clean -- bounded distance to every pole),
      2. ONE anchor ray, at the angle farthest from all poles,
         continues to the rim, and the whole rim row follows by
         cumulative arc integration along |z| = 1 (clean),
      3. every ray integrates INWARD from its exact rim value down to
         the ring (clean).
    The two determinations only meet inside the ring band, where the
    puncture mask discards the cells anyway."""
    d = spec['domain']
    r1 = float(_ev(d[2], p))
    dth = TAU / nv
    v = 0.5 * dth + np.arange(nv) * dth         # offset: no ray on a pole
    punct = _ev(spec['mask_punctures'], p)
    zc_a = np.array([zc for zc, _ in punct])
    r_ring = float(np.median(np.abs(zc_a)))
    # radial grid clustered around the ring (fine puncture rims)
    w = min(0.5 * min(r_ring, r1 - r_ring), 0.08)
    lo, hi = r_ring - w, r_ring + w
    n_band = max(8, int(0.35 * nu))
    n_rest = nu - n_band
    n_in = max(4, int(n_rest * lo / max(lo + r1 - hi, 1e-9)))
    n_out = max(4, n_rest - n_in)
    u = np.concatenate([
        np.linspace(0.0, lo, n_in, endpoint=False),
        lo + (hi - lo) * (0.5 - 0.5 * np.cos(
            math.pi * np.linspace(0.0, 1.0, n_band, endpoint=False))),
        np.linspace(hi, r1, n_out)])
    nu = len(u)
    R, TH = np.meshgrid(u, v, indexing='ij')
    z = R * np.exp(1j * TH)
    z[0, :] = 1e-3 * np.exp(1j * v)
    phi = _phi_fn(spec, p, theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = phi(z)
    Xr = np.real(F * np.exp(1j * TH)[..., None])
    Xr = np.where(np.isfinite(Xr), Xr, 0.0)
    dr = np.diff(R, axis=0)[..., None]
    inc = 0.5 * (Xr[1:] + Xr[:-1]) * dr
    X_in = np.concatenate(
        [np.zeros((1, nv, 3)), np.cumsum(inc, axis=0)], axis=0)
    # anchor ray: the grid angle farthest from every pole angle
    pang = np.angle(zc_a)
    dmin = np.min(np.abs(((v[:, None] - pang[None, :]) + math.pi)
                         % TAU - math.pi), axis=1)
    ja = int(np.argmax(dmin))
    t_fine = np.linspace(1e-3, r1, 4001)
    za = t_fine * np.exp(1j * v[ja])
    with np.errstate(divide='ignore', invalid='ignore'):
        Fa = phi(za) * np.exp(1j * v[ja])
    Fa = np.where(np.isfinite(Fa), Fa, 0.0)
    Xa_rim = np.sum(0.5 * (Fa[1:] + Fa[:-1]).real
                    * np.diff(t_fine)[:, None], axis=0)
    # rim row by cumulative Gauss-Legendre arcs from the anchor
    gx, gw = np.polynomial.legendre.leggauss(8)
    rim = np.zeros((nv, 3))
    rim[ja] = Xa_rim
    order_j = [(ja + k) % nv for k in range(nv)]
    for k in range(1, nv):
        j0, j1 = order_j[k - 1], order_j[k]
        t0 = v[j0]
        t1 = t0 + dth
        tg = 0.5 * (t0 + t1) + 0.5 * dth * gx
        zg = r1 * np.exp(1j * tg)
        dz = 1j * zg * 0.5 * dth * gw
        with np.errstate(divide='ignore', invalid='ignore'):
            Fg = phi(zg)
        rim[j1] = rim[j0] + np.real(np.sum(Fg * dz[:, None], axis=0))
    # inward from the rim for the outer rows
    X_out = np.zeros_like(X_in)
    X_out[-1] = rim
    for i in range(nu - 2, -1, -1):
        X_out[i] = X_out[i + 1] - inc[i]
    isplit = int(np.searchsorted(u, r_ring))
    X = np.where((np.arange(nu) < isplit)[:, None, None], X_in, X_out)
    mask = np.ones(z.shape, dtype=bool)
    for zc, rho in punct:
        mask &= np.abs(z - zc) > rho
    return X, mask


def symtail_crosscap_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """One-sided quotient mesh of a WE disk/annulus row under the free
    antipodal involution z -> -1/conj(z): sample the fundamental domain
    (|z| <= 1 for 'outer' rim, |z| >= 1 for 'inner'), verify the rim
    identity X(z) = X(-z) on |z| = 1, then weld each rim vertex to its
    antipode (v ~ v + pi) -- the cross-cap gluing.  Refuses to build if
    the measured rim identity fails (no fake quotients)."""
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    if 'solve' in spec:
        p = spec['solve'](p) or p
    rb = spec.get('res_boost')
    if rb:
        rb = rb(order) if callable(rb) else rb
        nu = max(3, int(round(nu * rb[0])))
        nv = max(3, int(round(nv * rb[1])))
    nv += nv % 2                                # even: antipode on-grid
    if 'Xexact' in spec:
        x, y, z, _, _, tail = _we_disk(spec, p, nu, nv, theta)
        X = np.stack([x, y, z], axis=-1)
        mask = tail if isinstance(tail, np.ndarray) \
            else np.ones((nu, nv), dtype=bool)
    else:
        X, mask = _symtail_disk_grid(spec, p, nu, nv, theta)
    nu = X.shape[0]                             # grid may have resized
    irow = 0 if spec.get('crosscap_rim', 'outer') == 'inner' else nu - 1
    h = nv // 2
    rim = X[irow]
    anti = np.roll(rim, -h, axis=0)
    mis = float(np.linalg.norm(rim - anti, axis=1).max())
    flat = X.reshape(-1, 3)
    diag = float(np.linalg.norm(flat.max(0) - flat.min(0)))
    if not np.isfinite(mis) or mis > 0.02 * diag:
        raise ValueError(
            f"crosscap: antipodal rim identity failed (mismatch "
            f"{mis:.3e} vs diagonal {diag:.3e}) -- not one-sided")
    X[irow] = 0.5 * (rim + anti)                # snap the identity exact
    # vertex ids with the rim's antipodal halves identified, and the
    # disk center (a single point sampled as a whole ring) welded
    vid = np.arange(nu * nv).reshape(nu, nv)
    vid[irow, h:] = vid[irow, :h]
    d0 = spec['domain']
    if irow != 0 and float(_ev(d0[1], p)) < 2e-3:
        vid[0, :] = vid[0, 0]
        X[0, :] = X[0, 0]
        mask[0, :] = True
    gu = np.arange(nu) / max(nu - 1, 1)
    gv = np.arange(nv) / nv
    UVg = np.stack(np.meshgrid(gu, gv, indexing='ij'),
                   axis=-1).reshape(-1, 2)
    vm = mask.reshape(-1)
    quads = []
    for i in range(nu - 1):
        for j in range(nv):
            j2 = (j + 1) % nv
            f = (vid[i, j], vid[i + 1, j], vid[i + 1, j2], vid[i, j2])
            if not (vm[f[0]] and vm[f[1]] and vm[f[2]] and vm[f[3]]):
                continue
            g = [int(f[0])]                     # collapse repeats -> tri
            for t in range(1, 4):
                if int(f[t]) != g[-1]:
                    g.append(int(f[t]))
            if len(g) > 3 and g[0] == g[-1]:
                g.pop()                         # wrap-around repeat
            if len(g) >= 3 and g[0] != g[-1] and len(set(g)) == len(g):
                quads.append(tuple(g))
    V = X.reshape(-1, 3)
    used = np.unique(np.fromiter((i for f in quads for i in f),
                                 dtype=np.int64))
    # optional object-space percentile clip (planar-end flares)
    pct = spec.get('crosscap_clip')
    if pct:
        cen = np.median(V[used], axis=0)
        rad = np.linalg.norm(V - cen, axis=1)
        thr = float(np.percentile(rad[used], pct))
        quads = [f for f in quads if all(rad[i] <= thr for i in f)]
        used = np.unique(np.fromiter((i for f in quads for i in f),
                                     dtype=np.int64))
    remap = np.full(nu * nv, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    V = V[used]
    UVg = UVg[used]
    quads = [tuple(int(remap[i]) for i in f) for f in quads]
    Vu, quads = _largest_component(np.hstack([V, UVg]), quads)
    V, UVg = Vu[:, :3], Vu[:, 3:]
    V = _smooth_boundary(V, quads)
    V = _center_fit(V, scale, V)
    return V, quads, UVg


# --- Lopez minimal Klein bottle -------------------------------------------

SYMTAIL_LOPEZ_A = 2.5447026679682394            # Lopez's branch value


def _symtail_lopez_om1(x):
    """Closed-form antiderivative of phi1 (the x coordinate)."""
    a = SYMTAIL_LOPEZ_A
    return 1j * np.sqrt(x - 1.0 / a) * np.sqrt(a + x) \
        * (a + 2.0 * (a - 2.0) * (a - 1.0) * x - a * x * x) \
        / (3.0 * x ** 1.5)


def _symtail_lopez_om3(x):
    """Closed-form antiderivative of phi3 = dh (the z coordinate)."""
    return 1j * math.sqrt(SYMTAIL_LOPEZ_A) * (x + 1.0 / x)


def _symtail_lopez_phi2(x):
    """phi2 = (i/2)(1/G + G) dh -- integrated numerically for the y
    coordinate (its antiderivative needs incomplete elliptic
    integrals; Lopez 1993, Weber's notebook).  Written in cancelled
    form: the (x - 1) zero of dh cancels G's pole at x = 1, so the
    grid corner x = 1 stays finite (a 0/0 there would poison the
    cumulative integration)."""
    a = SYMTAIL_LOPEZ_A
    q1 = np.sqrt(x) * np.sqrt(x - 1.0 / a)
    q2 = np.sqrt(x + a)
    return -0.5 / (x * x) * ((x - 1.0) ** 2 * q2 / q1
                             + a * (x + 1.0) ** 2 * q1 / q2)


def _symtail_lopez_rim_y(nsub=1600):
    """y along the rim |x| = 1 relative to y(x=1), by cumulative
    Gauss-Legendre arcs (the rim is free of singularities)."""
    vv = np.linspace(0.0, math.pi, nsub + 1)
    tt, wt = np.polynomial.legendre.leggauss(8)
    y = np.zeros(nsub + 1)
    acc = 0.0
    for i in range(nsub):
        t0, t1 = vv[i], vv[i + 1]
        t = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * tt
        zc = np.exp(1j * t)
        dz = 1j * zc * 0.5 * (t1 - t0) * wt
        acc += float(np.real(np.sum(_symtail_lopez_phi2(zc) * dz)))
        y[i + 1] = acc
    return vv, y


_SYMTAIL_LOPEZ_RIM = None


def symtail_lopez_vgrid(nv):
    """Rim-symmetric angular grid: v_j such that the rim involution
    (y -> -y at equal om1, the Klein deck on |x| = 1) maps the grid to
    itself as v_j <-> v_{nv-1-j}.  Built by sampling the rim y-values
    symmetrically about y = y_mid."""
    global _SYMTAIL_LOPEZ_RIM
    if _SYMTAIL_LOPEZ_RIM is None:
        _SYMTAIL_LOPEZ_RIM = _symtail_lopez_rim_y()
    vv, y = _SYMTAIL_LOPEZ_RIM
    if not np.all(np.diff(y) > 0):
        raise ValueError("lopez: rim y not monotone")
    ymid = 0.5 * (y[0] + y[-1])
    yc = y - ymid                               # symmetric range [-Y, Y]
    Y = yc[-1]
    # cosine-graded symmetric targets (denser near the axis endpoints)
    s = np.linspace(0.0, 1.0, nv)
    tgt = -Y * np.cos(math.pi * s)
    return np.interp(tgt, yc, vv)


def symtail_lopez_klein_mesh(spec, nu, nv, order, radius, scale,
                             theta=0.0):
    """Lopez's one-ended minimal Klein bottle (Lopez 1993): ONE
    conformal patch (upper half annulus 1 <= |x| <= rmax) with exact
    closed-form x/z coordinates and a numerically integrated y, tiled
    by the 180-degree rotations about the x- and y-axes (both are
    straight lines contained in the surface) and welded along the
    shared axis segments and the |x| = 1 Klein-deck rim.  The end at
    x = infinity is trimmed at |x| = rmax (the single boundary loop).
    The mesh is measurably one-sided (see the self-tests)."""
    A = SYMTAIL_LOPEZ_A
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    rmax = float(p.get('rmax', 3.0))
    rb = spec.get('res_boost')
    if rb:
        nu = max(3, int(round(nu * rb[0])))
        nv = max(3, int(round(nv * rb[1])))
    nv += 1 - (nv % 2)                          # odd: y = ymid on-grid
    # u-grid: log-radial with an exact node at the branch point |x| = a
    u = np.linspace(0.0, math.log(rmax), nu)
    iA = int(np.argmin(np.abs(u - math.log(A))))
    iA = min(max(iA, 1), nu - 2)
    u[iA] = math.log(A)
    v = symtail_lopez_vgrid(nv)
    U, V = np.meshgrid(u, v, indexing='ij')
    Xc = np.exp(U + 1j * V)
    with np.errstate(divide='ignore', invalid='ignore'):
        x1 = np.real(_symtail_lopez_om1(Xc))
        x3 = np.real(_symtail_lopez_om3(Xc))
        F2 = _symtail_lopez_phi2(Xc)
    # y: cumulative base row (v = v0, just above the real axis) plus
    # cumulative columns (dz = i x dv on each column)
    Fb = F2[:, 0] * Xc[:, 0]
    yb = np.zeros(nu)
    yb[1:] = np.cumsum(0.5 * np.real(Fb[1:] + Fb[:-1]) * np.diff(u))
    Fc = F2 * Xc * 1j
    dv = np.diff(v)[None, :]
    y = np.zeros((nu, nv))
    y[:, 1:] = np.cumsum(0.5 * np.real(Fc[:, 1:] + Fc[:, :-1]) * dv,
                         axis=1)
    y += yb[:, None]
    y = np.where(np.isfinite(y), y, 0.0)
    # anchor: the theta = pi edge beyond the branch point -a is a
    # straight line ON THE X-AXIS (Lopez's normalization f(-a) = 0);
    # its columns are regular, so their median y-value is the anchor
    far = np.arange(iA + 1, nu)
    y -= float(np.median(y[far, -1]))
    X = np.stack([x1, y, x3], axis=-1)
    # snap the symmetry sets exactly:
    #  * v -> 0 edge (x real in [1, rmax]) lies on the y-axis;
    #  * v -> pi edge, |x| <= a, lies on the y-axis;
    #  * v -> pi edge, |x| >= a, lies on the x-axis;
    #  * the rim |x| = 1 lies in the z = 0 plane, and the deck
    #    involution pairs rim vertex j with nv-1-j (mirror in y).
    X[:, 0, 0] = 0.0
    X[:, 0, 2] = 0.0
    X[:iA + 1, -1, 0] = 0.0
    X[:iA + 1, -1, 2] = 0.0
    X[iA:, -1, 1] = 0.0
    X[iA:, -1, 2] = 0.0
    rim = X[0]
    mir = rim[::-1] * np.array([1.0, -1.0, 1.0])
    mis = float(np.linalg.norm(rim - mir, axis=1).max())
    diag = float(np.linalg.norm(X.reshape(-1, 3).max(0)
                                - X.reshape(-1, 3).min(0)))
    if not np.isfinite(mis) or mis > 0.02 * diag:
        raise ValueError(
            f"lopez: Klein rim identity failed ({mis:.3e} vs "
            f"{diag:.3e})")
    X[0] = 0.5 * (rim + mir)
    X[0, :, 2] = 0.0
    # tile: I, Rx(pi), Ry(pi), Rz(pi)
    M1 = np.diag([1.0, -1.0, -1.0])
    M2 = np.diag([-1.0, 1.0, -1.0])
    M3 = np.diag([-1.0, -1.0, 1.0])
    gu = np.arange(nu) / max(nu - 1, 1)
    gv = np.arange(nv) / max(nv - 1, 1)
    uv0 = np.stack(np.meshgrid(gu, gv, indexing='ij'),
                   axis=-1).reshape(-1, 2)
    V0 = X.reshape(-1, 3)
    q0 = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            q0.append((i * nv + j, (i + 1) * nv + j,
                       (i + 1) * nv + j + 1, i * nv + j + 1))
    Vs, Fs, base = [], [], 0
    for M in (np.eye(3), M1, M2, M3):
        Vs.append(V0 @ M.T)
        for f in q0:
            Fs.append(tuple(i + base for i in f))
        base += len(V0)
    Vc = np.concatenate(Vs, axis=0)
    UVc = np.concatenate([uv0] * 4, axis=0)
    # weld ONLY the intended seam pairs (explicit index pairing, not a
    # geometric weld: the immersed Klein bottle SELF-INTERSECTS along
    # the axis lines, where a proximity weld would fuse crossing sheets
    # into non-manifold junk).  Seams, per the surface's symmetry
    # group {I, Rx, Ry, Rz} and the Klein-deck rim gluing:
    #   * theta->0 edge (on the y-axis):        I~Ry, Rx~Rz, same index;
    #   * theta->pi edge, |x| <= a (y-axis):    I~Ry, Rx~Rz, same index;
    #   * theta->pi edge, |x| >= a (x-axis):    I~Rx, Ry~Rz, same index;
    #   * rim |x| = 1 (Klein deck):             I~Rx, Ry~Rz, j <-> nv-1-j.
    N = nu * nv

    def cid(c, i, j):
        return c * N + i * nv + j
    pairs = []
    for i in range(nu):
        pairs.append((cid(0, i, 0), cid(2, i, 0)))
        pairs.append((cid(1, i, 0), cid(3, i, 0)))
    for i in range(iA + 1):
        pairs.append((cid(0, i, nv - 1), cid(2, i, nv - 1)))
        pairs.append((cid(1, i, nv - 1), cid(3, i, nv - 1)))
    for i in range(iA, nu):
        pairs.append((cid(0, i, nv - 1), cid(1, i, nv - 1)))
        pairs.append((cid(2, i, nv - 1), cid(3, i, nv - 1)))
    for j in range(nv):
        pairs.append((cid(0, 0, j), cid(1, 0, nv - 1 - j)))
        pairs.append((cid(2, 0, j), cid(3, 0, nv - 1 - j)))
    pmax = max(float(np.linalg.norm(Vc[a2] - Vc[b2]))
               for a2, b2 in pairs)
    if pmax > 1e-6 * diag:
        raise ValueError(f"lopez: seam pairing failed ({pmax:.3e} vs "
                         f"diagonal {diag:.3e})")
    parent = np.arange(4 * N)

    def find(a2):
        while parent[a2] != a2:
            parent[a2] = parent[parent[a2]]
            a2 = parent[a2]
        return a2
    for a2, b2 in pairs:
        ra, rb2 = find(a2), find(b2)
        if ra != rb2:
            parent[ra] = rb2
    roots = np.array([find(a2) for a2 in range(4 * N)])
    _, inv = np.unique(roots, return_inverse=True)
    inv = inv.ravel()
    Vw = np.zeros((int(inv.max()) + 1, 3))
    Vw[inv] = Vc
    UVw = np.zeros((len(Vw), 2))
    UVw[inv] = UVc
    flist = []
    for f in Fs:
        g = [int(inv[f[0]])]
        for t in range(1, 4):
            if int(inv[f[t]]) != g[-1]:
                g.append(int(inv[f[t]]))
        if len(g) > 3 and g[0] == g[-1]:
            g.pop()
        if len(g) >= 3 and g[0] != g[-1] and len(set(g)) == len(g):
            flist.append(tuple(g))
    used = np.unique(np.array([i for f in flist for i in f],
                              dtype=np.int64))
    remap = np.full(len(Vw), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    Vf = Vw[used]
    UVf = UVw[used]
    quads = [tuple(int(remap[i]) for i in f) for f in flist]
    Vu, quads = _largest_component(np.hstack([Vf, UVf]), quads)
    Vf, UVf = Vu[:, :3], Vu[:, 3:]
    Vf = _smooth_boundary(Vf, quads, iters=4)
    Vf = _center_fit(Vf, scale, Vf)
    return Vf, quads, UVf


# --- antiprismatic k-noid period solver -----------------------------------

def _symtail_ap_g0_dh(nn, b, a):
    def g0(z):
        zn = z ** nn
        return z ** (nn - 1) * (zn + a ** (-nn)) / (zn - a ** nn)

    def dh(z):
        zn = z ** nn
        return z ** (nn - 1) * (zn - a ** nn) * (zn + a ** (-nn)) \
            / ((zn - b ** nn) ** 2 * (zn + b ** (-nn)) ** 2)
    return g0, dh


def _symtail_ap_r2(nn, b, a):
    """(rho^2 from the inner end ring, rho^2 from the outer ring) --
    the Lopez-Ros closure rho^2 = -Im oint dh/g0 / Im oint g0 dh per
    ring; the free constant a must make them agree."""
    g0, dh = _symtail_ap_g0_dh(nn, b, a)
    ri = 0.3 * min(abs(b - a), abs(1.0 / b - b),
                   b * math.sin(math.pi / nn))
    zo = (1.0 / b) * np.exp(1j * math.pi / nn)
    ro = 0.3 * min(abs(1.0 / b - 1.0 / a), abs(1.0 / b - b),
                   (1.0 / b) * math.sin(math.pi / nn))
    Ai = period_integral(lambda z: dh(z) / g0(z), b, ri, ri)
    Bi = period_integral(lambda z: dh(z) * g0(z), b, ri, ri)
    Ao = period_integral(lambda z: dh(z) / g0(z), zo, ro, ro)
    Bo = period_integral(lambda z: dh(z) * g0(z), zo, ro, ro)
    return -Ai.imag / Bi.imag, -Ao.imag / Bo.imag


_SYMTAIL_AP_CACHE = {}


def symtail_antiprism_constants(nn, b):
    """Branch constant a and Lopez-Ros scale rho closing the period
    problem of the antiprismatic k-noid (2*nn ends in antiprism
    symmetry; Weber's notebook solves the same system with NSolve /
    FindRoot).  Verified < 1e-10 by the period gate in the self-tests
    for nn = 3..7; nn = 2 does not close and is rejected."""
    key = (nn, round(b, 12))
    if key in _SYMTAIL_AP_CACHE:
        return _SYMTAIL_AP_CACHE[key]

    def resid(a):
        r_in, r_out = _symtail_ap_r2(nn, b, a)
        return r_in - r_out
    xs = np.linspace(0.08, 0.95, 60)
    sol = None
    vals = [resid(float(a)) for a in xs]
    for i in range(len(xs) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if not (np.isfinite(v0) and np.isfinite(v1)) or v0 * v1 >= 0:
            continue
        a = solve_scalar(resid, float(xs[i]), float(xs[i + 1]))
        r_in, r_out = _symtail_ap_r2(nn, b, a)
        if r_in <= 0:
            continue
        rho = math.sqrt(r_in)
        # accept the root whose full period set closes best
        g0, dh = _symtail_ap_g0_dh(nn, b, a)

        def phi(z):
            g = rho * g0(z)
            h = dh(z)
            return np.stack(np.broadcast_arrays(
                0.5 * (1.0 / g - g) * h, 0.5j * (1.0 / g + g) * h, h),
                axis=-1)
        w = 0.0
        for kk in range(nn):
            for zc in (b * np.exp(2j * math.pi * kk / nn),
                       (1.0 / b) * np.exp(1j * (2 * math.pi * kk
                                                + math.pi) / nn)):
                rr = 0.3 * min(abs(b - a), abs(1.0 / b - b),
                               abs(zc) * math.sin(math.pi / nn))
                for c in range(3):
                    I = period_integral(
                        lambda z, c=c: phi(z)[..., c], zc, rr, rr)
                    w = max(w, abs(I.real))
        if sol is None or w < sol[2]:
            sol = (a, rho, w)
    if sol is None or sol[2] > 1e-8:
        raise ValueError(f"antiprism nn={nn} b={b}: period problem "
                         f"did not close (worst "
                         f"{sol[2] if sol else float('nan'):.2e})")
    _SYMTAIL_AP_CACHE[key] = (sol[0], sol[1])
    return _SYMTAIL_AP_CACHE[key]


# ==========================================================================
# SP SCHERK FAMILY (sscherk_* block): higher-genus singly periodic
# Scherk towers from the minimalsurfaces.blog notebook harvest
# ==========================================================================
# Four hyperelliptic Karcher/Scherk-type singly periodic minimal
# surfaces whose period problems were solved numerically (FindRoot) in
# Matthias Weber's Mathematica notebooks; the solved constants below are
# the literal high-precision values extracted from the raw notebooks
# (research/msblog_harvest/raw/, box-flattened by extract_nb.py), and
# every member re-verifies its period conditions here with independent
# quadrature before it ships:
#
#   * sscherk_six1_*   six-ended Scherk tower of genus 1
#       (Singly_6ended_Scherk_g1.nb):
#       G = rho sqrt(z-v2)/(sqrt(z-v1) sqrt(z-v3) sqrt(z-v4)),
#       dh = dz/(z^2-1) on the upper half plane; the genus-1
#       hyperelliptic curve w^2 = prod (z-v_i) carries a vertical handle
#       through the tower.  Period problem: Re int_{v1}^{v2} om2 = 0,
#       Re int_{v2}^{v3} om1 = 0, plus two closed-form residue
#       conditions (both horizontal Scherk ends must translate by the
#       same step); the notebook table solves (v1, v2, v3, rho) per
#       family parameter v4.  MEASURED quotient: chi = -6, 6 ends,
#       genus 1 (= 2 - 2g - e).
#   * sscherk_costa_*  Costa-Scherk tower of genus 1
#       (Singly_CostaScherk_g1.nb):
#       G = rho sqrt(z-b) sqrt(z-c) sqrt(z-d)/sqrt(z-a), dh =
#       dz/(z^2-1), a < b < -1 < 0 < c < d < 1: a six-ended Scherk
#       tower whose handle forms Costa-like saddles.  Four point-to-
#       point real-period conditions (notebook test[]) fix
#       (b, c, d, rho) per family parameter a.  MEASURED quotient:
#       chi = -6, 6 ends, genus 1.
#   * sscherk_eight_*  eight-ended Scherk tower of genus 2
#       (Singly_8ended_Scherk_g2.nb):
#       G = rho sqrt(z-1/a1) sqrt(z-a2) sqrt(z-1/a3)(z-c) /
#           (sqrt(z-a1) sqrt(z-1/a2) sqrt(z-a3)(z-1/c)),
#       dh = (z-1/c)(z-c) dz/(z(z-1/b)(z-b)),
#       rho = sqrt(a1 a3)/(sqrt(a2) c) (the closed form that makes
#       G(1)^2 = -1).  Two integral period conditions plus the residue
#       match Res(om2, 0) = -Res(om2, b) (checked here numerically to
#       ~1e-10); the translation is the closed form transy =
#       -2 pi (a1 a3 + a2 c^2)/(2 sqrt(a1 a2 a3) c), reproduced by the
#       built mesh to ~1e-9.  MEASURED quotient: chi = -10, 8 ends,
#       genus 2.
#   * sscherk_das_*    daSilva-Batista surface, genus 2 with 8 ends
#       (Singly_daSilvaBatista_g2.nb; L. daSilva and V. Ramos Batista,
#       2009): G and dh as in the block tables (hyperelliptic with
#       branch data a, d, e and ends b, c inside the unit half disk);
#       three integral period conditions verified to ~1e-9, with the
#       end normals G(b), G(c) reproducing the notebook's prescribed
#       inner/outer angles.  MEASURED quotient: chi = -10, 8 ends,
#       genus 2.
#
# Meshing scheme (same discipline as the sptail block): integrate the
# 1-forms over ONE conformal fundamental patch by compound Gauss-
# Legendre cells (no singular node is ever evaluated; fractional powers
# take the interior-limit nudge so no principal branch cut is crossed),
# snap the patch boundary exactly onto its measured symmetry planes,
# orbit under the isometry group plus storey translations, and weld
# bitwise-exactly (weld tolerance 1e-12 of the span -- every intended
# seam is exact after snapping, and the fine clusters must never be
# falsely merged).  Charts:
#   * six1/costa: two log-strips w with z = +-sqrt(1 + e^w) (the
#     notebooks' own charts) welded along the imaginary-axis seam;
#     integration spine moved off the singular corner z = 0.
#   * eight: the notebook chart z = ff(c0 - 1/p), ff the inverse
#     Joukowski map, pulling the half-disk domain to a polar annulus
#     whose inner/outer trims are exactly the z = 0 / z = b ends.
#   * dasilva: the notebook's Moebius + Joukowski chart sending the two
#     interior-segment ends b, c to the annulus radii 0 / infinity.
# Symmetry structure is MEASURED from the built patch (each boundary
# sub-row's constant coordinate), not assumed: two parallel vertical
# mirrors y = y0, y = ym half a period apart (their composition is the
# translation), a vertical mirror x = xm, and for the genus-2 pair a
# horizontal mirror z = zh on the |z| = 1 arc.  The Euler
# characteristic of the translation-wrapped quotient and of the S vs
# S+1 storey stacks is measured in the self-tests against
# chi = 2 - 2 genus - #ends.
#
# References:
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988) 83-114;
#   H. F. Scherk (1835), the classical tower these generalize;
#   K. Li, "Singly periodic minimal surfaces" (Indiana University PhD
#     thesis lineage cited by the blog for the 6/8-ended towers);
#   L. daSilva, V. Ramos Batista, "Costa-Hoffman-Meeks type embedded
#     minimal surfaces" lineage -- the daSilva-Batista singly periodic
#     surface (2009);
#   M. Weber, https://minimalsurfaces.blog/ -- notebooks
#     Singly_6ended_Scherk_g1.nb, Singly_CostaScherk_g1.nb,
#     Singly_8ended_Scherk_g2.nb, Singly_daSilvaBatista_g2.nb
#     (research/msblog_harvest/singly_periodic.json).
# --------------------------------------------------------------------------

# Solved period constants, extracted verbatim from the notebooks.
# six1: family parameter v4 -> (v1, v2, v3, v4, rho)
SSCHERK_SIX1_MEMBERS = (
    (-0.9999097463240165, -0.9995846218404437, 0.32722281347079313,
     0.4, 0.9311638650898755),
    (-0.999842581353628, -0.9990441104641703, 0.3947092606175121,
     0.45, 0.90617573822774),
    (-0.9997892166741251, -0.9982962652599652, 0.45891892793127825,
     0.5, 0.8774486270142349),
    (-0.9997707148120727, -0.9929889331850467, 0.6882048308313006,
     0.7, 0.7203224603580416),
    (-0.9998768763825345, -0.9584256703240235, 0.8955499209754002,
     0.9, 0.4425916550119159),
    (-0.9999200550022518, -0.8936905890458876, 0.9459601036545873,
     0.95, 0.32246552799548456),
)

# costa: family parameter a -> (a, b, c, d, rho)
SSCHERK_COSTA_MEMBERS = (
    (-1.01, -1.0014807442877984, 0.3754468359389136,
     0.5066016876271907, 1.1173158075304825),
    (-1.03, -1.0044731973790795, 0.3186247625445407,
     0.5477098649370288, 1.1187536194617387),
    (-1.1, -1.0134896339281911, 0.22686416763109937,
     0.6349754786834063, 1.1397736769892377),
    (-1.3, -1.0278863923801085, 0.1280930904161984,
     0.7599360066323362, 1.2174904307548113),
    (-1.6, -1.0342509923267171, 0.07410998330062755,
     0.8439283066495238, 1.3369857748270828),
    (-2.0, -1.034111462010235, 0.0444190235289843,
     0.895759295658659, 1.4870353450504705),
    (-3.0, -1.0276536423752907, 0.01909708900610511,
     0.9448847188685726, 1.8146317544905508),
)

# eight: family parameter b -> (a1, a2, a3, b, c); rho closed form
SSCHERK_EIGHT_MEMBERS = (
    (0.02, 0.05079558837191939, 0.24207555442197323, -0.4,
     -0.33429719165838),
    (0.02, 0.0516869008349762, 0.2645421110390311, -0.45,
     -0.3520006640761227),
    (0.02, 0.05188150165699855, 0.3411942059323637, -0.6,
     -0.4127725007154455),
    (0.02, 0.04963312235957757, 0.4047116866471815, -0.7,
     -0.46460413392763156),
    (0.02, 0.04510764776532291, 0.4869060986651275, -0.8,
     -0.5339075133085016),
    (0.02, 0.037642945984715515, 0.6092468800312116, -0.9,
     -0.6405186145093797),
)

# dasilva: (a, b, c, d, e) FindRoot members (ordered by neck angle)
SSCHERK_DAS_MEMBERS = (
    (-0.16538063249530707, -0.050186950657416095, 0.14280760356113087,
     0.14431373823719248, 0.2957720830704163),
    (-0.2318493801219483, -0.16955666911933556, 0.16955667364469934,
     0.1699296290799072, 0.43726314811269806),
    (-0.2641085050708083, -0.015204799559572081, 0.09423310060363765,
     0.09811254515942426, 0.18962901567819868),
    (-0.323143539872764, -0.010914867530614441, 0.07504706366931041,
     0.07935654240080936, 0.1549153647710601),
    (-0.5356990518612961, -0.0040643048547451665, 0.024479936412485897,
     0.02679936121082089, 0.06011882879492939),
    (-0.6094206570490024, -0.03032524297897274, 0.03032524309337132,
     0.030763209335820576, 0.11240090862288739),
)


def sscherk_cheb_seg(f, z0, z1, n=3000):
    """Gauss-Chebyshev quadrature of f along the straight segment
    z0 -> z1 (absorbs the inverse-square-root endpoint singularities of
    the hyperelliptic 1-forms)."""
    k = np.arange(1, n + 1)
    x = np.cos((2 * k - 1) * math.pi / (2 * n))
    w = math.pi / n * np.sqrt(1.0 - x * x)
    z = 0.5 * (z0 + z1) + 0.5 * (z1 - z0) * x
    return 0.5 * (z1 - z0) * np.sum(f(z) * w)


def _sscherk_grid1d(lo, hi, marks, nseg, cl, rel=False):
    """1-D grid from lo to hi: nseg points per inter-mark band plus
    geometric clusters around every mark (relative for log-scale polar
    radii, absolute for strip coordinates)."""
    marks = sorted(marks)
    parts = []
    prev = lo
    for mk in marks + [hi]:
        if rel:
            parts.append(np.exp(np.linspace(math.log(prev), math.log(mk),
                                            nseg, endpoint=False)))
        else:
            parts.append(np.linspace(prev, mk, nseg, endpoint=False))
        prev = mk
    parts.append([hi])
    for mk in marks:
        if rel:
            parts.extend([mk * (1.0 - cl), [mk], mk * (1.0 + cl[::-1])])
        else:
            parts.extend([mk - cl, [mk], mk + cl[::-1]])
    g = np.unique(np.concatenate([np.atleast_1d(np.asarray(p, float))
                                  for p in parts]))
    if rel:
        return g[np.concatenate([[True], np.diff(g) / g[:-1] > 1e-13])]
    return g[np.concatenate([[True], np.diff(g) > 1e-13])]


def _sscherk_tgrid(nt):
    """theta/y grid on (0, pi): cosine-clustered plus geometric edge
    clusters (the boundary rows carry the symmetry curves)."""
    base = math.pi * (0.5 - 0.5 * np.cos(np.pi * np.linspace(
        0, 1, 2 * max(6, nt // 2) + 1)))
    ex = sptail_cluster(0.06, hmin=1e-7, ratio=0.25)
    t = np.unique(np.concatenate([base, ex, math.pi - ex]))
    return t[np.concatenate([[True], np.diff(t) > 1e-12])]


def _sscherk_finish(pieces, rows, use_mz, storeys, extra_diag=None):
    """Shared classify/snap/orbit stage.

    pieces: list of (X (n1, n2, 3) arrays, already seam-aligned).
    rows:   list of (X, index-slices) picking each boundary sub-row that
            lies on a symmetry plane; the constant coordinate of every
            row is MEASURED (smallest ptp), grouped into planes (y
            splits into the two parallel mirrors y0/ym half a period
            apart), normalized to {x = 0, y0 = 0, z = 0} and snapped
            exactly.
    Returns (V, F, uv, diag): diag carries presnap (the max deviation
    of any row from its plane -- the machine-checked period residual),
    T, span."""
    kinds, consts, ptps = [], [], []
    for X, sl in rows:
        S = X[sl]
        p3 = [float(np.ptp(S[..., c])) for c in range(3)]
        k = int(np.argmin(p3))
        kinds.append(k)
        consts.append(float(np.median(S[..., k])))
        ptps.append(p3[k])
    # y-planes: split into two clusters (y0 holds rows[0] if y-kind,
    # else the cluster nearest zero span start)
    yv = [c for k, c in zip(kinds, consts) if k == 1]
    xv = [c for k, c in zip(kinds, consts) if k == 0]
    zv = [c for k, c in zip(kinds, consts) if k == 2]
    ys = sorted(yv)
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    if gaps and max(gaps) > 0.25 * (ys[-1] - ys[0] + 1e-30):
        cut = gaps.index(max(gaps))
        g1 = ys[:cut + 1]
        g2 = ys[cut + 1:]
    else:
        g1, g2 = ys, []
    # the base y-plane is the one containing the first y-kind row
    c0 = next(c for k, c in zip(kinds, consts) if k == 1)
    if g2 and any(abs(c0 - v) < 1e-12 + 0.0 or v == c0 for v in g2):
        g1, g2 = g2, g1
    y0 = float(np.mean(g1))
    ym = float(np.mean(g2)) if g2 else None
    xm = float(np.mean(xv)) if xv else 0.0
    zh = float(np.mean(zv)) if zv else 0.0
    spread = 0.0
    for grp, mean in ((g1, y0), ((g2 or [0]), ym or 0.0),
                      (xv, xm), (zv, zh)):
        for v in grp:
            spread = max(spread, abs(v - mean))
    shift = np.array([xm, y0, zh])
    for X in pieces:
        X -= shift
    if ym is not None:
        ym -= y0
    # snap every row onto its (shifted) plane
    planes = {0: 0.0, 2: 0.0}
    for (X, sl), k, c in zip(rows, kinds, consts):
        if k == 1:
            tgt = 0.0 if (ym is None or abs(c - y0 - 0.0)
                          < abs(c - y0 - ym)) else ym
        else:
            tgt = planes[k]
        X[sl + (k,)] = tgt
    T = np.array([0.0, 2.0 * ym, 0.0]) if ym is not None \
        else np.array([0.0, 0.0, 0.0])
    V0 = np.concatenate([X.reshape(-1, 3) for X in pieces], axis=0)
    quads = []
    uvs = []
    off = 0
    for X in pieces:
        n1, n2 = X.shape[0], X.shape[1]
        quads.extend(tuple(off + i for i in f)
                     for f in sptail_grid_quads(n1, n2))
        uvs.append(_sptail_grid_uv(n1, n2))
        off += n1 * n2
    uv0 = np.concatenate(uvs, axis=0)
    frames = []
    for mz in ((0, 1) if use_mz else (0,)):
        for mx in (0, 1):
            for my in (0, 1):
                M = np.diag([-1.0 if mx else 1.0, -1.0 if my else 1.0,
                             -1.0 if mz else 1.0])
                par = (-1.0) ** (mx + my + mz)
                for s in range(storeys):
                    frames.append((M, s * T, par))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, uv0, quads, frames, 1e-12 * span)
    diag = {'presnap': float(max(max(ptps), spread)), 'T': T,
            'span': span, 'ym': ym}
    if extra_diag:
        diag.update(extra_diag)
    return V, F, uv, diag


# ---- two-strip chart (six-ended g1 and Costa-Scherk g1) ------------------

_SSCHERK_SPINE = -2.0


def _sscherk_strip_patch(om, sign, x, y):
    """Rect patch in w = x + iy through the chart z = sign sqrt(1+e^w);
    dz = (z^2-1)/(2z) dw on both branches.  The integration spine is
    moved to x = -2 (a regular boundary point for every shipped member;
    the corner z = 0 at w = i pi is chart-singular)."""
    sp = _SSCHERK_SPINE

    def omw(w):
        w = w + sp
        z = sign * np.sqrt(1.0 + np.exp(w))
        z = np.where(np.imag(z) == 0.0,
                     z + 1e-14j * (1.0 + np.abs(z)), z)
        dz = (z * z - 1.0) / (2.0 * z)
        o = om(z)
        return tuple(c * dz for c in o)
    return sptail_rect_patch(omw, x - sp, y)


def _sscherk_twostrip_build(om, marks_a, marks_b0, marks_bpi, r1, r2,
                            nseg, nt, storeys):
    """Shared builder for the dh = dz/(z^2-1) towers: strip a covers
    the right half of the upper half plane, strip b the left, welded
    along the imaginary-axis seam (y = +-pi, x > 0); marks_* are the
    strip x-positions of the hyperelliptic branch points on the pi
    rows (x = log(1-v^2)) resp. the y = 0 rows (x = log(v^2-1))."""
    cl = sptail_cluster(0.25, hmin=1e-8, ratio=0.3)
    allm = sorted(set(list(marks_a) + list(marks_b0) + list(marks_bpi)
                      + [0.0, _SSCHERK_SPINE]))
    x = _sscherk_grid1d(r1, r2, allm, nseg, cl)
    y = _sscherk_tgrid(nt)
    Xa = _sscherk_strip_patch(om, +1.0, x, y)
    Xb = _sscherk_strip_patch(om, -1.0, x, y - math.pi)
    i0 = int(np.searchsorted(x, 0.0))
    d = Xa[:, -1, :] - Xb[:, 0, :]
    off = np.median(d[i0 + 1:], axis=0)
    seam_ptp = float(np.abs(d[i0 + 1:] - off).max())
    Xb += off
    Xb[i0:, 0, :] = Xa[i0:, -1, :]      # bitwise-exact seam

    def cuts(marks, lo_i, hi_i):
        idx = [lo_i] + [int(np.searchsorted(x, m)) for m in
                        sorted(marks)] + [hi_i]
        return [(slice(idx[i], idx[i + 1] + 1),)
                for i in range(len(idx) - 1)]

    rows = []
    rows.append((Xa, (slice(None), 0)))              # (1, inf)
    for sl in cuts(marks_a, 0, i0):                  # pi row, strip a
        rows.append((Xa, sl + (-1,)))
    for sl in cuts(marks_b0, 0, len(x) - 1):         # y=0 row, strip b
        rows.append((Xb, sl + (-1,)))
    for sl in cuts(marks_bpi, 0, i0):                # -pi row, strip b
        rows.append((Xb, sl + (0,)))
    V, F, uv, diag = _sscherk_finish(
        [Xa, Xb], rows, use_mz=False, storeys=storeys,
        extra_diag={'seam_ptp': seam_ptp})
    # exact seam weld happens inside the orbit because after snapping
    # the two seam rows are bitwise-identical copies:
    return V, F, uv, diag


def sscherk_six1_om(mi):
    v1, v2, v3, v4, rho = SSCHERK_SIX1_MEMBERS[mi]

    def om(z):
        z = sptail_nudge(z)
        G = rho * np.sqrt(z - v2) / (np.sqrt(z - v1) * np.sqrt(z - v3)
                                     * np.sqrt(z - v4))
        dh = 1.0 / (z * z - 1.0)
        p1 = G * dh
        p2 = dh / G
        return (0.5 * (p2 - p1), 0.5j * (p2 + p1), dh)
    return om


def sscherk_six1_check(mi):
    """Period residuals for a table member: the two integral conditions
    and the two closed-form residue conditions of the notebook's
    test[], plus Res(om2, +1) (the translation is 2 pi Im Res)."""
    v1, v2, v3, v4, rho = SSCHERK_SIX1_MEMBERS[mi]
    om = sscherk_six1_om(mi)
    # the near-degenerate members put the dh pole at -1 within ~1e-4 of
    # the branch interval; Gauss-Chebyshev needs depth there
    c1 = float(np.real(sscherk_cheb_seg(
        lambda z: om(z)[1], v1 + 1e-9j, v2 + 1e-9j, n=48000)))
    c2 = float(np.real(sscherk_cheb_seg(
        lambda z: om(z)[0], v2 + 1e-9j, v3 + 1e-9j, n=48000)))
    e3 = -((-1 + v1 + v3 - v1 * v3 + v4 - v1 * v4 - v3 * v4
            + v1 * v3 * v4 - rho ** 2 + v2 * rho ** 2)
           / (4 * math.sqrt(1 - v1) * math.sqrt(1 - v2)
              * math.sqrt(1 - v3) * math.sqrt(1 - v4))) - 0.5
    e4 = ((1 + v1 + v3 + v1 * v3 + v4 + v1 * v4 + v3 * v4
           + v1 * v3 * v4 + rho ** 2 + v2 * rho ** 2)
          / (4 * math.sqrt(1 + v1) * math.sqrt(1 + v2)
             * math.sqrt(1 + v3) * math.sqrt(1 + v4))) - 0.5
    rr = 0.25 * (1.0 - v4)
    res = period_integral(lambda z: om(z)[1], 1.0, rr, rr) \
        / (2j * math.pi)
    return c1, c2, float(e3), float(e4), complex(res)


def sscherk_six1_build(mi, nseg=10, nt=24, storeys=1, r1=-13.0, r2=8.0):
    v1, v2, v3, v4, rho = SSCHERK_SIX1_MEMBERS[mi]
    om = sscherk_six1_om(mi)
    marks_a = [math.log(1 - v4 * v4), math.log(1 - v3 * v3)]
    marks_bpi = [math.log(1 - v1 * v1), math.log(1 - v2 * v2)]
    r1 = min(r1, min(marks_bpi) - 2.5)
    V, F, uv, diag = _sscherk_twostrip_build(
        om, marks_a, [], marks_bpi, r1, r2, nseg, nt, storeys)
    res = sscherk_six1_check(mi)[4]
    diag['T_vs_residue'] = float(abs(abs(diag['T'][1])
                                     - abs(2 * math.pi * res.imag)))
    return V, F, uv, diag


def sscherk_costa_om(mi):
    a, b, c, d, rho = SSCHERK_COSTA_MEMBERS[mi]

    def om(z):
        z = sptail_nudge(z)
        G = rho * np.sqrt(z - b) * np.sqrt(z - c) * np.sqrt(z - d) \
            / np.sqrt(z - a)
        dh = 1.0 / (z * z - 1.0)
        p1 = G * dh
        p2 = dh / G
        return (0.5 * (p2 - p1), 0.5j * (p2 + p1), dh)
    return om


def sscherk_costa_check(mi):
    """The notebook test[] conditions (a -> b real period, a -> d and
    c -> 1.5 point-to-point real parts through i), plus Res(om2, 1)."""
    a, b, c, d, rho = SSCHERK_COSTA_MEMBERS[mi]
    om = sscherk_costa_om(mi)
    t1 = float(np.real(sscherk_cheb_seg(
        lambda z: om(z)[1], a + 1e-9j, b + 1e-9j)))
    t2 = float(np.real(sscherk_cheb_seg(lambda z: om(z)[0], a + 0j, 1j)
                       + sscherk_cheb_seg(lambda z: om(z)[0], 1j,
                                          d + 0j)))
    t3 = float(np.real(sscherk_cheb_seg(lambda z: om(z)[1], a + 0j, 1j)
                       + sscherk_cheb_seg(lambda z: om(z)[1], 1j,
                                          d + 0j)))
    t4 = float(np.real(sscherk_cheb_seg(lambda z: om(z)[1], c + 0j, 1j)
                       + sscherk_cheb_seg(lambda z: om(z)[1], 1j,
                                          1.5 + 1e-7j)))
    rr = 0.25 * (1.0 - d)
    res = period_integral(lambda z: om(z)[1], 1.0, rr, rr) \
        / (2j * math.pi)
    return t1, t2, t3, t4, complex(res)


def sscherk_costa_build(mi, nseg=10, nt=24, storeys=1, r1=-12.0,
                        r2=8.0):
    a, b, c, d, rho = SSCHERK_COSTA_MEMBERS[mi]
    om = sscherk_costa_om(mi)
    marks_a = [math.log(1 - d * d), math.log(1 - c * c)]
    marks_b0 = [math.log(b * b - 1), math.log(a * a - 1)]
    r1 = min(r1, min(marks_a + marks_b0) - 2.5)
    V, F, uv, diag = _sscherk_twostrip_build(
        om, marks_a, marks_b0, [], r1, r2, nseg, nt, storeys)
    res = sscherk_costa_check(mi)[4]
    diag['T_vs_residue'] = float(abs(abs(diag['T'][1])
                                     - abs(2 * math.pi * res.imag)))
    return V, F, uv, diag


# ---- eight-ended Scherk of genus 2 (half-disk polar chart) ---------------

def sscherk_eight_om(mi):
    a1, a2, a3, b, c = SSCHERK_EIGHT_MEMBERS[mi]
    rho = math.sqrt(a1) * math.sqrt(a3) / (math.sqrt(a2) * c)

    def om(z):
        z = sptail_nudge(z)
        G = rho * np.sqrt(z - 1 / a1) * np.sqrt(z - a2) \
            * np.sqrt(z - 1 / a3) * (z - c) \
            / (np.sqrt(z - a1) * np.sqrt(z - 1 / a2)
               * np.sqrt(z - a3) * (z - 1 / c))
        dh = (z - 1 / c) * (z - c) / (z * (z - 1 / b) * (z - b))
        p1 = G * dh
        p2 = dh / G
        return (0.5 * (p2 - p1), 0.5j * (p2 + p1), dh)
    return om


def sscherk_eight_check(mi):
    """Integral period conditions, the residue match of the two dh
    poles in the domain, and the closed-form translation transy."""
    a1, a2, a3, b, c = SSCHERK_EIGHT_MEMBERS[mi]
    om = sscherk_eight_om(mi)
    t1 = float(np.real(sscherk_cheb_seg(
        lambda z: om(z)[1], a1 + 1e-10j, a2 + 1e-10j)))
    t2 = float(np.real(sscherk_cheb_seg(
        lambda z: om(z)[0], a2 + 1e-10j, a3 + 1e-10j)))
    r0 = 0.4 * a1
    res0 = period_integral(lambda z: om(z)[1], 0.0, r0, r0) \
        / (2j * math.pi)
    rb = 0.2 * min(abs(b - c), abs(1.0 + b))
    resb = period_integral(lambda z: om(z)[1], b, rb, rb) \
        / (2j * math.pi)
    transy = -2 * math.pi * (a1 * a3 + a2 * c * c) \
        / (2 * math.sqrt(a1 * a2 * a3) * c)
    return t1, t2, complex(res0), complex(resb), float(transy)


def sscherk_eight_build(mi, nseg=10, nt=24, storeys=1, rmin=1e-3,
                        rmax=1e4):
    a1, a2, a3, b, c = SSCHERK_EIGHT_MEMBERS[mi]
    om = sscherk_eight_om(mi)
    c0 = -(b + 1 / b) / 2

    def omp(p):
        p = sptail_nudge(p)
        u = c0 - 1.0 / p
        u = np.where(np.imag(u) == 0.0,
                     u + 1e-14j * (1.0 + np.abs(u)), u)
        s = np.sqrt(u - 1.0) * np.sqrt(u + 1.0)
        z = -u + s
        dz = -(z / s) / (p * p)
        o = om(z)
        return tuple(cc * dz for cc in o)

    def invparm(z):
        return -(2 * b * z) / (-b + z + b * b * z - b * z * z)

    mka = [invparm(a1), invparm(a2), invparm(a3)]
    p1m, p2m, cm = 1.0 / (c0 + 1.0), 1.0 / (c0 - 1.0), -invparm(c)
    rmin = min(rmin, 0.25 * min(mka))
    rmax = max(rmax, 4.0 * max(p2m, cm))
    cl = sptail_cluster(0.12, hmin=1e-8, ratio=0.3)
    r = _sscherk_grid1d(rmin, rmax, mka + [p1m, p2m, cm], nseg, cl,
                        rel=True)
    t = _sscherk_tgrid(nt)
    X = sptail_polar_patch(omp, r, t)
    im = [int(np.searchsorted(r, v)) for v in mka]
    ip1 = int(np.searchsorted(r, p1m))
    ip2 = int(np.searchsorted(r, p2m))
    icm = int(np.searchsorted(r, cm))
    rows = [
        (X, (slice(0, im[0] + 1), 0)),          # (0, a1)  y-plane A
        (X, (slice(im[0], im[1] + 1), 0)),      # (a1, a2) x-plane
        (X, (slice(im[1], im[2] + 1), 0)),      # (a2, a3) y-plane A
        (X, (slice(im[2], ip1 + 1), 0)),        # (a3, 1)  x-plane
        (X, (slice(ip1, ip2 + 1), 0)),          # arc      z-plane
        (X, (slice(ip2, len(r)), 0)),           # (-1, b)  y-plane A
        (X, (slice(0, icm + 1), -1)),           # (0, c)   y-plane B
        (X, (slice(icm, len(r)), -1)),          # (c, b)   y-plane B
    ]
    V, F, uv, diag = _sscherk_finish([X], rows, use_mz=True,
                                     storeys=storeys)
    transy = sscherk_eight_check(mi)[4]
    diag['T_vs_transy'] = float(abs(abs(diag['T'][1]) - abs(transy)))
    return V, F, uv, diag


# ---- daSilva-Batista genus 2 (Moebius half-disk chart) -------------------

def sscherk_das_om(mi):
    a, b, c, d, e = SSCHERK_DAS_MEMBERS[mi]
    C = complex(np.sqrt(1 - a + 0j) * np.sqrt(1 - 1 / d + 0j)
                * np.sqrt(1 - 1 / e + 0j)
                / (np.sqrt(1 - 1 / a + 0j) * np.sqrt(1 - d + 0j)
                   * np.sqrt(1 - e + 0j)))

    def om(z):
        z = sptail_nudge(z)
        G = C * np.sqrt(z - 1 / a) * np.sqrt(z - d) * np.sqrt(z - e) \
            / (z * np.sqrt(z - a) * np.sqrt(z - 1 / d)
               * np.sqrt(z - 1 / e))
        dh = z / ((z - 1 / b) * (z - b) * (z - 1 / c) * (z - c))
        p1 = G * dh
        p2 = dh / G
        return (0.5 * (p2 - p1), 0.5j * (p2 + p1), dh)
    return om


def sscherk_das_check(mi):
    """The notebook test[] period conditions (through the upper half
    plane, avoiding the branch segments) plus Res(om2, b) (translation
    = |2 pi Im Res|)."""
    a, b, c, d, e = SSCHERK_DAS_MEMBERS[mi]
    om = sscherk_das_om(mi)
    t1 = float(np.real(
        sscherk_cheb_seg(lambda z: om(z)[1], 0.0 + 0j, e / 2 + 0.5j)
        + sscherk_cheb_seg(lambda z: om(z)[1], e / 2 + 0.5j, e + 0j)))
    t2 = float(np.real(
        sscherk_cheb_seg(lambda z: om(z)[1], a + 0j, 0.5j)
        + sscherk_cheb_seg(lambda z: om(z)[1], 0.5j, d + 0j)))
    t3 = float(np.real(
        sscherk_cheb_seg(lambda z: om(z)[0], a + 0j, 0.5j)
        + sscherk_cheb_seg(lambda z: om(z)[0], 0.5j, d + 0j)))
    rb = 0.25 * min(abs(b - a), abs(c - b), abs(b))
    resb = period_integral(lambda z: om(z)[1], b, rb, rb) \
        / (2j * math.pi)
    return t1, t2, t3, complex(resb)


def sscherk_das_build(mi, nseg=10, nt=24, storeys=1, cut1=0.02,
                      cut2=2e4):
    a, b, c, d, e = SSCHERK_DAS_MEMBERS[mi]
    om = sscherk_das_om(mi)

    def h(z):
        return -(z + 1.0 / z) / 2.0

    a1, b1 = h(b), h(c)

    def qmark(z):
        return abs((h(z) - a1) / (h(z) - b1))

    def omq(q):
        q = sptail_nudge(q)
        M = (b1 * q - a1) / (q - 1.0)
        M = np.where(np.imag(M) == 0.0,
                     M + 1e-14j * (1.0 + np.abs(M)), M)
        s = np.sqrt(M - 1.0) * np.sqrt(M + 1.0)
        z = -M + s
        dz = -(z / s) * (a1 - b1) / ((q - 1.0) ** 2)
        o = om(z)
        return tuple(cc * dz for cc in o)

    qa, qm1, qp1 = qmark(a), qmark(-1.0), qmark(1.0)
    qe, qd = qmark(e), qmark(d)
    cut1 = min(cut1, 0.25 * qa)
    cut2 = max(cut2, 4.0 * qd)
    marks = [qa, qm1, 1.0, qp1, qe, qd]
    cl = sptail_cluster(0.12, hmin=1e-8, ratio=0.3)
    r = _sscherk_grid1d(cut1, cut2, marks, nseg, cl, rel=True)
    t = _sscherk_tgrid(nt)
    X = sptail_polar_patch(omq, r, t)
    i0 = int(np.searchsorted(r, 1.0))
    ia = int(np.searchsorted(r, qa))
    im1 = int(np.searchsorted(r, qm1))
    ip1 = int(np.searchsorted(r, qp1))
    ie = int(np.searchsorted(r, qe))
    idm = int(np.searchsorted(r, qd))
    rows = [
        (X, (slice(0, i0 + 1), 0)),         # (b, 0)   y-plane A
        (X, (slice(i0, len(r)), 0)),        # (0, c)   y-plane A
        (X, (slice(0, ia + 1), -1)),        # (b, a)   y-plane B
        (X, (slice(ia, im1 + 1), -1)),      # (a, -1)  x-plane
        (X, (slice(im1, ip1 + 1), -1)),     # arc      z-plane
        (X, (slice(ip1, ie + 1), -1)),      # (1, e)   y-plane A
        (X, (slice(ie, idm + 1), -1)),      # (e, d)   x-plane
        (X, (slice(idm, len(r)), -1)),      # (d, c)   y-plane B
    ]
    V, F, uv, diag = _sscherk_finish([X], rows, use_mz=True,
                                     storeys=storeys)
    resb = sscherk_das_check(mi)[3]
    diag['T_vs_residue'] = float(abs(abs(diag['T'][1])
                                     - abs(2 * math.pi * resb.imag)))
    return V, F, uv, diag


# ---- toolkit meshers -----------------------------------------------------

def _sscherk_mesh(build, nmem, spec, nu, nv, order, radius, scale,
                  storeys):
    mi = int(np.clip(order, 1, nmem)) - 1
    S = int(np.clip(storeys, 1, 5))
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    kw = dict(p.get('build_kw', {}))
    V, F, uv, _ = build(mi, nseg=int(np.clip(int(0.22 * nu), 6, 18)),
                        nt=int(np.clip(int(0.5 * nv), 16, 60)),
                        storeys=S, **kw)
    V = _center_fit(V, scale, V)
    return V, F, uv


def sscherk_six1_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                      storeys=1):
    return _sscherk_mesh(sscherk_six1_build, len(SSCHERK_SIX1_MEMBERS),
                         spec, nu, nv, order, radius, scale, storeys)


def sscherk_costa_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                       storeys=1):
    return _sscherk_mesh(sscherk_costa_build, len(SSCHERK_COSTA_MEMBERS),
                         spec, nu, nv, order, radius, scale, storeys)


def sscherk_eight_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                       storeys=1):
    return _sscherk_mesh(sscherk_eight_build, len(SSCHERK_EIGHT_MEMBERS),
                         spec, nu, nv, order, radius, scale, storeys)


def sscherk_das_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                     storeys=1):
    return _sscherk_mesh(sscherk_das_build, len(SSCHERK_DAS_MEMBERS),
                         spec, nu, nv, order, radius, scale, storeys)


# ==========================================================================
# Translation-invariant catenoid / Costa towers and CHM-(1,2)
# (stinv_* block)
# ==========================================================================
# Four singly periodic minimal surfaces completing the translation-
# invariant family from the minimalsurfaces.blog harvest, all meshed by
# the sptail scheme (one fundamental patch integrated with compound
# Gauss-Legendre cells, boundary snapped exactly onto its measured
# symmetry elements, orbited under the isometry group plus `storeys`
# translations, welded seam-exactly; fractional powers always evaluated
# in the limit from the domain interior).
#
#   * STINV CATENOID + 1 HANDLE (genus 2 per period, 2 catenoid ends):
#     Karcher's fence of catenoids with one extra handle.  Data on the
#     upper half plane, dh = dz/z,
#       phi1 = sqrt(a/b) z^-1/2 (z-a)^-1/2 (z-b)^1/2 (z-1/b)^-1/2
#                                                     (z-1/a)^1/2,
#       phi2 = 1/sqrt(a/b) z^-3/2 (z-a)^1/2 (z-b)^-1/2 (z-1/b)^1/2
#                                                     (z-1/a)^-1/2,
#     branch values a in (0,1) and b in (-1,0) (the z -> 1/z symmetry
#     pairs them with 1/a, 1/b).  One period condition
#     Re int_{a->i->b} om1 = 0 fixes b(a); the solved pairs are
#     harvested verbatim from Weber's Singly_Catenoid_1Handle_g1
#     notebook (FindRoot output, all re-verified here to ~3e-9 by the
#     self-test):
#       a = 0.1  b = -0.7054086037971196
#       a = 0.2  b = -0.6309634409033806
#       a = 0.3  b = -0.5610309089063753
#       a = 0.5  b = -0.4191755389731957
#     Patch = upper half annulus a/r1 <= |z| <= 1; boundary: x-mirror
#     (segments (rmin, a) at t=0 and (-1, -|b|) at t=pi land in the SAME
#     vertical plane -- measured), y-mirror (a, 1), a parallel y-mirror
#     (-|b|, rmin...) at half-period distance, the horizontal mirror arc
#     |z| = 1, and the catenoid end trim at rmin.  8 isometries + the
#     horizontal translation T = (0, 2 dy, 0) tile the space.
#
#   * STINV CATENOID + 2 HANDLES (genus 3 per period, 2 catenoid ends):
#     same construction one level up; divisor gains (z-c)^{+-1/2}
#     (z-1/c)^{-+1/2} with c in (-1,0), b in (0,1).  Two period
#     conditions Re int_{a->b} om2 = 0 and Re int_{a->i->c} om2 = 0 fix
#     (b, c)(a); solved triples harvested verbatim from Weber's
#     Singly_Catenoid_2Handles_g3 notebook (re-verified to ~2e-9):
#       a = 0.06 b = 0.8313415275984382  c = -0.7834727487665828
#       a = 0.08 b = 0.8252014282724452  c = -0.7601944503629705
#       a = 0.1  b = 0.8207141842757188  c = -0.7382148901607947
#       a = 0.2  b = 0.8106797631408779  c = -0.6367415615956143
#       a = 0.4  b = 0.816529866798894   c = -0.44635393033532667
#       a = 0.6  b = 0.8409077253104242  c = -0.26280805739772073
#     Here the y-mirror holds THREE boundary segments; the two x-mirror
#     segments are parallel at half the translation T = (2 dx, 0, 0).
#
#   * TRANSLATION-INVARIANT COSTA I (genus 1 per period, 4 ends):
#     a Costa-type tower with two flat annular wings.  Data on the
#     upper half plane, dh = dz/((z-1)(z+1)),
#       phi1 = rho (z-a)^-1/2 (z-1)^-1/2 (z-b)^1/2  (z+1)^-1/2,
#       phi2 = 1/rho (z-a)^1/2 (z-1)^-3/2 (z-b)^-1/2 (z+1)^-3/2,
#     ends at z = +-1 (catenoid-type) and z = inf (two flat annular
#     wings after doubling); two period conditions
#     Re int_{b->i->r0} om2 = 0 and Re int_{a->i->b} om1 = 0 fix
#     (b, rho)(a); solved triples harvested verbatim from Weber's
#     Singly_TransInvCosta_I notebook (re-verified to ~4e-8):
#       a = -10   b = -0.03333055237361433  rho = 3.307161083209713
#       a = -5    b = -0.06664857639434493  rho = 2.336922594756168
#       a = -3    b = -0.11107380129022093  rho = 1.8071167824277032
#       a = -2    b = -0.16689594688623235  rho = 1.4701727470752084
#       a = -1.5  b = -0.22432453091517268  rho = 1.2657277793430088
#       a = -1.1  b = -0.3252466601711385   rho = 1.0666960764075604
#     Meshed on the Joukowski half-disk chart z = -(zeta + 1/zeta)/2
#     (zeta in the upper half unit disk): the z = inf end is the clean
#     inner trim |zeta| = rmin, the z = -+1 ends sit at the two corners
#     zeta = +-1 (masked off with small corner disks -- their funnels
#     are completed by the mirror frames).  The first period condition
#     makes the two y-mirror boundary planes coincide (measured in the
#     self-test); 4 isometries + T = (0, 2 dy, 0).
#     NOTE the interior-limit nudge here must shrink |zeta| RADIALLY
#     with a dominant weight: a naive +i epsilon nudge pushes arc
#     points with sin(theta) > 1/2 to the WRONG side of the Joukowski
#     image's branch cut (found the hard way; see stinv_costa_om).
#
#   * CHM-(1,2) (Callahan-Hoffman-Meeks family, second member; MEASURED
#     quotient genus 4 per period, 2 horizontal planar ends -- one
#     handle more than CHM-(1,1)'s genus 3, NOT the genus-2k+1 M_k
#     sequence).  Data on the real line with
#     branch values 0, +-1, +-a, +-b (1 < a < b),
#       phi1 = sqrt(x^2-1) / (sqrt x (x^2-a^2)^3/4 (x^2-b^2)^5/4),
#       phi2 = sqrt x (x^2-b^2)^1/4 / (sqrt(x^2-1) (x^2-a^2)^1/4),
#       dh   = dx / (sqrt(x^2-a^2) sqrt(x^2-b^2)).
#     The two period conditions of Weber's Singly_CHM_1_2 notebook (the
#     regularized A/P^(1/4) ratio conditions) at the notebook constants
#       a = 1.07378157681789798, b = 1.46816068659935883,
#       rho = 1.41294757571306188
#     leave a residual of ~1.4e-3; a Newton polish from that seed (same
#     conditions, higher-order quadrature) converges to
#       a = 1.0735928348392014, b = 1.467713226486139,
#       rho = 1.4120744657811857
#     with residual ~4e-6 (the quadrature noise floor); the polished
#     constants are shipped and the seed kept here for the record.
#     Meshed on the strip chart x = sqrt(b^2 + e^w) (so x^2 - b^2 = e^w
#     exactly; every principal fractional power is single-valued on
#     u in [umin, umax], t in [0, pi]).  The t = pi edge crosses THREE
#     branch corners xa = log(b^2-a^2) (x = a), xb = log(b^2-1)
#     (x = 1), x0 = log b^2 (x = 0) and continues onto the imaginary
#     x axis; its four sub-edges are a horizontal STRAIGHT line with
#     direction (1,-1,0) at height -d, the y = 0 mirror, the x = 0
#     mirror, and a second (1,-1,0) line through the origin.  t = 0 is
#     the y = 0 mirror again; u -> -inf is the planar end at x = b,
#     u -> +inf the planar end at x = inf (both trimmed by the u
#     range).  8 isometries {E, R_line} x {E, Mx} x {E, My} + the
#     vertical translation (0, 0, 2 z_line) tile one period -- the
#     direct CHM-(1,1) analog with the roles of the strip ends changed.
#
# Topology is MEASURED, never assumed (chi of the welded stacks and of
# the translation-wrapped quotient, manifoldness, orientability,
# connectivity -- all gated in the self-tests below.  CHM-(1,2)
# measures chi = -8 with 2 ends per period (genus 4): Weber's (1,j)
# index adds ONE handle per step over CHM-(1,1)'s genus 3, so this is
# NOT the M_k subsequence of genus 2k+1 -- the harvest metadata gives
# no usable chi here and the measurement is the authority).
#
# References:
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988) 83-114 -- the fence of
#     catenoids and its handle additions;
#   M. J. Callahan, D. Hoffman, W. H. Meeks III, "Embedded minimal
#     surfaces with an infinite number of ends", Invent. Math. 96
#     (1989) 459-505 -- the CHM_k family;
#   C. J. Costa (1984); D. Hoffman, W. H. Meeks III (1985) -- the Costa
#     surface the translation-invariant tower descends from;
#   M. Weber, https://minimalsurfaces.blog/ -- the harvested notebooks
#     (Singly_Catenoid_1Handle_g1, Singly_Catenoid_2Handles_g3,
#     Singly_TransInvCosta_I, Singly_CHM_1_2;
#     research/msblog_harvest/singly_periodic.json).
# --------------------------------------------------------------------------

# solved period constants, harvested verbatim from the notebooks
_STINV_G1 = {
    0.1: -0.7054086037971196,
    0.2: -0.6309634409033806,
    0.3: -0.5610309089063753,
    0.5: -0.4191755389731957,
}
_STINV_G3 = {
    0.06: (0.8313415275984382, -0.7834727487665828),
    0.08: (0.8252014282724452, -0.7601944503629705),
    0.1: (0.8207141842757188, -0.7382148901607947),
    0.2: (0.8106797631408779, -0.6367415615956143),
    0.4: (0.816529866798894, -0.44635393033532667),
    0.6: (0.8409077253104242, -0.26280805739772073),
}
_STINV_COSTA = {
    -10.0: (-0.03333055237361433, 3.307161083209713),
    -5.0: (-0.06664857639434493, 2.336922594756168),
    -3.0: (-0.11107380129022093, 1.8071167824277032),
    -2.0: (-0.16689594688623235, 1.4701727470752084),
    -1.5: (-0.22432453091517268, 1.2657277793430088),
    -1.1: (-0.3252466601711385, 1.0666960764075604),
}
# CHM-(1,2): Newton-polished from the notebook seed (see block header)
_STINV_CHM12_A = 1.0735928348392014
_STINV_CHM12_B = 1.467713226486139
_STINV_CHM12_RHO = 1.4120744657811857
_STINV_CHM12_SEED = (1.07378157681789798, 1.46816068659935883,
                     1.41294757571306188)


def stinv_prod(z, pref, factors):
    """pref * prod (z - v)^p with principal per-factor powers (z must
    already be nudged into the closed upper half plane)."""
    out = np.full(z.shape, pref, dtype=complex)
    for v, p in factors:
        out = out * np.exp(p * np.log(z - v))
    return out


def stinv_g1_facs(a, b):
    f1 = ((0.0, -0.5), (a, -0.5), (b, 0.5), (1.0 / b, -0.5),
          (1.0 / a, 0.5))
    f2 = ((0.0, -1.5), (a, 0.5), (b, -0.5), (1.0 / b, 0.5),
          (1.0 / a, -0.5))
    return complex(a / b) ** 0.5, f1, f2


def stinv_g3_facs(a, b, c):
    f1 = ((0.0, -0.5), (a, -0.5), (b, 0.5), (1.0 / b, -0.5),
          (1.0 / a, 0.5), (c, 0.5), (1.0 / c, -0.5))
    f2 = ((0.0, -1.5), (a, 0.5), (b, -0.5), (1.0 / b, 0.5),
          (1.0 / a, -0.5), (c, -0.5), (1.0 / c, 0.5))
    return complex(-a / b / c) ** 0.5, f1, f2


def stinv_cat_om(pref, f1, f2):
    """(om1, om2, om3) for the catenoid-tower data (dh = dz/z)."""
    def om(z):
        z = sptail_nudge(np.asarray(z, dtype=complex))
        p1 = stinv_prod(z, pref, f1)
        p2 = stinv_prod(z, 1.0 / pref, f2)
        return (-0.5 * (p1 - p2), 0.5j * (p1 + p2), 1.0 / z)
    return om


def stinv_rgrid(rmin, rmax, marks, nu, log_end=True):
    """Radial grid on [rmin, rmax]: per-segment spacing (log toward the
    trimmed end at rmin) + geometric clusters into every interior mark
    (branch points demand graded cells for the compound GL rows)."""
    cuts = [rmin] + sorted(marks) + [rmax]
    nseg = max(6, int(nu / max(1, len(cuts) - 1)))
    parts = []
    for i in range(len(cuts) - 1):
        lo, hi = cuts[i], cuts[i + 1]
        if i == 0 and log_end:
            parts.append(np.exp(np.linspace(math.log(lo), math.log(hi),
                                            nseg)))
        else:
            parts.append(np.linspace(lo, hi, nseg))
        w = hi - lo
        if cuts[i] in marks:
            parts.append(cuts[i] + sptail_cluster(0.12 * w, 2e-7 * w))
        if cuts[i + 1] in marks:
            parts.append(cuts[i + 1] - sptail_cluster(0.12 * w,
                                                      2e-7 * w))
    r = np.unique(np.concatenate(parts))
    r = r[(r >= rmin - 1e-12) & (r <= rmax + 1e-12)]
    return r[np.concatenate([[True], np.diff(r) > 1e-13])]


def stinv_tgrid(nt, marks=()):
    """t grid on [0, pi]: Chebyshev base + geometric clusters into both
    edges and into every interior mark."""
    base = 0.5 * math.pi * (1.0 - np.cos(np.pi * np.linspace(
        0, 1, 2 * (nt // 2) + 1)))
    ex = sptail_cluster(0.06, hmin=1e-7, ratio=0.25)
    parts = [base, ex, math.pi - ex]
    for m in marks:
        cl = sptail_cluster(0.05, hmin=1e-8, ratio=0.3)
        parts += [m - cl, np.array([m]), m + cl[::-1]]
    t = np.unique(np.concatenate(parts))
    t = t[(t >= 0.0) & (t <= math.pi)]
    return t[np.concatenate([[True], np.diff(t) > 1e-13])]


def stinv_g1_build(a=0.5, nu=48, nt=32, storeys=1, r1=3.0):
    """Catenoid tower with one handle.  Frames {E, sigma_z} x {E, Mx}
    x {E, sigma_yB} per storey, translation T = (0, 2 dy, 0)."""
    b = _STINV_G1[a]
    om = stinv_cat_om(*stinv_g1_facs(a, b))
    rmin = a / r1
    r = stinv_rgrid(rmin, 1.0, [a, abs(b)], nu)
    t = stinv_tgrid(nt)
    X = sptail_polar_patch(om, r, t)
    i_a = int(np.searchsorted(r, a))
    i_b = int(np.searchsorted(r, abs(b)))
    xL1 = float(np.median(X[:i_a + 1, 0, 0]))
    xL2 = float(np.median(X[i_b:, -1, 0]))
    yA = float(np.median(X[i_a:, 0, 1]))
    z1 = float(np.median(X[-1, :, 2]))
    diag = {
        'xL1_ptp': float(np.ptp(X[:i_a + 1, 0, 0])),
        'yA_ptp': float(np.ptp(X[i_a:, 0, 1])),
        'yB_ptp': float(np.ptp(X[:i_b + 1, -1, 1])),
        'xL2_ptp': float(np.ptp(X[i_b:, -1, 0])),
        'arc_z_ptp': float(np.ptp(X[-1, :, 2])),
        'xmirror_gap': abs(xL1 - xL2)}
    X = X.copy()
    X[..., 0] -= 0.5 * (xL1 + xL2)
    X[..., 1] -= yA
    X[..., 2] -= z1
    D = float(np.median(X[:i_b + 1, -1, 1]))
    X[:i_a + 1, 0, 0] = 0.0
    X[i_b:, -1, 0] = 0.0
    X[i_a:, 0, 1] = 0.0
    X[:i_b + 1, -1, 1] = D
    X[-1, :, 2] = 0.0
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(r), len(t))
    T = np.array([0.0, 2.0 * D, 0.0])
    frames = []
    for sh in (0, 1):
        for bx in (0, 1):
            for cy in (0, 1):
                M = np.diag([-1.0 if bx else 1.0, -1.0 if cy else 1.0,
                             -1.0 if sh else 1.0])
                tv = np.array([0.0, 2.0 * D if cy else 0.0, 0.0])
                par = (-1.0) ** (sh + bx + cy)
                for s in range(storeys):
                    frames.append((M, tv + s * T, par))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(t)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def stinv_g1_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                  storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 8))
    V, F, uv, _ = stinv_g1_build(
        p['a'], nu=int(np.clip(int(0.8 * nu), 24, 130)),
        nt=int(np.clip(int(0.6 * nv), 20, 90)), storeys=S, r1=p['r1'])
    V = _center_fit(V, scale, V)
    return V, F, uv


def stinv_g3_build(a=0.1, nu=48, nt=32, storeys=1, r1=3.5):
    """Catenoid tower with two handles.  Frames {E, sigma_z} x
    {E, sigma_xC} x {E, My} per storey, translation T = (2 dx, 0, 0)."""
    b, c = _STINV_G3[a]
    om = stinv_cat_om(*stinv_g3_facs(a, b, c))
    rmin = a / r1
    r = stinv_rgrid(rmin, 1.0, [a, b, abs(c)], nu)
    t = stinv_tgrid(nt)
    X = sptail_polar_patch(om, r, t)
    i_a = int(np.searchsorted(r, a))
    i_b = int(np.searchsorted(r, b))
    i_c = int(np.searchsorted(r, abs(c)))
    y1 = float(np.median(X[:i_a + 1, 0, 1]))
    y2 = float(np.median(X[i_b:, 0, 1]))
    y3 = float(np.median(X[i_c:, -1, 1]))
    xA = float(np.median(X[i_a:i_b + 1, 0, 0]))
    z1 = float(np.median(X[-1, :, 2]))
    diag = {
        'y1_ptp': float(np.ptp(X[:i_a + 1, 0, 1])),
        'y2_ptp': float(np.ptp(X[i_b:, 0, 1])),
        'y3_ptp': float(np.ptp(X[i_c:, -1, 1])),
        'xA_ptp': float(np.ptp(X[i_a:i_b + 1, 0, 0])),
        'xC_ptp': float(np.ptp(X[:i_c + 1, -1, 0])),
        'arc_z_ptp': float(np.ptp(X[-1, :, 2])),
        'ymirror_gap': max(abs(y1 - y2), abs(y1 - y3))}
    X = X.copy()
    X[..., 0] -= xA
    X[..., 1] -= float(np.median([y1, y2, y3]))
    X[..., 2] -= z1
    D = float(np.median(X[:i_c + 1, -1, 0]))
    X[:i_a + 1, 0, 1] = 0.0
    X[i_b:, 0, 1] = 0.0
    X[i_c:, -1, 1] = 0.0
    X[i_a:i_b + 1, 0, 0] = 0.0
    X[:i_c + 1, -1, 0] = D
    X[-1, :, 2] = 0.0
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(r), len(t))
    T = np.array([2.0 * D, 0.0, 0.0])
    frames = []
    for sh in (0, 1):
        for bx in (0, 1):
            for cy in (0, 1):
                M = np.diag([-1.0 if bx else 1.0, -1.0 if cy else 1.0,
                             -1.0 if sh else 1.0])
                tv = np.array([2.0 * D if bx else 0.0, 0.0, 0.0])
                par = (-1.0) ** (sh + bx + cy)
                for s in range(storeys):
                    frames.append((M, tv + s * T, par))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(t)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def stinv_g3_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                  storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 8))
    V, F, uv, _ = stinv_g3_build(
        p['a'], nu=int(np.clip(int(0.8 * nu), 24, 130)),
        nt=int(np.clip(int(0.6 * nv), 20, 90)), storeys=S, r1=p['r1'])
    V = _center_fit(V, scale, V)
    return V, F, uv


def stinv_costa_om(a, b, rho):
    """(om1, om2, om3) on the Joukowski half-disk chart.  The interior
    nudge shrinks |zeta| with a 100x dominant radial weight -- the +i
    epsilon term alone puts arc points with sin(theta) > 1/2 BELOW the
    real z axis (Im z = eps1 sin t - eps2 sin^2 t), silently flipping
    every branch there.  The cap zeroes the divergent corner-end cells
    (masked off the mesh) so they cannot poison the cumulative sums."""
    def om(zeta):
        zeta = np.asarray(zeta, dtype=complex)
        zeta = (zeta + 1e-13j * (1.0 + np.abs(zeta))) * (1.0 - 1e-11)
        z = -0.5 * (zeta + 1.0 / zeta)
        dz = -0.5 * (1.0 - 1.0 / (zeta * zeta))
        p1 = rho * np.exp(-0.5 * np.log(z - a) - 0.5 * np.log(z - 1.0)
                          + 0.5 * np.log(z - b) - 0.5 * np.log(z + 1.0))
        p2 = (1.0 / rho) * np.exp(
            0.5 * np.log(z - a) - 1.5 * np.log(z - 1.0)
            - 0.5 * np.log(z - b) - 1.5 * np.log(z + 1.0))
        dh = 1.0 / ((z - 1.0) * (z + 1.0))
        o = (-0.5 * (p1 - p2) * dz, 0.5j * (p1 + p2) * dz, dh * dz)
        return tuple(np.where(np.isfinite(v) & (np.abs(v) < 1e7),
                              v, 0.0) for v in o)
    return om


def stinv_costa_build(a=-10.0, nu=52, nt=44, storeys=1, rmin=0.02,
                      delta=0.12):
    """Translation-invariant Costa I.  Frames {E, sigma_y} x {E, Mx}
    per storey, translation T = (0, 2 dy, 0); the z = -+1 catenoid ends
    are corner-masked disks of chart radius `delta`, the z = inf wing
    pair is the clean inner trim at |zeta| = rmin."""
    b, rho = _STINV_COSTA[a]
    om = stinv_costa_om(a, b, rho)
    zeta_a = -a - math.sqrt(a * a - 1.0)
    theta_b = math.acos(max(-1.0, min(1.0, -b)))
    cl1 = sptail_cluster(0.08, 1e-7)
    r = stinv_rgrid(rmin, 1.0, [zeta_a], nu)
    r = np.unique(np.concatenate([r, 1.0 - cl1]))
    r = r[(r >= rmin) & (r <= 1.0)]
    t = stinv_tgrid(nt, marks=(theta_b,))
    X = sptail_polar_patch(om, r, t)
    ZG = r[:, None] * np.exp(1j * t[None, :])
    valid = ((np.abs(ZG - 1.0) > delta) & (np.abs(ZG + 1.0) > delta)
             ).reshape(-1)
    i_a = int(np.searchsorted(r, zeta_a))
    jb = int(np.argmin(np.abs(t - theta_b)))
    vg = valid.reshape(len(r), len(t))

    def _med(vals, ok):
        vals = vals[ok]
        return (float(np.median(vals)), float(np.ptp(vals))) \
            if len(vals) else (0.0, 0.0)

    yA, yA_ptp = _med(X[:i_a + 1, 0, 1], vg[:i_a + 1, 0])
    xB1, xB1_ptp = _med(X[i_a:, 0, 0], vg[i_a:, 0])
    yC1, yC1_ptp = _med(X[:, -1, 1], vg[:, -1])
    xB2, xB2_ptp = _med(X[-1, jb:, 0], vg[-1, jb:])
    yC2, yC2_ptp = _med(X[-1, :jb + 1, 1], vg[-1, :jb + 1])
    diag = {
        'yA_ptp': yA_ptp, 'xB1_ptp': xB1_ptp, 'yC1_ptp': yC1_ptp,
        'xB2_ptp': xB2_ptp, 'yC2_ptp': yC2_ptp,
        'xmirror_gap': abs(xB1 - xB2),
        'ymirror_gap': abs(yC1 - yC2)}
    X = X.copy()
    X[..., 0] -= 0.5 * (xB1 + xB2)
    X[..., 1] -= 0.5 * (yC1 + yC2)
    yAc = float(yA - 0.5 * (yC1 + yC2))
    X[:i_a + 1, 0, 1] = np.where(vg[:i_a + 1, 0], yAc,
                                 X[:i_a + 1, 0, 1])
    X[i_a:, 0, 0] = np.where(vg[i_a:, 0], 0.0, X[i_a:, 0, 0])
    X[:, -1, 1] = np.where(vg[:, -1], 0.0, X[:, -1, 1])
    X[-1, jb:, 0] = np.where(vg[-1, jb:], 0.0, X[-1, jb:, 0])
    X[-1, :jb + 1, 1] = np.where(vg[-1, :jb + 1], 0.0,
                                 X[-1, :jb + 1, 1])
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(r), len(t), valid=valid)
    T = np.array([0.0, 2.0 * yAc, 0.0])
    frames = []
    for bx in (0, 1):
        for cy in (0, 1):
            M = np.diag([-1.0 if bx else 1.0, -1.0 if cy else 1.0,
                         1.0])
            par = (-1.0) ** (bx + cy)
            for s in range(storeys):
                frames.append((M, s * T, par))
    # span over USED vertices only -- the masked corner-zone vertices
    # carry meaningless (capped) coordinates and must not inflate the
    # weld tolerance
    used = np.unique(np.array([i for f in q0 for i in f],
                              dtype=np.int64))
    Vu = V0[used]
    span = float(np.linalg.norm(Vu.max(0) - Vu.min(0)))
    # every frame is a sign-diagonal matrix plus an exact multiple of T,
    # so seam partners are bitwise equal after snapping -- a near-exact
    # tolerance keeps the branch-point clusters from being fused
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(t)),
                                 q0, frames, 1e-12 * span)
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def stinv_costa_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                     storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 8))
    V, F, uv, _ = stinv_costa_build(
        p['a'], nu=int(np.clip(int(0.8 * nu), 28, 130)),
        nt=int(np.clip(int(0.7 * nv), 24, 100)), storeys=S,
        rmin=p['rmin'], delta=p['delta'])
    V = _center_fit(V, scale, V)
    return V, F, uv


# ---- CHM-(1,2) strip chart ------------------------------------------------

def stinv_chm12_forms(w):
    """(om1, om2, om3) pulled back to the strip chart (times dx/dw =
    e^w / 2x).  With x^2 - b^2 = e^w the (x^2-b^2)^{5/4} factor becomes
    an exact exponential; x stays in the first quadrant and x^2 - 1,
    x^2 - a^2 in the closed upper half plane, so every principal power
    is single-valued.  Strip corners at (xa, pi), (xb, pi), (x0, pi)."""
    a, b, rho = _STINV_CHM12_A, _STINV_CHM12_B, _STINV_CHM12_RHO
    w = np.asarray(w, dtype=complex)
    xa = math.log(b * b - a * a)
    xb = math.log(b * b - 1.0)
    x0 = math.log(b * b)
    # exact corner factorizations: b^2-a^2+e^w = -e^{xa} expm1(w-xa-ipi)
    # etc. -- the naive sums cancel catastrophically in the deep corner
    # clusters (za ~ e^{xa} h at distance h, below float resolution of
    # the direct difference once h < 1e-13)
    za = -math.exp(xa) * _cwce_expm1c(w - xa - 1j * math.pi)
    zm1 = -math.exp(xb) * _cwce_expm1c(w - xb - 1j * math.pi)
    x = np.sqrt(-math.exp(x0) * _cwce_expm1c(w - x0 - 1j * math.pi))
    q1 = np.sqrt(zm1) * np.exp(-0.25 * w) / (2.0 * x ** 1.5
                                             * za ** 0.75)
    q2 = np.exp(1.25 * w) / (2.0 * np.sqrt(x) * np.sqrt(zm1)
                             * za ** 0.25)
    om3 = np.exp(0.5 * w) / (2.0 * x * np.sqrt(za))
    return (-0.5 * (rho * q1 - q2 / rho),
            0.5j * (rho * q1 + q2 / rho), om3)


def stinv_chm12_ugrid(nu, umin, umax):
    """u grid: flare-uniform toward both trimmed ends (conformal end
    radii ~ e^{-u/4} and ~ e^{+u/4}), geometric clusters into the three
    singular corners xa, xb, x0 (exact grid nodes)."""
    b2 = _STINV_CHM12_B * _STINV_CHM12_B
    xa = math.log(b2 - _STINV_CHM12_A * _STINV_CHM12_A)
    xb = math.log(b2 - 1.0)
    x0 = math.log(b2)
    nE = max(10, int(0.22 * nu))
    uf1 = -4.0 * np.log(np.linspace(math.exp(-0.25 * umin),
                                    math.exp(-0.25 * (xa - 0.7)), nE))
    uf2 = 4.0 * np.log(np.linspace(math.exp(0.25 * (x0 + 0.7)),
                                   math.exp(0.25 * umax), nE))
    parts = [uf1, uf2]
    marks = [xa, xb, x0]
    segs = [(xa, xb), (xb, x0)]
    for lo, hi in segs:
        parts.append(np.linspace(lo, hi, max(6, int(0.12 * nu))))
    for m in marks:
        # deep clusters: the corner cells' quadrature bias scales as
        # hmin^(1/4) (measured), so hmin = 1e-14 buys real accuracy --
        # possible only with the exact expm1 corner factorizations in
        # stinv_chm12_forms; image spacing shrinks as hmin^(1/4) too,
        # so the deep nodes never collapse under the near-exact weld
        cl = sptail_cluster(0.03, hmin=1e-14, ratio=0.25)
        parts += [m - cl, np.array([m]), m + cl[::-1]]
    u = np.unique(np.concatenate(parts))
    u = u[(u >= umin) & (u <= umax)]
    return u[np.concatenate([[True], np.diff(u) > 1e-16])]


def stinv_chm12_build(nu=64, nt=40, storeys=1, umin=-4.0, umax=4.0,
                      wrap=False):
    """CHM-(1,2): one strip patch, 8 isometries {E, R_line} x {E, Mx}
    x {E, My} per storey, vertical translation T = (0, 0, 2 z_line).
    The spine column is pinned at u = umin + 0.3 (xa - umin), far from
    the singular corners (a spine near u = 0 sits within 3e-3 of the
    xa corner and its quadrature error pollutes every mirror offset)."""
    b2 = _STINV_CHM12_B * _STINV_CHM12_B
    xa = math.log(b2 - _STINV_CHM12_A * _STINV_CHM12_A)
    xb = math.log(b2 - 1.0)
    x0 = math.log(b2)
    u = stinv_chm12_ugrid(nu, umin, umax)
    t = _chmp_tgrid(nt)
    us = u[int(np.argmin(np.abs(u - (umin + 0.3 * (xa - umin)))))]
    X = sptail_rect_patch(lambda w: stinv_chm12_forms(w + us),
                          u - us, t)
    i_a = int(np.searchsorted(u, xa))
    i_b = int(np.searchsorted(u, xb))
    i_0 = int(np.searchsorted(u, x0))
    F0 = X[i_0, -1, :].copy()                 # the x = 0 corner
    X = X - F0[None, None, :]
    sL1 = X[:i_a + 1, -1, :]
    zL = float(np.median(sL1[:, 2]))
    diag = {
        'L1_z_ptp': float(np.ptp(sL1[:, 2])),
        'L1_diag_ptp': float(np.ptp(sL1[:, 0] + sL1[:, 1])),
        'y_mid_ptp': float(np.ptp(X[i_a:i_b + 1, -1, 1])),
        'y_mid_off': abs(float(np.median(X[i_a:i_b + 1, -1, 1]))),
        'x_mid_ptp': float(np.ptp(X[i_b:i_0 + 1, -1, 0])),
        'x_mid_off': abs(float(np.median(X[i_b:i_0 + 1, -1, 0]))),
        'L0_z_ptp': float(np.ptp(X[i_0:, -1, 2])),
        'L0_diag_ptp': float(np.ptp(X[i_0:, -1, 0] + X[i_0:, -1, 1])),
        't0_y_ptp': float(np.ptp(X[:, 0, 1])),
        't0_y_off': abs(float(np.median(X[:, 0, 1]))),
        'zL': zL}
    # snap every boundary sub-edge exactly onto its symmetry element
    m = 0.5 * (X[:i_a + 1, -1, 0] - X[:i_a + 1, -1, 1])
    X[:i_a + 1, -1, 0] = m
    X[:i_a + 1, -1, 1] = -m
    X[:i_a + 1, -1, 2] = zL
    X[i_a:i_b + 1, -1, 1] = 0.0
    X[i_b:i_0 + 1, -1, 0] = 0.0
    m = 0.5 * (X[i_0:, -1, 0] - X[i_0:, -1, 1])
    X[i_0:, -1, 0] = m
    X[i_0:, -1, 1] = -m
    X[i_0:, -1, 2] = 0.0
    X[:, 0, 1] = 0.0
    X[i_a, -1, :] = (0.0, 0.0, zL)             # x = a corner: line meets
    #                                            the y = 0 mirror on the
    #                                            vertical axis (CHM-(1,1)
    #                                            axis-point analog)
    X[i_b, -1, :] = (0.0, 0.0, X[i_b, -1, 2])  # x = 1 corner: Mx + My
    X[i_0, -1, :] = (0.0, 0.0, 0.0)            # x = 0 corner: origin
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(u), len(t))
    T = np.array([0.0, 0.0, 2.0 * zL])
    RL = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0]])
    frames = []
    for e in (0, 1):
        for bx in (0, 1):
            for cy in (0, 1):
                M = np.eye(3)
                if e:
                    M = RL.copy()
                if bx:
                    M = np.diag([-1.0, 1.0, 1.0]) @ M
                if cy:
                    M = np.diag([1.0, -1.0, 1.0]) @ M
                par = (-1.0) ** (e + bx + cy)
                for s in range(storeys):
                    frames.append((M, s * T, par))
    # COMBINATORIAL weld (the CHM-(1,1) scheme): every seam pairs a
    # frame with its partner frame at IDENTICAL within-patch indices.
    # No coincidence tolerance is involved, so the deep corner clusters
    # (grid steps down to 1e-14, image spacing |om| h ~ 1e-13) can
    # never be fused into sliver-collapse holes.  Pairing algebra for
    # g = My^cy Mx^bx R^e (+ sT):  R My = Mx R,  R Mx = My R,
    # R T = T^-1 R; the L1 line is fixed by T R.
    #   y = 0 seams (t=0 row, (xa,xb) row): toggle cy if e=0 else bx
    #   x = 0 seam  ((xb,x0) row):          toggle bx if e=0 else cy
    #   L0 line     ((x0,umax) row):        toggle e
    #   L1 line     ((umin,xa) row):        toggle e; s+1 (e=0) / s-1
    nu_, nt_ = len(u), len(t)
    S = storeys

    def fidx(e, bx, cy, s):
        return ((e * 2 + bx) * 2 + cy) * S + s

    t0r = np.arange(nu_) * nt_
    ymid = np.arange(i_a, i_b + 1) * nt_ + (nt_ - 1)
    xmid = np.arange(i_b, i_0 + 1) * nt_ + (nt_ - 1)
    l0r = np.arange(i_0, nu_) * nt_ + (nt_ - 1)
    l1r = np.arange(0, i_a + 1) * nt_ + (nt_ - 1)
    seams = []
    for e in (0, 1):
        for bx in (0, 1):
            for cy in (0, 1):
                for s in range(S):
                    me = fidx(e, bx, cy, s)
                    ytog = fidx(e, bx, 1 - cy, s) if e == 0 \
                        else fidx(e, 1 - bx, cy, s)
                    xtog = fidx(e, 1 - bx, cy, s) if e == 0 \
                        else fidx(e, bx, 1 - cy, s)
                    seams.append((t0r, me, ytog))
                    seams.append((ymid, me, ytog))
                    seams.append((xmid, me, xtog))
                    seams.append((l0r, me, fidx(1 - e, bx, cy, s)))
                    s2 = s + 1 if e == 0 else s - 1
                    if wrap:
                        # test-only translation-wrapped quotient: close
                        # the L1 period seam cyclically (measures the
                        # true per-period chi and end count)
                        seams.append((l1r, me,
                                      fidx(1 - e, bx, cy, s2 % S)))
                    elif 0 <= s2 < S:
                        seams.append((l1r, me, fidx(1 - e, bx, cy, s2)))
    UV0 = _sptail_grid_uv(nu_, nt_)
    nV = len(V0)
    Vp, Fp = [], []
    for fr, (M, tv, par) in enumerate(frames):
        Vp.append(V0 @ M.T + tv)
        off = fr * nV
        if par < 0:
            Fp.extend(tuple(off + i for i in f[::-1]) for f in q0)
        else:
            Fp.extend(tuple(off + i for i in f) for f in q0)
    Vall = np.concatenate(Vp, axis=0)
    UVall = np.tile(UV0, (len(frames), 1))
    parent = np.arange(len(Vall))

    def _find(a2):
        while parent[a2] != a2:
            parent[a2] = parent[parent[a2]]
            a2 = parent[a2]
        return a2

    for idx, fA, fB in seams:
        if fA >= fB:
            continue
        for i in idx:
            ra, rb = _find(fA * nV + int(i)), _find(fB * nV + int(i))
            if ra != rb:
                parent[ra] = rb
    roots = np.array([_find(a2) for a2 in range(len(Vall))])
    uniq, inv = np.unique(roots, return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    UVw = np.zeros((len(uniq), 2))
    cw = np.zeros(len(uniq))
    np.add.at(Vw, inv, Vall)
    np.add.at(UVw, inv, UVall)
    np.add.at(cw, inv, 1)
    Vw /= cw[:, None]
    UVw /= cw[:, None]
    F = []
    for f in Fp:
        g = [int(inv[i]) for i in f]
        h = [g[0]]
        for sgi in range(1, len(g)):
            if g[sgi] != h[-1]:
                h.append(g[sgi])
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            F.append(tuple(h))
    used = sorted({i for f in F for i in f})
    remap = np.full(len(uniq), -1, dtype=np.int64)
    remap[np.array(used, dtype=np.int64)] = np.arange(len(used))
    V = Vw[np.array(used, dtype=np.int64)]
    uv = UVw[np.array(used, dtype=np.int64)]
    F = [tuple(int(remap[i]) for i in f) for f in F]
    span = float(np.linalg.norm(V.max(0) - V.min(0)))
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def stinv_chm12_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                     storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = stinv_chm12_build(
        nu=int(np.clip(nu, 28, 200)),
        nt=int(np.clip(int(0.8 * nv), 20, 160)), storeys=S,
        umin=p['umin'], umax=p['umax'])
    V = _center_fit(V, scale, V)
    return V, F, uv


# ---- screw-motion CHM (theta-function data on the tau-torus) --------------
# Weber's screw-motion deformation of CHM-(1,1): dh = i dz on the
# rectangular torus C/<1, tau> (tau = i Im tau), Gauss map
#   G(z) = G0(z)/G0(tau/4),
#   G0   = th(z-a)^{-3/2} th(z+a+tau/2)^{3/4} th(z+a-tau/2)^{3/4}
#          th(z+b)^{1/2} th(z-b-tau/2)^{-1/4} th(z-b+tau/2)^{-1/4}
# (th = Jacobi theta_11).  The two solved constants (a, b) = (u, v) per
# tau are harvested verbatim from Singly_ScrewMotion_CHM.nb (FindRoot
# output; both period conditions re-verified here to ~1e-5).  The
# fundamental rectangle D = [0,1] x [0, Im tau/2] carries two ends (the
# theta divisor points z = a and z = 1-a+tau/2, trimmed as masked
# disks), two vertical-normal points z = 1-b, b+tau/2, and FOUR
# horizontal straight lines on its edges: Lt1 (top edge across the
# corner), Lt2 (top middle), Lb1 (bottom across the corner), Lb2
# (bottom middle).  Per-factor principal theta powers are continuous
# on D away from the end disks (verified numerically), so no branch
# tracking is needed.
#
# Group structure (measured, then EXACTIFIED): the x -> x+1 deck sigma
# is an exact half-turn about a VERTICAL axis (angle pi, rise 0!); the
# half-turn axes A := Rot(Lt1) and C := Rot(Lb1) are horizontal lines
# meeting sigma's axis, so sigma is central and the TOWER SCREW is
# lambda = A o C -- a vertical screw with rise exactly Im tau whose
# twist angle is twice the angle between the two line directions.  The
# generators are rebuilt exactly in that form (axes forced through the
# common vertical, displacement ~ the closure residual ~1e-6), every
# frame is the exact word lambda^s A^a sigma^e ({E, A, sigma, A sigma}
# per storey), and every seam welds COMBINATORIALLY -- the Lt2/Lb2
# continuations resolve to the words A sigma and C sigma (axis
# distance ~1e-7, printed by the self-test), the x-seam pairs frame g
# with g sigma.  A and C are Schwarz continuations (anti-conformal on
# the domain), so they flip the face winding; sigma and lambda do not.
# Topology is MEASURED in the self-tests: chi = -6 with 2 ends per
# lambda-period (quotient genus 3 -- the screw-deformed CHM-(1,1),
# same genus as its undeformed limit), manifold and oriented; the
# planar-end trim rims SPIRAL across period boundaries (a screw
# surface's rims close only in the full tower), so rim loops are not
# gated per period.
# References: H. Karcher (Manuscripta Math. 62, 1988); M. Callahan,
# D. Hoffman, W. H. Meeks III (Invent. Math. 96, 1989); M. Weber,
# https://minimalsurfaces.blog/ (Screw Motion CHM notebook).

_STINV_SCREW = {
    0.6: (0.20086439129905187, 0.3748366987539473),
    0.7: (0.18511053052190832, 0.39446970599451003),
    0.8: (0.16378924815066304, 0.4140041171462015),
    0.9: (0.13429097753439428, 0.43453946982282166),
    1.0: (0.0890762704535419, 0.4593474241183917),
    1.05: (0.04910253352674421, 0.4782496107553167),
}


def stinv_screw_om(timag, cap=1e6):
    """(om1, om2, om3) on the tau-torus; principal per-factor theta
    powers (continuous on the fundamental rectangle away from the two
    end disks); divergent end-zone values are capped to zero (their
    cells are masked off the mesh)."""
    tau = 1j * timag
    ua, vb = _STINV_SCREW[timag]
    th = genus1helicoid_theta11
    fac = ((-ua, -1.5), (ua + tau / 2, 0.75), (ua - tau / 2, 0.75),
           (vb, 0.5), (-vb - tau / 2, -0.25), (-vb + tau / 2, -0.25))

    def G0(z):
        out = np.ones(np.shape(z), dtype=complex)
        for s, e in fac:
            out = out * th(z + s, tau) ** e
        return out
    g0c = complex(G0(np.array(tau / 4)))

    def om(z):
        z = np.asarray(z, dtype=complex)
        G = G0(z) / g0c
        o1 = 0.5j * (1.0 / G - G)
        o2 = -0.5 * (1.0 / G + G)
        o3 = np.full(z.shape, 1j)
        return tuple(np.where(np.isfinite(c) & (np.abs(c) < cap),
                              c, 0.0) for c in (o1, o2, o3))
    return om, ua, vb


def stinv_screw_patch(om, x, y):
    """Integrate om over the rect grid (x_i, y_j) with the SPINE along
    the middle row y ~ h/2 (no boundary singularity ever sits on the
    spine; the masked end disks touch only the y-edges, so column-wise
    integration reaches every unmasked cell cleanly)."""
    xg, wg = _SPTAIL_GL
    nx, ny = len(x), len(y)
    jm = int(np.argmin(np.abs(y - 0.5 * (y[0] + y[-1]))))
    dx = np.diff(x)
    xmid = 0.5 * (x[1:] + x[:-1])
    zs = (xmid[:, None] + 0.5 * dx[:, None] * xg[None, :]) \
        + 1j * y[jm]
    o = om(zs)
    incS = np.stack([np.sum(c * wg, axis=-1) for c in o], axis=-1) \
        * (0.5 * dx)[:, None]
    S = np.zeros((nx, 3), complex)
    S[1:] = np.cumsum(incS, axis=0)
    dy = np.diff(y)
    ymid = 0.5 * (y[1:] + y[:-1])
    Z = x[:, None, None] + 1j * (
        ymid[None, :, None] + 0.5 * dy[None, :, None] * xg)
    o = om(Z)
    incC = np.stack([np.sum(c * (1j) * wg, axis=-1) for c in o],
                    axis=-1) * (0.5 * dy)[None, :, None]
    C = np.zeros((nx, ny, 3), complex)
    C[:, 1:, :] = np.cumsum(incC, axis=1)
    C = C - C[:, jm, :][:, None, :]
    return np.real(S[:, None, :] + C)


def _stinv_line_fit(P):
    """(point, unit direction, max residual) of the best-fit 3D line."""
    c = P.mean(axis=0)
    d = P - c
    _, _, Vt = np.linalg.svd(d, full_matrices=False)
    n = Vt[0]
    res = float(np.max(np.linalg.norm(
        d - np.outer(d @ n, n), axis=-1))) if len(P) > 1 else 0.0
    return c, n, res


def _stinv_rot180(c, n):
    """(M, tvec) of the 180-degree rotation about the line c + s n."""
    M = 2.0 * np.outer(n, n) - np.eye(3)
    return M, c - M @ c


def _stinv_kabsch(P, Q):
    """Rigid motion (M, t) minimizing |M P + t - Q| (proper rotation)."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    M = Vt.T @ D @ U.T
    return M, cq - M @ cp


def stinv_screw_build(timag=0.8, nu=64, nt=40, storeys=1, delta=0.10):
    """Screw-motion CHM: one rectangle patch, frames
    {sigma^s} x {E, A, C, CA}.  Returns (V, F, uv, diag)."""
    om, ua, vb = stinv_screw_om(timag)
    h = 0.5 * timag
    e1 = complex(ua, 0.0)
    e2 = complex(1.0 - ua, h)
    zb1 = 1.0 - vb                     # theta zero on the bottom edge
    zt1 = vb                           # theta zero on the top edge
    marks_x = sorted({ua, zb1, zt1, 1.0 - ua})
    nseg = max(5, int(nu / (len(marks_x) + 1)))
    parts = []
    cuts = [0.0] + marks_x + [1.0]
    for i in range(len(cuts) - 1):
        lo, hi = cuts[i], cuts[i + 1]
        parts.append(np.linspace(lo, hi, nseg))
        w = hi - lo
        if cuts[i] in marks_x:
            parts.append(cuts[i] + sptail_cluster(0.12 * w, 2e-7 * w))
        if cuts[i + 1] in marks_x:
            parts.append(cuts[i + 1] - sptail_cluster(0.12 * w,
                                                      2e-7 * w))
    x = np.unique(np.concatenate(parts))
    x = x[(x >= 0.0) & (x <= 1.0)]
    x = x[np.concatenate([[True], np.diff(x) > 1e-12])]
    ycl = sptail_cluster(0.10 * h, 2e-7 * h)
    y = np.unique(np.concatenate(
        [np.linspace(0.0, h, max(12, nt)), ycl, h - ycl]))
    y = y[(y >= 0.0) & (y <= h)]
    y = y[np.concatenate([[True], np.diff(y) > 1e-12])]
    X = stinv_screw_patch(om, x, y)
    nx, ny = len(x), len(y)
    ZG = x[:, None] + 1j * y[None, :]
    # lattice-aware end masks: a disk crossing the x = 0 / x = 1 screw
    # seam must cut BOTH sides identically, or the seam weld leaves
    # unpaired faces
    valid = np.ones(ZG.shape, dtype=bool)
    for ec in (e1, e2):
        for sh in (-1.0, 0.0, 1.0):
            valid &= np.abs(ZG - (ec + sh)) > delta
    valid = valid.reshape(-1)
    vg = valid.reshape(nx, ny)
    i_u = int(np.searchsorted(x, ua))
    i_b1 = int(np.searchsorted(x, zb1))
    i_t1 = int(np.searchsorted(x, zt1))
    i_u2 = int(np.searchsorted(x, 1.0 - ua))
    # measured straight-line axes (masked verts excluded)
    selLb1 = np.concatenate([np.arange(0, i_u + 1),
                             np.arange(i_b1, nx)])
    selLb1 = selLb1[vg[selLb1, 0]]
    Lb1 = _stinv_line_fit(X[selLb1, 0, :])
    selLb2 = np.arange(i_u, i_b1 + 1)
    selLb2 = selLb2[vg[selLb2, 0]]
    Lb2 = _stinv_line_fit(X[selLb2, 0, :])
    selLt1 = np.concatenate([np.arange(0, i_t1 + 1),
                             np.arange(i_u2, nx)])
    selLt1 = selLt1[vg[selLt1, -1]]
    Lt1 = _stinv_line_fit(X[selLt1, -1, :])
    selLt2 = np.arange(i_t1, i_u2 + 1)
    selLt2 = selLt2[vg[selLt2, -1]]
    Lt2 = _stinv_line_fit(X[selLt2, -1, :])
    diag = {'Lb1_res': Lb1[2], 'Lb2_res': Lb2[2],
            'Lt1_res': Lt1[2], 'Lt2_res': Lt2[2]}
    # ---- EXACTIFIED group -------------------------------------------
    # sigma (the x -> x+1 deck) measures as an exact half-turn about a
    # VERTICAL axis (angle pi, rise 0), so the group is generated by
    # sigma and the two edge-line half-turns A (Lt1), C (Lb1) with
    # sigma central and A, C axes both meeting sigma's axis; the tower
    # screw is lambda = A o C (vertical screw, rise 2(zA - zC)).  Force
    # that structure exactly: both line axes are made horizontal and
    # are translated (by ~ the period-closure residual) to pass through
    # the common vertical axis q; sigma becomes diag(-1,-1,1) about q
    # -- an exact involution.  Every frame is then an exact WORD
    # lambda^s A^a sigma^e and every seam welds combinatorially.
    nT = Lt1[1].copy()
    nT[2] = 0.0
    nT /= np.linalg.norm(nT)
    nB = Lb1[1].copy()
    nB[2] = 0.0
    nB /= np.linalg.norm(nB)
    # q = intersection of the two axes' xy-projections
    Amat = np.array([[nT[0], -nB[0]], [nT[1], -nB[1]]])
    rhs = np.array([Lb1[0][0] - Lt1[0][0], Lb1[0][1] - Lt1[0][1]])
    t12 = np.linalg.solve(Amat, rhs)
    q = np.array([Lt1[0][0] + t12[0] * nT[0],
                  Lt1[0][1] + t12[0] * nT[1]])
    zA = float(np.median(X[selLt1, -1, 2]))
    zC = float(np.median(X[selLb1, 0, 2]))
    cA = np.array([q[0], q[1], zA])
    cC = np.array([q[0], q[1], zC])
    Afr = _stinv_rot180(cA, nT)
    Cfr = _stinv_rot180(cC, nB)
    Sfr = (np.diag([-1.0, -1.0, 1.0]),
           np.array([2.0 * q[0], 2.0 * q[1], 0.0]))
    diag['axis_shift'] = float(max(
        np.linalg.norm(np.cross(Lt1[0] - cA, nT)),
        np.linalg.norm(np.cross(Lb1[0] - cC, nB))))

    def _mul(g1, g2):
        return (g1[0] @ g2[0], g1[0] @ g2[1] + g1[1])

    def _word(s, a, e):
        """Frame of lambda^s A^a sigma^e (lambda = A o C)."""
        out = (np.eye(3), np.zeros(3))
        lam = _mul(Afr, Cfr)
        lami = _mul(Cfr, Afr)          # lambda^-1 = C o A (involutions)
        for _ in range(abs(s)):
            out = _mul(out, lam if s > 0 else lami)
        if a:
            out = _mul(out, Afr)
        if e:
            out = _mul(out, Sfr)
        return out

    # snap the four line edges onto their EXACT axes: Lt1 -> A-axis,
    # Lb1 -> C-axis; Lt2/Lb2 -> the axes of their continuation WORDS
    # (searched below), so those seams weld combinatorially too
    def _snap_line(sel, jj, c, n):
        P = X[sel, jj, :]
        X[sel, jj, :] = c + np.outer((P - c) @ n, n)

    _snap_line(selLt1, -1, cA, nT)
    _snap_line(selLb1, 0, cC, nB)

    def _axis_of(fr):
        """Axis (point, dir) of a 180-degree rotation frame."""
        M, tv = fr
        w_, vec_ = np.linalg.eigh(0.5 * (M + M.T))
        n_ = vec_[:, np.argmax(w_)]
        p_ = np.linalg.lstsq(np.eye(3) - M, tv, rcond=None)[0]
        return p_, n_

    def _line_dist(fr, c, n, m=None):
        p_, n_ = _axis_of(fr)
        d1 = np.linalg.norm(np.cross(c - p_, n_))
        d2 = 1.0 - abs(float(n @ n_))
        return d1 + d2

    def _search_word(cfit, nfit):
        best = None
        for k in (-1, 0, 1, 2, -2):
            for e in (0, 1):
                fr = _word(k, 1, e)
                d = _line_dist(fr, cfit, nfit)
                if best is None or d < best[0]:
                    best = (d, k, e, fr)
        return best

    dB, kB, eB, frB = _search_word(Lt2[0], Lt2[1])
    dD, kD, eD, frD = _search_word(Lb2[0], Lb2[1])
    diag['wordB'] = (kB, eB, dB)
    diag['wordD'] = (kD, eD, dD)
    pB, nBax = _axis_of(frB)
    pD, nDax = _axis_of(frD)
    _snap_line(selLt2, -1, pB, nBax)
    _snap_line(selLb2, 0, pD, nDax)
    # screw seam: x=1 edge := sigma(x=0 edge), bitwise
    sel0 = np.arange(ny)[vg[0, :]]
    Msg, tsg = Sfr
    diag['screw_fit'] = float(np.max(np.linalg.norm(
        X[0, sel0, :] @ Msg.T + tsg - X[-1, sel0, :], axis=-1)))
    X[-1, :, :] = X[0, :, :] @ Msg.T + tsg
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(nx, ny, valid=valid)
    # ---- frames: lambda^s A^a sigma^e -------------------------------
    S = storeys
    lab = []                          # frame order: (s, a, e)
    for s in range(S):
        for a in (0, 1):
            for e in (0, 1):
                lab.append((s, a, e))
    lidx = {t: i for i, t in enumerate(lab)}
    frames = []
    for (s, a, e) in lab:
        M, tv = _word(s, a, e)
        frames.append((M, tv, (-1.0) ** a))

    def _rmul(t, gen):
        """Right-multiply label t = (s, a, e) by a generator."""
        s, a, e = t
        if gen == 'sig':
            return (s, a, 1 - e)
        if gen == 'A':
            return (s, 1 - a, e)
        if gen == 'lam':
            return (s + (1 if a == 0 else -1), a, e)
        if gen == 'lami':
            return (s - (1 if a == 0 else -1), a, e)
        if gen == 'C':                 # C = lambda^-1 A
            s2, a2, e2 = _rmul(t, 'lami')
            return (s2, 1 - a2, e2)
        raise ValueError(gen)

    def _rmul_word(t, k, e2):
        """Right-multiply by lambda^k A sigma^e2."""
        for _ in range(abs(k)):
            t = _rmul(t, 'lam' if k > 0 else 'lami')
        t = _rmul(t, 'A')
        for _ in range(e2):
            t = _rmul(t, 'sig')
        return t

    # ---- combinatorial seams ----------------------------------------
    idx_x1 = (nx - 1) * ny + np.arange(ny)
    idx_x0 = np.arange(ny)
    iLt1 = selLt1 * ny + (ny - 1)
    iLb1 = selLb1 * ny
    iLt2 = selLt2 * ny + (ny - 1)
    iLb2 = selLb2 * ny
    seams = []                        # (idxA, idxB, frameA, frameB)
    for t in lab:
        me = lidx[t]

        def _pair(idxA, idxB, t2):
            if t2 in lidx:
                seams.append((idxA, idxB, me, lidx[t2]))
        _pair(idx_x1, idx_x0, _rmul(t, 'sig'))
        _pair(iLt1, iLt1, _rmul(t, 'A'))
        _pair(iLb1, iLb1, _rmul(t, 'C'))
        _pair(iLt2, iLt2, _rmul_word(t, kB, eB))
        _pair(iLb2, iLb2, _rmul_word(t, kD, eD))
    nV = len(V0)
    Vp, Fp = [], []
    for fr, (M, tv, par) in enumerate(frames):
        Vp.append(V0 @ M.T + tv)
        off = fr * nV
        if par < 0:
            Fp.extend(tuple(off + i for i in f[::-1]) for f in q0)
        else:
            Fp.extend(tuple(off + i for i in f) for f in q0)
    Vall = np.concatenate(Vp, axis=0)
    UVall = np.tile(_sptail_grid_uv(nx, ny), (len(frames), 1))
    parent = np.arange(len(Vall))

    def _find(a2):
        while parent[a2] != a2:
            parent[a2] = parent[parent[a2]]
            a2 = parent[a2]
        return a2

    for idxA, idxB, fA, fB in seams:
        if fA == fB:
            continue
        for iA, iB in zip(idxA, idxB):
            ra = _find(fA * nV + int(iA))
            rb = _find(fB * nV + int(iB))
            if ra != rb:
                parent[ra] = rb
    roots = np.array([_find(a2) for a2 in range(len(Vall))])
    uniq, inv = np.unique(roots, return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    UVw = np.zeros((len(uniq), 2))
    cw = np.zeros(len(uniq))
    np.add.at(Vw, inv, Vall)
    np.add.at(UVw, inv, UVall)
    np.add.at(cw, inv, 1)
    Vw /= cw[:, None]
    UVw /= cw[:, None]
    F = []
    for f in Fp:
        g = [int(inv[i]) for i in f]
        hh = [g[0]]
        for sgi in range(1, len(g)):
            if g[sgi] != hh[-1]:
                hh.append(g[sgi])
        if len(hh) >= 3 and hh[0] != hh[-1] and len(set(hh)) == len(hh):
            F.append(tuple(hh))
    used2 = sorted({i for f in F for i in f})
    remap = np.full(len(uniq), -1, dtype=np.int64)
    remap[np.array(used2, dtype=np.int64)] = np.arange(len(used2))
    V = Vw[np.array(used2, dtype=np.int64)]
    uv = UVw[np.array(used2, dtype=np.int64)]
    F = [tuple(int(remap[i]) for i in f) for f in F]
    span = float(np.linalg.norm(V.max(0) - V.min(0)))
    diag['span'] = span
    lamfr = _mul(Afr, Cfr)
    diag['lambda'] = lamfr
    diag['rise'] = 2.0 * (zA - zC)
    return V, F, uv, diag


def stinv_screw_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                     storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 8))
    V, F, uv, _ = stinv_screw_build(
        p['timag'], nu=int(np.clip(int(0.8 * nu), 32, 130)),
        nt=int(np.clip(int(0.6 * nv), 20, 90)), storeys=S,
        delta=p['delta'])
    V = _center_fit(V, scale, V)
    return V, F, uv


def stinv_screw_residuals(n=3000):
    """The notebook's two period conditions per shipped tau (det
    conditions on the horizontal period vectors; see the header)."""
    out = {}
    for timag, (ua, vb) in _STINV_SCREW.items():
        tau = 1j * timag
        om, _, _ = stinv_screw_om(timag, cap=1e12)

        def seg(wp, e_end=(0.0, 0.0)):
            tot = np.zeros(2)
            m = len(wp) - 1
            for i in range(m):
                A2, B2 = complex(wp[i]), complex(wp[i + 1])
                t = np.linspace(0.0, 1.0, n)
                if (e_end[0] if i == 0 else 0.0) > 0:
                    t = t ** 3.0
                if (e_end[1] if i == m - 1 else 0.0) > 0:
                    t = 1.0 - (1.0 - t) ** 3.0
                z = A2 + t * (B2 - A2)
                o1, o2, _ = om(z)
                dz = np.diff(z)
                tot[0] += np.real(np.sum(0.5 * (o1[1:] + o1[:-1]) * dz))
                tot[1] += np.real(np.sum(0.5 * (o2[1:] + o2[:-1]) * dz))
            return tot
        p0 = 0.5 + tau / 4
        I1 = seg([p0, vb + tau / 2], (0.0, 0.5))
        om1v, om2v, _ = om(np.array(complex(p0)))
        # want I1 parallel to the horizontal normal projection at p0
        # recover G from the om pair: om2 + i om1 = -1/G
        Gv = complex(-1.0 / (om2v + 1j * om1v))
        d = abs(Gv) ** 2 + 1.0
        n2 = np.array([2.0 * Gv.real / d, 2.0 * Gv.imag / d])
        c1 = I1[0] * n2[1] - I1[1] * n2[0]
        Ia = seg([tau / 2, tau / 4, vb + tau / 2], (0.0, 0.5))
        Ib = seg([tau / 2, 0.5 + tau / 4, 1 + tau / 2])
        c2 = Ia[0] * Ib[1] - Ia[1] * Ib[0]
        out[f'screw tau={timag}i'] = max(abs(c1), abs(c2))
    return out


def stinv_period_residuals(n=600):
    """Machine-check every shipped constant set against its notebook
    period conditions (the honesty gate; see the block header)."""
    out = {}
    for a, b in _STINV_G1.items():
        pref, f1, f2 = stinv_g1_facs(a, b)

        def om1(z):
            z = sptail_nudge(np.asarray(z, dtype=complex))
            return -0.5 * (stinv_prod(z, pref, f1)
                           - stinv_prod(z, 1.0 / pref, f2))
        out[f'g1 a={a}'] = abs(np.real(cwce_path_int(
            om1, [a, 1j, b], (0.5, 0.5), n)))
    for a, (b, c) in _STINV_G3.items():
        pref, f1, f2 = stinv_g3_facs(a, b, c)

        def om2(z):
            z = sptail_nudge(np.asarray(z, dtype=complex))
            return 0.5j * (stinv_prod(z, pref, f1)
                           + stinv_prod(z, 1.0 / pref, f2))
        r1 = np.real(cwce_path_int(
            om2, [a, 0.5 * (a + b) + 0.25j, b], (0.5, 0.5), n))
        r2 = np.real(cwce_path_int(om2, [a, 1j, c], (0.5, 0.5), n))
        out[f'g3 a={a}'] = max(abs(r1), abs(r2))
    for a, (b, rho) in _STINV_COSTA.items():
        f1 = ((a, -0.5), (1.0, -0.5), (b, 0.5), (-1.0, -0.5))
        f2 = ((a, 0.5), (1.0, -1.5), (b, -0.5), (-1.0, -1.5))

        def omj(z, j):
            z = sptail_nudge(np.asarray(z, dtype=complex))
            p1 = stinv_prod(z, complex(rho), f1)
            p2 = stinv_prod(z, complex(1.0 / rho), f2)
            return -0.5 * (p1 - p2) if j == 1 else 0.5j * (p1 + p2)
        r1 = np.real(cwce_path_int(lambda z: omj(z, 2),
                                   [b, 1j, 200.0], (0.5, 0.0), n))
        r2 = np.real(cwce_path_int(lambda z: omj(z, 1),
                                   [a, 1j, b], (0.5, 0.5), n))
        out[f'costa a={a}'] = max(abs(r1), abs(r2))
    # CHM-(1,2): the notebook's regularized ratio conditions
    a, b = _STINV_CHM12_A, _STINV_CHM12_B

    def P(z):
        return (z ** 2 * (z ** 2 - 1.0) ** 2 * (z ** 2 - a * a) ** 3
                * (z ** 2 - b * b))

    def Areg(z):
        return ((b * b - z ** 2 - 2 * a * a * z ** 2 - b * b * z ** 2
                 + 3 * z ** 4) / (b * b * (b - a) * (a + b)))

    def om1i(z):
        z = np.asarray(z, dtype=complex)
        return Areg(z) * np.asarray(P(z), complex) ** -0.25

    def om2i(z):
        z = np.asarray(z, dtype=complex)
        return (z ** 2 * (z ** 2 - 1.0 + 0j) ** -2.0
                * (z ** 2 - a * a + 0j) ** -1.0
                * (z ** 2 - b * b + 0j)) ** 0.25
    o1a = cwce_path_int(om1i, [0.0, 1.0], (0.5, 0.5), n)
    o1b = cwce_path_int(om1i, [1.0, a], (0.5, 0.75), n)
    o1c = cwce_path_int(om1i, [a, b], (0.75, 0.25), n)
    o2a = cwce_path_int(om2i, [0.0, 1.0], (0.0, 0.5), n)
    o2b = cwce_path_int(om2i, [1.0, a], (0.5, 0.25), n)
    o2c = cwce_path_int(om2i, [a, b], (0.25, 0.0), n)
    out['chm12'] = max(abs(np.real(o2b / o2a + o1b / o1a)),
                       abs(np.real(o2c / o2b + o1c / o1b)))
    return out


# ==========================================================================
# SFK TAIL (sfk_* block) -- Fischer-Koch towers, Hackman surfaces, and
# annular-ended genus-1 tori (singly periodic)
# ==========================================================================
# Completes the singly periodic catalog with the members that were
# deferred pending per-notebook constant extraction:
#
#   * sfk_e2a2_*  translation-invariant torus with 2 Enneper + 2 annular
#     ends.  Rational data on the z-sphere chart (upper half disk),
#       G  = z (1 + a z)(b z - 1) rho / (a b (b - z)(a + z)),
#       dh = (a + z)(1 + a z)/(a z^2) dz,     rho = a b,
#     with the period problem CLOSED IN CLOSED FORM: the residues of
#     om2 at z = 0 and z = b cancel iff a = b/(1 - b^2 + sqrt(1 - b^2))
#     (the notebook's Solve), leaving b in (0, 1) as a free modulus.
#     The translation is the full om2 residue loop at z = b; the chart
#     boundary lands on the mirrors y = 0 / y = -Y1 (2 Y1 = period) and
#     the horizontal mirror plane |z| = 1 (|G| = 1 there exactly).
#   * sfk_c1a2_*  singly periodic torus with 1 catenoid + 2 annular
#     ends.  Square-root data branched at {0, a, 1, b} (genus-1 double
#     cover),
#       G  = rho sqrt(z) sqrt(z-1) / (sqrt(z-a) sqrt(z-b)),
#       dh = dz / z,
#     with TWO period conditions Re Int_a^1 om1 = Re Int_1^b om2 = 0
#     solved here by Newton (endpoint sqrt singularities removed by a
#     sine substitution) from the notebook's harvested seeds -- the
#     solver reproduces the notebook triples (e.g. a = 0.2574798...,
#     b = 1.5922917..., rho = 1 for the parallel-end member) and the
#     measured mirror spacing reproduces the residue translation
#     pi (1 + rho^2)/rho to integration accuracy (gated).
#   * sfk_fkt_*   translation-invariant Fischer-Koch surfaces
#     (theta-function data on a rhombic torus; constants k, b0, t0 from
#     the notebook's solved period problem) -- see the builder header.
#   * sfk_fkf_*   Fischer-Koch-Freese screw-motion family (mu-twisted
#     theta data; solved (mu, b0, t0) continuation tables) -- see the
#     builder header.
#   * sfk_hack_*  Hackman surfaces (toroidal 1-noids) -- see the
#     builder header for status.
#
# All meshed with the sptail_* machinery above (interior-limit nudge for
# every fractional power ON a boundary cut, compound GL per cell,
# boundary snapping onto measured symmetry elements, orbit + weld,
# translation-wrapped quotient for MEASURED Euler characteristic).
#
# References:
#   W. Fischer, E. Koch, "Spanning minimal surfaces", Phil. Trans. R.
#     Soc. Lond. A 354 (1996) 2105-2142 -- the Fischer-Koch family;
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988) -- the tower construction
#     language these singly periodic pieces live in;
#   M. Weber, https://minimalsurfaces.blog/ -- harvested notebooks
#     "Fischer-Koch Translational", "Fischer-Koch-Freese" (after
#     R. Freese), "Hackman Surfaces" (after M. Hackman), "Singly
#     Periodic Torus with One Catenoid and Two Annular Ends",
#     "Translation Invariant Torus with Two Enneper and Two Annular
#     Ends" (research/msblog_harvest/singly_periodic.json);
#   B. C. Carlson, "Numerical computation of real or complex elliptic
#     integrals", Numer. Algorithms 10 (1995) 13-26 -- the R_F used by
#     the torus uniformization.
# --------------------------------------------------------------------------

_SFK_GL24 = np.polynomial.legendre.leggauss(24)


# ---- 2 Enneper + 2 annular ends, genus-1 quotient ------------------------

def sfk_e2a2_om(b):
    """Weierstrass 1-form triple for the 2-Enneper 2-annular torus;
    the closed-form residue closure a = b/(1 - b^2 + sqrt(1 - b^2))."""
    a = b / (1.0 - b * b + math.sqrt(1.0 - b * b))
    rho = a * b

    def om(z):
        z = sptail_nudge(z)
        f1 = rho * (1.0 / a + z) ** 2 * (z - 1.0 / b) / (z * (z - b))
        f2 = b * (a + z) ** 2 * (z - b) \
            / (z ** 3 * (b * z - 1.0) * rho)
        dh = (a + z) * (1.0 + a * z) / (a * z * z)
        return (0.5 * (f2 - f1), 0.5j * (f2 + f1), -dh)
    return om, a, rho


def sfk_e2a2_build(b=0.5, nu=52, nt=40, storeys=1, rmin=0.14,
                   epsb=0.10):
    """Quarter chart = upper half of the unit disk, polar about the
    Enneper end z = 0; annular end at the boundary puncture z = b
    (masked at |z - b| < epsb).  Frames {E, M_z(h)} x {E, M_y} per
    storey, translation (0, -2 Y1, 0)."""
    om, a, rho = sfk_e2a2_om(b)
    cl = sptail_cluster(0.12, hmin=2e-3, ratio=0.45)
    r1 = np.exp(np.linspace(math.log(rmin),
                            math.log(b - cl[0] - 0.02),
                            max(10, int(0.45 * nu))))
    r2 = np.exp(np.linspace(math.log(b + cl[0] + 0.02), 0.0,
                            max(8, int(0.35 * nu))))
    r = np.unique(np.concatenate([r1, b - cl, [b], b + cl[::-1], r2]))
    th = math.pi * 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, nt)))
    X = sptail_polar_patch(om, r, th)
    R_, T_ = np.meshgrid(r, th, indexing='ij')
    Z = R_ * np.exp(1j * T_)
    mask = (np.abs(Z - b) > epsb) & np.isfinite(X).all(axis=-1)
    ib = int(np.searchsorted(r, b))
    m0 = mask[:, 0]
    yA = float(np.median(X[:ib, 0, 1][m0[:ib]]))
    yB = float(np.median(X[ib + 1:, 0, 1][m0[ib + 1:]]))
    X[..., 1] -= yA
    yB -= yA
    Y1 = -yB
    h = float(np.median(X[-1, :, 2]))
    # the translation must reproduce the om2 residue loop at z = b
    res = period_integral(lambda z: om(z)[1], b, 0.25 * b, 0.25 * b)
    diag = {
        'yA_ptp': float(np.ptp(X[:ib, 0, 1][m0[:ib]])),
        'yB_ptp': float(np.ptp(X[ib + 1:, 0, 1][m0[ib + 1:]])),
        'ypi_vs_yB': abs(float(np.median(X[:, -1, 1])) - yB),
        'circ_z_ptp': float(np.ptp(X[-1, :, 2])),
        'trans_vs_residue': abs(2.0 * Y1 - abs(float(np.real(res)))),
        'a': a, 'rho': rho}
    X[:ib, 0, 1] = 0.0
    X[ib:, 0, 1] = np.where(m0[ib:], -Y1, X[ib:, 0, 1])
    X[:, -1, 1] = -Y1
    X[-1, :, 2] = h
    X[~mask] = 0.0
    V0 = X.reshape(-1, 3)
    vm = mask.reshape(-1)
    q0 = sptail_grid_quads(len(r), len(th), vm)
    T = np.array([0.0, -2.0 * Y1, 0.0])
    frames = []
    for ez in (0, 1):
        for ey in (0, 1):
            M = np.diag([1.0, -1.0 if ey else 1.0,
                         -1.0 if ez else 1.0])
            tv = np.array([0.0, 0.0, 2.0 * h if ez else 0.0])
            par = (-1.0) ** (ez + ey)
            for s in range(storeys):
                frames.append((M, tv + s * T, par))
    vu = V0[vm]
    span = float(np.linalg.norm(vu.max(0) - vu.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(th)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def sfk_e2a2_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                  storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sfk_e2a2_build(
        b=p['b'], nu=int(np.clip(nu, 30, 120)),
        nt=int(np.clip(int(0.8 * nv), 24, 100)), storeys=S,
        rmin=p['rmin'])
    V = _smooth_boundary(V, F, iters=4)
    V = _center_fit(V, scale, V)
    return V, F, uv


# ---- 1 catenoid + 2 annular ends, genus-1 quotient -----------------------

def sfk_c1a2_om(a, b, rho):
    """Square-root Weierstrass data branched at {0, a, 1, b}."""
    def om(z):
        z = sptail_nudge(z)
        f1 = rho * z ** -0.5 * (z - a) ** -0.5 * (z - 1.0) ** 0.5 \
            * (z - b) ** -0.5
        f2 = (1.0 / rho) * z ** -1.5 * (z - a) ** 0.5 \
            * (z - 1.0) ** -0.5 * (z - b) ** 0.5
        return (0.5 * (f2 - f1), 0.5j * (f1 + f2), 1.0 / z)
    return om


def _sfk_c1a2_seg(om, comp, lo, hi):
    """Re Int of om[comp] over (lo, hi) just above the cut; the sine
    substitution kills both endpoint 1/sqrt singularities."""
    xg, wg = _SFK_GL24
    s = 0.5 * math.pi * xg
    z = lo + (hi - lo) * 0.5 * (1.0 + np.sin(s))
    dz = (hi - lo) * 0.25 * math.pi * np.cos(s)
    return float(np.real(np.sum(om(z + 0j)[comp] * dz * wg)))


# Newton seeds (b, rho) by neck modulus a -- the notebook's harvested
# continuation table (Weber, "Singly Periodic Torus with One Catenoid
# and Two Annular Ends"); a = 0.2574798... is the parallel-end member
# (rho = 1).
_SFK_C1A2_SEEDS = (
    (0.47, 1.4633227747553412, 0.8560003563161298),
    (0.40, 1.5116073321252144, 0.9072805483464664),
    (0.30, 1.5713693328892708, 0.9739124409967519),
    (0.25747983928707496, 1.592291695522628, 1.0),
    (0.10, 1.62, 1.06),
    (0.052445797436667364, 1.5956627329623914, 1.1),
    (0.02, 1.5357253576689303, 1.1040875033336026),
)
_SFK_C1A2_CACHE = {}


def sfk_c1a2_constants(a):
    """Solve the two period conditions Re Int_a^1 om1 = 0 and
    Re Int_1^b om2 = 0 for (b, rho) by damped Newton from the nearest
    harvested seed."""
    key = round(float(a), 12)
    if key in _SFK_C1A2_CACHE:
        return _SFK_C1A2_CACHE[key]
    sd = min(_SFK_C1A2_SEEDS, key=lambda s: abs(s[0] - a))
    b, rho = sd[1], sd[2]

    def resid(bv, rv):
        om = sfk_c1a2_om(a, bv, rv)
        return np.array([_sfk_c1a2_seg(om, 0, a, 1.0),
                         _sfk_c1a2_seg(om, 1, 1.0, bv)])
    F = resid(b, rho)
    for _ in range(40):
        if float(np.max(np.abs(F))) < 1e-12:
            break
        hstep = 1e-7
        J = np.empty((2, 2))
        J[:, 0] = (resid(b + hstep, rho) - F) / hstep
        J[:, 1] = (resid(b, rho + hstep) - F) / hstep
        db, dr = np.linalg.solve(J, -F)
        scl = min(1.0, 0.2 / max(abs(db), abs(dr)))
        b += scl * db
        rho += scl * dr
        F = resid(b, rho)
    worst = float(np.max(np.abs(F)))
    if worst > 1e-8 or not (1.0 < b and 0.0 < rho):
        raise ValueError(f"c1a2 period problem did not close at a={a}"
                         f" (residual {worst:.2e})")
    _SFK_C1A2_CACHE[key] = (b, rho, worst)
    return _SFK_C1A2_CACHE[key]


def sfk_c1a2_build(a=0.25747983928707496, nu=52, nt=40, storeys=1,
                   rmin=0.05, rmax=12.0):
    """Half chart = upper half plane, polar about the catenoid end
    z = 0; annular ends at z = infinity (rmax trim).  Boundary rays:
    (0,a) and (1,b) lie in the SAME vertical mirror x = 0 (that IS the
    first period condition, measured); (a,1) and (b,inf) lie at y = 0;
    the negative axis lies at y = -trans/2 where trans =
    pi (1 + rho^2)/rho is the om2 residue translation at infinity
    (measured against the closed form).  Frames {E, M_x} x
    {E, M_y(-trans/2)} per storey, translation (0, trans, 0)."""
    b, rho, pres = sfk_c1a2_constants(a)
    om = sfk_c1a2_om(a, b, rho)
    trans = math.pi * (1.0 + rho * rho) / rho
    cl = sptail_cluster(0.06, hmin=1e-4, ratio=0.35)
    n1 = max(8, int(0.22 * nu))
    r = np.unique(np.concatenate([
        np.exp(np.linspace(math.log(rmin), math.log(a), n1)),
        a + cl[::-1] * (1.0 - a), [a],
        np.linspace(a, 1.0, n1)[1:-1], 1.0 - cl * (1.0 - a), [1.0],
        1.0 + cl[::-1] * (b - 1.0),
        np.linspace(1.0, b, max(6, int(0.15 * nu)))[1:-1],
        b - cl * (b - 1.0), [b], b + cl[::-1],
        np.exp(np.linspace(math.log(1.05 * b), math.log(rmax), n1))]))
    th = math.pi * 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, nt)))
    X = sptail_polar_patch(om, r, th)
    ia = int(np.searchsorted(r, a))
    i1 = int(np.searchsorted(r, 1.0))
    ib = int(np.searchsorted(r, b))
    x0 = float(np.median(np.concatenate([X[:ia + 1, 0, 0],
                                         X[i1:ib + 1, 0, 0]])))
    y1 = float(np.median(np.concatenate([X[ia:i1 + 1, 0, 1],
                                         X[ib:, 0, 1]])))
    X[..., 0] -= x0
    X[..., 1] -= y1
    yN = float(np.median(X[:, -1, 1]))
    diag = {
        'period_res': pres,
        'xseg_gap': abs(float(np.median(X[:ia + 1, 0, 0]))
                        - float(np.median(X[i1:ib + 1, 0, 0]))),
        'yseg_gap': abs(float(np.median(X[ia:i1 + 1, 0, 1]))
                        - float(np.median(X[ib:, 0, 1]))),
        'mirror_vs_residue': abs(yN + 0.5 * trans),
        'b': b, 'rho': rho}
    X[:ia + 1, 0, 0] = 0.0
    X[i1:ib + 1, 0, 0] = 0.0
    X[ia:i1 + 1, 0, 1] = 0.0
    X[ib:, 0, 1] = 0.0
    X[:, -1, 1] = -0.5 * trans
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(r), len(th))
    T = np.array([0.0, trans, 0.0])
    frames = []
    for ex in (0, 1):
        for ey in (0, 1):
            M = np.diag([-1.0 if ex else 1.0,
                         -1.0 if ey else 1.0, 1.0])
            tv = np.array([0.0, -trans if ey else 0.0, 0.0])
            par = (-1.0) ** (ex + ey)
            for s in range(storeys):
                frames.append((M, tv + s * T, par))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(th)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def sfk_c1a2_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                  storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sfk_c1a2_build(
        a=p['a'], nu=int(np.clip(nu, 30, 120)),
        nt=int(np.clip(int(0.8 * nv), 24, 100)), storeys=S,
        rmin=p['rmin'], rmax=p['rmax'])
    V = _smooth_boundary(V, F, iters=4)
    V = _center_fit(V, scale, V)
    return V, F, uv


# ---- theta / elliptic helpers for the Fischer-Koch block -----------------

def sfk_theta1(v, q):
    """Jacobi theta_1(v, nome q); complex v (vectorized) and nome."""
    v = np.asarray(v, complex)
    s = np.zeros_like(v)
    for n in range(10):
        s = s + (-1.0) ** n * q ** ((n + 0.5) ** 2) \
            * np.sin((2 * n + 1) * v)
    return 2.0 * s


def sfk_theta1p(v, q):
    """d theta_1 / dv."""
    v = np.asarray(v, complex)
    s = np.zeros_like(v)
    for n in range(10):
        s = s + (-1.0) ** n * q ** ((n + 0.5) ** 2) * (2 * n + 1) \
            * np.cos((2 * n + 1) * v)
    return 2.0 * s


def sfk_TH(z, tau):
    """The notebooks' Theta[z, tau] = theta_1(pi z, e^(i pi tau))."""
    return sfk_theta1(np.pi * np.asarray(z, complex),
                      np.exp(1j * np.pi * tau))


def sfk_rf(x, y, z):
    """Carlson symmetric elliptic R_F (1995), complex-capable."""
    x, y, z = np.broadcast_arrays(*[np.asarray(t, complex)
                                    for t in (x, y, z)])
    x, y, z = x.copy(), y.copy(), z.copy()
    for _ in range(60):
        lam = np.sqrt(x) * np.sqrt(y) + np.sqrt(y) * np.sqrt(z) \
            + np.sqrt(z) * np.sqrt(x)
        x, y, z = 0.25 * (x + lam), 0.25 * (y + lam), 0.25 * (z + lam)
        mu = (x + y + z) / 3.0
        rel = float(np.max(np.abs(np.stack([x - mu, y - mu, z - mu]))
                           / np.maximum(np.abs(mu), 1e-300)))
        if rel < 1e-12:
            break
    mu = (x + y + z) / 3.0
    X, Y, Z = 1 - x / mu, 1 - y / mu, 1 - z / mu
    e2 = X * Y - Z * Z
    e3 = X * Y * Z
    return (1.0 - e2 / 10.0 + e3 / 14.0 + e2 * e2 / 24.0
            - 3.0 * e2 * e3 / 44.0) / np.sqrt(mu)


def sfk_ellipk(m):
    """Complete elliptic K(m), parameter convention, m < 1."""
    return complex(sfk_rf(0.0, 1.0 - m, 1.0)).real


def sfk_ellipf(z, m):
    """Incomplete F(arcsin z | m) via Carlson R_F, complex z."""
    z = np.asarray(z, complex)
    return z * sfk_rf(1.0 - z * z, 1.0 - m * z * z, 1.0)


def sfk_seg_patch(om, Z, jm=None):
    """Cumulative GL-8 integration of the 1-form triple om along the
    straight segments of an arbitrary curvilinear grid Z[i, j] (path
    independence inside the chart); spine = column jm, rows outward."""
    xg, wg = _SPTAIL_GL
    nr, nt = Z.shape
    if jm is None:
        jm = nt // 2
    P = np.zeros((nr, nt, 3), complex)

    def inc(z0, z1):
        dz = z1 - z0
        zs = 0.5 * (z0 + z1)[..., None] + 0.5 * dz[..., None] * xg
        o = om(zs)
        return np.stack([np.sum(c * wg, axis=-1) for c in o], -1) \
            * (0.5 * dz)[..., None]

    P[1:, jm, :] = np.cumsum(inc(Z[:-1, jm], Z[1:, jm]), axis=0)
    for j in range(jm, nt - 1):
        P[:, j + 1, :] = P[:, j, :] + inc(Z[:, j], Z[:, j + 1])
    for j in range(jm, 0, -1):
        P[:, j - 1, :] = P[:, j, :] + inc(Z[:, j], Z[:, j - 1])
    return np.real(P)


# ---- translation-invariant Fischer-Koch surfaces -------------------------
# Theta-function Weierstrass data on the rhombic torus C/(1, tau0),
# tau0 = e^(i pi t0), with (b0, t0) the notebook's SOLVED period problem
# (FindRoot) for k = 3, 4, 5.  The Gauss map is multivalued on the
# torus (it gains e^(+-2 pi i/k) around the lattice cycles -- measured
# in the self-tests), so one translational period carries k torus
# charts; the chart boundary is a skeleton of 2-fold rotation AXES
# measured straight to ~1e-9:
#   horizontal axes along y at (x = 0, z = 0) and (x = 0, z = -1);
#   horizontal axes along (cos(pi/2 - pi/k), sin(pi/2 - pi/k)) at
#     z = -1/k and z = -1 - 1/k;
#   vertical axis segments on the z axis, z in [-1/k, 0] and
#     [-1 - 1/k, -1].
# Compositions of these half-turns generate the screw
# S = Rz(-2 pi/k) + (0, 0, -2/k) with S^k = the translation (0,0,-2),
# and the quotient group {E, R_z-axis} x {S^i}: 2k chart copies per
# period.  Verified gates: the vertical rise Re Int_0^{(tau0-1)/2} dh
# = -1/k (the notebook's period equation) to ~1e-14, and every axis
# straightness/placement residual above.

_SFK_FKT_SOLN = {
    # k: (b0, t0) -- FindRoot literals from Weber's notebook
    3: (0.5500677399890902, 0.18208994257933972),
    4: (0.5208186836558262, 0.1195089689509489),
    5: (0.5111052672256567, 0.08665409066790537),
}
_SFK_FKT_CACHE = {}


def sfk_fkt_constants(k):
    """Uniformization constants for the k-wing member: the strip
    w -> torus map g(w) = tst(tr(e^w)) (1 + tau0)/2 and the corner
    abscissas alpha1..alpha4."""
    if k in _SFK_FKT_CACHE:
        return _SFK_FKT_CACHE[k]
    b0, t0 = _SFK_FKT_SOLN[k]
    tau0 = complex(np.exp(1j * math.pi * t0))
    taup = (tau0 - 1.0) / (tau0 + 1.0)
    target = taup.imag

    def fr(r):
        return (r * sfk_ellipk(1.0 - r * r)
                / (2.0 * sfk_ellipk(1.0 / r ** 2))) - target
    lo, hi = 1.0 + 1e-12, 12.0
    flo = fr(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if flo * fr(mid) < 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-14 * hi:
            break
    r0 = 0.5 * (lo + hi)
    m = 1.0 / r0 ** 2
    quot0 = 2.0 * sfk_ellipk(m)

    def tst_r(x):
        return complex(sfk_ellipf(x, m)).real / quot0 + 0.5

    lo, hi = -1.0 + 1e-12, 1.0 - 1e-12
    flo = tst_r(lo) - (1.0 - b0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if flo * (tst_r(mid) - (1.0 - b0)) < 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-15:
            break
    a0 = 0.5 * (lo + hi)

    def zeta_of(s):
        return (s - a0) / (r0 + s * a0)
    al1 = math.log(abs(zeta_of(1.0)))
    al2 = math.log(abs(zeta_of(r0)))
    al3 = math.log(abs(zeta_of(-r0)))
    al4 = math.log(abs(zeta_of(-1.0)))
    c = {'k': k, 'b0': b0, 'tau0': tau0, 'taup': taup, 'r0': r0,
         'm': m, 'quot0': quot0, 'a0': a0, 'al1': al1, 'al2': al2,
         'al3': al3, 'al4': al4, 'sym': al1 + al3}
    _SFK_FKT_CACHE[k] = c
    return c


def sfk_fkt_gmap(w, c):
    """Strip -> torus coordinate."""
    zeta = np.exp(np.asarray(w, complex))
    trv = (-c['a0'] - c['r0'] * zeta) / (-1.0 + c['a0'] * zeta)
    return (sfk_ellipf(trv, c['m']) / c['quot0'] + 0.5) \
        * (1.0 + c['tau0']) / 2.0


def sfk_fkt_om(k):
    """The 1-form triple (om1, om2, om3) on the torus chart."""
    c = sfk_fkt_constants(k)
    b0, tau0 = c['b0'], c['tau0']
    astar = 0.5 * (1.0 - 1.0 / k)
    taup = c['taup']
    q = np.exp(1j * np.pi * tau0)
    N = (1j * sfk_theta1(
        -((-1 + k + 2 * b0 * k) * np.pi * (1 + tau0)) / (4 * k), q)
        * sfk_theta1(
            -((1 + (-1 + 2 * b0) * k) * np.pi * (1 + tau0)) / (4 * k),
            q)) \
        / (sfk_theta1(-b0 * np.pi * (1 + tau0), q)
           * sfk_theta1p(0.0, q))
    CG = (sfk_TH(-astar / 2, taup)
          * sfk_TH(0.5 - (astar / 2 + 0.5 * taup), taup)) \
        / (sfk_TH(astar / 2, taup)
           * sfk_TH(0.5 - (1 - astar / 2 + 0.5 * taup), taup))

    def om(z):
        z = np.asarray(z, complex)
        u = z / (tau0 + 1.0)
        G = ((sfk_TH(u - 0.5 * (1 + astar), taup)
              * sfk_TH(u - (astar / 2 + 0.5 * taup), taup))
             / (sfk_TH(u - 0.5 * (1 - astar), taup)
                * sfk_TH(u - (1 - astar / 2 + 0.5 * taup), taup))) / CG
        d = ((sfk_TH(z - 0.5 * (1 - astar) * (1 + tau0), tau0)
              * sfk_TH(z - 0.5 * (1 + astar) * (1 + tau0), tau0))
             / (sfk_TH(z - 0.5 * (1 - b0) * (1 + tau0), tau0)
                * sfk_TH(z - 0.5 * (1 + b0) * (1 + tau0), tau0))) / N
        return (-(G * d - d / G) / 2.0, 1j * (G * d + d / G) / 2.0, d)
    return om


def sfk_fkt_build(k=3, nu=64, nt=44, storeys=1, rmin=-3.5,
                  rmax=10.0 / 12.0):
    """One strip chart, snapped onto its measured 2-fold-axis skeleton,
    orbited under {E, Rz(pi)} x {screw S^i} x storey translations."""
    c = sfk_fkt_constants(k)
    om = sfk_fkt_om(k)
    al1, al2, al3, al4 = c['al1'], c['al2'], c['al3'], c['al4']
    xs0 = np.linspace(rmin, rmax, nu)
    refl = c['sym'] - xs0
    xs = np.unique(np.concatenate(
        [xs0, [al4, al3, al1, al2],
         refl[(refl >= rmin) & (refl <= rmax)]]))
    ys = np.linspace(1e-9, math.pi - 1e-9, nt)
    Z = sfk_fkt_gmap(xs[:, None] + 1j * ys[None, :], c)
    X = sfk_seg_patch(om, Z)
    i4 = int(np.argmin(np.abs(xs - al4)))
    i3 = int(np.argmin(np.abs(xs - al3)))
    i1 = int(np.argmin(np.abs(xs - al1)))
    i2 = int(np.argmin(np.abs(xs - al2)))
    X = X - X[i4, -1, :][None, None, :]
    # the vertical rise over the half (tau0 - 1) cycle must equal -1/k
    # (the notebook's period equation) -- integrate dh along the cycle
    xg, wg = _SPTAIL_GL
    zc = (c['tau0'] - 1.0) / 2.0
    tot = 0.0j
    ss = np.linspace(0.0, 1.0, 33)
    for s0, s1 in zip(ss[:-1], ss[1:]):
        za, zb = zc * s0, zc * s1
        zn = 0.5 * (za + zb) + 0.5 * (zb - za) * xg
        tot += np.sum(om(zn)[2] * wg) * 0.5 * (zb - za)
    u = np.array([math.cos(math.pi / 2 - math.pi / k),
                  math.sin(math.pi / 2 - math.pi / k)])
    up = np.array([-u[1], u[0]])
    diag = {
        'rise_res': abs(tot.real + 1.0 / k),
        'ax_y0': float(max(np.max(np.abs(X[:i4 + 1, -1, 0])),
                           np.max(np.abs(X[:i4 + 1, -1, 2])))),
        'ax_y1': float(max(np.max(np.abs(X[:i1 + 1, 0, 0])),
                           np.max(np.abs(X[:i1 + 1, 0, 2] + 1.0)))),
        'ax_v': float(max(np.max(np.abs(X[i1:i2 + 1, 0, :2])),
                          np.max(np.abs(X[i4:i3 + 1, -1, :2])))),
        'ax_u0': float(max(np.max(np.abs(X[i3:, -1, :2] @ up)),
                           np.max(np.abs(X[i3:, -1, 2] + 1.0 / k)))),
        'ax_u1': float(max(np.max(np.abs(X[i2:, 0, :2] @ up)),
                           np.max(np.abs(X[i2:, 0, 2] + 1.0
                                         + 1.0 / k))))}
    # snap the six boundary runs onto the exact axes
    X[:i4 + 1, -1, 0] = 0.0
    X[:i4 + 1, -1, 2] = 0.0
    X[:i1 + 1, 0, 0] = 0.0
    X[:i1 + 1, 0, 2] = -1.0
    X[i1:i2 + 1, 0, :2] = 0.0
    X[i4:i3 + 1, -1, :2] = 0.0
    for sl, zv in ((np.s_[i2:, 0], -1.0 - 1.0 / k),
                   (np.s_[i3:, -1], -1.0 / k)):
        d = X[sl][:, :2] @ u
        X[sl][:, 0] = d * u[0]
        X[sl][:, 1] = d * u[1]
        X[sl][:, 2] = zv
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(xs), len(ys))
    frames = []
    for s in range(storeys):
        for i in range(k):
            phi = -TAU * i / k
            cp, sp = math.cos(phi), math.sin(phi)
            Rz = np.array([[cp, -sp, 0.0], [sp, cp, 0.0],
                           [0.0, 0.0, 1.0]])
            tz = -2.0 * i / k + 2.0 * s
            for e in (0, 1):
                M = Rz @ (np.diag([-1.0, -1.0, 1.0]) if e
                          else np.eye(3))
                frames.append((M, np.array([0.0, 0.0, tz]),
                               (-1.0) ** e))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(xs), len(ys)),
                                 q0, frames, 1e-7 * span)
    diag['T'] = np.array([0.0, 0.0, 2.0])
    diag['span'] = span
    return V, F, uv, diag


def sfk_fkt_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                 storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sfk_fkt_build(
        k=p['k'], nu=int(np.clip(nu, 36, 120)),
        nt=int(np.clip(int(0.8 * nv), 24, 90)), storeys=S,
        rmin=p['rmin'])
    V = _smooth_boundary(V, F, iters=4)
    V = _center_fit(V, scale, V)
    return V, F, uv


# ---- Fischer-Koch-Freese screw-motion family (after R. Freese) -----------
# The mu-twisted generalization: G0 gains four theta factors raised to
# the REAL power mu (a multivalued Gauss map; a continuous branch of
# log H is carried across the chart by grid unwrapping + per-segment
# seed unwrapping), dh0's exponent a* becomes
# a* = (mu (2 b - 1 - 1/k) - 1/k + 1)/2, and the surface is invariant
# under the period SCREW Rz(-2 pi mu) + (0, 0, -2) rather than a pure
# translation.  The measured 2-fold-axis skeleton (gated):
#   theta_1 = pi/2 (z = -1),  theta_4 = pi/2 - pi (1 - mu) (z = 0),
#   theta_6 = pi/2 - (pi/k)(1 - (k-1) mu) (z = -1/k),
#   theta_3 = pi/2 - (pi/k)(1 + mu) (z = -1 - 1/k),
#   vertical axis segments z in [-1/k, 0] and [-1 - 1/k, -1];
# generating the chart screw S = Rz(-2 pi (1+mu)/k) + (0, 0, -2/k),
# S^k = the period screw.  Only k = 3 ships: for even k the copies
# S^(k/2) map the theta_4 axis exactly onto the theta_1 axis (any mu),
# a genuine line self-intersection of the immersed family (that is the
# classical "embedded iff k odd" restriction) which fails the manifold
# gate; see BACKLOG.md.  Constants: the notebook's soln3 continuation
# table (k = 3), mu from -0.2 to +0.3, with mu = 0 exactly the
# translational Fischer-Koch member.

_SFK_FKF_SOLN3 = {
    # mu: (b0, t0) -- k = 3 continuation table (Weber's notebook)
    -0.20: (0.4790492996907018, 0.13000057825607578),
    -0.15: (0.5027694738349303, 0.15409423607280687),
    -0.10: (0.5213425526437409, 0.16811202081570092),
    -0.05: (0.5367941484624708, 0.17680521707217497),
    0.05: (0.561737511034624, 0.18492786546038673),
    0.10: (0.5721951419037229, 0.18583805096179182),
    0.15: (0.5817275808719873, 0.18509006142392043),
    0.20: (0.5905586477939514, 0.18278339259132123),
    0.25: (0.598876035234557, 0.17887021275265938),
    0.30: (0.6068569675493184, 0.1731275508391721),
}
_SFK_FKF_CACHE = {}


def _sfk_strip_constants(b0, t0):
    """Shared strip -> torus uniformization for the FK family: from
    (b0, tau0 = e^(i pi t0)) to (r0, m, quot0, a0, alphas)."""
    tau0 = complex(np.exp(1j * math.pi * t0))
    taup = (tau0 - 1.0) / (tau0 + 1.0)
    target = taup.imag

    def bisect(f, lo, hi, it=200):
        flo = f(lo)
        for _ in range(it):
            mid = 0.5 * (lo + hi)
            if flo * f(mid) < 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)
    r0 = bisect(lambda r: (r * sfk_ellipk(1.0 - r * r)
                           / (2.0 * sfk_ellipk(1.0 / r ** 2)))
                - target, 1.0 + 1e-12, 12.0)
    m = 1.0 / r0 ** 2
    quot0 = 2.0 * sfk_ellipk(m)

    def tst_r(x):
        return complex(sfk_ellipf(x, m)).real / quot0 + 0.5
    a0 = bisect(lambda x: tst_r(x) - (1.0 - b0),
                -1.0 + 1e-12, 1.0 - 1e-12)

    def zof(s):
        return (s - a0) / (r0 + s * a0)
    al1 = math.log(abs(zof(1.0)))
    al2 = math.log(abs(zof(r0)))
    al3 = math.log(abs(zof(-r0)))
    al4 = math.log(abs(zof(-1.0)))
    return {'b0': b0, 'tau0': tau0, 'taup': taup, 'r0': r0, 'm': m,
            'quot0': quot0, 'a0': a0, 'al1': al1, 'al2': al2,
            'al3': al3, 'al4': al4, 'sym': al1 + al3}


def sfk_fkf_data(k, mu):
    """(constants dict, om(z, L), logH(z)) for the Freese member."""
    key = (k, round(mu, 6))
    if key in _SFK_FKF_CACHE:
        return _SFK_FKF_CACHE[key]
    b0, t0 = _SFK_FKF_SOLN3[round(mu, 6)]
    c = _sfk_strip_constants(b0, t0)
    tau0, taup = c['tau0'], c['taup']
    astar = 0.5 * (mu * (2 * b0 - 1 - 1.0 / k) - 1.0 / k + 1)
    q = np.exp(1j * np.pi * tau0)
    Nmu = (1j * sfk_theta1(
        (np.pi * (-1 + (-1 + 2 * b0) * k * (-1 + mu) - mu)
         * (1 + tau0)) / (4 * k), q)
        * sfk_theta1(
            -(np.pi * (-1 + k - mu - k * mu + 2 * b0 * k * (1 + mu))
              * (1 + tau0)) / (4 * k), q)) \
        / (sfk_theta1(-b0 * np.pi * (1 + tau0), q)
           * sfk_theta1p(0.0, q))
    CB = (sfk_TH(-astar / 2, taup)
          * sfk_TH(0.5 - (astar / 2 + 0.5 * taup), taup)) \
        / (sfk_TH(astar / 2, taup)
           * sfk_TH(0.5 - (1 - astar / 2 + 0.5 * taup), taup))
    CH = (sfk_TH(b0 / 2, taup)
          * sfk_TH(0.5 - ((2 - b0) / 2 + 0.5 * taup), taup)) \
        / (sfk_TH(-b0 / 2, taup)
           * sfk_TH(0.5 - (b0 / 2 + 0.5 * taup), taup))

    def parts(z):
        z = np.asarray(z, complex)
        u = z / (tau0 + 1.0)
        B = ((sfk_TH(u - 0.5 * (1 + astar), taup)
              * sfk_TH(u - (astar / 2 + 0.5 * taup), taup))
             / (sfk_TH(u - 0.5 * (1 - astar), taup)
                * sfk_TH(u - (1 - astar / 2 + 0.5 * taup), taup))) / CB
        H = ((sfk_TH(u - 0.5 * (1 - b0), taup)
              * sfk_TH(u - ((2 - b0) / 2 + 0.5 * taup), taup))
             / (sfk_TH(u - 0.5 * (1 + b0), taup)
                * sfk_TH(u - (b0 / 2 + 0.5 * taup), taup))) / CH
        d = ((sfk_TH(z - 0.5 * (1 - astar) * (1 + tau0), tau0)
              * sfk_TH(z - 0.5 * (1 + astar) * (1 + tau0), tau0))
             / (sfk_TH(z - 0.5 * (1 - b0) * (1 + tau0), tau0)
                * sfk_TH(z - 0.5 * (1 + b0) * (1 + tau0), tau0))) / Nmu
        return B, H, d

    def om(z, L):
        B, H, d = parts(z)
        G = B * np.exp(mu * L)
        return (-(G * d - d / G) / 2.0, 1j * (G * d + d / G) / 2.0, d)

    def logH(z):
        return np.log(parts(z)[1])
    _SFK_FKF_CACHE[key] = (c, om, logH)
    return _SFK_FKF_CACHE[key]


def _sfk_unwrap2d(L, jm):
    """Continuous branch of a grid of principal logs: unwrap the spine
    column jm, then each row outward from the spine."""
    Lu = L.copy()
    Lu[:, jm] = np.real(L[:, jm]) + 1j * np.unwrap(np.imag(L[:, jm]))
    for i in range(L.shape[0]):
        newim = np.empty(L.shape[1])
        right = np.unwrap(np.concatenate(
            [[np.imag(Lu[i, jm])], np.imag(L[i, jm + 1:])]))
        left = np.unwrap(np.concatenate(
            [[np.imag(Lu[i, jm])], np.imag(L[i, :jm][::-1])]))
        newim[jm:] = right
        newim[:jm] = left[1:][::-1]
        Lu[i, :] = np.real(L[i, :]) + 1j * newim
    return Lu


def sfk_seg_patch_br(om, logH, Z, jm=None):
    """sfk_seg_patch with a carried branch: om(z, L) receives the
    continuous log of the multivalued factor (grid-unwrapped, then
    seed-unwrapped on each segment's quadrature nodes)."""
    xg, wg = _SPTAIL_GL
    nr, nt = Z.shape
    if jm is None:
        jm = nt // 2
    Lg = _sfk_unwrap2d(logH(Z), jm)
    P = np.zeros((nr, nt, 3), complex)

    def inc(z0, z1, L0):
        dz = z1 - z0
        zs = 0.5 * (z0 + z1)[..., None] + 0.5 * dz[..., None] * xg
        Lp = logH(zs)
        Ls = Lp - 2j * np.pi * np.round(
            np.imag(Lp - L0[..., None]) / (2 * np.pi))
        o = om(zs, Ls)
        return np.stack([np.sum(c * wg, axis=-1) for c in o], -1) \
            * (0.5 * dz)[..., None]

    P[1:, jm, :] = np.cumsum(inc(Z[:-1, jm], Z[1:, jm], Lg[:-1, jm]),
                             axis=0)
    for j in range(jm, nt - 1):
        P[:, j + 1, :] = P[:, j, :] + inc(Z[:, j], Z[:, j + 1],
                                          Lg[:, j])
    for j in range(jm, 0, -1):
        P[:, j - 1, :] = P[:, j, :] + inc(Z[:, j], Z[:, j - 1],
                                          Lg[:, j])
    return np.real(P)


def _sfk_rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])


def sfk_fkf_build(k=3, mu=0.15, nu=64, nt=44, storeys=1, rmin=-3.5,
                  rmax=10.0 / 12.0):
    """One branch-tracked strip chart snapped onto the mu-twisted axis
    skeleton, orbited under {E, Rz(pi)} x {S^i} x period screws."""
    c, om, logH = sfk_fkf_data(k, mu)
    al1, al2, al3, al4 = c['al1'], c['al2'], c['al3'], c['al4']
    xs0 = np.linspace(rmin, rmax, nu)
    refl = c['sym'] - xs0
    xs = np.unique(np.concatenate(
        [xs0, [al4, al3, al1, al2],
         refl[(refl >= rmin) & (refl <= rmax)]]))
    ys = np.linspace(1e-9, math.pi - 1e-9, nt)
    Z = sfk_fkt_gmap(xs[:, None] + 1j * ys[None, :], c)
    X = sfk_seg_patch_br(om, logH, Z)
    i4 = int(np.argmin(np.abs(xs - al4)))
    i3 = int(np.argmin(np.abs(xs - al3)))
    i1 = int(np.argmin(np.abs(xs - al1)))
    i2 = int(np.argmin(np.abs(xs - al2)))
    X = X - X[i4, -1, :][None, None, :]
    xg, wg = _SPTAIL_GL
    zc = (c['tau0'] - 1.0) / 2.0
    tot = 0.0j
    ss = np.linspace(0.0, 1.0, 33)
    for s0, s1 in zip(ss[:-1], ss[1:]):
        za, zb = zc * s0, zc * s1
        zn = 0.5 * (za + zb) + 0.5 * (zb - za) * xg
        tot += np.sum(om(zn, logH(zn))[2] * wg) * 0.5 * (zb - za)
    th4 = math.pi / 2 - math.pi * (1.0 - mu)
    th1 = math.pi / 2
    th6 = math.pi / 2 - (math.pi / k) * (1.0 - (k - 1) * mu)
    th3 = math.pi / 2 - (math.pi / k) * (1.0 + mu)

    def line_res(sl, th, zv):
        u = np.array([math.cos(th), math.sin(th)])
        up = np.array([-u[1], u[0]])
        return float(max(np.max(np.abs(X[sl][:, :2] @ up)),
                         np.max(np.abs(X[sl][:, 2] - zv))))
    diag = {
        'rise_res': abs(tot.real + 1.0 / k),
        'ax_th4': line_res(np.s_[:i4 + 1, -1], th4, 0.0),
        'ax_th1': line_res(np.s_[:i1 + 1, 0], th1, -1.0),
        'ax_v': float(max(np.max(np.abs(X[i1:i2 + 1, 0, :2])),
                          np.max(np.abs(X[i4:i3 + 1, -1, :2])))),
        'ax_th6': line_res(np.s_[i3:, -1], th6, -1.0 / k),
        'ax_th3': line_res(np.s_[i2:, 0], th3, -1.0 - 1.0 / k)}

    def snap(sl, th, zv):
        u = np.array([math.cos(th), math.sin(th)])
        d = X[sl][:, :2] @ u
        X[sl][:, 0] = d * u[0]
        X[sl][:, 1] = d * u[1]
        X[sl][:, 2] = zv
    snap(np.s_[:i4 + 1, -1], th4, 0.0)
    snap(np.s_[:i1 + 1, 0], th1, -1.0)
    X[i1:i2 + 1, 0, :2] = 0.0
    X[i4:i3 + 1, -1, :2] = 0.0
    snap(np.s_[i3:, -1], th6, -1.0 / k)
    snap(np.s_[i2:, 0], th3, -1.0 - 1.0 / k)
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(xs), len(ys))
    Sang = -TAU * (1.0 + mu) / k
    frames = []
    for s in range(storeys):
        Pm = _sfk_rz(-TAU * mu * s)
        for i in range(k):
            M0 = Pm @ _sfk_rz(Sang * i)
            tz = -2.0 * s - 2.0 * i / k
            for e in (0, 1):
                M = M0 @ (np.diag([-1.0, -1.0, 1.0]) if e
                          else np.eye(3))
                frames.append((M, np.array([0.0, 0.0, tz]),
                               (-1.0) ** e))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(xs), len(ys)),
                                 q0, frames, 1e-7 * span)
    diag['screw'] = (-TAU * mu, -2.0)
    diag['span'] = span
    return V, F, uv, diag


def sfk_screw_quotient(V, F, ang, tz, span, kmax=4):
    """sptail_quotient under a SCREW v -> Rz(k ang) v + (0,0,k tz):
    test-only (measures the per-period chi of a screw-periodic
    stack)."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for f in F:
        mm = len(f)
        for t2 in range(mm):
            a2, b2 = f[t2], f[(t2 + 1) % mm]
            cnt[(a2, b2) if a2 < b2 else (b2, a2)] += 1
    bnd = sorted({v for e, cc in cnt.items() if cc == 1 for v in e})
    B = V[bnd]
    tol = 1e-6 * span
    key = {tuple(np.round(B[i] / tol).astype(np.int64)): bnd[i]
           for i in range(len(bnd))}
    parent = list(range(len(V)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for kk in range(1, kmax + 1):
        BM = B @ _sfk_rz(ang * kk).T + np.array([0.0, 0.0, tz * kk])
        for i in range(len(bnd)):
            j = key.get(tuple(np.round(BM[i] / tol).astype(np.int64)))
            if j is not None:
                ra, rb = find(bnd[i]), find(j)
                if ra != rb:
                    parent[ra] = rb
    roots = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    Fq = []
    for f in F:
        g = [int(inv[i]) for i in f]
        h = [g[0]]
        for s_ in g[1:]:
            if s_ != h[-1]:
                h.append(s_)
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            Fq.append(tuple(h))
    return V[uniq], Fq


def sfk_fkf_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                 storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sfk_fkf_build(
        k=3, mu=p['mu'], nu=int(np.clip(nu, 36, 120)),
        nt=int(np.clip(int(0.8 * nv), 24, 90)), storeys=S,
        rmin=p['rmin'])
    V = _smooth_boundary(V, F, iters=4)
    V = _center_fit(V, scale, V)
    return V, F, uv


# ==========================================================================
# Kusner spheres with 2n planar ends (immersed minimal S^2), built to
# Weber's Kusner.nb recipe and REGISTERED against his PoVRay exports.
# Weber's own framing (the live page; the mirrored chapter kept only
# its navigation): "Rob Kusner discovered an interesting class of
# immersed minimal spheres with an even number 2n of planar ends",
# and for ODD n the immersion commutes with the antipodal map of S^2,
# descending to a projective plane with n planar ends -- so the
# even-n restriction of THIS row is Kusner's mathematics (the odd
# members ARE the KUSNER_RP2 row), not an arbitrary limit.
#
# Weierstrass data (rho = 1 is the classical member; s = sqrt(2p-1)):
#
#     G  = rho z^(p-1) (z^p - s) / (s z^p + 1),
#     dh = i z^(p-1) (z^p - s)(1 + s z^p) / poly^2 dz,
#     poly = z^(2p) + 2 s z^p/(p-1) - 1,
#
# an immersed sphere with 2p planar ends at the roots of poly (p ends
# inside the unit disk at z^p = (p-s)/(p-1) =: r0, their p partners
# outside at z^p = -1/r0).  ALL 2p end residues vanish (gated at
# ~1e-14), so the immersion is single-valued with no period problem.
# For ODD p the immersion commutes with the antipodal map z ->
# -1/conj(z) (measured: X(-1/conj z) = X(z) to 1e-16), which is
# Kusner's projective-plane family -- that one-sided quotient is the
# separate KUSNER_RP2 row; THIS row is the full immersed sphere, and
# exists for every p >= 2 (even p included).
#
# MESHED TO THE NOTEBOOK'S OWN CHART: the fundamental patch is the
# image of the upper half annulus xmin <= |zeta| <= 1 under
# w = ((r0 + zeta)/(1 + r0 zeta))^(1/p) -- a curvilinear sector of
# angle pi/p with the plate hole around the end w = r0^(1/p) cut out
# by the |zeta| = xmin circle (Weber's fft/tr/st mesh, reproduced
# exactly: radial nodes graded by t^(1/p) <-> t^p around r0, angular
# nodes by the Im Log[(r0 - e^(i p t))/(r0 e^(i p t) - 1)] map).  The
# exterior chart is the same grid pushed through the MEASURED domain
# symmetry w -> e^(i pi/p)/w and integrated independently; the two
# charts agree along |w| = 1 to ~1e-12 (gated), which re-proves the
# vanishing residues as assembled geometry.  Space frames, MEASURED
# (not assumed) at 1e-16 by least squares over random domain pairs:
#     X(conj w)         = diag(-1, 1, -1) X(w)      (patch edge line)
#     X(e^(2 pi i/p) w) = Rz(-2 pi/p) X(w)
# and the assembly is the 4p-piece orbit of the two patches under the
# group they generate.  X(w = 0) = 0 (base normalization f0 - f0(0)
# of the notebook).
#
# GROUND TRUTH: Weber's p = 2 export (kusners-spheres-with-planar-
# ends, dummy.pov) IS the disk-chart half of this surface at the
# notebook window xmin = 0.2 -- his page says so outright ("the
# cases n=2 (showing one half of the surface)"), so the operator's
# FULL sphere legitimately shows more lobes than his picture.  Our
# half-assembly registers against it at 0.11% GT -> ours mean of
# span (his exports normalise the y half-extent to 1; the axis map
# is the identity), bbox ratios agreeing to 4 digits.  The zoo gate
# pins that member's extent ratios (x/y = 0.4430, z/y = 0.5353)
# plus the closed-form landmark below.  His p = 3 and p = 5 exports
# are the SAME kind of object, not artistic compositions (an
# earlier note here claimed 'three nested copies at relative
# scales', which was measured off his copies.pov SCENE arrangement,
# not the dummy.pov meshes -- each dummy.pov is ONE connected mesh):
# the two p = 3 exports are Weber's "two views ... with different
# cutoffs for the planar ends", and each registers against our
# half-assembly at its own notebook window (p = 3: 0.088% mean at
# xmin = 0.35; p = 5: 0.061% at xmin = 0.45).
#
# Landmark, derived numerically and gated in closed form: the rim
# corner w = 1 (junction of the two charts on the y-axis line) sits
# at X(1) = (0, (p-1)/(2 sqrt(2p-1)), 0) for rho = 1 -- 1/sqrt(12),
# 1/sqrt(5), 2/3 for p = 2, 3, 5; Weber's p = 2 export carries its
# sphere markers on the same y-axis line.
#
# References:
# - R. Kusner, "Conformal geometry and complete minimal surfaces",
#   Bull. Amer. Math. Soc. 17 (1987) 291-295 -- the immersed minimal
#   spheres with 2n planar ends and their projective-plane quotients.
# - R. Bryant, "A duality theorem for Willmore surfaces", J. Diff.
#   Geom. 20 (1984) 23-53 -- planar-end spheres as Willmore surfaces;
#   the inversion of the 3-ended member is Boy's surface (the
#   Oberwolfach sculpture).
# - M. Weber, "Kusner's spheres with planar ends", minimalsurfaces.
#   blog (notebook `Kusner.nb` -- the Weierstrass data, the fft/tr/st
#   chart and the per-p windows transcribed above; PoVRay exports =
#   the registration ground truth).
# ==========================================================================

# per-p plate windows from the notebook (p = 4 interpolated)
KUSNER_XMIN = {2: 0.20, 3: 0.35, 4: 0.40, 5: 0.45, 6: 0.48}


def kusner_forms(p, rho=1.0):
    """(om1, om2, om3)(w) of Kusner's sphere, poles only at the ends."""
    s = math.sqrt(2.0 * p - 1.0)

    def om(w):
        w = np.asarray(w, dtype=complex)
        wp = w ** p
        poly = w ** (2 * p) + 2.0 * s * wp / (p - 1.0) - 1.0
        phi1 = 1j * rho * w ** (2 * p - 2) * (wp - s) ** 2 / poly ** 2
        phi2 = (1j / rho) * (1.0 + s * wp) ** 2 / poly ** 2
        dh = 1j * w ** (p - 1) * (wp - s) * (1.0 + s * wp) / poly ** 2
        return np.stack([-(phi1 - phi2) / 2.0,
                         1j * (phi1 + phi2) / 2.0, dh], axis=-1)
    return om


def _kus_gl(om, z0, z1, n=10):
    """Gauss-Legendre integral of om along the straight chord z0->z1.
    Broadcasts over equal-shaped complex arrays z0, z1."""
    gx, gw = np.polynomial.legendre.leggauss(n)
    z0 = np.asarray(z0, dtype=complex)
    z1 = np.asarray(z1, dtype=complex)
    mid = 0.5 * (z0 + z1)
    half = 0.5 * (z1 - z0)
    acc = 0.0
    for x_, w_ in zip(gx, gw):
        acc = acc + om(mid + x_ * half) * w_
    return acc * half[..., None]


def _kus_wgrid(p, xmin, nu, nv, mirror=False):
    """Weber's graded chart of the fundamental sector (see header).
    Returns (W, i_r0): the grid and the radial index of the r0 node
    (the boundary corner that maps to w = 0).  `mirror` reverses the
    angular grading (the exterior chart uses the mirrored grid so its
    rim nodes coincide with the interior rim nodes)."""
    s = math.sqrt(2.0 * p - 1.0)
    r0 = (p - s) / (p - 1.0)
    eps = 1e-12
    nx1 = max(6, nu // 3)
    nx2 = max(12, nu - nx1)
    t1 = np.linspace(xmin ** (1.0 / p), r0 ** (1.0 / p), nx1,
                     endpoint=False)
    t2 = np.linspace(r0 ** (1.0 / p), (1.0 - eps) ** (1.0 / p), nx2 + 1)
    XR = np.concatenate([t1, t2]) ** p
    tt = np.linspace(eps, math.pi / p - eps, nv)
    e = np.exp(1j * p * tt)
    YR = np.angle((r0 - e) / (r0 * e - 1.0))
    YR = np.where(YR < 0, YR + TAU, YR)
    YR[0] = 0.0
    YR[-1] = math.pi
    if mirror:
        YR = math.pi - YR[::-1]
    Z = XR[:, None] * np.exp(1j * YR[None, :])
    W = ((r0 + Z) / (1.0 + r0 * Z)) ** (1.0 / p)
    return W, nx1


def _kus_integrate(om, W):
    """Cumulative Re-integration of om over the grid W: anchor the rim
    mid-column by a radial ray from w = 0 (X(0) = 0 normalization),
    sweep the rim row, then every column rim -> inward.  All chords
    stay inside the chart, away from the end poles."""
    nu2, nv2 = W.shape
    F = np.zeros((nu2, nv2, 3), dtype=complex)
    jm, i0 = nv2 // 2, nu2 - 1
    zt = W[i0, jm]
    seg = np.linspace(0.0, 1.0, 33)
    val = np.zeros(3, dtype=complex)
    for k in range(32):
        val = val + _kus_gl(om, zt * seg[k], zt * seg[k + 1], 12)
    F[i0, jm] = val
    for j in range(jm + 1, nv2):
        F[i0, j] = F[i0, j - 1] + _kus_gl(om, W[i0, j - 1], W[i0, j])
    for j in range(jm - 1, -1, -1):
        F[i0, j] = F[i0, j + 1] + _kus_gl(om, W[i0, j + 1], W[i0, j])
    for i in range(i0 - 1, -1, -1):
        F[i] = F[i + 1] + _kus_gl(om, W[i + 1], W[i])
    return np.real(F)


def kusner_forms_ext(p, rho=1.0):
    """The forms pushed to the outer chart u = 1/w (w = infinity is a
    regular point; the substitution gives polynomial-stable forms):
    polyu = 1 + 2 s u^p/(p-1) - u^(2p), and
        phi1_u = -i rho (1 - s u^p)^2 / polyu^2,
        phi2_u = -(i/rho) u^(2p-2) (u^p + s)^2 / polyu^2,
        dh_u   = -i u^(p-1) (1 - s u^p)(u^p + s) / polyu^2."""
    s = math.sqrt(2.0 * p - 1.0)

    def om(u):
        u = np.asarray(u, dtype=complex)
        up = u ** p
        polyu = 1.0 + 2.0 * s * up / (p - 1.0) - u ** (2 * p)
        phi1 = -1j * rho * (1.0 - s * up) ** 2 / polyu ** 2
        phi2 = -(1j / rho) * u ** (2 * p - 2) * (up + s) ** 2 / polyu ** 2
        dh = -1j * u ** (p - 1) * (1.0 - s * up) * (up + s) / polyu ** 2
        return np.stack([-(phi1 - phi2) / 2.0,
                         1j * (phi1 + phi2) / 2.0, dh], axis=-1)
    return om


def kusner_patches(p, xmin, rho=1.0, nu=40, nv=None, xmin_ext=None):
    """(X_int, X_ext, i_r0): the two fundamental patches (nu', nv, 3)
    and the radial index of the w = 0 / u = 0 boundary corner.  The
    interior patch integrates the w-chart forms on Weber's sector
    grid; the exterior patch integrates the u = 1/w forms on the
    mirrored grid rotated into the u-plane (u = W e^(-i pi/p)), so
    its rim nodes land exactly on the interior rim nodes (reversed
    order) and u = 0 is the regular point w = infinity with X = 0."""
    if nv is None:
        nv = 30 * p
    if xmin_ext is None:
        xmin_ext = xmin
    Wi, i_r0 = _kus_wgrid(p, xmin, nu, nv)
    Xi = _kus_integrate(kusner_forms(p, rho), Wi)
    Wm, _ = _kus_wgrid(p, xmin_ext, nu, nv, mirror=True)
    U = Wm * np.exp(-1j * math.pi / p)
    Xe = _kus_integrate(kusner_forms_ext(p, rho), U)
    return Xi, Xe, i_r0


def kusner_landmark(p, rho=1.0):
    """X(w = 1) by a mid-sector ray + rim arc (chart-pole-free path)."""
    om = kusner_forms(p, rho)
    w0 = np.exp(1j * math.pi / (2.0 * p))
    seg = np.linspace(0.0, 1.0, 201)
    val = np.zeros(3, dtype=complex)
    for k in range(200):
        val = val + _kus_gl(om, w0 * seg[k], w0 * seg[k + 1], 14)
    th = np.linspace(math.pi / (2.0 * p), 0.0, 401)
    arc = np.exp(1j * th)
    val = val + _kus_gl(om, arc[:-1], arc[1:], 14).sum(axis=0)
    return np.real(val)


def _kus_frames(p):
    """[(M, parity), ...]: the measured dihedral space frames."""
    Sc = np.diag([-1.0, 1.0, -1.0])
    out = []
    for k in range(p):
        c, sn = math.cos(TAU * k / p), math.sin(TAU * k / p)
        Rk = np.array([[c, sn, 0.0], [-sn, c, 0.0], [0.0, 0.0, 1.0]])
        out.append((Rk, 1))
        out.append((Rk @ Sc, -1))
    return out


def _kus_grid_quads(nu2, nv2, off=0):
    return [(off + i * nv2 + j, off + (i + 1) * nv2 + j,
             off + (i + 1) * nv2 + j + 1, off + i * nv2 + j + 1)
            for i in range(nu2 - 1) for j in range(nv2 - 1)]


def _kus_snap(Xi, Xe, i_r0, p):
    """Snap the symmetry-line boundaries exactly (the weld then only
    absorbs float rounding): the arg-0 edges lie on the y-axis line,
    the arg-pi/p edges on the in-plane C2 axis at azimuth
    pi/2 - pi/p, the w = 0 / w = infinity boundary corners at the
    origin, and the exterior rim IS the interior rim reversed."""
    al = math.pi / 2.0 - math.pi / p
    u = np.array([math.cos(al), math.sin(al), 0.0])

    def to_y(P):
        Q = np.zeros_like(P)
        Q[..., 1] = P[..., 1]
        return Q

    def to_al(P):
        return (P @ u)[..., None] * u

    Xi[:, 0] = to_y(Xi[:, 0])                    # w real in [.., 1]
    Xi[:i_r0 + 1, -1] = to_y(Xi[:i_r0 + 1, -1])  # w real in [0, ..]
    Xi[i_r0 + 1:, -1] = to_al(Xi[i_r0 + 1:, -1])
    Xi[i_r0, -1] = 0.0                           # w = 0
    Xe[:, 0] = to_al(Xe[:, 0])
    Xe[:i_r0 + 1, -1] = to_al(Xe[:i_r0 + 1, -1])
    Xe[i_r0 + 1:, -1] = to_y(Xe[i_r0 + 1:, -1])
    Xe[i_r0, -1] = 0.0                           # w = infinity
    Xe[-1, :] = Xi[-1, ::-1]                     # shared |w| = 1 rim


def _kus_weld(V0, UV0, quads0, cls0, frames, tol):
    """Orbit-tile V0 under `frames` and weld coincident vertices OF
    THE SAME CLASS (two offset quantization passes).  The class keeps
    the two sheets of a genuine self-intersection point (w = 0 and
    w = infinity both map to the origin) from being fused into a
    non-manifold vertex: rim vertices are class 0 (they weld across
    the two charts), interior-chart vertices +1, exterior -1."""
    n0 = len(V0)
    Vp, Fp = [], []
    for i, (M, par) in enumerate(frames):
        Vp.append(V0 @ M.T)
        off = i * n0
        Fp.extend(tuple(off + k for k in (f[::-1] if par < 0 else f))
                  for f in quads0)
    V = np.concatenate(Vp, axis=0)
    UV = np.tile(UV0, (len(frames), 1))
    C = np.tile(np.asarray(cls0, dtype=np.int64), len(frames))
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for off_ in (0.0, 0.5):
        key = np.round(V / tol + off_).astype(np.int64)
        key = np.concatenate([key, C[:, None]], axis=1)
        _, first, inv = np.unique(key, axis=0, return_index=True,
                                  return_inverse=True)
        for i_ in range(len(V)):
            ra, rb = find(i_), find(int(first[inv[i_]]))
            if ra != rb:
                parent[ra] = rb
    root = np.array([find(i_) for i_ in range(len(V))])
    used = np.unique(root)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    idx = remap[root]
    F = []
    for f in Fp:
        g = [int(idx[f[0]])]
        for a_ in f[1:]:
            if int(idx[a_]) != g[-1]:
                g.append(int(idx[a_]))
        if len(g) > 3 and g[0] == g[-1]:
            g.pop()
        if len(g) >= 3 and len(set(g)) == len(g):
            F.append(tuple(g))
    return V[used], F, UV[used]


def kusner_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """Kusner sphere with 2n planar ends, n = 2 * order (order 1 ->
    the n = 2 member registered against Weber's export; odd n
    descends to the projective plane and ships as KUSNER_RP2).
    `radius` works the plate window: 1.2 = the notebook's own xmin
    for that n, larger = the flat ends follow further out."""
    del spec, theta
    p = int(np.clip(2 * order, 2, 8))
    xm0 = KUSNER_XMIN.get(p, 0.5)
    xm = float(np.clip(xm0 * (1.2 / max(float(radius), 0.2)) ** 1.5,
                       0.02, 0.85))
    nu_e = max(18, int(nu * 0.75))
    nv_e = max(24, int(nv * 0.25 * p))
    Xi, Xe, i_r0 = kusner_patches(p, xm, 1.0, nu_e, nv_e)
    _kus_snap(Xi, Xe, i_r0, p)
    nu2, nv2 = Xi.shape[:2]
    V0 = np.concatenate([Xi.reshape(-1, 3), Xe.reshape(-1, 3)], axis=0)
    quads0 = (_kus_grid_quads(nu2, nv2)
              + _kus_grid_quads(nu2, nv2, off=nu2 * nv2))
    gu = np.tile(np.arange(nu2)[:, None], (1, nv2)).reshape(-1)
    gv = np.tile(np.arange(nv2)[None, :], (nu2, 1)).reshape(-1)
    UV0 = np.stack([gu / max(nu2 - 1, 1), gv / max(nv2 - 1, 1)],
                   axis=-1)
    UV0 = np.concatenate([UV0, UV0], axis=0)
    cls_g = np.ones((nu2, nv2), dtype=np.int64)
    cls_g[-1, :] = 0                             # the shared rim row
    cls0 = np.concatenate([cls_g.reshape(-1), -cls_g.reshape(-1)])
    cls0[len(cls_g.reshape(-1)) + (nu2 - 1) * nv2:] = 0
    diag = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, UV = _kus_weld(V0, UV0, quads0, cls0, _kus_frames(p),
                         tol=1e-7 * diag)
    V = _center_fit(V, scale, V)
    return V, F, UV


# ==========================================================================
# The Horgan surface -- A MINIMAL SURFACE THAT DOES NOT EXIST, shipped
# as the honest near-miss.  In 1993 Hoffman and Karcher set up the
# Weierstrass data of a genus-2 Costa variant and named it after John
# Horgan's Scientific American piece suggesting computer experiments
# could replace proof: the numerical example looks utterly convincing,
# and the period problem provably cannot be closed.  This generator
# draws Weber's own near-miss illustration and MEASURES the failure
# instead of hiding it.
#
# Data (Weber's Horgan.nb, transcribed):
#     phi1 = rho sqrt(z^2-1) / (sqrt(z) sqrt(z^2-a^2)),
#     phi2 = sqrt(z) / (rho (z^2-a^2)^(3/2) sqrt(z^2-1)),
#     dh   = dz / (z^2-a^2),
# on the strip chart z = sqrt(a^2 + e^w), w = x + iy, y in (0, pi)
# (all square roots pointwise principal -- the strip maps into the
# closed upper half plane, where they are continuous).  rho is the
# notebook's Lopez-Ros balance int_0^1 phi1 = int_0^1 phi2, solved
# here by graded Gauss quadrature (endpoint substitutions at the
# z^(-1/2) and (1-z^2)^(-1/2) singularities).
#
# THE PERIOD PROBLEM, MEASURED (this is the point of the row): the
# strip boundary carries two planar symmetry curves --
#     y = 0   edge (z real > a):      x-mirror curve at   y = disy,
#     y = pi  edge, z real in (1, a): y-mirror curve at   x = disx
# (both constant along their edges to ~1e-7, measured).  Closing the
# surface under the two mirrors needs ONE translation t with
# disx - t = 0 AND disy - t = 0, i.e. disx = disy.  Measured across
# the family (see the HORGAN_GAP table in the zoo gate): disx(a)
# crosses zero near a ~ 1.115, disy(a) > 0 everywhere and vanishes
# only in the degenerate a -> 1 limit, and |disx - disy| has a
# MINIMUM of ~0.0093 near a ~ 1.06 -- it never closes.  The assembly
# translates by t = -disy (the catenoid-edge curve lands exactly in
# its mirror plane) and leaves the second seam open by the measured
# defect |disx - disy|, VISIBLE in the geometry exactly as in Weber's
# renders.
#
# GROUND TRUTH: registered against Weber's own PoVRay exports of all
# three members he renders (a = 1.01, 1.1, 1.5): one-sided means
# 0.15-0.16% (GT -> ours) / 0.33-0.39% (ours -> GT) of span.  For
# a = 1.1 and 1.5 his translation equals our -disy to fit precision;
# for a = 1.01 his dis came from NIntegrate straight through the
# near-collision of the branch points z = 1 and z = a = 1.01 at
# PrecisionGoal -> 5, and the registration-fitted value (-0.005)
# confirms his export used that (inaccurate) number, not the exact
# offset -- our edge-median measurement replaces the singular path.
#
# References:
# - D. Hoffman and H. Karcher, "Complete embedded minimal surfaces of
#   finite total curvature", in Geometry V (Encycl. Math. Sci. 90),
#   Springer 1997, sec. 3.4 -- the Horgan surface as the cautionary
#   example: the period problem that looks solvable and is not.
# - J. Horgan, "The death of proof", Scientific American 269:4 (1993)
#   92-103 -- the article the non-existent surface answers.
# - M. Weber, "The Horgan surface", minimalsurfaces.blog, repository
#   of non-existent surfaces (notebook `Horgan.nb` -- the data, the
#   strip chart, the member windows and the gap presentation
#   transcribed above; PoVRay exports = registration ground truth).
# ==========================================================================

HORGAN_A = (1.01, 1.1, 1.5)                     # Weber's three members
HORGAN_PADS = {1.01: (1.0, 3.0), 1.1: (2.0, 4.0), 1.5: (2.5, 4.5)}


def horgan_rho(a):
    """The notebook's Lopez-Ros balance rho = sqrt(I2 / I1) with
    I1 = int_0^1 phi1, I2 = int_0^1 phi2 at rho = 1 (both real and
    positive on (0, 1) with the upper-half-plane branches)."""
    gx, gw = np.polynomial.legendre.leggauss(200)
    t = 0.5 * (gx + 1.0)
    wt = 0.5 * gw

    def f1(z):
        return np.sqrt(1.0 - z * z) / (np.sqrt(z)
                                       * np.sqrt(a * a - z * z))

    def f2(z):
        return np.sqrt(z) / ((a * a - z * z) * np.sqrt(1.0 - z * z)
                             * np.sqrt(a * a - z * z))
    z1 = t * t                                   # z^(-1/2) endpoint
    I1 = float(np.sum(f1(z1) * 2.0 * t * wt))
    z2 = 1.0 - (1.0 - t) ** 2                    # (1-z^2)^(-1/2) endpoint
    I2 = float(np.sum(f2(z2) * 2.0 * (1.0 - t) * wt))
    return math.sqrt(I2 / I1)


def horgan_forms_w(a, rho):
    """(om1, om2, om3)(w) on the strip chart, dz/dw folded in:
    dz/dw = (z^2 - a^2)/(2z) with z = sqrt(a^2 + e^w)."""
    def om(w):
        w = np.asarray(w, dtype=complex)
        z = np.sqrt(a * a + np.exp(w))
        s1 = np.sqrt(z * z - 1.0)
        sa = np.sqrt(z * z - a * a)
        sz = np.sqrt(z)
        phi1 = rho * s1 / (sz * sa)
        phi2 = sz / (rho * (z * z - a * a) * s1 * sa)
        om3 = 1.0 / (z * z - a * a)
        jac = ((z * z - a * a) / (2.0 * z))[..., None]
        return np.stack([-(phi1 - phi2) / 2.0,
                         1j * (phi1 + phi2) / 2.0, om3], axis=-1) * jac
    return om


def horgan_forms_z(a, rho):
    """The forms in the sphere coordinate (for the anchor paths)."""
    def phi(z):
        z = np.asarray(z, dtype=complex)
        s1 = np.sqrt(z * z - 1.0)
        sa = np.sqrt(z * z - a * a)
        sz = np.sqrt(z)
        phi1 = rho * s1 / (sz * sa)
        phi2 = sz / (rho * (z * z - a * a) * s1 * sa)
        om3 = 1.0 / (z * z - a * a)
        return np.stack([-(phi1 - phi2) / 2.0,
                         1j * (phi1 + phi2) / 2.0, om3], axis=-1)
    return phi


def horgan_patch(a, nu=(24, 24, 36), ny=48, pad=None):
    """The fundamental strip patch.  Returns (F, X, meta): the real
    immersion grid (nx, ny, 3) normalized to X(z = 0) = 0, the radial
    node vector, and a dict with rho, the measured mirror-curve
    offsets disx / disy (edge medians, std ~1e-7) and the rotation
    axis image delta = -f(i)."""
    rho = horgan_rho(a)
    x0 = math.log(a * a)
    x1 = math.log(a * a - 1.0)
    if pad is None:
        pad = HORGAN_PADS.get(a, (2.0, 4.0))
    xmin, xmax = x1 - pad[0], x0 + pad[1]
    nx1, nx2, nx3 = nu
    X = np.unique(np.concatenate([
        np.linspace(xmin, x1, nx1), np.linspace(x1, x0, nx2),
        np.linspace(x0, xmax, nx3)]))
    eps = 1e-6
    Y = np.linspace(eps, math.pi - eps, ny)
    W = X[:, None] + 1j * Y[None, :]
    om = horgan_forms_w(a, rho)
    phi = horgan_forms_z(a, rho)
    ia = int(np.argmin(np.abs(X - x0)))
    jm = ny // 2
    za = np.sqrt(a * a + np.exp(W[ia, jm]))
    # anchor: straight z-path i -> z(anchor), clear of the real axis
    path = np.linspace(1j, za, 400)
    F = np.zeros((len(X), ny, 3), dtype=complex)
    F[ia, jm] = _kus_gl(phi, path[:-1], path[1:], 10).sum(axis=0)
    # ONE horizontal sweep along the mid row (clear of the two
    # integrable boundary singularities z = 1, z = 0 at y = pi), then
    # vertical sweeps down each column: every chord's distance to a
    # singular corner is at least |x - x1| / |x - x0|, so quadrature
    # error stays confined to the two columns AT the corners instead
    # of contaminating whole boundary rows (measured: a horizontal
    # boundary sweep shifted the mirror-curve offset disx by ~2e-2
    # depending on ny; the vertical scheme is grid-independent).
    for i in range(ia + 1, len(X)):
        F[i, jm] = F[i - 1, jm] + _kus_gl(om, W[i - 1, jm], W[i, jm])
    for i in range(ia - 1, -1, -1):
        F[i, jm] = F[i + 1, jm] + _kus_gl(om, W[i + 1, jm], W[i, jm])
    for j in range(jm + 1, ny):
        F[:, j] = F[:, j - 1] + _kus_gl(om, W[:, j - 1], W[:, j])
    for j in range(jm - 1, -1, -1):
        F[:, j] = F[:, j + 1] + _kus_gl(om, W[:, j + 1], W[:, j])
    # normalization X(z = 0) = 0: z-path i -> 0 down the imaginary
    # axis (upper-side branches, all integrable)
    zp = 1j * np.linspace(1.0, 1e-10, 1500)
    delta = _kus_gl(phi, zp[:-1], zp[1:], 10).sum(axis=0)
    Fr = np.real(F) - np.real(delta)
    x1m = x1 - 1e-9
    sel1 = X < x1m
    meta = {
        'rho': rho,
        'disy': float(np.median(Fr[:, 0, 1])),
        'disx': float(np.median(Fr[sel1, -1, 0])),
        'ey_std': float(Fr[:, 0, 1].std()),
        'ex_std': float(Fr[sel1, -1, 0].std()),
        'delta': -np.real(delta),
    }
    return Fr, X, meta


def horgan_gap(a, nu=(14, 14, 20), ny=28):
    """(disx, disy) of member a -- the two mirror-curve offsets whose
    difference is the unclosable period defect."""
    _F, _X, meta = horgan_patch(a, nu=nu, ny=ny)
    return meta['disx'], meta['disy']


def horgan_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """Weber's Horgan near-miss illustration: order picks the member
    (a = 1.01, 1.1, 1.5 -- his three renders), radius stretches the
    end windows.  The assembly closes the catenoid-edge mirror seam
    exactly (t = -disy) and leaves the other seam open by the
    measured period defect |disx - disy| -- the gap IS the point."""
    del spec, theta
    a = HORGAN_A[int(np.clip(order - 1, 0, len(HORGAN_A) - 1))]
    p0 = HORGAN_PADS[a]
    fac = float(np.clip(radius / 1.2, 0.4, 2.0))
    n = max(12, int(nu / 3))
    F, Xg, meta = horgan_patch(
        a, nu=(n, n, int(1.5 * n)), ny=max(24, int(nv * 0.8)),
        pad=(p0[0] * fac, p0[1] * fac))
    nx, ny2 = F.shape[:2]
    t = -meta['disy']
    x0_ = math.log(a * a)
    x1_ = math.log(a * a - 1.0)
    # snap each boundary arc onto its own measured symmetry element
    # (medians; std ~1e-7), so the two seams that DO close weld
    # vertex-to-vertex and the two that cannot stay open by exactly
    # the measured defect:
    #   y = 0 edge: the catenoid-edge mirror curve, plane y = disy;
    #   y = pi, x > x0: the (1,1,0) rotation-axis line x = y, z = 0;
    #   y = pi, x < x1: the gap mirror curve, plane x = disx (OPEN);
    #   y = pi, x1..x0: its R-image family, plane y = const (OPEN).
    F[:, 0, 1] = meta['disy']
    s1 = Xg <= x1_ + 1e-12
    s2 = (Xg >= x1_ - 1e-12) & (Xg <= x0_ + 1e-12)
    s3 = Xg >= x0_ - 1e-12
    F[s1, -1, 0] = meta['disx']
    F[s2, -1, 1] = float(np.median(F[s2, -1, 1]))
    ax_ = 0.5 * (F[s3, -1, 0] + F[s3, -1, 1])
    F[s3, -1, 0] = ax_
    F[s3, -1, 1] = ax_
    F[s3, -1, 2] = 0.0
    P0 = F.reshape(-1, 3)
    # rotate about the (1,1,0) symmetry line through f(0) = 0 FIRST,
    # then translate both copies (the notebook's order)
    R = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    tv = np.array([t, t, 0.0])
    parts = [P0 + tv, P0 @ R.T + tv]
    parts = parts + [P_ * np.array([-1.0, 1.0, 1.0]) for P_ in parts]
    parts = parts + [P_ * np.array([1.0, -1.0, 1.0]) for P_ in parts]
    # winding parities solved from the two welded seam families (the
    # in-surface Schwarz elements reverse the attached copy)
    flips = (False, True, True, False, True, False, False, True)
    quads0 = _kus_grid_quads(nx, ny2)
    NV = nx * ny2
    V = np.concatenate(parts, axis=0)
    Fc = []
    for k_, fl_ in enumerate(flips):
        for q in quads0:
            qq = tuple(int(i) + k_ * NV for i in q)
            Fc.append(qq[::-1] if fl_ else qq)
    # weld the two closable seam families by exact grid-index pairs:
    #   y = 0 edge (in the y = 0 plane after the translation) glues
    #   each copy to its y-mirror image; the axis arc glues each copy
    #   to its 180-degree rotation.  The gap seams are NOT welded --
    #   the x = +-|disx - disy| and y = +-|disx - disy| arc pairs
    #   stay open by twice the measured period defect, which is the
    #   finding this row ships.
    pairs = []

    def gid(part_, i_, j_):
        return part_ * NV + i_ * ny2 + j_

    iax = [i_ for i_ in range(nx) if s3[i_]]
    for pa_, pb_ in ((0, 4), (1, 5), (2, 6), (3, 7)):
        for i_ in range(nx):
            pairs.append((gid(pa_, i_, 0), gid(pb_, i_, 0)))
    for pa_, pb_ in ((0, 1), (2, 3), (4, 5), (6, 7)):
        for i_ in iax:
            pairs.append((gid(pa_, i_, ny2 - 1), gid(pb_, i_, ny2 - 1)))
    V, Fc, _first = _g1h_weld_pairs(V, Fc, pairs)
    V = _center_fit(V, scale, V)
    return V, Fc, None


# ==========================================================================
# Lopez-Martin slab surface -- the b = 1/2 member of the genus-one
# helicoid family, where the unsolved vertical period closes up again
# one full translation late.
#
# Lopez and Martin ("Minimal surfaces in a wedge of a slab") construct
# a translation-invariant minimal surface with planar ends that is
# NEITHER EMBEDDED NOR ORIENTABLE: its only self-intersection is along
# the z-axis, and it solves a Plateau problem in a wedge of a slab.
# Per Martin's observation on Weber's page, it belongs to the
# translation-invariant helicoid-with-handle family with the vertical
# period condition left unsolved: the same rhombic-torus theta data
# (G, dh as in the genus1helicoid block above) at tau = e^(i alpha0)
# but with the end parameter b FREE.  The period structure, measured
# here (and re-derived by the zoo gate at the notebook's own member):
#
#   * horizontal closure of BOTH lattice cycles holds for EVERY
#     (tau, b) once arg(dhper) = arg(A B)/2 and |rho1|^2 = |B|/|A|
#     (A = oint G~ dh~, B = oint dh~/G~ over the z -> z+1 cycle);
#     |dhper| then normalizes the z -> z+tau deck to (0, 0, -2);
#   * the z -> z+1 cycle translates by (0, 0, s(b)) -- the SLIDE the
#     blog animates.  b0 = 0.6290650983... is the root s = 0 (the
#     helicoid with handle; our solver reproduces the notebook's
#     rho_abs to 13 digits and dhper to 7);
#   * at b = 1/2 the slide is EXACTLY 2 = one full period, so the
#     torus quotient closes again, shifted one translation -- and the
#     data degenerates beautifully: theta factors pair up, dh becomes
#     CONSTANT (dh = dz/dhper, CHM-style) and G a perfect square with
#     double zero/pole, so both ends turn PLANAR.  That member is the
#     Lopez-Martin slab: flat plates at consecutive integer heights
#     joined by necks, self-intersecting along the vertical axis
#     (measured: far plate points cluster at z = 0, +-1; the axis
#     line lies in two sheets of the surface).
#
# GROUND TRUTH: the slab itself has no PoVRay export (the page's
# resource links are dead text), so the family CODE is registered at
# the helicoid member instead, against Weber's export of the
# translation-invariant helicoid with handle: built through THIS
# solver/sheet path, it registers at 0.245% (GT -> ours of span,
# export scale exactly 3.0 = his three periods normalized to height
# 2; the residual outlier fraction is his decorative low-resolution
# line sub-meshes, nt = 2).  The slab member is then the same
# verified code at the measured b = 1/2 constants, gated on its own
# structure (slide exactly 2, dh constant, planar-end flatness).
#
# References:
# - F. J. Lopez and F. Martin, "Minimal surfaces in a wedge of a
#   slab", Comm. Anal. Geom. 9 (2001) 683-723 -- the construction the
#   page presents.
# - D. Hoffman, H. Karcher, F. Wei, "The singly periodic genus-one
#   helicoid", Comment. Math. Helv. 74 (1999) 248-279 -- the family
#   whose vertical period condition is left unsolved here.
# - M. Weber, "Lopez-Martin slab surface" and "The translation
#   invariant helicoid with handle", minimalsurfaces.blog (notebook
#   `Translation-Helicoid-g-1.nb`: the theta data, the solved member
#   constants and the strip chart; his helicoid export = the
#   registration ground truth).
# ==========================================================================

_LMS_CACHE = {}


def lm_slab_member(alpha_deg=_G1H_ALPHA0, b=0.5, n=20001):
    """Solve the constants chain of the (tau = e^(i alpha), b) member:
    returns dict(tau, c, b, rho_abs, psi, dhper, slide, r0, a0,
    XA..XD, sym).  See the block header for the conditions."""
    key = (round(alpha_deg, 10), round(b, 12))
    if key in _LMS_CACHE:
        return _LMS_CACHE[key]
    tau = complex(np.exp(1j * np.pi * alpha_deg / 180.0))
    c = 0.5 * (1.0 + tau)
    th = genus1helicoid_theta11

    def om_raw(z):
        t1 = th(z + (b - 2.0) * c, tau)
        t2 = th(z - (1.0 + b) * c, tau)
        t3 = th(z + (b - 1.0) * c, tau)
        t4 = th(z - b * c, tau)
        e = np.exp(1j * np.pi * (b - 2.0 * z + 2.0 * tau + b * tau))
        return e * t1 * t2 / (t3 * t4), (t1 * t4) / (t3 * t2)

    z0 = 0.13 + 0.27j * tau.imag
    t = np.linspace(0.0, 1.0, n)

    def cyc(dz):
        z = z0 + dz * t
        Gt, dh = om_raw(z)
        dzs = np.diff(z)
        A = np.sum(0.5 * ((Gt * dh)[1:] + (Gt * dh)[:-1]) * dzs)
        B = np.sum(0.5 * ((dh / Gt)[1:] + (dh / Gt)[:-1]) * dzs)
        P3 = np.sum(0.5 * (dh[1:] + dh[:-1]) * dzs)
        return A, B, P3
    A1, B1, P31 = cyc(1.0)
    _At, _Bt, P3t = cyc(tau)
    psi = 0.5 * float(np.angle(A1 * B1))
    rho_abs = float(np.sqrt(np.abs(B1) / np.abs(A1)))
    r_abs = float(np.real(P3t * np.exp(-1j * psi))) / (-2.0)
    dhper = r_abs * np.exp(1j * psi)
    slide = float(np.real(P31 / dhper))

    # domain chart constants: r0 from the notebook's rectangle-shape
    # condition, a0 from tst(a0) = 1 - b
    def tst(z, r0_):
        m_ = 1.0 / (r0_ * r0_)
        z = np.asarray(z, dtype=complex)
        K2 = 2.0 * float(np.real(
            _g1h_ellf(np.array(1.0 - 1e-15 + 0j), m_)))
        return (z * _g1h_rf(1.0 - z * z + 0j, 1.0 - m_ * z * z + 0j,
                            np.ones_like(z)) / K2 + 0.5)

    def h(r_):
        return float(np.imag((1.0 + tau) / 2.0
                             * (1.0 + complex(tst(-r_ + 1e-14j, r_)))
                             - tau))
    lo, hi = 1.05, 8.0
    flo = h(lo)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        fm = h(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    r0 = 0.5 * (lo + hi)

    def g(a_):
        return float(np.real(complex(tst(a_ + 0j, r0)))) - (1.0 - b)
    lo2, hi2 = -0.999, 0.999
    flo2 = g(lo2)
    for _ in range(80):
        mid = 0.5 * (lo2 + hi2)
        fm = g(mid)
        if flo2 * fm <= 0:
            hi2 = mid
        else:
            lo2, flo2 = mid, fm
    a0 = 0.5 * (lo2 + hi2)

    def corner(tg):
        s_ = (tg - a0) / (r0 + tg * a0)
        return math.log(abs(s_))
    XA, XB = corner(1.0), corner(r0)
    XC, XD = corner(-1.0), corner(-r0)
    mem = dict(tau=tau, c=c, b=b, rho_abs=rho_abs, psi=psi,
               dhper=complex(dhper), slide=slide, r0=r0, a0=a0,
               m=1.0 / (r0 * r0), XA=XA, XB=XB, XC=XC, XD=XD,
               sym=XA + XD)
    _LMS_CACHE[key] = mem
    return mem


def _lms_omega(z, mem, rho1):
    tau, b, c = mem['tau'], mem['b'], mem['c']
    th = genus1helicoid_theta11
    t1 = th(z + (b - 2.0) * c, tau)
    t2 = th(z - (1.0 + b) * c, tau)
    t3 = th(z + (b - 1.0) * c, tau)
    t4 = th(z - b * c, tau)
    e = np.exp(1j * np.pi * (b - 2.0 * z + 2.0 * tau + b * tau))
    G = rho1 * e * t1 * t2 / (t3 * t4)
    o3 = (t1 * t4) / (t3 * t2) / mem['dhper']
    return 0.5 * (1.0 / G - G) * o3, 0.5j * (1.0 / G + G) * o3, o3


def _lms_zmap(w, mem):
    ew = np.exp(np.asarray(w, dtype=complex))
    s = (-mem['a0'] - mem['r0'] * ew) / (-1.0 + mem['a0'] * ew)
    z = np.asarray(s, dtype=complex)
    K2 = 2.0 * float(np.real(_g1h_ellf(np.array(1.0 - 1e-15 + 0j),
                                       mem['m'])))
    return (z * _g1h_rf(1.0 - z * z + 0j, 1.0 - mem['m'] * z * z + 0j,
                        np.ones_like(z)) / K2 + 0.5) * mem['c']


def lm_slab_sheet(mem, rho1, r1=-2.5, nu=101, nv=41, K=8, eps=1e-7):
    """Fundamental sheet of the (tau, b) member over the half strip
    [r1, sym - r1] x (0, pi) -- the generalized genus1helicoid sheet
    (same chart, member constants instead of the harvested ones)."""
    x_hi = mem['sym'] - r1
    corners = (mem['XA'], mem['XB'], mem['XC'], mem['XD'])
    spec = sorted(set(list(corners)
                      + [mem['sym'] - c_ for c_ in corners]))
    xs = _g1h_graded(r1, x_hi, nu, spec)
    xs = np.unique(np.round(np.concatenate(
        [xs, mem['sym'] - xs, spec,
         [mem['sym'] - s_ for s_ in spec]]), 12))
    # enforce EXACT mirror symmetry about SYM/2: the assembly welds by
    # the index map i <-> n-1-i, and the 1e-12 rounding of the deduped
    # union can otherwise leave near-twin nodes whose mirrors collapse
    # (a non-bijective mirror map -> slit seams in the weld)
    c_half = mem['sym'] / 2.0
    lo_ = xs[xs < c_half - 1e-9]
    xs = np.concatenate([lo_, [c_half], (c_half - lo_)[::-1] + c_half])
    t = np.linspace(0.0, 1.0, nv)
    ys = eps + (np.pi - 2 * eps) * (0.5 - 0.5 * np.cos(np.pi * t))
    nu2 = len(xs)
    j0 = nv // 2
    i0 = int(np.argmin(np.abs(xs - mem['sym'] / 2.0)))

    def seg(wa, wb):
        tt = np.linspace(0.0, 1.0, K + 1)
        W = wa[:, None] + (wb - wa)[:, None] * tt[None, :]
        Z = _lms_zmap(W, mem)
        o = np.stack(_lms_omega(Z, mem, rho1), axis=-1)
        dZ = np.diff(Z, axis=1)
        return np.sum(0.5 * (o[:, 1:] + o[:, :-1]) * dZ[..., None],
                      axis=1)

    F = np.zeros((nu2, nv, 3), complex)
    row = np.concatenate([np.zeros((1, 3), complex),
                          np.cumsum(seg(xs[:-1] + 1j * ys[j0],
                                        xs[1:] + 1j * ys[j0]), axis=0)])
    F[:, j0] = row - row[i0]
    for j in range(j0 + 1, nv):
        F[:, j] = F[:, j - 1] + seg(xs + 1j * ys[j - 1],
                                    xs + 1j * ys[j])
    for j in range(j0 - 1, -1, -1):
        F[:, j] = F[:, j + 1] - seg(xs + 1j * ys[j],
                                    xs + 1j * ys[j + 1])

    def pint(za, zb, n=20001):
        tt = np.linspace(0.0, 1.0, n)
        p = za + (zb - za) * tt
        o = np.stack(_lms_omega(p, mem, rho1), axis=-1)
        dz = np.diff(p)
        return np.sum(0.5 * (o[1:] + o[:-1]) * dz[:, None], axis=0)
    C = pint(1.0 + 0j, mem['tau'] / 2.0) \
        + pint(mem['tau'] / 2.0,
               complex(_lms_zmap(xs[i0] + 1j * ys[j0], mem)))
    return xs, ys, np.real(F + C[None, None, :])


def lm_slab_assemble(mem, rho1, storeys=1, r1=-2.5, nu=101, nv=41):
    """Finished (V, quads) of `storeys` translational cells of the
    b = 1/2 member: each cell is the strip sheet plus its 180-degree
    rotation about the z axis, stacked by (0, 0, 2) and welded by
    EXACT grid-index pairs, following `genus1helicoid_assemble` (the
    generic member of the same family) with the b = 1/2 degeneracy
    folded in.  At b = 1/2 the slide is a FULL period, so (i) the two
    chart axis segments (E0: x in [XA, XB] of the y = 0 edge, E1:
    x in [XC, XD] of the y = pi edge) land on the SAME z-axis segment
    pointwise -- welding each sheet<->rotation pair separately keeps
    the two sheets through the axis as distinct crossing walls (the
    surface's genuine self-intersection) -- and (ii) the in-cell
    ruling of the generic member has migrated onto the cell boundary:
    the plates carry horizontal straight rays (Schwarz lines in the
    plate planes), and ALL FOUR ray arcs of a level weld
    cell-to-cell, none in-cell.  The x-grid is symmetric about SYM/2
    with the corner values sample-exact, so every partner of sample i
    is sample nu' - 1 - i and no positional tolerance is involved:
      * axis welds, cell k: A(i, y=0) <-> B(i, y=0) for i in
        [iA, iB], and likewise on the y = pi edge;
      * level welds between cells k, k+1 (the plate line, where the
        sagging top plate of cell k crosses the bulging bottom plate
        of cell k+1 transversally -- the smooth Schwarz continuation
        swaps sheet and side): A_k.E0[x <= XA] <-> B_{k+1}.E1[sym-x]
        plus the three 180-degree-rotation mates."""
    xs, ys, X = lm_slab_sheet(mem, rho1, r1, nu, nv)
    nu2, nv2 = X.shape[:2]
    # center the cell at z = 0 (the sheet lands on [-2 - s, -s])
    zc = 0.5 * (X[..., 2].max() + X[..., 2].min())
    X = X - np.array([0.0, 0.0, zc])
    P0 = X.reshape(-1, 3)
    quads0 = _kus_grid_quads(nu2, nv2)
    parts, flips = [], []
    zoff = -(storeys - 1)
    for s_ in range(storeys):
        off = np.array([0.0, 0.0, 2.0 * s_ + zoff])
        parts.append(P0 + off)
        flips.append(False)
        parts.append(P0 * np.array([-1.0, -1.0, 1.0]) + off)
        flips.append(True)
    V = np.concatenate(parts, axis=0)
    NV = nu2 * nv2
    F = []
    for k_, fl_ in enumerate(flips):
        for q in quads0:
            qq = tuple(int(i) + k_ * NV for i in q)
            F.append(qq[::-1] if fl_ else qq)
    # the weld table (all exact index pairs; positions averaged)
    iA = int(np.argmin(np.abs(xs - min(mem['XA'], mem['XB']))))
    iB = int(np.argmin(np.abs(xs - max(mem['XA'], mem['XB']))))
    # mirror partner of sample i (x -> SYM - x); looked up rather than
    # assumed to be nu' - 1 - i, because the rounding that dedups the
    # concatenated grid can leave the index symmetry off by one
    mir = np.argmin(np.abs(xs[None, :]
                           - (mem['sym'] - xs)[:, None]), axis=1)

    def gid(sheet, i, j):
        return sheet * NV + i * nv2 + j

    pairs = []
    for k_ in range(storeys):
        p, r = 2 * k_, 2 * k_ + 1
        for i in range(iA, iB + 1):            # the two axis walls
            pairs.append((gid(p, i, 0), gid(r, i, 0)))
            pairs.append((gid(p, i, nv2 - 1), gid(r, i, nv2 - 1)))
        if k_ + 1 < storeys:                   # plate-line welds
            p2, r2 = 2 * (k_ + 1), 2 * (k_ + 1) + 1
            for i in range(0, iA + 1):
                m = int(mir[i])
                pairs.append((gid(p, i, 0), gid(r2, m, nv2 - 1)))
                pairs.append((gid(r, i, 0), gid(p2, m, nv2 - 1)))
                pairs.append((gid(p, i, nv2 - 1), gid(r2, m, 0)))
                pairs.append((gid(r, i, nv2 - 1), gid(p2, m, 0)))
    Vw, qw, _first = _g1h_weld_pairs(V, F, pairs)
    return Vw, qw


def lm_slab_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """Lopez-Martin slab: order = stacked periods, radius sets how far
    the flat plates follow their planar ends.  Immersed and one-sided
    as a complete surface -- the mesh keeps the two sheets through the
    z-axis as separate walls (a genuine self-intersection, exactly as
    in Weber's and the authors' pictures)."""
    del spec, theta
    storeys = int(np.clip(order, 1, 6))
    mem = lm_slab_member(_G1H_ALPHA0, 0.5)
    rho1 = mem['rho_abs'] * np.exp(1j * math.atan2(
        -62.8417365006266681, 108.369522264594063))
    r1 = -(1.7 + 0.8 * float(np.clip(radius / 1.2, 0.5, 2.5)))
    pnu = int(np.clip(nu * 1.4, 70, 200))
    pnv = int(np.clip(nv * 0.7, 30, 80))
    V, F = lm_slab_assemble(mem, rho1, storeys, r1, pnu, pnv)
    V = _center_fit(V, scale, V)
    return V, F, None


# ==========================================================================
# Weber-Wolf surfaces: the borderline case of the Hoffman-Meeks
# conjecture at genus 3 (two catenoidal + three planar ends), and its
# higher dihedral symmetrizations.
#
# The Hoffman-Meeks conjecture bounds an embedded finite-total-
# curvature surface of genus g by g + 2 ends; the borderline
# realizations are the catenoid (g = 0), Costa (g = 1), Wohlgemuth
# (g = 2), and at g = 3 the Weber-Wolf surface: two catenoidal ends,
# three planar ends, the connections between consecutive planar
# levels realized by Costa saddles.  The k-fold dihedral versions
# (k = 2 is the genus-3 surface; higher k gives genus 3(k - 1) --
# the k-cover of the sphere is totally branched over the 8 points
# 0, +-1, +-a, +-b, infinity, so Riemann-Hurwitz gives
# 2 - 2g = 2k - 8(k - 1); the meshes measure chi = 3 - 6k with 5 end
# rims, exactly 2 - 2(3k - 3) - 5) come from Weber's DH11 notebook
# (higher-symmetry portion by Ramazan Yol).
#
# Data (DH11.nb, transcribed; all powers pointwise principal):
#     phi1 = z^(1/k-1) (z^2-a^2)^(1/k-1) (z^2-1)^(1-1/k)
#            (z^2-b^2)^(-1-1/k),
#     phi2 = z^(1-1/k) (z^2-b^2)^(1+1/k) (z^2-1)^(1/k-1)
#            (z^2-a^2)^(-1-1/k),
#     dh   = dz/(z^2 - a^2),
#     om1  = -(rho phi1 - phi2/rho)/2,  om2 = i(rho phi1 + phi2/rho)/2,
#     rho  = sqrt( int_0^1 e^(-i pi/k) phi2 / int_0^1 e^(i pi/k) phi1 )
# (rho comes out REAL, gated), with (a, b) the two-parameter period
# problem.  The notebook's own test function is
#     tst = ( Re int_{(1+a)/2}^{i -> 10} om2,
#             Re int_{1/2}^{i -> (a+b)/2} (om1, om2)
#                 . (-sin(-pi/k), cos(-pi/k)) )
# and the members are its roots.  The notebook's stored (a, b) values
# satisfy tst only to 1e-8 (k = 2) .. 5e-3 (k = 5) -- quadrature-
# converged plateaus, so those are the notebook's own FindRoot
# tolerances, not our error; WW_MEMBERS stores the roots RE-SOLVED
# from the same test to ~1e-11 (Newton; k = 2 moved by 1e-8, k = 5
# by 6e-4).  The zoo gate re-derives tst at the stored roots.
#
# MESHED TO THE NOTEBOOK'S LOG CHART: w = log((z^2-a^2)^2/(z^2-b^2)),
# whose two inverse branches z = sqrt((2a^2 + e^w +- e^(w/2)
# sqrt(4a^2-4b^2+e^w))/2) cover the fundamental piece as two strips
# (f2 on y in (0, pi), f3 on y in (-pi, 0), Weber's graded windows
# with breaks at the critical values x1, x2, x3 = the logs of the
# images of z = 1, 0 and the branch-merge).  dz/dw is used IN CLOSED
# FORM (dw/dz = 2z(2/(z^2-a^2) - 1/(z^2-b^2))); integration is one
# horizontal sweep along a mid row plus vertical column sweeps, the
# anchor by a straight z-path from the base z = i for F2, the F3
# strip CONTINUED from F2 across the y = 0 seam (see ww_patches),
# and the normalization X(0) = 0 by the imaginary-axis path
# (upper-side principal branches throughout).  Assembly: 180-degree
# rotation about the horizontal line at azimuth -pi/(2k), mirror
# across y = 0, then the k vertical rotations -- the notebook's
# mp2/mp3/mp4 -- WELDED along the measured seam families into one
# manifold surface (see ww_mesh; chi = 3 - 6k with 5 end rims and a
# consistent orientation, measured at every k).
#
# GROUND TRUTH: registered against Weber's own PoVRay exports of the
# three members he renders (k = 2, 3, 4): GT -> ours one-sided means
# 0.27% / 0.23% / 0.20% of span at moderate resolution (0.17% at
# high; the residual outlier fraction, 6-15%, is his decorative
# low-resolution FR sub-meshes exactly as in his other packages),
# and the assembled extent ratios z/x match his exports to 4 digits
# (0.8928 / 0.9836 / 0.8829 for k = 2 / 3 / 4) -- pinned in the zoo
# gate.
#
# References:
# - M. Weber and M. Wolf, "Teichmueller theory and handle addition
#   for minimal surfaces", Ann. of Math. 156 (2002) 713-795 -- the
#   handle-addition machinery behind the family.
# - D. Hoffman and W. H. Meeks III, "The asymptotic behavior of
#   properly embedded minimal surfaces of finite topology", J. Amer.
#   Math. Soc. 2 (1989) 667-682 -- the conjecture whose g = 3
#   borderline case this surface realizes.
# - M. Weber, "Weber-Wolf surface of genus 3 with 5 ends",
#   minimalsurfaces.blog (notebook `DH11.nb`, higher-symmetry portion
#   by Ramazan Yol -- the data, the solved members and the log chart
#   transcribed above; PoVRay exports = registration ground truth).
# ==========================================================================

# (a, b) per k, re-solved from the notebook's own tst to ~1e-11
# (the notebook's stored values, satisfying tst to 1e-8..5e-3, are
# k=2: (1.03243674045806521, 1.09547100064006697),
# k=3: (1.0261070260032843, 1.0785891849884828),
# k=4: (1.0203659563370262, 1.0608027225248307),
# k=5: (1.0162839641877608, 1.0477361415441295))
WW_MEMBERS = {2: (1.0324367538, 1.0954710181),
              3: (1.0261022604, 1.0785622028),
              4: (1.0203291491, 1.0605994668),
              5: (1.0161738755, 1.0471651251)}
WW_WINDOWS = {2: (13.0, 0.2, 6.0), 3: (13.0, 0.2, 7.0),
              4: (13.0, 0.2, 8.0), 5: (13.0, 0.2, 8.0)}


def _ww_pow(z, e):
    return np.exp(e * np.log(z))


def ww_phis(k, a, b):
    def phi1(z):
        z = np.asarray(z, dtype=complex)
        return (_ww_pow(z, 1.0 / k - 1.0)
                * _ww_pow(z * z - a * a, 1.0 / k - 1.0)
                * _ww_pow(z * z - 1.0, 1.0 - 1.0 / k)
                * _ww_pow(z * z - b * b, -1.0 - 1.0 / k))

    def phi2(z):
        z = np.asarray(z, dtype=complex)
        return (_ww_pow(z, 1.0 - 1.0 / k)
                * _ww_pow(z * z - b * b, 1.0 + 1.0 / k)
                * _ww_pow(z * z - 1.0, 1.0 / k - 1.0)
                * _ww_pow(z * z - a * a, -1.0 - 1.0 / k))
    return phi1, phi2


def ww_rho(k, a, b, n=400):
    """The Lopez-Ros balance on (0, 1); real for the true members."""
    phi1, phi2 = ww_phis(k, a, b)
    gx, gw = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (gx + 1.0)
    wt = 0.5 * gw
    u = 3 * t * t - 2 * t ** 3
    du = 6 * t - 6 * t * t
    z = u + 0j
    I1 = np.sum(np.exp(1j * np.pi / k) * phi1(z) * du * wt)
    I2 = np.sum(np.exp(-1j * np.pi / k) * phi2(z) * du * wt)
    return complex(np.sqrt(I2 / I1))


def ww_forms(k, a, b, rho):
    phi1, phi2 = ww_phis(k, a, b)

    def om(z):
        z = np.asarray(z, dtype=complex)
        p1 = rho * phi1(z)
        p2 = phi2(z) / rho
        return np.stack([-(p1 - p2) / 2.0, 1j * (p1 + p2) / 2.0,
                         1.0 / (z * z - a * a)], axis=-1)
    return om


def ww_forms_w(k, a, b, rho, branch):
    """The forms in the log chart, dz/dw in closed form."""
    om = ww_forms(k, a, b, rho)

    def zfn(w):
        w = np.asarray(w, dtype=complex)
        ew = np.exp(w)
        s = np.sqrt(4 * a * a - 4 * b * b + ew)
        return np.sqrt(0.5 * (2 * a * a + ew
                              + branch * np.exp(0.5 * w) * s))

    def omw(w):
        z = zfn(w)
        dwdz = 2.0 * z * (2.0 / (z * z - a * a)
                          - 1.0 / (z * z - b * b))
        return om(z) / dwdz[..., None]
    return zfn, omw


def ww_tst(k, a, b, n=3000):
    """The notebook's own 2-component period test (see header)."""
    rho = ww_rho(k, a, b)
    om = ww_forms(k, a, b, rho)

    def path_int(waypts):
        tot = np.zeros(3, dtype=complex)
        for z0, z1 in zip(waypts[:-1], waypts[1:]):
            t = np.linspace(0.0, 1.0, n // len(waypts))
            u = 3 * t * t - 2 * t ** 3
            pts = z0 + (z1 - z0) * u
            tot = tot + _kus_gl(om, pts[:-1], pts[1:], 12).sum(axis=0)
        return tot
    t1 = float(np.real(path_int(
        [(1 + a) / 2.0 + 0j, 1j, 10.0 + 0j]))[1])
    v = np.real(path_int([0.5 + 0j, 1j, (a + b) / 2.0 + 0j]))[:2]
    d = np.array([-math.sin(-math.pi / k), math.cos(-math.pi / k)])
    return t1, float(v @ d)


def ww_patches(k, nx=10, ny=26, windows=None):
    """The two log-chart strips of the fundamental piece.  F2 is
    integrated as before (mid-row sweep + vertical columns, z-path
    anchor from z = i, X(0) = 0 normalization).  F3 is anchored by
    DIRECT CONTINUATION from F2 across the y = 0 seam: the two
    inverse branches agree on y = 0 for x below the branch merge
    log(4(b^2 - a^2)) (measured: the +-delta rows differ by
    O(delta) there, and by O(1) beyond the merge where the branches
    are genuinely distinct real-z arcs), so the two strips share one
    rigid frame instead of each trusting its own z-anchor path.  Its
    y grid mirrors F2's quadratic grading toward the seam -- the
    earlier sqrt grading (a mis-transcription of the notebook's
    NRange[eps^4, .]^(1/2)) truncated the strip 0.042 pi short of
    y = 0, which is exactly why the F3 pieces could never weld.
    Returns ([F2, F3], (a, b, rho), (X1, X2, Y2, Y1))."""
    a, b = WW_MEMBERS[k]
    rho = ww_rho(k, a, b).real
    plan1, plan2, cat = windows or WW_WINDOWS[k]
    x1, x2, x3 = sorted([
        math.log((a * a - 1.0) ** 2 / (b * b - 1.0)),
        math.log(a ** 4 / (b * b)),
        math.log(4 * (b * b - a * a))])
    om = ww_forms(k, a, b, rho)
    eps = 1e-11

    def xgrid(spec):
        xs = [np.linspace(lo, hi, nx, endpoint=False)
              for lo, hi in zip(spec[:-1], spec[1:])]
        return np.unique(np.concatenate(xs + [[spec[-1]]]))
    X1 = xgrid([-plan1, x1, x2, x3, plan2, cat])
    X2 = xgrid([-plan1, x1, x2, x3, plan2])
    Y2 = np.pi * np.linspace(eps ** 0.25, 1.0 - eps, ny) ** 2
    Y1 = -Y2[::-1]
    zfn2, omw2 = ww_forms_w(k, a, b, rho, +1)
    zfn3, omw3 = ww_forms_w(k, a, b, rho, -1)
    # ---- F2: anchored from z = i --------------------------------
    W2 = X1[:, None] + 1j * Y2[None, :]
    n2x, n2y = W2.shape
    F2 = np.zeros((n2x, n2y, 3), dtype=complex)
    jm = n2y // 2
    im = int(np.argmin(np.abs(X1 - 0.5 * (x3 + plan2))))
    za = complex(zfn2(W2[im, jm]))
    path = np.linspace(1j, za, 1200)
    F2[im, jm] = _kus_gl(om, path[:-1], path[1:], 10).sum(axis=0)
    for i in range(im + 1, n2x):
        F2[i, jm] = F2[i - 1, jm] + _kus_gl(omw2, W2[i - 1, jm],
                                            W2[i, jm])
    for i in range(im - 1, -1, -1):
        F2[i, jm] = F2[i + 1, jm] + _kus_gl(omw2, W2[i + 1, jm],
                                            W2[i, jm])
    for j in range(jm + 1, n2y):
        F2[:, j] = F2[:, j - 1] + _kus_gl(omw2, W2[:, j - 1], W2[:, j])
    for j in range(jm - 1, -1, -1):
        F2[:, j] = F2[:, j + 1] + _kus_gl(omw2, W2[:, j + 1], W2[:, j])
    # ---- F3: continued from F2 across the y = 0 seam ------------
    W3 = X2[:, None] + 1j * Y1[None, :]
    n3x, n3y = W3.shape
    F3 = np.zeros((n3x, n3y, 3), dtype=complex)
    xm = math.log(4.0 * (b * b - a * a))
    i_s = int(np.argmin(np.abs(X2 - (xm - 1.5))))
    zs2 = complex(zfn2(W2[i_s, 0]))
    zs3 = complex(zfn3(W3[i_s, -1]))
    seg = np.linspace(zs2, zs3, 9)
    F3[i_s, -1] = F2[i_s, 0] + _kus_gl(om, seg[:-1], seg[1:],
                                       10).sum(axis=0)
    jm3 = n3y // 2
    for j in range(n3y - 2, jm3 - 1, -1):
        F3[i_s, j] = F3[i_s, j + 1] + _kus_gl(omw3, W3[i_s, j + 1],
                                              W3[i_s, j])
    for i in range(i_s + 1, n3x):
        F3[i, jm3] = F3[i - 1, jm3] + _kus_gl(omw3, W3[i - 1, jm3],
                                              W3[i, jm3])
    for i in range(i_s - 1, -1, -1):
        F3[i, jm3] = F3[i + 1, jm3] + _kus_gl(omw3, W3[i + 1, jm3],
                                              W3[i, jm3])
    for j in range(jm3 + 1, n3y):
        F3[:, j] = F3[:, j - 1] + _kus_gl(omw3, W3[:, j - 1], W3[:, j])
    for j in range(jm3 - 1, -1, -1):
        F3[:, j] = F3[:, j + 1] + _kus_gl(omw3, W3[:, j + 1], W3[:, j])
    zp = 1j * np.linspace(1.0, 1e-9, 3000) ** 2
    delta = np.real(_kus_gl(om, zp[:-1], zp[1:], 10).sum(axis=0))
    return ([np.real(F2) - delta, np.real(F3) - delta], (a, b, rho),
            (X1, X2, Y2, Y1))


def ww_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """Weber-Wolf surface: order picks k = order + 1 (order 1 = the
    genus-3, 5-end borderline surface; higher = the k-fold dihedral
    versions of genus 3(k - 1)).  `radius` follows the catenoid and
    planar ends
    further out.  The orbit of the two-strip fundamental piece under
    the dihedral group is WELDED into one surface by exact grid-index
    pairs along its measured seam families:
      * the F2/F3 chart continuation across y = 0, x below the
        branch merge (same group element, same column);
      * the y = 0-plane mirror arcs of both strips beyond the merge
        (partner g My);
      * F2's y = pi edge, measured as THREE arcs: x < x1 (z real in
        (1, a)) lies in the y = 0 mirror plane (partner g My);
        x1 < x < log(a^4/b^2) (z real in (0, 1)) lies in the
        mirror plane at azimuth -pi/k (partner g Q,
        Q = Rz(-2 pi/k) My; at k = 2 this is the x = 0 plane);
        x beyond that (z imaginary) lies ON the in-surface straight
        line at azimuth -pi/(2 k) (partner g R) -- with the arc
        junctions at the z-axis (x = x1) and at f(0) = 0;
      * F3's whole y = -pi edge (z real in (a, b)), in the same
        azimuth -pi/k mirror plane (partner g Q).
    The ends (z = a catenoid, z = infinity, z = b planar) stay open
    rims.  Winding parity of each copy is the parity of its point
    group element (R and My each reverse)."""
    del spec, theta
    k = int(np.clip(order + 1, 2, 5))
    p1, p2, cat = WW_WINDOWS[k]
    fac = float(np.clip(radius / 1.2, 0.5, 1.6))
    nx = max(6, int(nu / 6))
    ny = max(16, int(nv * 0.55))
    patches, meta2, grids = ww_patches(k, nx=nx, ny=ny,
                                       windows=(p1 * fac, p2,
                                                cat * fac))
    a, b, _rho = meta2
    X1, X2, Y2, Y1 = grids
    F2g, F3g = patches
    xm = math.log(4.0 * (b * b - a * a))
    xr = math.log(a ** 4 / (b * b))
    u = np.array([math.cos(math.pi / (2 * k)),
                  -math.sin(math.pi / (2 * k)), 0.0])
    R = 2.0 * np.outer(u, u) - np.eye(3)
    My = np.diag([1.0, -1.0, 1.0])
    cq, sq = math.cos(-TAU / k), math.sin(-TAU / k)
    Q = np.array([[cq, -sq, 0.0], [sq, cq, 0.0],
                  [0.0, 0.0, 1.0]]) @ My
    nq = np.array([math.sin(math.pi / k), math.cos(math.pi / k),
                   0.0])
    x1s = math.log((a * a - 1.0) ** 2 / (b * b - 1.0))
    # snap each boundary arc onto its measured symmetry element
    s2sel = X1 > xm + 1e-9                       # F2 y=0 mirror arc
    F2g[s2sel, 0, 1] = 0.0
    s3sel = X2 > xm + 1e-9                       # F3 y=0 mirror arc
    F3g[s3sel, -1, 1] = 0.0
    mysel = X1 < x1s - 1e-9                      # F2 y=pi, x < x1
    F2g[mysel, -1, 1] = 0.0
    qsel = (X1 > x1s + 1e-9) & (X1 < xr - 1e-9)  # F2 y=pi Q-plane arc
    P_ = F2g[qsel, -1]
    F2g[qsel, -1] = P_ - (P_ @ nq)[:, None] * nq[None, :]
    rsel = X1 > xr + 1e-9                        # F2 y=pi R-line arc
    P_ = F2g[rsel, -1]
    F2g[rsel, -1] = (P_ @ u)[:, None] * u[None, :]
    ic1 = int(np.argmin(np.abs(X1 - x1s)))       # corner on the z axis
    if abs(X1[ic1] - x1s) < 1e-9:
        F2g[ic1, -1, 0] = 0.0
        F2g[ic1, -1, 1] = 0.0
    icr = int(np.argmin(np.abs(X1 - xr)))        # corner z = 0: f = 0
    if abs(X1[icr] - xr) < 1e-9:
        F2g[icr, -1] = 0.0
    P_ = F3g[:, 0]                               # F3 y=-pi Q-plane arc
    F3g[:, 0] = P_ - (P_ @ nq)[:, None] * nq[None, :]
    # orbit the two strips; keep each copy's transform for the welds
    n2x, n2y = F2g.shape[:2]
    n3x, n3y = F3g.shape[:2]
    NV2 = n2x * n2y
    NV3 = n3x * n3y
    q2 = _kus_grid_quads(n2x, n2y)
    q3 = _kus_grid_quads(n3x, n3y)
    Vs, Fs, recs = [], [], []
    off = 0
    for kk in range(k):
        th = TAU * kk / k
        c_, s_ = math.cos(th), math.sin(th)
        Rz = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
        for E, par in ((np.eye(3), 0), (R, 1), (My, 1), (My @ R, 0)):
            M = Rz @ E
            for si, Xg, qq, nvv in ((0, F2g, q2, NV2),
                                    (1, F3g, q3, NV3)):
                P0 = Xg.reshape(-1, 3)
                Vs.append(P0 @ M.T)
                Fs.extend(tuple(i + off for i in
                                (q[::-1] if par else q))
                          for q in qq)
                recs.append((si, M, off))
                off += nvv
    V = np.concatenate(Vs, axis=0)

    def findpart(si, M):
        for sj, Mj, o_ in recs:
            if sj == si and float(np.abs(Mj - M).max()) < 1e-9:
                return o_
        return -1

    pairs = []
    i_seam = [i_ for i_ in range(n3x) if X2[i_] <= xm + 1e-9]
    i_s2 = [i_ for i_ in range(n2x) if X1[i_] > xm - 1e-9]
    i_s3 = [i_ for i_ in range(n3x) if X2[i_] > xm - 1e-9]
    i_my = [i_ for i_ in range(n2x) if X1[i_] <= x1s + 1e-9]
    i_q = [i_ for i_ in range(n2x)
           if x1s - 1e-9 <= X1[i_] <= xr + 1e-9]
    i_r = [i_ for i_ in range(n2x) if X1[i_] >= xr - 1e-9]
    for si, M, o_ in recs:
        if si == 0:
            o3 = findpart(1, M)                  # chart continuation
            for i_ in i_seam:
                pairs.append((o_ + i_ * n2y,
                              o3 + i_ * n3y + n3y - 1))
            oMy = findpart(0, M @ My)            # y = 0 mirror
            for i_ in i_s2:
                pairs.append((o_ + i_ * n2y, oMy + i_ * n2y))
            for i_ in i_my:                      # y = pi, x < x1
                pairs.append((o_ + i_ * n2y + n2y - 1,
                              oMy + i_ * n2y + n2y - 1))
            oQ = findpart(0, M @ Q)              # azimuth pi/k mirror
            for i_ in i_q:
                pairs.append((o_ + i_ * n2y + n2y - 1,
                              oQ + i_ * n2y + n2y - 1))
            oR = findpart(0, M @ R)              # R line
            for i_ in i_r:
                pairs.append((o_ + i_ * n2y + n2y - 1,
                              oR + i_ * n2y + n2y - 1))
        else:
            oMy = findpart(1, M @ My)            # y = 0 mirror
            for i_ in i_s3:
                pairs.append((o_ + i_ * n3y + n3y - 1,
                              oMy + i_ * n3y + n3y - 1))
            oQ = findpart(1, M @ Q)              # azimuth pi/k mirror
            for i_ in range(n3x):
                pairs.append((o_ + i_ * n3y, oQ + i_ * n3y))
    V, Fs, _first = _g1h_weld_pairs(V, Fs, pairs)
    V = _center_fit(V, scale, V)
    return V, Fs, None



# ==========================================================================
# Kapouleas surfaces -- finite-total-curvature desingularizations of
# two coaxial catenoids, from Weber's repository page (notebook
# `Kapouleas.nb` by Ramazan Yol).
#
# Kapouleas (1997) constructed embedded finite-total-curvature minimal
# surfaces with arbitrarily many ends by taking coaxial unions of
# catenoids and planes and desingularizing the circular intersections
# with bent singly periodic Scherk surfaces.  This family is the
# simplest case: TWO coaxial catenoids whose two intersection circles
# are each replaced by a ring of k Scherk-type handles (k-fold
# dihedral symmetry).  STATUS, exactly as the page states it: "All
# period problems here have been solved numerically, so there is no
# simple existence proof for these surfaces yet."  Also per the page:
# for 2-fold symmetry no embedded examples are believed to exist (the
# k = 2 member is an immersed illustration); the first embedded ones
# appear at 3-fold symmetry; and the 3-dimensional period problem
# often has two solutions for the same pair of catenoidal growth
# rates (which is why Weber exports two members for the same k).
#
# Data (Yol's notebook, transcribed verbatim; th = theta11 on the
# rectangular torus tau = i t):
#     G0 = [th(z-((tau+1)/2+d)) th(z-(1/2-c))^(1/k) th(z-(1/2-b))
#           th(z-(1/2+a))^(1/k)] /
#          [th(z-((tau+1)/2-d)) th(z-(1/2+c))^(1/k) th(z-(1/2+b))
#           th(z-(1/2-a))^(1/k)],
#     dh0 = [th(z-((tau+1)/2+d)) th(z-((tau+1)/2-d)) th(z-(1/2-b))
#            th(z-(1/2+b))] /
#           [th(z-(1/2-c)) th(z-(1/2+c)) th(z-(1/2-a))
#            th(z-(1/2+a+tau))],
#     G = G0/G0(0),  dh = dh0/dh0(0),
# with the linear constraint b = (d + (a-c)/k) - 1/(2k) (satisfied
# EXACTLY by every stored row) and the free parameters (a, c, d, t)
# solved by the notebook's FindRoot on its printed 3-component test:
#     tst1 = Re int_{tau/2}^{tau/2+1/2} dh / c,
#     tst2 = Re int_0^{1/4+tau/4}^{1/2} om2 / (a-1/2),
#     tst3 = Re int_{1/2-b}^{(tau+1)/2-d} (om1, om2)
#            . (-sin pi/k, cos pi/k) / (a-1/2).
# G carries 1/k-fractional theta powers, so every path evaluation
# must be branch-tracked CONTINUOUSLY (log-unwrap along the whole
# polyline, anchored at the normalization point z = 0): the pointwise
# principal product jumps a k-th-root phase partway along the test
# paths, and independently-anchored path legs jump sheets as the
# branch point 1/2 - a crosses the path corner near a = 1/4 -- both
# produce phantom residuals of order 1e-1 that look exactly like
# unsolved members.  Tracked correctly, Yol's stored tables satisfy
# the notebook's own test to 1.3e-7 (worst, k = 2) and typically
# 1e-8..1e-10 -- far tighter than DH11's tables, so they are kept
# VERBATIM (nothing re-solved).
#
# The quotient of the full surface by its k-fold rotation is the
# (a,b,c,d,tau) torus with FOUR catenoidal ends (Weber's related
# page: "Tori with four catenoidal ends"): dh has simple poles at
# 1/2 +- c (the middle catenoid) and 1/2 - a, 1/2 + a + tau (the
# outer catenoid), and G has k-th-root branch points at those four
# points, so the full surface is the k-cover totally branched there:
# Riemann-Hurwitz gives chi_closed = -4(k-1), genus 2k - 1, with 4
# catenoidal ends (chi = 2 - 2(2k-1) - 4 = -4k once the end disks
# are cut).  `kap_growth` is the notebook's closed-form theta-product
# ratio of the two catenoidal growth rates (the embeddedness knob).
#
# MEASURED assembly topology (kap_mesh): 1 component, chi = -4k with
# 4 catenoid rims, manifold, oriented -- EXACT for k = 3, 4, 6 (and
# gated).  OPEN QUESTION at k = 2: the mesh measures chi = -6 (genus
# 2) against the naive 2-cover target -8; at k = 2 the cover's deck
# transformation coincides with the dihedral rotation Rz(pi) (the
# tau-cycle monodromy of G is e^(2 pi i/k) by the b-constraint), so
# the notebook's own 4k-copy assembly may genuinely realize a
# quotient with one fewer handle there.  The k = 2 member is the one
# the page says admits no embedded example; its geometry registers
# against Weber's export like the rest (all six exports land at
# 0.2-0.4% median of span with per-export cutoff radii, r0 ~ 1e-2.4
# .. 1e-1.6, r1 ~ 1e1.6 .. 1e2.4 -- his exports truncate the ends
# closer in than the notebook's r0 = 0.01, r1 = 1000 cell).
#
# References:
# - N. Kapouleas, "Complete embedded minimal surfaces of finite total
#   curvature", J. Diff. Geom. 47 (1997) 95-169 -- the
#   desingularization construction this family illustrates.
# - M. Weber, "Kapouleas surfaces", minimalsurfaces.blog (notebook
#   `Kapouleas.nb` by Ramazan Yol -- the theta data, the solved
#   member tables and the 3-component period test transcribed above;
#   PoVRay exports = registration ground truth; the page's
#   numerical-only status is recorded as stated).
# ==========================================================================

# solved members, {k: ((a, b, c, d, t), ...)}, tau = i t -- Yol's
# tables verbatim (every row satisfies b = (d + (a-c)/k) - 1/(2k)
# exactly and the notebook's own period test to <= 1.3e-7, measured)
KAP_SOLS = {
    2: (
        (0.1755, 0.06698019974822289, 0.010057381750312164, 0.234258890623379, 0.7543457613853664),
        (0.176, 0.06727996568588351, 0.010120642536458193, 0.2343402869541126, 0.7547849133737129),
        (0.18, 0.06966403973665058, 0.010623659602824723, 0.23497586953806296, 0.7580549552839374),
        (0.2, 0.08130442498240442, 0.01306721313256788, 0.23783803154868838, 0.7690179488802832),
        (0.22, 0.0926593744173635, 0.015407459561082441, 0.24036310419790474, 0.7730964459414758),
    ),
    3: (
        (0.075, 0.03315207605693751, 0.01019564080267277, 0.17821728965782843, 0.41807116769369357),
        (0.08, 0.03678303000587341, 0.012201223677745657, 0.18085010456512196, 0.4271423133078388),
        (0.09, 0.043520973027908, 0.016015280624279836, 0.1855260665693346, 0.44239645301951497),
        (0.1, 0.049880751248852995, 0.01964157681891243, 0.18976127685515712, 0.45543105573571396),
        (0.11, 0.05602100706268037, 0.02309837104971218, 0.1937204640792511, 0.467035073618209),
        (0.12, 0.062013249491964184, 0.02638368389120975, 0.19747447745570076, 0.47755683388872017),
        (0.13, 0.0678955565957585, 0.02949245583819962, 0.20105970854182503, 0.4871796522425911),
        (0.14, 0.07369024573622202, 0.03242078456956554, 0.2044971739260772, 0.49600976452426043),
        (0.16, 0.08506771141996647, 0.037731047816367644, 0.210978060692089, 0.5115259277708725),
        (0.18, 0.09621026945709352, 0.04232314269176511, 0.21698465035434855, 0.524376008919746),
        (0.2, 0.10715051913771009, 0.04622736859204474, 0.22255964200172498, 0.5346435891702486),
        (0.22, 0.11790861532077165, 0.049484046045620955, 0.22773663066931196, 0.5423188577334739),
    ),
    4: (
        (0.0497, 0.023167936059970945, 0.010009894307294539, 0.13824540963679458, 0.27276381351319257),
        (0.0498, 0.023246531765591383, 0.010073907966680826, 0.1383150087572616, 0.27293258937511927),
        (0.04984079422299173, 0.02327853777835248, 0.0101, 0.13834333922260456, 0.2730012034058101),
        (0.04999732121030994, 0.023401046673533932, 0.0102, 0.13845171637095643, 0.27326322448816487),
        (0.05, 0.02340313922230222, 0.010201709829358982, 0.13845356667964195, 0.2732676916410874),
        (0.06, 0.03063679435349026, 0.016364061393047687, 0.1447278097017522, 0.2873793808672695),
        (0.07, 0.03734712211647173, 0.022271189612112184, 0.15041491951949978, 0.29879708835732705),
        (0.08, 0.043873705614721786, 0.027963785309997574, 0.15586465194222118, 0.3088837557049866),
        (0.09, 0.05031822504189021, 0.03342343241940676, 0.1611740831467419, 0.31812745944669935),
        (0.1, 0.056716169221540574, 0.03863104939408024, 0.16637393157006064, 0.3267679901942672),
        (0.12, 0.06940322441399571, 0.048236806295359785, 0.17646242598783565, 0.34272048276533806),
        (0.14, 0.08192523463103524, 0.05670910270164892, 0.18610251030644745, 0.35728093865363536),
        (0.16, 0.09423525625436652, 0.06401540290776463, 0.19523910698130767, 0.37062246284001005),
        (0.18, 0.10628992919323757, 0.07015069815117489, 0.2038276037310313, 0.38275254294082556),
        (0.2, 0.11805719626772238, 0.07513257193002774, 0.2118403392502293, 0.39358595170004596),
        (0.22, 0.1295160921992753, 0.0789965853873448, 0.21926523854611146, 0.4029776971248489),
    ),
    6: (
        (0.032, 0.016200786461698563, 0.010289631050414, 0.0959157249701009, 0.15722030652968705),
        (0.035, 0.018434046777331875, 0.012743194382458503, 0.09805791250774162, 0.1603584159202951),
        (0.04, 0.022056004395236375, 0.016704055251645703, 0.10150668027051066, 0.16496033658295503),
        (0.05, 0.029240937339811923, 0.024303961581307085, 0.1082915976033631, 0.1729849754367468),
        (0.06, 0.03647800143689704, 0.031582192329031235, 0.11507503349173558, 0.18020438103946748),
        (0.07, 0.04376526226324175, 0.03859540435544804, 0.12186449632248308, 0.18693576118370575),
        (0.08, 0.051071679823866165, 0.045369661993360226, 0.12863329015609287, 0.19329638261389598),
        (0.09, 0.05836954526027237, 0.05191596615720512, 0.13535553961980656, 0.1993444574105409),
        (0.1, 0.06563779615749361, 0.0582365842272164, 0.142010560195363, 0.20511745085237681),
        (0.11, 0.07286058482594719, 0.06432829429841681, 0.14858196720901665, 0.2106440375142692),
        (0.12, 0.08002556348410157, 0.07018419339650228, 0.1550562623835186, 0.21594812439323155),
        (0.13, 0.0871226087555169, 0.07579476953473728, 0.16142173701130644, 0.2210502636791432),
        (0.14, 0.09414297236711953, 0.08114858251661612, 0.1676677361198889, 0.22596814453122957),
        (0.15, 0.10107873918593759, 0.08623273543542435, 0.17378419509184165, 0.2307167316573859),
        (0.16, 0.10792249192816548, 0.0910332372210538, 0.17976136479834112, 0.23530824916418258),
        (0.17, 0.11466711353295854, 0.09553531298346923, 0.18558966569687008, 0.2397520824133301),
        (0.18, 0.12130568238739499, 0.09972369376134649, 0.1912596313476194, 0.24405462820266519),
        (0.19, 0.1278314311496745, 0.10358290130106061, 0.19676191469985127, 0.24821911062874272),
        (0.2, 0.13423774913670772, 0.10709753215706078, 0.20208733782955118, 0.25224537733859204),
        (0.21, 0.1405182133558348, 0.11025253669610757, 0.20722696947185273, 0.2561296911795149),
        (0.22, 0.14666663593386703, 0.11303348179663357, 0.21217221623330595, 0.25986453239272184),
        (0.23, 0.1526771171247635, 0.11542678124907915, 0.21691491399961005, 0.2634384248492769),
        (0.24, 0.15854409411464976, 0.11741987545601075, 0.22144740669065152, 0.26683579565860704),
        (0.25, 0.16426237708431518, 0.1190013424442995, 0.22576260082503175, 0.27003687050085373),
        (0.26, 0.1698271657664664, 0.12016092547872072, 0.22985398667958648, 0.273017597764656),
        (0.27, 0.17523404213100718, 0.12088946836344626, 0.23371562019158157, 0.2757495837753267),
        (0.275, 0.17787697953605952, 0.12108950219461014, 0.23555856323516117, 0.2770122191791659),
        (0.28, 0.1804789376851667, 0.12117875688337308, 0.2373420638323955, 0.2782000103565516),
        (0.29, 0.18555807685837872, 0.12102127251470916, 0.24072828894416354, 0.28033149577237854),
        (0.3, 0.1904679006612544, 0.12040987119226158, 0.24386954585996468, 0.2821018514530964),
        (0.32, 0.19976590484501155, 0.11779630308492174, 0.24939862202583182, 0.2843637520081887),
        (0.34, 0.2083453226885899, 0.11327320103607691, 0.2538908561946027, 0.28453072294831167),
        (0.36, 0.21617639605094147, 0.10675520139463286, 0.25730226295004693, 0.28201569438082746),
        (0.38, 0.22322595171841264, 0.09812424682337755, 0.2595799928556422, 0.2760319982019401),
        (0.4, 0.2294593393414061, 0.08722387868447055, 0.2606633191221512, 0.26547497908130346),
        (0.42, 0.2348500039031039, 0.07387573341816778, 0.2604959594727985, 0.24871710423565305),
        (0.43, 0.23722952140877002, 0.06624029914267435, 0.25993623793254905, 0.237278567223596),
        (0.44, 0.23940795611233073, 0.057965941645410676, 0.25906894638656586, 0.22325269255353622),
        (0.445, 0.2404269415350389, 0.05359935930823804, 0.2585268347530786, 0.21510225308342867),
        (0.45, 0.24140337996789774, 0.0490908074914815, 0.2579185145498113, 0.20608544713773394),
    ),
    8: (
        (0.07, 0.04859172227753544, 0.0457808575834959, 0.10806432947547243, 0.1366065915306091),
        (0.1, 0.07206763376832387, 0.06717469199902004, 0.13046447026820138, 0.15045681363976926),
        (0.12, 0.08745814141616517, 0.08059300756772426, 0.1450322673621307, 0.15841867927959682),
        (0.15, 0.10999248303015252, 0.09920420397465858, 0.16614300852698485, 0.16901444351789563),
        (0.2, 0.1455209402791164, 0.1248906788646222, 0.1986322751371942, 0.18431065528004176),
    ),
    10: (
        (0.07, 0.0520380486996463, 0.0499847948043074, 0.10003652818007705, 0.10809701628142294),
        (0.2, 0.15362209322749176, 0.1368086722665547, 0.19730296045414722, 0.144626758198563),
    ),
    12: (
        (0.07, 0.05454792127545446, 0.05287460747421045, 0.09478747189830533, 0.0895387516392219),
        (0.1, 0.07978920595331249, 0.0766217658381284, 0.11950768643982318, 0.0984699793700975),
        (0.12, 0.09638971202770821, 0.09187516107369417, 0.13571264211718273, 0.1034139931820036),
        (0.15, 0.1208199802678355, 0.11361881339327531, 0.15945488138394176, 0.1098058805781098),
    ),
}


def kap_shifts(k, a, b, c, d, tau):
    """(shift, exponent) factor list of G0."""
    return [((tau + 1.0) / 2.0 + d, 1.0), (0.5 - c, 1.0 / k),
            (0.5 - b, 1.0), (0.5 + a, 1.0 / k),
            ((tau + 1.0) / 2.0 - d, -1.0), (0.5 + c, -1.0 / k),
            (0.5 + b, -1.0), (0.5 - a, -1.0 / k)]


def kap_G0_pv(Z, k, a, b, c, d, tau):
    """G0 as the pointwise principal-branch product (patch use only;
    NOT continuous along arbitrary paths -- see the block header)."""
    th = genus1helicoid_theta11
    out = np.ones_like(np.asarray(Z, dtype=complex))
    for s, e in kap_shifts(k, a, b, c, d, tau):
        v = th(Z - s, tau)
        if e == 1.0:
            out = out * v
        elif e == -1.0:
            out = out / v
        else:
            out = out * np.exp(e * np.log(v))
    return out


def kap_G0_path(zp, k, a, b, c, d, tau):
    """G0 along a 1-D path, every factor's log unwrapped (the branch
    is anchored at the path's FIRST node)."""
    th = genus1helicoid_theta11
    zp = np.asarray(zp, dtype=complex)
    tot = np.zeros_like(zp)
    for s, e in kap_shifts(k, a, b, c, d, tau):
        v = th(zp - s, tau)
        lg = np.log(np.abs(v)) + 1j * np.unwrap(np.angle(v))
        tot = tot + e * lg
    return np.exp(tot)


def kap_dh0(Z, a, b, c, d, tau):
    th = genus1helicoid_theta11
    Z = np.asarray(Z, dtype=complex)

    def f(s):
        return th(Z - s, tau)
    return (f((tau + 1.0) / 2.0 + d) * f((tau + 1.0) / 2.0 - d)
            * f(0.5 - b) * f(0.5 + b)) / (
        f(0.5 - c) * f(0.5 + c) * f(0.5 - a) * f(0.5 + a + tau))


def kap_tst(k, a, b, c, d, t, n=20001):
    """The notebook's printed 3-component period test, every G
    evaluation branch-tracked continuously from z = 0 (independently
    anchored legs jump sheets for a > ~1/4; see the block header)."""
    tau = 1j * t
    g00 = complex(kap_G0_pv(np.array([0j]), k, a, b, c, d, tau)[0])
    dh00 = complex(kap_dh0(np.array([0j]), a, b, c, d, tau)[0])

    def om_on(zp):
        G = kap_G0_path(zp, k, a, b, c, d, tau) / g00
        dh = kap_dh0(zp, a, b, c, d, tau) / dh00
        return np.stack([(-G * dh + dh / G) / 2.0,
                         1j * (G * dh + dh / G) / 2.0, dh], axis=-1)
    u = np.linspace(0.0, 1.0, n)
    w = u * u * (3.0 - 2.0 * u)
    # tst1: dh alone (single-valued) along the top mid-line
    zp = tau / 2.0 + 0.5 * u
    dh = kap_dh0(zp, a, b, c, d, tau) / dh00
    t1 = float(np.trapezoid(dh * 0.5, u).real / c)
    # tst2: ONE continuous branch along the polyline 0 -> 1/4+tau/4
    # -> 1/2
    zp = np.concatenate([(0.25 + tau / 4.0) * w,
                         (0.25 + tau / 4.0)
                         + (0.25 - tau / 4.0) * w[1:]])
    om = om_on(zp)
    t2 = float(np.trapezoid(om[:, 1], zp).real / (a - 0.5))
    # tst3: branch carried from z = 0 via an interior approach; the
    # leg endpoints are G zeros/poles (the om limit is finite), nodes
    # stay 1e-9 inside
    ws = w * (1.0 - 2e-9) + 1e-9
    leg0 = 0.5 - b
    leg1 = (tau + 1.0) / 2.0 - d
    appr = (0.25 + tau / 4.0) * w
    appr2 = (0.25 + tau / 4.0) + (
        (leg0 + 0.02 * (leg1 - leg0)) - (0.25 + tau / 4.0)) * w[1:]
    leg = leg0 + (leg1 - leg0) * ws
    zp = np.concatenate([appr, appr2, leg])
    om = om_on(zp)
    nl = len(leg)
    I3 = np.trapezoid(om[-nl:, :2], leg[:, None], axis=0)
    t3 = float((I3[0].real * (-math.sin(math.pi / k))
                + I3[1].real * math.cos(math.pi / k)) / (a - 0.5))
    return t1, t2, t3


def kap_growth(k, a, b, c, d, t):
    """The notebook's closed-form catenoid growth-rate ratio (the
    embeddedness check of the page)."""
    th = genus1helicoid_theta11
    tau = 1j * t

    def f(s):
        return complex(th(np.array([s], dtype=complex), tau)[0])
    num = (f(-a - b) * f(-a + b) * f(a - c) * f(-2 * c)
           * f(0.5 - a - d + 0.5 * (-1 - tau))
           * f(0.5 - a + d + 0.5 * (-1 - tau)) * f(-a - c - tau))
    den = (f(-a - c) * f(-b - c) * f(b - c) * f(-a + c)
           * f(0.5 - c - d + 0.5 * (-1 - tau))
           * f(0.5 - c + d + 0.5 * (-1 - tau)) * f(-2 * a - tau))
    return float((num / den).real)



# member knob order: Weber's six exported members first
# (k=2 a=.22 | k=3 a=.14 | k=4 a=.07 | k=4 a=.22 | k=6 a=.11 |
#  k=6 a=.27), then one representative per remaining k
KAP_MEMBERS = ((2, 4), (3, 7), (4, 6), (4, 15), (6, 9), (6, 25),
               (8, 1), (10, 1), (12, 3))


def _kap_ellK(m):
    return float(np.real(_g1h_rf(np.array([0j]),
                                 np.array([1.0 - m + 0j]),
                                 np.array([1.0 + 0j]))[0]))


def _kap_ellF(z, m):
    z = np.asarray(z, dtype=complex)
    return z * _g1h_rf(1.0 - z * z, 1.0 - m * z * z, np.ones_like(z))


def kap_chart(k, a, b, c, d, t):
    """The notebook's EllipticF rectangle chart: lambda from its
    aspect condition, rect mapping the upper half plane onto the
    quarter torus [0, 1/2] x [0, t/2], the Moebius trf placing the
    polar grid's r -> 0 at the middle end 1/2 - c and r -> infinity
    at the outer end 1/2 - a, and the special boundary preimages
    used as mesh grading breaks.  Returns (rect, trf, invtrf,
    consts)."""
    y0 = t / 2.0

    def lam_eq(lam):
        return (lam * _kap_ellK(1.0 - lam * lam) / 2.0
                / (2.0 * _kap_ellK(1.0 / lam ** 2)) - y0)
    lo, hi = 1.0 + 1e-9, 50.0
    flo = lam_eq(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if flo * lam_eq(mid) <= 0:
            hi = mid
        else:
            lo = mid
    lam = 0.5 * (lo + hi)
    m = 1.0 / lam ** 2
    K = _kap_ellK(m)

    def rect(z):
        return (_kap_ellF(z, m) + K) / K / 4.0

    def rect_real(x):
        return rect(np.asarray(x, dtype=complex) + 1e-14j)

    def solve_bottom(target):
        lo_, hi_ = -1.0 + 1e-15, 1.0 - 1e-15
        for _ in range(200):
            mid_ = 0.5 * (lo_ + hi_)
            if float(np.real(rect_real(mid_))) < target:
                lo_ = mid_
            else:
                hi_ = mid_
        return 0.5 * (lo_ + hi_)

    def solve_top(target_x):
        lo_, hi_ = lam * (1.0 + 1e-12), 1e8
        for _ in range(220):
            mid_ = math.sqrt(lo_ * hi_)
            if float(np.real(rect_real(mid_))) > target_x:
                lo_ = mid_
            else:
                hi_ = mid_
        return math.sqrt(lo_ * hi_)
    alpha = solve_bottom(0.5 - a)
    beta = solve_bottom(0.5 - c)
    consts = dict(lam=lam, m=m, K=K, alpha=alpha, beta=beta,
                  bn=solve_bottom(0.5 - b), dn=solve_top(0.5 - d),
                  eta=solve_bottom((1.0 - c) / 2.0),
                  xi=solve_bottom((0.5 - a) / 2.0))

    def trf(w):
        w = np.asarray(w, dtype=complex)
        return (-beta + alpha * w) / (-1.0 + w)

    def invtrf(z):
        return (z - beta) / (z - alpha)
    return rect, trf, invtrf, consts


def kap_sheet(k, mi, nx=4, ny=18, r0=0.01, r1=1000.0, subdiv=10):
    """Fundamental patch of Kapouleas member (k, mi) over the
    notebook's polar grid (upper half w-plane; r0/r1 truncate the
    middle/outer catenoid ends).  Every omega evaluation is
    branch-tracked: the anchor node continues G from z = 0 (where
    G = 1 by normalization) and each grid sweep carries the branch
    forward node to node, so the whole patch sits on ONE sheet of
    the k-cover.  Returns (W, Z, F, meta) with F the real immersion
    (nr, ntheta, 3) after the notebook's two normalizations (f0(0)
    subtracted via the anchor at z = 0; ff1 sliding the d-line onto
    the origin)."""
    a, b, c, d, t = KAP_SOLS[k][mi]
    tau = 1j * t
    rect, trf, invtrf, C = kap_chart(k, a, b, c, d, t)
    lam = C['lam']
    br = [float(np.real(invtrf(x)))
          for x in (-lam, -1.0, 1.0, lam, C['dn'], C['eta'], C['xi'])]
    br += [-float(np.real(invtrf(x)))
           for x in (C['bn'], C['alpha'] + 0.001, C['beta'] - 0.001)]
    xspec = sorted(set([r0, r1] + [abs(x) for x in br
                                   if np.isfinite(x)
                                   and r0 < abs(x) < r1]))
    xs = [np.exp(np.linspace(math.log(lo), math.log(hi), nx,
                             endpoint=False))
          for lo, hi in zip(xspec[:-1], xspec[1:])]
    xr = np.unique(np.concatenate(xs + [np.array([xspec[-1]])]))
    eps = 1e-6
    yr = math.pi * np.linspace(eps, 1.0 - eps, ny) ** 2
    W = xr[:, None] * np.exp(1j * yr[None, :])
    Z = rect(trf(W))
    g00 = complex(kap_G0_pv(np.array([0j]), k, a, b, c, d, tau)[0])
    dh00 = complex(kap_dh0(np.array([0j]), a, b, c, d, tau)[0])
    th = genus1helicoid_theta11
    shifts = kap_shifts(k, a, b, c, d, tau)

    def om_batch(zp, Gstart=None, axis=-1):
        """omega and G along subdivided paths (last axis = path);
        per-factor log-unwrap along `axis`, branch corrected to
        Gstart at the first node when given."""
        tot = np.zeros_like(zp)
        for s_, e_ in shifts:
            v = th(zp - s_, tau)
            tot = tot + e_ * (np.log(np.abs(v))
                              + 1j * np.unwrap(np.angle(v),
                                               axis=axis))
        G = np.exp(tot) / g00
        if Gstart is not None:
            corr = Gstart / np.take(G, 0, axis=axis)
            G = G * np.expand_dims(corr, axis)
        dh = kap_dh0(zp, a, b, c, d, tau) / dh00
        om = np.stack([(-G * dh + dh / G) / 2.0,
                       1j * (G * dh + dh / G) / 2.0, dh], axis=-1)
        return om, G

    def seg_int(zp, Gstart=None):
        """integral over subdivided paths zp (..., K+1) -> (..., 3),
        plus G at the last node."""
        om, G = om_batch(zp, Gstart, axis=-1)
        dz = np.diff(zp, axis=-1)
        I = np.sum(0.5 * (om[..., 1:, :] + om[..., :-1, :])
                   * dz[..., None], axis=-2)
        return I, np.take(G, -1, axis=-1)

    nu2, nv2 = Z.shape
    F = np.zeros((nu2, nv2, 3), dtype=complex)
    Gn = np.zeros((nu2, nv2), dtype=complex)
    jm = nv2 // 2
    im = int(np.argmin(np.abs(np.log(xr))))
    uu = np.linspace(0.0, 1.0, subdiv + 1)
    # anchor: z = 0 (G = 1, f = 0 by the f0(0) normalization) -> mid
    # node, graded straight path
    apath = (np.linspace(0.0, 1.0, 400) ** 1.5) * Z[im, jm]
    I0, G0v = seg_int(apath[None, :], Gstart=np.array([1.0 + 0j]))
    F[im, jm] = I0[0]
    Gn[im, jm] = G0v[0]
    # radial sweep along the mid column (theta = yr[jm])
    for i in list(range(im + 1, nu2)) + list(range(im - 1, -1, -1)):
        i0 = i - 1 if i > im else i + 1
        wseg = W[i0, jm] + (W[i, jm] - W[i0, jm]) * uu
        zseg = rect(trf(wseg))
        I_, G_ = seg_int(zseg[None, :], Gstart=Gn[i0, jm][None])
        F[i, jm] = F[i0, jm] + I_[0]
        Gn[i, jm] = G_[0]
    # angular sweeps, batched over all radii
    for j in list(range(jm + 1, nv2)) + list(range(jm - 1, -1, -1)):
        j0 = j - 1 if j > jm else j + 1
        wseg = (W[:, j0])[:, None] + ((W[:, j] - W[:, j0]))[:, None]             * uu[None, :]
        zseg = rect(trf(wseg))
        I_, G_ = seg_int(zseg, Gstart=Gn[:, j0])
        F[:, j] = F[:, j0] + I_
        Gn[:, j] = G_
    Fr = np.real(F)
    # ff1: slide the d-symmetry line (through f((tau+1)/2 - d), at
    # azimuth pi/k) onto the origin, exactly as the notebook does
    P = (tau + 1.0) / 2.0 - d
    u4 = np.linspace(0.0, 1.0, 3001)
    w4 = u4 * u4 * (3.0 - 2.0 * u4)
    ws = w4 * (1.0 - 2e-9) + 1e-9
    mid_ = 0.25 + tau / 4.0
    zp = np.concatenate([mid_ * w4, mid_ + (P - mid_) * ws[1:]])
    om, _G = om_batch(zp[None, :], Gstart=np.array([1.0 + 0j]))
    om = om[0]
    dz = np.diff(zp)
    f1P = np.real(np.sum(0.5 * (om[1:] + om[:-1]) * dz[:, None],
                         axis=0))
    ff1 = np.array([f1P[0] - f1P[1] / math.tan(math.pi / k),
                    0.0, 0.0])
    Fr = Fr - ff1[None, None, :]
    return W, Z, Fr, dict(a=a, b=b, c=c, d=d, t=t, consts=C,
                          f1P=f1P, ff1=ff1, y0=t / 2.0)


def kap_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """Kapouleas surface (two coaxial catenoids desingularized by a
    ring of k Scherk handles at each of the two intersection
    circles).  `order` walks KAP_MEMBERS (Weber's six exported
    members first); `radius` follows the catenoid ends further out.
    The 4k-copy orbit of the fundamental patch (mirror across z = 0,
    mirror across y = 0, k rotations -- the notebook's mp2/mp3/mp4)
    is welded by exact grid-index pairs along its measured seam
    families: the quarter's left and right edges lie in the z = 0
    plane (partner g Mz), the bottom-edge segments in the y = 0
    plane (partner g My), and the top edge plus the inter-end
    segment in the vertical plane at azimuth pi/k (partner g Q,
    Q = Rz(2 pi/k) My).  The four catenoid end rims (two middle at
    1/2 +- c, two outer at 1/2 - a and 1/2 + a + tau) stay open."""
    del spec, theta
    ki, mi = KAP_MEMBERS[int(np.clip(order - 1, 0,
                                     len(KAP_MEMBERS) - 1))]
    fac = float(np.clip(radius / 1.2, 0.4, 2.5))
    r0 = 0.01 * (1.0 / fac) ** 2
    r1 = 1000.0 * fac ** 2
    nx = max(3, int(nu / 12))
    ny = max(12, int(nv * 0.45))
    W, Z, F, meta = kap_sheet(ki, mi, nx=nx, ny=ny, r0=r0, r1=r1)
    y0 = meta['y0']
    nu2, nv2 = F.shape[:2]
    # classify the theta = 0 boundary nodes by their chart image
    Zb = Z[:, 0]
    selL = np.abs(np.real(Zb)) < 1e-4
    selR = np.abs(np.real(Zb) - 0.5) < 1e-4
    selT = np.imag(Zb) > y0 - 1e-4
    # the flags are INDEPENDENT: a corner node (e.g. the chart origin,
    # on the bottom edge AND the left edge) belongs to both of its
    # symmetry elements and must join both seams -- classifying it
    # into one leaves a one-edge slit at every corner of every copy
    selB = (np.imag(Zb) < 1e-4) & (~selT)
    # snap each arc onto its symmetry element
    F[selL | selR, 0, 2] = 0.0                   # z = 0 plane
    F[selB, 0, 1] = 0.0                          # y = 0 plane
    nq = np.array([-math.sin(math.pi / ki), math.cos(math.pi / ki),
                   0.0])
    P_ = F[selT, 0]
    F[selT, 0] = P_ - (P_ @ nq)[:, None] * nq[None, :]
    P_ = F[:, -1]                                # theta = pi edge
    F[:, -1] = P_ - (P_ @ nq)[:, None] * nq[None, :]
    # orbit: I, Mz, My, MzMy per rotation (the notebook's order)
    Mz = np.diag([1.0, 1.0, -1.0])
    My = np.diag([1.0, -1.0, 1.0])
    cq, sq = math.cos(TAU / ki), math.sin(TAU / ki)
    Q = np.array([[cq, -sq, 0.0], [sq, cq, 0.0],
                  [0.0, 0.0, 1.0]]) @ My
    NV = nu2 * nv2
    quads0 = _kus_grid_quads(nu2, nv2)
    P0 = F.reshape(-1, 3)
    Vs, Fs, recs = [], [], []
    off = 0
    for kk in range(ki):
        thr = TAU * kk / ki
        c_, s_ = math.cos(thr), math.sin(thr)
        Rz = np.array([[c_, -s_, 0.0], [s_, c_, 0.0],
                       [0.0, 0.0, 1.0]])
        for E, par in ((np.eye(3), 0), (Mz, 1), (My, 1),
                       (My @ Mz, 0)):
            M = Rz @ E
            Vs.append(P0 @ M.T)
            Fs.extend(tuple(i_ + off for i_ in
                            (q[::-1] if par else q))
                      for q in quads0)
            recs.append((M, off))
            off += NV
    V = np.concatenate(Vs, axis=0)

    def findpart(M):
        for Mj, o_ in recs:
            if float(np.abs(Mj - M).max()) < 1e-9:
                return o_
        return -1

    iL = np.where(selL)[0]
    iR = np.where(selR)[0]
    iB = np.where(selB)[0]
    iT = np.where(selT)[0]
    pairs = []
    for M, o_ in recs:
        oMz = findpart(M @ Mz)
        for i_ in np.concatenate([iL, iR]):
            pairs.append((o_ + i_ * nv2, oMz + i_ * nv2))
        oMy = findpart(M @ My)
        for i_ in iB:
            pairs.append((o_ + i_ * nv2, oMy + i_ * nv2))
        oQ = findpart(M @ Q)
        for i_ in iT:
            pairs.append((o_ + i_ * nv2, oQ + i_ * nv2))
        for i_ in range(nu2):                    # theta = pi edge
            pairs.append((o_ + i_ * nv2 + nv2 - 1,
                          oQ + i_ * nv2 + nv2 - 1))
    V, Fs, _first = _g1h_weld_pairs(V, Fs, pairs)
    V = _center_fit(V, scale, V)
    return V, Fs, None



# --------------------------------------------------------------------------
# Extension plumbing (no Blender UI of its own; the toolkit owns it)
# --------------------------------------------------------------------------

ADD_MENU = True


def register():
    pass


def unregister():
    pass


def _selftest():
    # engine self-tests (numpy only)
    ok = True
    # period integral: winding integral of 1/z
    pi1 = period_integral(lambda z: 1.0 / z, 0.0, 1.0, 0.5)
    e = abs(pi1 - TAU * 1j)
    print(f"period_integral 1/z: {pi1:.6f} err={e:.2e} "
          f"{'OK' if e < 1e-10 else 'FAIL'}")
    ok &= e < 1e-10
    # solve_scalar: root of cos on [1, 2]
    r = solve_scalar(math.cos, 1.0, 2.0)
    e = abs(r - math.pi / 2)
    print(f"solve_scalar cos: {r:.12f} err={e:.2e} "
          f"{'OK' if e < 1e-10 else 'FAIL'}")
    ok &= e < 1e-10
    # disk engine vs the closed-form Enneper antiderivative
    spec = {'g': lambda z, p: z, 'dh': lambda z, p: z,
            'domain': ('disk', 0.0, 1.2), 'offset_rays': False,
            'clip': False}
    x, y, z, wu, wv, _ = we_surface(spec, 220, 48, 1, 1.2)
    u = np.linspace(1e-3, 1.2, 220)
    v = np.arange(48) * (TAU / 48)
    R, TH = np.meshgrid(u, v, indexing='ij')
    zz = R * np.exp(1j * TH)
    xe = np.real(0.5 * (zz - zz ** 3 / 3.0))
    ye = np.real(0.5j * (zz + zz ** 3 / 3.0))
    ze = np.real(0.5 * zz ** 2)
    err = max(np.max(np.abs(x - xe)), np.max(np.abs(y - ye)),
              np.max(np.abs(z - ze)))
    print(f"disk engine vs Enneper closed form: err={err:.2e} "
          f"{'OK' if err < 2e-3 else 'FAIL'}")
    ok &= err < 2e-3
    # rect engine on the same data (path independence check)
    specr = {'g': lambda z, p: z, 'dh': lambda z, p: z,
             'domain': ('rect', 0.1, 1.0, 0.1, 0.9)}
    x, y, z, _, _, _ = we_surface(specr, 200, 200, 1, 1.0)
    u = np.linspace(0.1, 1.0, 200)
    v = np.linspace(0.1, 0.9, 200)
    U, V = np.meshgrid(u, v, indexing='ij')
    zz = U + 1j * V
    F0 = 0.1 + 0.1j
    xe = np.real(0.5 * (zz - zz ** 3 / 3.0) - 0.5 * (F0 - F0 ** 3 / 3.0))
    err = np.max(np.abs(x - xe))
    print(f"rect engine vs closed form: err={err:.2e} "
          f"{'OK' if err < 2e-3 else 'FAIL'}")
    ok &= err < 2e-3
    # Bjorling: cycloid seed must reproduce the Catalan surface
    bj = {'curve': lambda w, p: (w - np.sin(w), 1.0 - np.cos(w),
                                 0.0 * w),
          'normal': lambda w, p: (np.cos(w / 2), -np.sin(w / 2),
                                  0.0 * w),
          't_range': (-math.pi, 3 * math.pi), 'v_half': 1.2}
    x, y, z, _, _, _ = bjorling_surface(bj, 160, 161, 1, 1.2)
    u = np.linspace(-math.pi, 3 * math.pi, 160)
    v = np.linspace(-1.2, 1.2, 161)
    U, V = np.meshgrid(u, v, indexing='ij')
    xe = U - np.sin(U) * np.cosh(V)
    ye = 1 - np.cos(U) * np.cosh(V)
    ze = 4 * np.sin(U / 2) * np.sinh(V / 2)
    err = max(np.max(np.abs(x - xe)), np.max(np.abs(y - ye)),
              np.max(np.abs(z - ze)))
    print(f"Bjorling cycloid == Catalan: err={err:.2e} "
          f"{'OK' if err < 2e-3 else 'FAIL'}")
    ok &= err < 2e-3
    # Bjorling seed row: v = 0 must reproduce the curve exactly
    jm = 161 // 2
    err = max(np.max(np.abs(x[:, jm] - (u - np.sin(u)))),
              np.max(np.abs(y[:, jm] - (1 - np.cos(u)))),
              np.max(np.abs(z[:, jm])))
    print(f"Bjorling seed row: err={err:.2e} "
          f"{'OK' if err < 1e-9 else 'FAIL'}")
    ok &= err < 1e-9

    # ---- exact P / Gyroid / D associate family (Bonnet angle) --------------
    # (1) the three coordinate 1-forms are a NULL (conformal minimal) triple
    zt = np.array([0.3 + 0.7j, -0.5 + 1.2j, 2.0 + 0.4j, -2.5 + 0.9j,
                   0.1 + 3.0j, 1.4 + 2.2j])
    o1, o2, o3 = _pgd_om(zt)
    nullerr = float(np.max(np.abs(o1 ** 2 + o2 ** 2 + o3 ** 2)))
    print(f"P/G/D null identity |om1^2+om2^2+om3^2|={nullerr:.2e} "
          f"{'OK' if nullerr < 1e-12 else 'FAIL'}")
    ok &= nullerr < 1e-12
    # (2) every boundary edge is a proper 180-degree 2-fold axis, all theta,
    #     and its snapped linear part is one of the 24 cube rotations
    for thd in (0.0, 38.0148, 90.0):
        th = math.radians(thd)
        worst_ang, worst_res, worst_snap = 0.0, 0.0, 0.0
        for i in range(6):
            C = _pgd_edge_curve(th, i)
            M, b, res = _pgd_fit_twofold(C)
            ang = math.degrees(math.acos(
                max(-1.0, min(1.0, (np.trace(M) - 1.0) / 2.0))))
            span = float(np.linalg.norm(C.max(0) - C.min(0)))
            worst_ang = max(worst_ang, abs(ang - 180.0))
            worst_res = max(worst_res, res / max(span, 1e-9))
            worst_snap = max(worst_snap,
                             np.abs(_pgd_snap_rot(M) - M).max())
        good = worst_ang < 3.0 and worst_res < 0.02 and worst_snap < 0.35
        ok &= good
        print(f"P/G/D 2-fold edges theta={thd:7.3f}deg: "
              f"max|angle-180|={worst_ang:.2f} max_res={worst_res:.4f} "
              f"snap<{worst_snap:.2f} {'OK' if good else 'FAIL'}")
    # (3) the tiling closes into a cubic lattice, independent of theta
    for thd in (0.0, 38.0148, 90.0):
        gens = pgd_gluings(math.radians(thd))
        lat = pgd_lattice(gens)
        norms = np.sort(np.linalg.norm(lat, axis=1)) if len(lat) else \
            np.array([0.0])
        a = float(norms.min()) if len(lat) else 0.0
        good = len(lat) >= 3 and 2.5 < a < 6.0
        ok &= good
        print(f"P/G/D lattice theta={thd:7.3f}deg: a_min={a:.3f} "
              f"n={len(lat)} {'OK' if good else 'FAIL'}")
    # (4) the build fits the 2 m cube and is finite across the whole morph;
    #     P (0 deg) and D (90 deg) assemble a recognizable *filled cell* --
    #     one connected component, near-manifold (few non-manifold edges
    #     relative to its size) -- while the gyroid / intermediate angles keep
    #     the exact fundamental piece (strictly edge-manifold, no edge used
    #     more than twice).

    def _edge_stats(T):
        ec = {}
        for (x0, x1, x2) in T:
            for a2, b2 in ((x0, x1), (x1, x2), (x2, x0)):
                e = (a2, b2) if a2 < b2 else (b2, a2)
                ec[e] = ec.get(e, 0) + 1
        return sum(1 for c in ec.values() if c > 2)

    def _ncomp(nV, T):
        parent = list(range(nV))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for (x, y, z) in T:
            for u, v in ((x, y), (y, z)):
                ra, rb = find(u), find(v)
                if ra != rb:
                    parent[ra] = rb
        used = set(i for f in T for i in f)
        return len({find(i) for i in used})

    for thd in (0.0, 19.0, 38.0148, 64.0, 90.0):
        V, T = pgd_build(1, 44, 1.0, math.radians(thd))
        finite = bool(np.all(np.isfinite(V)))
        lo, hi = V.min(0), V.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        nonman = _edge_stats(T)
        tiled = thd in (0.0, 90.0)
        if tiled:                                  # filled P / D cell
            comps = _ncomp(len(V), T)
            good = (finite and len(T) > 4000 and cen < 1e-6
                    and abs(ext - 2.0) < 1e-6 and comps == 1
                    and nonman < 0.12 * len(T))
            tag = f"cell comps={comps}"
        else:                                      # exact fundamental piece
            good = (finite and len(T) > 500 and cen < 1e-6
                    and abs(ext - 2.0) < 1e-6 and nonman == 0)
            tag = "piece"
        ok &= good
        print(f"P/G/D {tag} theta={thd:7.3f}deg: {len(V):6d}v {len(T):6d}t "
              f"fit[|c|={cen:.1e} ext={ext:.4f}] nonman={nonman} "
              f"{'OK' if good else 'FAIL'}")
    # (5) the Bonnet morph is continuous (a small angle step gives a bounded,
    #     non-degenerate change) and the three iconic members are distinct
    base = _pgd_grid_pts(0.0)
    d1 = float(np.nanmax(np.abs(_pgd_grid_pts(0.03) - base)))
    d2 = float(np.nanmax(np.abs(_pgd_grid_pts(0.06) - base)))
    cont = (np.isfinite(d1) and np.isfinite(d2)
            and 1e-4 < d1 < 1.0 and d2 < 3.0 * d1 + 1e-6)
    gpat = _pgd_grid_pts(math.radians(38.0148))
    dpat = _pgd_grid_pts(math.radians(90.0))
    distinct = (float(np.nanmax(np.abs(gpat - base))) > 0.05
                and float(np.nanmax(np.abs(dpat - base))) > 0.05)
    good = cont and distinct
    ok &= good
    print(f"P/G/D Bonnet morph: d(.03)={d1:.2e} d(.06)={d2:.2e} "
          f"distinct={distinct} {'OK' if good else 'FAIL'}")

    # ---- higher-genus Chen-Gackstatter: watertight D2d assembly ------------
    # gates: exact chi = 1 - 2g, edge-manifold, ONE boundary loop (the
    # trimmed Enneper end), one component, globally consistent winding
    for gg, nu_t, arcn_t, R_t in ((2, 100, 44, 5.5), (4, 100, 40, 6.0),
                                  (5, 100, 36, 8.0)):
        Vg, Fg, _uv = cg_higher_assemble(gg, nu_t, arcn_t, R_t)
        ecc, dcc = {}, {}
        for f in Fg:
            m = len(f)
            for kk in range(m):
                a2, b2 = f[kk], f[(kk + 1) % m]
                e2 = (a2, b2) if a2 < b2 else (b2, a2)
                ecc[e2] = ecc.get(e2, 0) + 1
                dcc[(a2, b2)] = dcc.get((a2, b2), 0) + 1
        chi = len(Vg) - len(ecc) + len(Fg)
        nonman = sum(1 for c in ecc.values() if c > 2)
        bed = [e2 for e2, c in ecc.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a2, b2 in bed:
            ra, rb = bfind(a2), bfind(b2)
            if ra != rb:
                par[ra] = rb
        loops = len({bfind(a2) for a2, b2 in bed})
        orient = all(c == 1 for c in dcc.values())
        good = (chi == 1 - 2 * gg and nonman == 0 and loops == 1
                and orient and bool(np.all(np.isfinite(Vg))))
        ok &= good
        print(f"CG higher genus {gg}: {len(Vg):6d}v {len(Fg):6d}f "
              f"chi={chi} (want {1 - 2 * gg}) nonman={nonman} "
              f"loops={loops} orient={orient} {'OK' if good else 'FAIL'}")

    # ---- genus-one helicoid: theta engine, harvested closure, topology ----
    # (1) theta_11 identities: odd, quasi-periodic under z+1 and z+tau
    zt = np.array([0.13 + 0.21j, -0.4 + 0.7j, 0.9 - 0.3j, 2.3 + 1.9j])
    tt = _G1H_TAU
    t0 = genus1helicoid_theta11(zt, tt)
    # relative errors (theta grows like e^{pi Im(z)^2 / Im tau}, so the
    # identities are compared against the values' own magnitude)
    e1 = float(np.max(np.abs(genus1helicoid_theta11(zt + 1.0, tt) + t0)
                      / np.abs(t0)))
    e2 = float(np.max(np.abs(
        genus1helicoid_theta11(zt + tt, tt)
        + np.exp(-1j * np.pi * tt - 2j * np.pi * zt) * t0)
        / np.abs(genus1helicoid_theta11(zt + tt, tt))))
    e3 = float(np.max(np.abs(genus1helicoid_theta11(-zt, tt) + t0)
                      / np.abs(t0)))
    good = max(e1, e2, e3) < 1e-10
    ok &= good
    print(f"g1-helicoid theta11: |z+1|={e1:.2e} |z+tau|={e2:.2e} "
          f"odd={e3:.2e} {'OK' if good else 'FAIL'}")
    # (2) the domain map reproduces the notebook's solved constants:
    # tst(a0) = 1 - b0 and the r0 root residual
    ra = abs(complex(_g1h_tst(np.array(_G1H_A0 + 0j))) - (1.0 - _G1H_B))
    v = complex(_g1h_tst(np.array(-_G1H_R0 + 1e-12j)))
    rb = abs((0.5 * (1.0 + tt) * (1.0 + v) - tt).imag)
    good = ra < 1e-6 and rb < 1e-6
    ok &= good
    print(f"g1-helicoid domain: |tst(a0)-(1-b0)|={ra:.2e} "
          f"r0-residual={rb:.2e} {'OK' if good else 'FAIL'}")
    # (3) period closure with the harvested constants: the z -> z+1
    # cycle is the exact vertical translation (0, 0, 2); z -> z+tau
    # closes (both to the FindRoot precision of the constants)
    z0 = 0.311 + 0.077j
    PA = _g1h_path_int(z0, z0 + 1.0, 8001)
    PB = _g1h_path_int(z0, z0 + tt, 8001)
    eh = max(abs(PA[0].real), abs(PA[1].real),
             abs(PB[0].real), abs(PB[1].real))
    ev = abs(PA[2].real - 2.0)
    eb = abs(PB[2].real)
    good = eh < 1e-6 and ev < 1e-6 and eb < 1e-5
    ok &= good
    print(f"g1-helicoid periods: horiz={eh:.2e} |A_v-2|={ev:.2e} "
          f"B_v={eb:.2e} {'OK' if good else 'FAIL'}")
    # (4) the sheet contains the z axis and the horizontal rulings
    xs_t, ys_t, Xs = genus1helicoid_sheet(-2.5, 111, 45)
    max_ax = (xs_t > _G1H_XA + 1e-6) & (xs_t < _G1H_XB - 1e-6)
    axdev = float(np.max(np.abs(Xs[max_ax, 0, :2])))
    rul = Xs[xs_t < _G1H_XA - 1e-6, 0]
    rdev = max(float(np.max(np.abs(rul[:, 0]))),
               float(np.max(np.abs(rul[:, 2] + 1.0))))
    # weld-correspondence residuals (the assembly merges these pairs)
    Rzv = np.array([-1.0, -1.0, 1.0])
    ax_r = float(np.max(np.linalg.norm(
        Xs[max_ax, 0] - Rzv * Xs[max_ax, 0], axis=1)))
    mA = xs_t <= _G1H_XA + 1e-6
    rl_r = float(np.max(np.linalg.norm(
        Xs[mA, 0] - Rzv * Xs[::-1][mA, -1], axis=1)))
    mC = xs_t <= _G1H_XC + 1e-6
    cr_r = float(np.max(np.linalg.norm(
        Xs[mC, -1] - (Rzv * Xs[::-1][mC, 0] + np.array([0, 0, 2.0])),
        axis=1)))
    wres = max(ax_r, rl_r, cr_r)
    good = axdev < 5e-3 and rdev < 5e-3 and wres < 2e-3
    ok &= good
    print(f"g1-helicoid lines: axis-dev={axdev:.2e} "
          f"ruling-dev={rdev:.2e} weld-residual={wres:.2e} "
          f"{'OK' if good else 'FAIL'}")
    # (5) topology: a stack of S cells is one connected, manifold,
    # consistently oriented surface of genus exactly S (one handle per
    # translational period): chi = 2 - 2S - boundary_loops
    for S in (1, 2):
        Vg, Fg, uvg = genus1helicoid_assemble(S, -2.5, 111, 45)
        ecc, dcc = {}, {}
        for f in Fg:
            m = len(f)
            for kk in range(m):
                a2, b2 = f[kk], f[(kk + 1) % m]
                e2k = (a2, b2) if a2 < b2 else (b2, a2)
                ecc[e2k] = ecc.get(e2k, 0) + 1
                dcc[(a2, b2)] = dcc.get((a2, b2), 0) + 1
        chi = len(Vg) - len(ecc) + len(Fg)
        nonman = sum(1 for c in ecc.values() if c > 2)
        orient = all(c == 1 for c in dcc.values())
        bed = [e2k for e2k, c in ecc.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a2, b2 in bed:
            ra2, rb2 = bfind(a2), bfind(b2)
            if ra2 != rb2:
                par[ra2] = rb2
        loops = len({bfind(a2) for a2, b2 in bed})
        parc = list(range(len(Vg)))

        def cfind(a2):
            while parc[a2] != a2:
                parc[a2] = parc[parc[a2]]
                a2 = parc[a2]
            return a2

        for f in Fg:
            for i2 in range(1, len(f)):
                ra2, rb2 = cfind(f[0]), cfind(f[i2])
                if ra2 != rb2:
                    parc[ra2] = rb2
        ncomp = len({cfind(i2) for f in Fg for i2 in f})
        genus = (2 - chi - loops) / 2.0
        uv_ok = bool(np.all(np.isfinite(uvg))) and len(uvg) == len(Vg)
        good = (genus == S and nonman == 0 and orient and ncomp == 1
                and uv_ok and bool(np.all(np.isfinite(Vg))))
        ok &= good
        print(f"g1-helicoid S={S}: {len(Vg):6d}v {len(Fg):6d}f chi={chi} "
              f"loops={loops} genus={genus:.0f} (want {S}) "
              f"nonman={nonman} orient={orient} ncomp={ncomp} "
              f"{'OK' if good else 'FAIL'}")

    # ---- symmetrized Chen-Gackstatter towers (k-fold D_kd assembly) --------
    # (a) period data: the gn rho quadrature must reproduce the closed
    #     Gamma form; g2n / g3k per-segment rho estimates must agree
    #     (that consistency IS the harvested-period validation)
    for kk in (2, 4, 6):
        _SYMMCG_RHO_CACHE.pop(('gn', kk), None)
        a0t, ft, rho_cf = _symmcg_data('gn', kk)
        est0, _ests = symmcg_rho_solve('gn', kk)
        e2 = abs(est0 - rho_cf)
        good = e2 < 1e-6
        ok &= good
        print(f"symmcg gn rho k={kk}: quad={est0:.9f} "
              f"closed={rho_cf:.9f} err={e2:.1e} "
              f"{'OK' if good else 'FAIL'}")
    for tower, kk in (('g2n', 3), ('g2n', 4), ('g3k', 2), ('g3k', 3)):
        a0t, ft, rho_t = _symmcg_data(tower, kk)
        est0, ests = symmcg_rho_solve(tower, kk)
        spread = max(ests) - min(ests)
        good = spread < 5e-5 and (rho_t is None or abs(est0 - rho_t) < 1e-4)
        ok &= good
        tgt = f" harvested={rho_t:.6f}" if rho_t is not None else ""
        print(f"symmcg {tower} rho k={kk}: per-seg "
              f"{['%.6f' % x for x in ests]}{tgt} spread={spread:.1e} "
              f"{'OK' if good else 'FAIL'}")
    # (b) watertight assembly: exact chi = 1 - 2g (one trimmed end),
    #     edge-manifold, ONE boundary loop, oriented, and small pre-snap
    #     period residuals at the branch images
    for tower, kk, gen in (('gn', 2, 1), ('gn', 3, 2), ('gn', 4, 3),
                           ('gn', 6, 5), ('g2n', 3, 4), ('g3k', 2, 3)):
        Vg, Fg, _uv, diag = symmcg_assemble(tower, kk, 90, 24)
        ecc, dcc = {}, {}
        for f in Fg:
            m = len(f)
            for t in range(m):
                a2, b2 = f[t], f[(t + 1) % m]
                e3 = (a2, b2) if a2 < b2 else (b2, a2)
                ecc[e3] = ecc.get(e3, 0) + 1
                dcc[(a2, b2)] = dcc.get((a2, b2), 0) + 1
        chi = len(Vg) - len(ecc) + len(Fg)
        nonman = sum(1 for c in ecc.values() if c > 2)
        bed = [e3 for e3, c in ecc.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a2, b2 in bed:
            ra, rb = bfind(a2), bfind(b2)
            if ra != rb:
                par[ra] = rb
        loops = len({bfind(a2) for a2, b2 in bed})
        orient = all(c == 1 for c in dcc.values())
        pres = max(v for kk2, v in diag['res'].items()
                   if kk2.startswith('branch'))
        good = (chi == 1 - 2 * gen and nonman == 0 and loops == 1
                and orient and pres < 5e-4
                and bool(np.all(np.isfinite(Vg))))
        ok &= good
        print(f"symmcg {tower} k={kk} (genus {gen}): {len(Vg):6d}v "
              f"{len(Fg):6d}f chi={chi} (want {1 - 2 * gen}) "
              f"nonman={nonman} loops={loops} orient={orient} "
              f"period_res={pres:.1e} {'OK' if good else 'FAIL'}")
    # ---- singly periodic Callahan-Hoffman-Meeks (k = 2) --------------------
    # (1) the pulled-back 1-forms are a null triple (exact minimality)
    rng = np.random.default_rng(7)
    wt = (rng.uniform(-6.0, _CHMP_UMAX, 300)
          + 1j * rng.uniform(1e-3, math.pi - 1e-3, 300))
    o1, o2, o3 = chm_periodic_forms(wt)
    nullerr = float(np.max(np.abs(o1 ** 2 + o2 ** 2 + o3 ** 2)))
    print(f"CHM periodic null |sum om^2|={nullerr:.2e} "
          f"{'OK' if nullerr < 1e-12 else 'FAIL'}")
    ok &= nullerr < 1e-12
    # (2) patch boundary structure BEFORE snapping: the t = 0 / t = pi
    # rows are planar geodesics (constant y / x), the arc column is a
    # horizontal planar geodesic, the pre-xa t = pi row is a straight
    # (1,1,0) line at constant height, and the half-spacing d matches
    # the real-axis elliptic integral (= 2d).  Residual tolerances sit
    # at the harvested constants' own closure precision (~1e-3).
    ug = _chmp_ugrid(90, -5.0)
    tg = _chmp_tgrid(64)
    Xp = chm_periodic_patch(ug, tg)
    iLt = int(np.searchsorted(ug, _CHMP_XA))
    ry = float(np.ptp(Xp[:, 0, 1]))
    rx = float(np.ptp(Xp[iLt:, -1, 0]))
    rz = float(np.ptp(Xp[-1, :, 2]))
    rl = float(np.ptp(Xp[:iLt + 1, -1, 0] - Xp[:iLt + 1, -1, 1]))
    rlz = float(np.ptp(Xp[:iLt + 1, -1, 2]))
    dmeas = float(np.median(Xp[-1, :, 2]) - np.median(Xp[:iLt + 1, -1, 2]))
    th_e = np.linspace(0.0, 0.5 * math.pi, 4001)
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    ell = _trapz(1.0 / np.sqrt(_CHMP_A ** 2 - np.sin(th_e) ** 2), th_e)
    good = (ry < 1e-3 and rx < 1e-3 and rz < 1e-6 and rl < 1e-3
            and rlz < 1e-3 and abs(2.0 * dmeas - ell) < 1e-3)
    ok &= good
    print(f"CHM periodic patch: dy={ry:.1e} dx={rx:.1e} dz_arc={rz:.1e} "
          f"line[d(x-y)={rl:.1e} dz={rlz:.1e}] d={dmeas:.6f} "
          f"(elliptic/2={0.5 * ell:.6f}) {'OK' if good else 'FAIL'}")
    # (3) assembled periods: edge-manifold, oriented, one component,
    # chi = -6 S (the quotient by the 4d translation is genus 3 = 2k+1
    # with two planar ends -- measured here, and matching the CHM theorem
    # for k = 1), 2 S end rims + 2 outer horizontal cut loops
    for S in (1, 2):
        Xs, iLs, dS = chm_periodic_snap(Xp, ug)
        Vg, Fg, uvg = chm_periodic_assemble(ug, tg, Xs, iLs, dS, S)
        ecc, dcc = {}, {}
        for f in Fg:
            m = len(f)
            for kk in range(m):
                a2, b2 = f[kk], f[(kk + 1) % m]
                e2 = (a2, b2) if a2 < b2 else (b2, a2)
                ecc[e2] = ecc.get(e2, 0) + 1
                dcc[(a2, b2)] = dcc.get((a2, b2), 0) + 1
        chi = len(Vg) - len(ecc) + len(Fg)
        nonman = sum(1 for c in ecc.values() if c > 2)
        bed = [e2 for e2, c in ecc.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a2, b2 in bed:
            ra, rb = bfind(a2), bfind(b2)
            if ra != rb:
                par[ra] = rb
        loops = len({bfind(a2) for a2, b2 in bed})
        orient = all(c == 1 for c in dcc.values())
        good = (chi == -6 * S and nonman == 0 and loops == 2 * S + 2
                and orient and bool(np.all(np.isfinite(Vg)))
                and bool(np.all(np.isfinite(uvg))))
        ok &= good
        print(f"CHM periodic S={S}: {len(Vg):6d}v {len(Fg):6d}f "
              f"chi={chi} (want {-6 * S}) nonman={nonman} "
              f"loops={loops} (want {2 * S + 2}) orient={orient} "
              f"{'OK' if good else 'FAIL'}")
    # ---- Catenoid-Enneper / Costa-Wohlgemuth / Wohlgemuth (cwce block) -----
    # (1) the stored period constants close the period problem: every
    #     notebook condition vanishes under independent quadrature
    for gg in (2, 3, 4):
        r = float(np.max(np.abs(cwce_ce_residual(gg))))
        good = r < 1e-6
        ok &= good
        print(f"catenoid-enneper g={gg} period residual: {r:.1e} "
              f"{'OK' if good else 'FAIL'}")
    for fam, kk in (('cw', 2), ('cw', 3), ('cw', 4), ('cw', 5),
                    ('cw', 7), ('w2', 2)):
        r = float(np.max(np.abs(cwce_cw_residual(fam, kk))))
        good = r < 1e-8
        ok &= good
        print(f"{'costa-wohlgemuth' if fam == 'cw' else 'wohlgemuth-2'} "
              f"k={kk} period residual: {r:.1e} "
              f"{'OK' if good else 'FAIL'}")
    # (2) null identity: phi1 phi2 = dh^2 makes sum omega_i^2 vanish
    #     identically -- checked on the actual meshing integrands
    rt = np.linspace(0.31, 4.7, 40)
    omn = cwce_ce_omega(3, rt, 0.71, np.full(40, -1), np.zeros(40),
                        list(_CWCE_CE[3][1]))
    nerr = float(np.max(np.abs(np.sum(omn ** 2, axis=-1))))
    good = nerr < 1e-12
    ok &= good
    print(f"catenoid-enneper null |sum om^2|={nerr:.1e} "
          f"{'OK' if good else 'FAIL'}")
    datn = _cwce_cw_data('w2', 2)
    omn = cwce_cw_omega_row('w2', 2, datn, np.linspace(-0.9, 2.9, 40),
                            1.1, np.full(40, -1), np.zeros(40),
                            ['x0', 'xinf', 'xc', 'x1'])
    nerr = float(np.max(np.abs(np.sum(omn ** 2, axis=-1))))
    good = nerr < 1e-12
    ok &= good
    print(f"wohlgemuth null |sum om^2|={nerr:.1e} "
          f"{'OK' if good else 'FAIL'}")
    # (3) watertight assemblies: exact chi = 2 - 2g - (open end rims),
    #     edge-manifold, oriented, one component, small pre-snap period
    #     residuals (the snap only removes quadrature noise)
    for fam, kk, gen, nrim in (('ce', 2, 2, 2), ('ce', 3, 3, 2),
                               ('ce', 4, 4, 2), ('cw', 2, 2, 4),
                               ('cw', 3, 4, 4), ('cw', 5, 8, 4),
                               ('w2', 2, 3, 4)):
        if fam == 'ce':
            Vg, Fg, _uv, diag = cwce_ce_assemble(kk, 100, 29)
        else:
            Vg, Fg, _uv, diag = cwce_cw_assemble(fam, kk, 110, 21)
        ecc, dcc = {}, {}
        for f in Fg:
            m = len(f)
            for tq in range(m):
                a2, b2 = f[tq], f[(tq + 1) % m]
                e2 = (a2, b2) if a2 < b2 else (b2, a2)
                ecc[e2] = ecc.get(e2, 0) + 1
                dcc[(a2, b2)] = dcc.get((a2, b2), 0) + 1
        chi = len(Vg) - len(ecc) + len(Fg)
        nonman = sum(1 for cq in ecc.values() if cq > 2)
        bed = [e2 for e2, cq in ecc.items() if cq == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a2, b2 in bed:
            ra, rb = bfind(a2), bfind(b2)
            if ra != rb:
                par[ra] = rb
        loops = len({bfind(a2) for a2, b2 in bed})
        orient = all(cq == 1 for cq in dcc.values())
        parc = list(range(len(Vg)))

        def cfind(a2):
            while parc[a2] != a2:
                parc[a2] = parc[parc[a2]]
                a2 = parc[a2]
            return a2

        for f in Fg:
            for i2 in range(1, len(f)):
                ra, rb = cfind(f[0]), cfind(f[i2])
                if ra != rb:
                    parc[ra] = rb
        ncomp = len({cfind(f[0]) for f in Fg})
        pres = max(diag['res'].values())
        want = 2 - 2 * gen - nrim
        good = (chi == want and nonman == 0 and loops == nrim
                and orient and ncomp == 1 and pres < 5e-3
                and bool(np.all(np.isfinite(Vg))))
        ok &= good
        nm = {'ce': 'catenoid-enneper', 'cw': 'costa-wohlgemuth',
              'w2': 'wohlgemuth-2'}[fam]
        print(f"{nm} {'g' if fam == 'ce' else 'k'}={kk} "
              f"(genus {gen}): {len(Vg):6d}v {len(Fg):6d}f chi={chi} "
              f"(want {want}) nonman={nonman} loops={loops} "
              f"(want {nrim}) orient={orient} ncomp={ncomp} "
              f"period_res={pres:.1e} {'OK' if good else 'FAIL'}")
    # ---- doubly periodic hyperelliptic tiler (KMR + Wei) -------------------
    # (1) Wei's period problem: the harvested FindRoot constants close the
    #     notebook's own residual Im int_a^b (G + 1/G) dz/z, and the
    #     residual moves off-solution (the gate is not vacuous)
    for bb, aa in ((0.3, 0.23640853975826828), (0.5, 0.07082564803712697),
                   (0.7, 0.0030808434547059727)):
        rres = abs(dperiodic_wei_residual(bb, aa))
        roff = abs(dperiodic_wei_residual(bb, 0.8 * aa))
        good = rres < 1e-5 and roff > 50 * max(rres, 1e-9)
        ok &= good
        print(f"dperiodic Wei period b={bb}: |res|={rres:.2e} "
              f"off-solution={roff:.2e} {'OK' if good else 'FAIL'}")
    # (2) per member: wall/period closure residuals small; a genuinely 2-D
    #     lattice; quotient topology chi = 2 - 2g - 4 with exactly 4
    #     Scherk end rims, edge-manifold, oriented, one component; and a
    #     2 x 2 lattice tiling stays ONE connected manifold block
    for key, p, gen in (('kmr2', {'a': 0.4, 'rmin': 0.05}, 1),
                        ('kmr2', {'a': 0.55, 'rmin': 0.05}, 1),
                        ('wei', {'b': 0.3, 'a': 0.23640853975826828,
                                 'rmin': 0.05}, 2),
                        ('wei', {'b': 0.5, 'a': 0.07082564803712697,
                                 'rmin': 0.05}, 2),
                        ('kmr3', {'xmin': 0.05}, 1)):
        P = dperiodic_patch(key, p, 48, 48)
        wres = max(v for kk2, v in P['res'].items() if kk2 != 'x3=u')
        lat = float(np.linalg.norm(np.cross(P['T1'], P['T2'])))
        chi, loops, nonman, orient, ncomp = dperiodic_quotient(P)
        V2, Q2, UV2 = dperiodic_assemble(P, (2, 2))
        par2 = list(range(len(V2)))

        def find2(a2):
            while par2[a2] != a2:
                par2[a2] = par2[par2[a2]]
                a2 = par2[a2]
            return a2

        ec2 = {}
        for f in Q2:
            m2 = len(f)
            for t2 in range(m2):
                a2, b2 = f[t2], f[(t2 + 1) % m2]
                e2 = (a2, b2) if a2 < b2 else (b2, a2)
                ec2[e2] = ec2.get(e2, 0) + 1
                ra2, rb2 = find2(a2), find2(b2)
                if ra2 != rb2:
                    par2[ra2] = rb2
        ncomp2 = len({find2(i2) for f in Q2 for i2 in f})
        nonman2 = sum(1 for c in ec2.values() if c > 2)
        chi_want = 2 - 2 * gen - 4
        good = (wres < 5e-3 and lat > 1.0
                and chi == chi_want and loops == 4 and nonman == 0
                and orient and ncomp == 1 and ncomp2 == 1 and nonman2 == 0
                and bool(np.all(np.isfinite(V2)))
                and bool(np.all(np.isfinite(UV2))))
        ok &= good
        print(f"dperiodic {key:5s} (genus {gen}): wall_res={wres:.1e} "
              f"|T1xT2|={lat:.2f} chi={chi} (want {chi_want}) ends={loops} "
              f"nonman={nonman} orient={orient} 2x2[comp={ncomp2} "
              f"nonman={nonman2}] {'OK' if good else 'FAIL'}")
    # (3) KMR-3 external cross-check: the measured lattice reproduces the
    #     notebook's printed translation constants (disx, disy, disz)
    P3 = dperiodic_patch('kmr3', {'xmin': 0.01}, 60, 48)
    e1 = abs(abs(P3['T1'][0]) - 2.0 * 2.6417540147391194)
    e2 = abs(abs(P3['T2'][1]) - 2.0 * 1.2779360827691162)
    e3 = abs(abs(P3['T2'][2]) - 2.0 * 2.4000944407384024)
    good = max(e1, e2, e3) < 5e-3
    ok &= good
    print(f"dperiodic KMR-3 lattice vs notebook: |dT|=({e1:.1e},"
          f"{e2:.1e},{e3:.1e}) {'OK' if good else 'FAIL'}")
    # ---- doubly periodic long tail (KS / RTW / Wei tower / Connor) --------
    # (1) RTW M1+ period solve: the re-solved b closes the notebook's own
    #     SolvePeriod condition, and the residual moves off-solution
    rres = abs(dptail_rtwmp_residual(DPTAIL_RTWMP_B))
    roff = abs(dptail_rtwmp_residual(0.9 * DPTAIL_RTWMP_B))
    good = rres < 1e-9 and roff > 1e3 * max(rres, 1e-12)
    ok &= good
    print(f"dptail RTW M1+ period b={DPTAIL_RTWMP_B:.6f}: |res|={rres:.2e} "
          f"off-solution={roff:.2e} {'OK' if good else 'FAIL'}")
    # (2) every shipped member: wall/period residuals small, x3 = u exact,
    #     a genuinely 2-D lattice, quotient topology chi = 2 - 2g - 4 with
    #     exactly 4 Scherk end rims, edge-manifold, oriented, connected;
    #     and a 2 x 2 lattice tiling stays ONE connected manifold block
    dpt_ship = ('ksg2', 'ksg3', 'ksg3x',
                'rtwmp', 'rtwm1pm_0', 'rtwm1pm_1',
                'wei13_0', 'wei13_1', 'wei13_2', 'wei13_3',
                'wei14_0', 'wei14_1', 'wei23', 'wei16',
                'conn_asym', 'conn78', 'conn80', 'conn82',
                'conn84', 'conn85')
    for key in dpt_ship:
        mem = dptail_member(key)
        gen = mem['genus']
        P = dptail_patch(key, {}, 56, 40)
        wres = max(v for k2, v in P['res'].items() if k2 != 'x3=u')
        lat = float(np.linalg.norm(np.cross(P['T1'], P['T2'])))
        chi, loops, nonman, orient, ncomp = dptail_quotient(P)
        V2, Q2, UV2 = dptail_assemble(P, (2, 2))
        par2 = list(range(len(V2)))

        def find2(a2):
            while par2[a2] != a2:
                par2[a2] = par2[par2[a2]]
                a2 = par2[a2]
            return a2

        ec2 = {}
        for f in Q2:
            m2 = len(f)
            for t2 in range(m2):
                a2, b2 = f[t2], f[(t2 + 1) % m2]
                e2 = (a2, b2) if a2 < b2 else (b2, a2)
                ec2[e2] = ec2.get(e2, 0) + 1
                ra2, rb2 = find2(a2), find2(b2)
                if ra2 != rb2:
                    par2[ra2] = rb2
        ncomp2 = len({find2(i2) for f in Q2 for i2 in f})
        nonman2 = sum(1 for c in ec2.values() if c > 2)
        chi_want = 2 - 2 * gen - 4
        good = (wres < 5e-3 and P['res']['x3=u'] < 1e-9 and lat > 1.0
                and chi == chi_want and loops == 4 and nonman == 0
                and orient and ncomp == 1 and ncomp2 == 1
                and nonman2 == 0 and bool(np.all(np.isfinite(V2)))
                and bool(np.all(np.isfinite(UV2))))
        ok &= good
        print(f"dptail {key:10s} ({mem['style']:5s} genus {gen}): "
              f"wall_res={wres:.1e} |T1xT2|={lat:.2f} chi={chi} "
              f"(want {chi_want}) ends={loops} nonman={nonman} "
              f"orient={orient} 2x2[comp={ncomp2} nonman={nonman2}] "
              f"{'OK' if good else 'FAIL'}")
    print("dptail deferred (see BACKLOG.md): ksg1 ksg4 dp_catenoid "
          "lubeck_batista chm_g3")
    # ---- singly periodic long tail (sptail block) --------------------------
    # (1) null identity phi1 phi2 = dh^2 <=> sum om_i^2 = 0, on the actual
    #     meshing integrands of every family
    rng2 = np.random.default_rng(11)
    zt2 = rng2.uniform(0.05, 3.0, 200) + 1j * rng2.uniform(1e-3, 3.0, 200)
    for nm2, omf in (('six-scherk', sptail_six_om(30.0)[2]),
                     ('alt-fence', sptail_fencealt_om(2.0)),
                     ('fence-cat', sptail_fencecat_om(0.2)),
                     ('helicoidal-KS', sptail_hks_om(4, 0.1, 0.8))):
        o1, o2, o3 = omf(zt2)
        nerr = float(np.max(np.abs(o1 * o1 + o2 * o2 + o3 * o3)))
        good = nerr < 1e-12
        ok &= good
        print(f"sptail null {nm2}: |sum om^2|={nerr:.1e} "
              f"{'OK' if good else 'FAIL'}")
    # (2) six-ended Scherk: boundary symmetry residuals pre-snap, the
    #     translation vs the om2 residue closed form, and measured
    #     chi(S+1) - chi(S) = -4 = 2 - 2*0 - 6 (genus 0, 6 ends/period)
    for phid in (30.0, 60.0):
        V1, F1, _u, dg = sptail_six_build(phid, storeys=1)
        pres = max(dg['line_dy'], dg['line_dz'], dg['mx_dx'],
                   dg['my_dy'], dg['pi_dy'])
        V2, F2, _u2, _d2 = sptail_six_build(phid, storeys=2)
        c1 = sptail_topology(V1, F1)
        c2 = sptail_topology(V2, F2)
        good = (pres < 1e-3 and dg['dy_vs_residue'] < 1e-6
                and dg['pi_vs_2dy'] < 1e-6
                and c2[0] - c1[0] == -4
                and c1[1] == 0 and c2[1] == 0 and c1[2] and c2[2]
                and c1[4] == 1 and c2[4] == 1
                and bool(np.all(np.isfinite(V2))))
        ok &= good
        print(f"sptail six-scherk phi={phid:.0f}: presnap={pres:.1e} "
              f"dy_res={dg['dy_vs_residue']:.1e} chi {c1[0]}->{c2[0]} "
              f"(dchi -4) nonman={c2[1]} orient={c2[2]} "
              f"{'OK' if good else 'FAIL'}")
    # (3) alternating fence: rho = 1/sqrt(a) closes the horizontal period
    #     (the two vertical mirrors coincide); quotient chi = -2, 2 ends
    #     (genus 1); stacks stay manifold/oriented/connected
    V1, F1, _u, dg = sptail_fencealt_build(2.0, storeys=1)
    V2, F2, _u2, _d2 = sptail_fencealt_build(2.0, storeys=2)
    c1 = sptail_topology(V1, F1)
    c2 = sptail_topology(V2, F2)
    Vq, Fq = sptail_quotient(V1, F1, dg['T'], dg['span'])
    cq = sptail_topology(Vq, Fq)
    good = (dg['period_x'] < 1e-9 and c2[0] - c1[0] == -2
            and cq[0] == -2 and cq[3] == 2 and cq[1] == 0 and cq[2]
            and c2[1] == 0 and c2[2] and c2[4] == 1)
    ok &= good
    print(f"sptail alt-fence a=2: period_x={dg['period_x']:.1e} "
          f"chi {c1[0]}->{c2[0]} (dchi -2) quotient chi={cq[0]} "
          f"ends={cq[3]} (want -2, 2) {'OK' if good else 'FAIL'}")
    # (4) fence of catenoids: |G| = 1 identically on the arc (the
    #     Lopez-Ros closure rho = sqrt(a)); quotient chi = -2 with 2
    #     ends (genus 1 per period, measured)
    aa = 0.2
    tha = np.linspace(0.01, math.pi - 0.01, 400)
    za = math.sqrt(aa) * np.exp(1j * tha)
    Ga = math.sqrt(aa) * np.sqrt(za - 1.0) / (np.sqrt(za)
                                              * np.sqrt(za - aa))
    gerr = float(np.max(np.abs(np.abs(Ga) - 1.0)))
    V1, F1, _u, dg = sptail_fencecat_build(aa, storeys=1)
    V2, F2, _u2, _d2 = sptail_fencecat_build(aa, storeys=2)
    c1 = sptail_topology(V1, F1)
    c2 = sptail_topology(V2, F2)
    Vq, Fq = sptail_quotient(V1, F1, dg['T'], dg['span'])
    cq = sptail_topology(Vq, Fq)
    pres = max(dg['y_ptp'], dg['xA_ptp'], dg['xB_ptp'],
               dg['arc_z_ptp'])
    good = (gerr < 1e-12 and pres < 5e-3 and c2[0] - c1[0] == -2
            and cq[0] == -2 and cq[3] == 2 and cq[1] == 0 and cq[2]
            and c2[1] == 0 and c2[2] and c2[4] == 1)
    ok &= good
    print(f"sptail fence-cat a={aa}: max||G|-1|={gerr:.1e} "
          f"presnap={pres:.1e} chi {c1[0]}->{c2[0]} quotient "
          f"chi={cq[0]} ends={cq[3]} (want -2, 2) "
          f"{'OK' if good else 'FAIL'}")
    # (5) helicoidal Karcher-Scherk: the geometric period solve closes
    #     the k-fold axis (and is not vacuous -- off-solution R misses),
    #     the measured screw rise reproduces pi R^2/(1+R^4), and
    #     chi(S+1) - chi(S) = 2 - 2k (genus 0, 2k ends per period)
    for kk, av in ((2, 0.2), (4, 0.1)):
        Rs = sptail_hks_solveR(kk, av)
        res_on = float(np.linalg.norm(
            sptail_hks_axis_residual(kk, av, Rs)))
        res_off = float(np.linalg.norm(
            sptail_hks_axis_residual(kk, av, 0.97 * Rs)))
        V1, F1, _u, dg = sptail_hks_build(kk, av, storeys=1, R=Rs)
        V2, F2, _u2, _d2 = sptail_hks_build(kk, av, storeys=2, R=Rs)
        c1 = sptail_topology(V1, F1)
        c2 = sptail_topology(V2, F2)
        good = (res_on < 1e-6 and res_off > 100 * max(res_on, 1e-9)
                and dg['trans_vs_theory'] < 1e-8
                and c2[0] - c1[0] == 2 - 2 * kk
                and c2[1] == 0 and c2[2] and c2[4] == 1
                and bool(np.all(np.isfinite(V2))))
        ok &= good
        print(f"sptail helicoidal-KS k={kk} a={av}: R={Rs:.8f} "
              f"axis_res={res_on:.1e} (off {res_off:.1e}) "
              f"rise_vs_theory={dg['trans_vs_theory']:.1e} "
              f"chi {c1[0]}->{c2[0]} (dchi {2 - 2 * kk}) "
              f"{'OK' if good else 'FAIL'}")
    # (6) translation-invariant Enneper, 3 annular ends: the closed-form
    #     boundary constants (mirrors at y = 0 / -Y1, line at -Y1/2),
    #     quotient chi = -2 with 4 end rims (MEASURED genus 0 -- see the
    #     block note on the harvest's genus-1 annotation)
    V1, F1, _u, dg = sptail_e3a_build(storeys=1)
    V2, F2, _u2, _d2 = sptail_e3a_build(storeys=2)
    c1 = sptail_topology(V1, F1)
    c2 = sptail_topology(V2, F2)
    Vq, Fq = sptail_quotient(V1, F1, dg['T'], dg['span'])
    cq = sptail_topology(Vq, Fq)
    pres = max(dg['y01_ptp'], dg['y1R_vs_const'],
               dg['line_y_vs_const'], dg['line_z_ptp'])
    good = (pres < 1e-9 and c2[0] - c1[0] == -2
            and cq[0] == -2 and cq[3] == 4 and cq[1] == 0 and cq[2]
            and c2[1] == 0 and c2[2] and c2[4] == 1)
    ok &= good
    print(f"sptail enneper-3ann: const_res={pres:.1e} chi "
          f"{c1[0]}->{c2[0]} (dchi -2) quotient chi={cq[0]} "
          f"ends={cq[3]} (want -2, 4) {'OK' if good else 'FAIL'}")
    # (7) periodic Enneper: one turn of the built grid is the EXACT
    #     translation (0, -pi, 0); the wrapped quotient is a cylinder
    #     (chi = 0, 2 rims -- genus 0, 2 ends)
    V2t, q2g, _u2, dg2 = sptail_penneper_build(storeys=2)
    nturn = dg2['ncol_per_turn']
    nvv = len(V2t) // 64
    G1 = V2t.reshape(64, nvv, 3)
    dT = G1[:, nturn:, :] - G1[:, :-nturn, :]
    terr = float(np.max(np.abs(dT - dg2['T'][None, None, :])))
    V1, q1g, _u, dg = sptail_penneper_build(storeys=1)
    span1 = float(np.linalg.norm(V1.max(0) - V1.min(0)))
    Vq, Fq = sptail_quotient(V1, q1g, dg['T'], span1, kmax=1)
    cq = sptail_topology(Vq, Fq)
    good = (terr < 1e-12 and cq[0] == 0 and cq[3] == 2
            and cq[1] == 0 and cq[2] and cq[4] == 1)
    ok &= good
    print(f"sptail periodic-enneper: |T - (0,-pi,0)|={terr:.1e} "
          f"quotient chi={cq[0]} ends={cq[3]} (want 0, 2) "
          f"{'OK' if good else 'FAIL'}")

    # ---- SP SCHERK FAMILY engine gates (sscherk block) -----------------
    # (1) null identity sum om_i^2 = 0 on every family's integrand
    zt3 = rng2.uniform(1.2, 3.0, 160) + 1j * rng2.uniform(0.1, 2.5, 160)
    for nm3, omf in (('six1', sscherk_six1_om(3)),
                     ('costa', sscherk_costa_om(1)),
                     ('eight', sscherk_eight_om(2)),
                     ('dasilva', sscherk_das_om(1))):
        o1, o2, o3 = omf(zt3)
        nerr = float(np.max(np.abs(o1 * o1 + o2 * o2 + o3 * o3)))
        good = nerr < 1e-10
        ok &= good
        print(f"sscherk null {nm3}: |sum om^2|={nerr:.1e} "
              f"{'OK' if good else 'FAIL'}")
    # (2) the notebook-solved constants close their period problems:
    #     independent quadrature of every table member (integral
    #     conditions ~ quadrature-limited for the nearly degenerate
    #     branch clusters; the closed-form residue conditions are exact)
    w_int = w_cf = 0.0
    for mi3 in range(len(SSCHERK_SIX1_MEMBERS)):
        c1r, c2r, e3r, e4r, _r = sscherk_six1_check(mi3)
        w_int = max(w_int, abs(c1r), abs(c2r))
        w_cf = max(w_cf, abs(e3r), abs(e4r))
    good = w_int < 5e-4 and w_cf < 1e-9
    ok &= good
    print(f"sscherk six1 periods (all {len(SSCHERK_SIX1_MEMBERS)} "
          f"members): worst integral={w_int:.1e} closed-form="
          f"{w_cf:.1e} {'OK' if good else 'FAIL'}")
    w_int = 0.0
    for mi3 in range(len(SSCHERK_COSTA_MEMBERS)):
        t1r, t2r, t3r, t4r, _r = sscherk_costa_check(mi3)
        w_int = max(w_int, abs(t1r), abs(t2r), abs(t3r), abs(t4r))
    good = w_int < 5e-4
    ok &= good
    print(f"sscherk costa periods (all {len(SSCHERK_COSTA_MEMBERS)} "
          f"members): worst residual={w_int:.1e} "
          f"{'OK' if good else 'FAIL'}")
    w_int = w_res = 0.0
    for mi3 in range(len(SSCHERK_EIGHT_MEMBERS)):
        t1r, t2r, r0r, rbr, _t = sscherk_eight_check(mi3)
        w_int = max(w_int, abs(t1r), abs(t2r))
        w_res = max(w_res, abs(r0r + rbr))
    good = w_int < 1e-5 and w_res < 1e-8
    ok &= good
    print(f"sscherk eight periods (all {len(SSCHERK_EIGHT_MEMBERS)} "
          f"members): worst integral={w_int:.1e} residue match="
          f"{w_res:.1e} {'OK' if good else 'FAIL'}")
    w_int = 0.0
    for mi3 in range(len(SSCHERK_DAS_MEMBERS)):
        t1r, t2r, t3r, _r = sscherk_das_check(mi3)
        w_int = max(w_int, abs(t1r), abs(t2r), abs(t3r))
    good = w_int < 1e-6
    ok &= good
    print(f"sscherk dasilva periods (all {len(SSCHERK_DAS_MEMBERS)} "
          f"members): worst residual={w_int:.1e} "
          f"{'OK' if good else 'FAIL'}")
    # (3) topology per family (one representative member; the full
    #     member sweep was verified during development): measured
    #     chi(S+1) - chi(S) and the translation-wrapped quotient must
    #     equal chi = 2 - 2 genus - #ends with the right end count;
    #     stacks manifold, oriented, one component; the mesh
    #     translation reproduces the analytic residue/transy value
    for nm3, build3, mi3, wchi, wend, ptol, ttol in (
            ('six1', sscherk_six1_build, 3, -6, 6, 8e-3, 1.5e-2),
            ('costa', sscherk_costa_build, 1, -6, 6, 1e-4, 1e-5),
            ('eight', sscherk_eight_build, 2, -10, 8, 5e-4, 1e-7),
            ('dasilva', sscherk_das_build, 1, -10, 8, 1e-5, 1e-7)):
        V1, F1, u1, dg = build3(mi3, storeys=1)
        V2, F2, _u2, _d2 = build3(mi3, storeys=2)
        c1 = sptail_topology(V1, F1)
        c2 = sptail_topology(V2, F2)
        Vq, Fq = sptail_quotient(V1, F1, dg['T'], dg['span'] * 1e-6)
        cq = sptail_topology(Vq, Fq)
        tres = dg.get('T_vs_residue', dg.get('T_vs_transy', 0.0))
        good = (dg['presnap'] < ptol and tres < ttol
                and c2[0] - c1[0] == wchi and cq[0] == wchi
                and cq[3] == wend and cq[1] == 0 and cq[2]
                and c2[1] == 0 and c2[2] and c2[4] == 1
                and bool(np.all(np.isfinite(V2)))
                and bool(np.all(np.isfinite(u1))))
        ok &= good
        print(f"sscherk {nm3} member {mi3}: presnap={dg['presnap']:.1e} "
              f"T_res={tres:.1e} chi {c1[0]}->{c2[0]} (dchi {wchi}) "
              f"quotient chi={cq[0]} ends={cq[3]} (want {wchi}, {wend}) "
              f"nonman={c2[1]} orient={c2[2]} "
              f"{'OK' if good else 'FAIL'}")

    # ---- SYMM/NONORIENT TAIL engine gates ------------------------------
    # (1) validate the one-sidedness checker itself on two synthetic
    #     meshes: a Mobius band (must read one-sided, chi = 0, 1 loop)
    #     and a cylinder (two-sided, chi = 0, 2 loops)
    nuq, nvq = 24, 8

    def _strip(mobius):
        V, F = [], []
        for i in range(nuq):
            t = TAU * i / nuq
            for j in range(nvq):
                s = j / (nvq - 1) - 0.5
                if mobius:
                    V.append([(1 + s * math.cos(t / 2)) * math.cos(t),
                              (1 + s * math.cos(t / 2)) * math.sin(t),
                              s * math.sin(t / 2)])
                else:
                    V.append([math.cos(t), math.sin(t), s])
        for i in range(nuq):
            i2 = (i + 1) % nuq
            for j in range(nvq - 1):
                if mobius and i2 == 0:
                    # glue with the half-twist flip j -> nvq-1-j
                    F.append((i * nvq + j, (nvq - 1 - j),
                              (nvq - 2 - j), i * nvq + j + 1))
                else:
                    F.append((i * nvq + j, i2 * nvq + j,
                              i2 * nvq + j + 1, i * nvq + j + 1))
        return np.array(V), F
    Vm, Fm = _strip(True)
    chi_m, nm_m, lo_m, os_m = symtail_edge_stats(Vm, Fm)
    Vc, Fc = _strip(False)
    chi_c, nm_c, lo_c, os_c = symtail_edge_stats(Vc, Fc)
    good = (os_m and chi_m == 0 and lo_m == 1 and nm_m == 0
            and (not os_c) and chi_c == 0 and lo_c == 2 and nm_c == 0)
    ok &= good
    print(f"symtail checker: mobius(chi={chi_m} loops={lo_m} "
          f"one_sided={os_m}) cylinder(chi={chi_c} loops={lo_c} "
          f"one_sided={os_c}) {'OK' if good else 'FAIL'}")
    # (2) antiprismatic solver reproduces the harvested nn=5 member and
    #     closes the full period set for the slider range
    ah, rh = symtail_antiprism_constants(5, 0.2)
    e = max(abs(ah - 0.2748767946679093),
            abs(rh - 0.0015692436842339352))
    good = e < 1e-8
    ok &= good
    print(f"symtail antiprism nn=5 b=0.2: a={ah:.12f} rho={rh:.6e} "
          f"err={e:.2e} {'OK' if good else 'FAIL'}")
    # (3) Lopez Klein bottle numerics: the rim y-profile is monotone
    #     (the deck pairing is well defined) and the symmetric v-grid
    #     is strictly increasing
    vg = symtail_lopez_vgrid(61)
    vvr, yr = _SYMTAIL_LOPEZ_RIM
    good = bool(np.all(np.diff(vg) > 0)) and bool(np.all(np.diff(yr) > 0))
    ok &= good
    print(f"symtail lopez rim: y monotone={bool(np.all(np.diff(yr) > 0))} "
          f"vgrid monotone={bool(np.all(np.diff(vg) > 0))} "
          f"y range [{yr[0] - 0.5 * (yr[0] + yr[-1]):.4f},"
          f"{yr[-1] - 0.5 * (yr[0] + yr[-1]):.4f}] "
          f"{'OK' if good else 'FAIL'}")
    # ---- STINV translation-invariant towers + CHM variants -----------
    # (1) every harvested/polished constant set closes its notebook
    #     period conditions (machine-checked residuals)
    resid = stinv_period_residuals()
    worst = {}
    for k, v in resid.items():
        fam = k.split()[0]
        worst[fam] = max(worst.get(fam, 0.0), v)
    good = (worst['g1'] < 1e-7 and worst['g3'] < 1e-7
            and worst['costa'] < 1e-6 and worst['chm12'] < 1e-4)
    ok &= good
    print(f"stinv period residuals: g1={worst['g1']:.1e} "
          f"g3={worst['g3']:.1e} costa={worst['costa']:.1e} "
          f"chm12={worst['chm12']:.1e} {'OK' if good else 'FAIL'}")
    sres = stinv_screw_residuals(n=1500)
    wsc = max(sres.values())
    good = wsc < 1e-3
    ok &= good
    print(f"stinv screw-CHM residuals (6 tau values): worst={wsc:.1e} "
          f"{'OK' if good else 'FAIL'}")
    # (2) the Weierstrass null identity om1^2+om2^2+om3^2 = 0 on random
    #     interior points, for every member's integrand
    rng = np.random.default_rng(11)
    zs = (rng.uniform(0.1, 0.8, 40)
          * np.exp(1j * rng.uniform(0.1, 3.0, 40)))
    nullw = 0.0
    for om in (stinv_cat_om(*stinv_g1_facs(0.5, _STINV_G1[0.5])),
               stinv_cat_om(*stinv_g3_facs(0.1, *_STINV_G3[0.1])),
               stinv_costa_om(-10.0, *_STINV_COSTA[-10.0])):
        o1, o2, o3 = om(zs)
        nullw = max(nullw, float(np.max(np.abs(
            o1 * o1 + o2 * o2 + o3 * o3))))
    ws = rng.uniform(-2.0, 2.0, 40) + 1j * rng.uniform(0.2, 2.9, 40)
    o1, o2, o3 = stinv_chm12_forms(ws)
    nullw = max(nullw, float(np.max(np.abs(o1 * o1 + o2 * o2
                                           + o3 * o3))))
    omscrew, _, _ = stinv_screw_om(0.8)
    zt = rng.uniform(0.25, 0.35, 30) + 1j * rng.uniform(0.1, 0.3, 30)
    o1, o2, o3 = omscrew(zt)
    nullw = max(nullw, float(np.max(np.abs(o1 * o1 + o2 * o2
                                           + o3 * o3))))
    good = nullw < 1e-10
    ok &= good
    print(f"stinv null identity: worst |om.om| = {nullw:.1e} "
          f"{'OK' if good else 'FAIL'}")
    # (3) per member: stacks at S = 1, 2 (manifold, oriented, one
    #     component; chi(2)-chi(1) = the quotient chi), the translation-
    #     wrapped quotient (true per-period chi and end count), and the
    #     measured mirror/period-closure diagnostics
    for name, fn, kw, want_dchi, want_ends, gaps in (
            ('g1 (a=0.5)', stinv_g1_build, dict(a=0.5), -4, 2,
             ('xmirror_gap',)),
            ('g3 (a=0.1)', stinv_g3_build, dict(a=0.1), -6, 2,
             ('ymirror_gap',)),
            ('costa (a=-10)', stinv_costa_build, dict(a=-10.0), -4, 4,
             ('xmirror_gap', 'ymirror_gap'))):
        chis = []
        good = True
        keep = None
        for S in (1, 2):
            V, F, uv, d = fn(nu=44, nt=30, storeys=S, **kw)
            chi, nonman, orient, loops, ncomp = sptail_topology(V, F)
            chis.append(chi)
            good &= (nonman == 0 and orient and ncomp == 1)
            if S == 1:
                keep = (V, F, d)
        V1, F1, d = keep
        Vq, Fq = sptail_quotient(V1, F1, d['T'], d['span'] * 1e-3)
        cq = sptail_topology(Vq, Fq)
        gap = max(d[g] for g in gaps)
        good &= (chis[1] - chis[0] == want_dchi
                 and cq[0] == want_dchi and cq[1] == 0 and cq[2]
                 and cq[3] == want_ends and cq[4] == 1
                 and gap < 5e-4 * d['span'])
        ok &= good
        print(f"stinv {name:14s}: dchi={chis[1] - chis[0]} "
              f"(want {want_dchi}) wrapped chi={cq[0]} "
              f"ends={cq[3]} (want {want_ends}) gap={gap:.1e} "
              f"{'OK' if good else 'FAIL'}")
    # CHM-(1,2): combinatorial weld; quotient via the cyclic wrap flag
    chis = []
    good = True
    for S in (1, 2):
        V, F, uv, d = stinv_chm12_build(nu=56, nt=34, storeys=S)
        chi, nonman, orient, loops, ncomp = sptail_topology(V, F)
        chis.append(chi)
        good &= (nonman == 0 and orient and ncomp == 1)
    Vq, Fq, _, dq = stinv_chm12_build(nu=56, nt=34, storeys=1,
                                      wrap=True)
    cq = sptail_topology(Vq, Fq)
    good &= (chis[1] - chis[0] == -8 and cq[0] == -8 and cq[1] == 0
             and cq[2] and cq[3] == 2 and cq[4] == 1)
    ok &= good
    print(f"stinv chm12         : dchi={chis[1] - chis[0]} (want -8) "
          f"wrapped chi={cq[0]} ends={cq[3]} (want 2) "
          f"line/mirror ptps < {max(dq[k] for k in dq if k.endswith('_ptp')):.1e} "
          f"{'OK' if good else 'FAIL'}")
    # screw-motion CHM: dchi = -6 per screw period; the exactified
    # group's word identifications must land on the fitted lines
    chis = []
    good = True
    for S in (1, 2):
        V, F, uv, d = stinv_screw_build(timag=0.8, nu=48, nt=32,
                                        storeys=S)
        chi, nonman, orient, loops, ncomp = sptail_topology(V, F)
        chis.append(chi)
        good &= (nonman == 0 and orient and ncomp == 1)
    good &= (chis[1] - chis[0] == -6 and d['wordB'][2] < 1e-4
             and d['wordD'][2] < 1e-4 and d['screw_fit'] < 1e-4
             and abs(d['rise'] + 0.8) < 1e-9)
    ok &= good
    print(f"stinv screw (t=0.8i): dchi={chis[1] - chis[0]} (want -6) "
          f"wordB axis-dist={d['wordB'][2]:.1e} "
          f"screw fit={d['screw_fit']:.1e} rise={d['rise']:+.3f} "
          f"{'OK' if good else 'FAIL'}")
    # ---- SFK TAIL engine gates -----------------------------------------
    # (1) 2 Enneper + 2 annular torus: null identity, the closed-form
    #     residue closure (om1/om2 residues at 0 and b cancel), the
    #     measured boundary constants, and quotient chi = -2 with 4 end
    #     rims (genus 0 MEASURED -- see the catalog note on the
    #     harvest's genus-1 annotation)
    omE, aE, rhoE = sfk_e2a2_om(0.5)
    zt3 = rng2.uniform(0.1, 2.0, 120) + 1j * rng2.uniform(1e-3, 2.0,
                                                          120)
    o1, o2, o3 = omE(zt3)
    nerr = float(np.max(np.abs(o1 * o1 + o2 * o2 + o3 * o3)))
    rsum = max(abs(period_integral(lambda z, i=i: omE(z)[i], 0.0,
                                   0.05, 0.05)
                   + period_integral(lambda z, i=i: omE(z)[i], 0.5,
                                     0.05, 0.05))
               for i in (0, 1))
    V1, F1, _u, dg = sfk_e2a2_build(storeys=1)
    V2, F2, _u2, _d2 = sfk_e2a2_build(storeys=2)
    c1 = sptail_topology(V1, F1)
    c2 = sptail_topology(V2, F2)
    Vq, Fq = sptail_quotient(V1, F1, dg['T'], dg['span'])
    cq = sptail_topology(Vq, Fq)
    pres = max(dg['yA_ptp'], dg['yB_ptp'], dg['ypi_vs_yB'],
               dg['circ_z_ptp'], dg['trans_vs_residue'])
    good = (nerr < 1e-10 and rsum < 1e-10 and pres < 1e-6
            and c2[0] - c1[0] == -2 and cq[0] == -2 and cq[3] == 4
            and cq[1] == 0 and cq[2] and c2[1] == 0 and c2[2]
            and c2[4] == 1)
    ok &= good
    print(f"sfk 2enn-2ann b=0.5: null={nerr:.1e} residue_sum={rsum:.1e} "
          f"presnap={pres:.1e} chi {c1[0]}->{c2[0]} quotient "
          f"chi={cq[0]} ends={cq[3]} (want -2, 4) "
          f"{'OK' if good else 'FAIL'}")
    # (2) 1 catenoid + 2 annular torus: the Newton period solve
    #     reproduces the notebook's parallel-end member (rho = 1) and
    #     closes both periods; the mirror spacing reproduces the
    #     residue translation; quotient chi = -3 with 3 end rims
    #     (genus 1 -- matches the harvest expected chi)
    aP = 0.25747983928707496
    bP, rhoP, resP = sfk_c1a2_constants(aP)
    nb_err = max(abs(bP - 1.592291695522628), abs(rhoP - 1.0))
    V1, F1, _u, dg = sfk_c1a2_build(a=aP, storeys=1)
    V2, F2, _u2, _d2 = sfk_c1a2_build(a=aP, storeys=2)
    c1 = sptail_topology(V1, F1)
    c2 = sptail_topology(V2, F2)
    Vq, Fq = sptail_quotient(V1, F1, dg['T'], dg['span'])
    cq = sptail_topology(Vq, Fq)
    pres = max(dg['xseg_gap'], dg['yseg_gap'], dg['mirror_vs_residue'])
    good = (resP < 1e-8 and nb_err < 1e-7 and pres < 5e-3
            and c2[0] - c1[0] == -3 and cq[0] == -3 and cq[3] == 3
            and cq[1] == 0 and cq[2] and c2[1] == 0 and c2[2]
            and c2[4] == 1)
    ok &= good
    print(f"sfk 1cat-2ann a={aP:.4f}: period_res={resP:.1e} "
          f"vs_notebook={nb_err:.1e} presnap={pres:.1e} chi "
          f"{c1[0]}->{c2[0]} quotient chi={cq[0]} ends={cq[3]} "
          f"(want -3, 3) {'OK' if good else 'FAIL'}")
    # (3) Fischer-Koch translational k = 3, 5: dh is exactly elliptic,
    #     the rise Re Int dh = -1/k closes (the notebook's period
    #     equation), the 2-fold-axis skeleton is straight/placed to
    #     ~1e-3, and chi/period = -2k (genus 1, 2k wing ends)
    for kk in (3, 5):
        omF = sfk_fkt_om(kk)
        zt4 = rng2.uniform(0.1, 0.4, 40) + 1j * rng2.uniform(
            0.05, 0.3, 40)
        tau0k = sfk_fkt_constants(kk)['tau0']
        eerr = float(max(
            np.max(np.abs(omF(zt4 + 1.0)[2] / omF(zt4)[2] - 1.0)),
            np.max(np.abs(omF(zt4 + tau0k)[2] / omF(zt4)[2] - 1.0))))
        V1, F1, _u, dg = sfk_fkt_build(k=kk, storeys=1)
        V2, F2, _u2, _d2 = sfk_fkt_build(k=kk, storeys=2)
        c1 = sptail_topology(V1, F1)
        c2 = sptail_topology(V2, F2)
        Vq, Fq = sptail_quotient(V1, F1, dg['T'], dg['span'])
        cq = sptail_topology(Vq, Fq)
        pres = max(v for k_, v in dg.items() if k_.startswith('ax'))
        good = (eerr < 1e-10 and dg['rise_res'] < 1e-10
                and pres < 1e-3 and c2[0] - c1[0] == -2 * kk
                and cq[0] == -2 * kk and cq[1] == 0 and cq[2]
                and c2[1] == 0 and c2[2] and c2[4] == 1)
        ok &= good
        print(f"sfk fischer-koch k={kk}: elliptic={eerr:.1e} "
              f"rise_res={dg['rise_res']:.1e} axes={pres:.1e} chi "
              f"{c1[0]}->{c2[0]} (dchi {-2 * kk}) quotient "
              f"chi={cq[0]} {'OK' if good else 'FAIL'}")
    # (4) Fischer-Koch-Freese k = 3: branch-tracked screw family; the
    #     rise closes, the mu-twisted axis skeleton matches its closed
    #     forms, and the SCREW-wrapped quotient has chi = -6 (genus 1,
    #     6 wing ends per screw period Rz(-2 pi mu) + (0,0,-2))
    for muF in (0.15, -0.10):
        V1, F1, _u, dg = sfk_fkf_build(k=3, mu=muF, storeys=1)
        V2, F2, _u2, _d2 = sfk_fkf_build(k=3, mu=muF, storeys=2)
        c1 = sptail_topology(V1, F1)
        c2 = sptail_topology(V2, F2)
        ang, tzs = dg['screw']
        Vq, Fq = sfk_screw_quotient(V1, F1, ang, tzs, dg['span'])
        cq = sptail_topology(Vq, Fq)
        pres = max(v for k_, v in dg.items() if k_.startswith('ax'))
        good = (dg['rise_res'] < 1e-10 and pres < 1e-3
                and c2[0] - c1[0] == -6 and cq[0] == -6
                and cq[1] == 0 and cq[2] and c2[1] == 0 and c2[2]
                and c2[4] == 1)
        ok &= good
        print(f"sfk fk-freese mu={muF:+.2f}: "
              f"rise_res={dg['rise_res']:.1e} axes={pres:.1e} chi "
              f"{c1[0]}->{c2[0]} (dchi -6) screw-quotient chi={cq[0]} "
              f"{'OK' if good else 'FAIL'}")
    print("sfk deferred (see BACKLOG.md): hackman_surfaces, even-k "
          "Fischer-Koch/Freese (self-intersecting), Freese k=4 branch")

    # ---- Karcher's 4-noid with two symmetry planes ------------------
    # Its whole claim is that the period problem is solved in closed
    # form -- no FindRoot anywhere in the source notebook -- so the
    # gate is the four end periods, measured across the family rather
    # than at the rendered member.  A wrong digit in tau(lambda) or
    # rho(lambda) breaks these while still meshing something.
    p4 = 0.0
    for lamq in (1.2, 1.5, 1.8, 2.4):
        p4 = max(p4, float(np.abs(four_noid_sym2_end_periods(lamq)).max()))
    good = p4 < 1e-9
    ok &= good
    print(f"4-noid sym2: max |Re period|, four ends x four members "
          f"= {p4:.1e} {'OK' if good else 'FAIL'}")
    # The assembly is gated two ways, at both members Weber renders.
    # TOPOLOGY: one sheet, chi = -2, exactly four boundary loops,
    # manifold, consistently oriented -- four loose discs are 4
    # components with chi = +4, a mis-welded seam breaks chi = -2 or
    # leaks boundary loops.  Necessary, not sufficient: a sphere with
    # four slits passes it too.  So, SHAPE: the rigid-motion-invariant
    # end statistics, measured off Weber's own PoVRay exports of this
    # family (the `dummy.pov` mesh in each member directory of
    # 4-noids-with-two-symmetry-planes__uq9HA8Va IS the assembled
    # surface; our mesh registers onto it at ~0.2% of span).  The
    # references below are those measurements: every pairwise angle
    # between the four outward end axes, the wide/narrow end radius
    # ratio, and the bounding-box proportions at the notebook's own
    # truncation window.  A wrong member, a wrong mu mapping, or any
    # assembly that merely has the right topology moves the angles by
    # tens of degrees.
    for (orderq, radq, ryx, rzx, awide, anarrow, across, rr) in (
            (5, 4.0, 0.93058, 0.92205, 151.82, 73.88, 101.22, 4.4929),
            (1, 1.2, 0.97224, 0.62681, 154.97, 135.30, 94.73, 3.5983)):
        V4, F4 = four_noid_sym2_mesh(None, 80, 60, orderq, radq, 1.0)
        chi4, nm4, or4, loops4, ncomp4 = sptail_topology(V4, F4)
        ext = V4.max(axis=0) - V4.min(axis=0)
        myx, mzx = float(ext[1] / ext[0]), float(ext[2] / ext[0])
        st = _fournoid_end_stats(V4, F4)
        good = (ncomp4 == 1 and chi4 == -2 and loops4 == 4
                and nm4 == 0 and or4 and len(st) == 4
                and abs(myx - ryx) < 0.02 and abs(mzx - rzx) < 0.02)
        if good:
            def _ang(u, v):
                return math.degrees(math.acos(
                    float(np.clip(np.dot(u, v), -1.0, 1.0))))
            mrr = (st[0][0] + st[1][0]) / (st[2][0] + st[3][0])
            mwide = _ang(st[0][2], st[1][2])
            mnarrow = _ang(st[2][2], st[3][2])
            mcross = max(_ang(st[i][2], st[j][2])
                         for i in (0, 1) for j in (2, 3))
            mcross_min = min(_ang(st[i][2], st[j][2])
                             for i in (0, 1) for j in (2, 3))
            good = (abs(mwide - awide) < 1.0
                    and abs(mnarrow - anarrow) < 1.0
                    and abs(mcross - across) < 1.0
                    and abs(mcross_min - across) < 1.0
                    and abs(mrr / rr - 1.0) < 0.02)
            print(f"4-noid sym2 shape order={orderq}: axis angles "
                  f"wide-wide={mwide:.2f} (ref {awide}) "
                  f"narrow-narrow={mnarrow:.2f} (ref {anarrow}) "
                  f"wide-narrow={mcross_min:.2f}..{mcross:.2f} "
                  f"(ref {across}) radius ratio={mrr:.4f} (ref {rr}) "
                  f"{'OK' if good else 'FAIL'}")
        ok &= good
        print(f"4-noid sym2 assembly order={orderq}: comps={ncomp4} "
              f"chi={chi4} loops={loops4} nonman={nm4} "
              f"oriented={or4} bbox y/x={myx:.4f} (ref {ryx}) "
              f"z/x={mzx:.4f} (ref {rzx}) {'OK' if good else 'FAIL'}")

    print("\nRESULT:", "ALL OK" if ok else "FAILURES in weierstrass")
    assert ok
