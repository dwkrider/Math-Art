"""The default build-to-finished pipeline, with fitted parameters.

The geometric ratios come from a reference surface -- disks of diameter 60
spaced 83 apart, bands 20 wide -- and the finishing constants were fitted by
searching for the smallest Chamfer distance to a reference trefoil surface,
measured after normalising both to unit RMS radius and aligning by principal
axes plus ICP.

The fitted mix is worth noting: a *light* relaxation followed by heavy
mean-curvature fairing.  Long relaxations move away from the reference, because
the particle model's equilibrium is not the same object as a minimal surface.

Optional in-loop grooming (``groom_cycles > 0``): after the fitted fairing the
triangulated surface is remeshed by the shared solver core
(:mod:`solver.groom`) -- Botsch-Kobbelt split/collapse toward a uniform edge
length, Delaunay flips, and El Topo null-space smoothing -- with the rim
pinned and every topology operation behind static safety vetoes, then briefly
re-faired.  Each groom cycle is verified: the exact self-intersection count
must not rise and the topological signature ``(chi, boundaries,
orientability)`` must be unchanged, else the cycle is reverted wholesale
(refused, not repaired).  Grooming changes vertex and face counts, so the
result is a triangle mesh; the default ``groom_cycles=0`` leaves the historic
quad pipeline bitwise untouched.

References for the grooming stage: M. Botsch, L. Kobbelt, "A Remeshing
Approach to Multiresolution Modeling", SGP 2004; T. Brochu, R. Bridson,
"Robust Topological Operations for Dynamic Explicit Surfaces", SIAM J. Sci.
Comput. 31(4), 2009 (El Topo, BSD-2-Clause); K. A. Brakke, "The Surface
Evolver", Experimental Mathematics 1(2), 1992.
"""

from __future__ import annotations

import numpy as np

from .braid import Braid
from .build import SurfaceParams, seifert_surface, state_surface
from .fair import minimal_surface, smooth_boundary
from .mesh import Mesh
from .relax import RelaxParams, relax
from .states import State, StateSurface
from .subdivide import catmull_clark

try:
    from ..solver import groom as _solver_groom
except ImportError:                      # flat (path-based) headless import
    try:
        from solver import groom as _solver_groom
    except ImportError:
        _solver_groom = None

__all__ = ["finish", "groom_mesh", "surface"]


def groom_mesh(mesh: Mesh, use_groups: bool = True, **groom_kwargs):
    """One safety-vetoed grooming pass over the triangulated surface.

    Triangulates, pins every boundary vertex, and runs
    :func:`solver.groom.groom_topo` (split / collapse / flip /
    null-space smooth, every topology op behind its static vetoes,
    the whole cycle behind an exact crossing verification).  The
    ``face_groups`` labels ride along as label classes, so a collapse
    can never rewire a disk/band seam.  After grooming the topological
    signature ``(chi, boundaries, orientability)`` is re-checked and
    the ungroomed mesh is returned unchanged if it moved -- refusal,
    not repair.

    Returns ``(mesh2, info)``; ``mesh2`` has triangle faces.
    """
    if _solver_groom is None:
        return mesh, {"skipped": "solver.groom unavailable"}
    tri = mesh.triangulated()
    V = np.asarray(tri.vertices, dtype=float).copy()
    T = np.asarray(tri.faces, dtype=np.int64)
    fixed = np.zeros(len(V), dtype=bool)
    for loop in tri.boundary_loops():
        fixed[np.asarray(loop, dtype=int)] = True
    groups = None
    names: list[str] = []
    if use_groups and tri.face_groups:
        names = sorted(set(tri.face_groups))
        gmap = {gname: gi for gi, gname in enumerate(names)}
        groups = np.array([gmap[gname] for gname in tri.face_groups])
    kwargs = dict(groom_kwargs)
    kwargs.setdefault("verify_selfx", True)
    sig0 = tri.info()
    key0 = (sig0.euler_characteristic, sig0.n_boundaries, sig0.orientable)
    V2, T2, _, groups2, info = _solver_groom.groom_topo(
        V, T, fixed=fixed, tri_groups=groups, **kwargs)
    out_groups = ([names[gi] for gi in groups2]
                  if groups2 is not None else [])
    mesh2 = Mesh(V2, [tuple(int(i) for i in t) for t in T2], out_groups)
    sig1 = mesh2.info()
    key1 = (sig1.euler_characteristic, sig1.n_boundaries, sig1.orientable)
    if key1 != key0:
        # cannot happen while the vetoes hold; refuse rather than ship
        info = dict(info)
        info["reverted_topology"] = True
        return mesh, info
    return mesh2, info


def finish(
    mesh: Mesh,
    *,
    relax_steps: int = 100,
    rim_steps: int = 0,
    levels: int = 2,
    fair_steps: int = 10,
    fair_strength: float = 2.0,
    relax_params: RelaxParams | None = None,
    groom_cycles: int = 0,
    groom_fair_steps: int = 2,
    groom_kwargs: dict | None = None,
) -> Mesh:
    """Relax, refine and fair a freshly built surface.

    With ``groom_cycles > 0`` the faired surface is additionally
    groomed (see :func:`groom_mesh`) with a ``groom_fair_steps``-long
    fairing touch-up after each cycle; the output then has triangle
    faces.  The default keeps the historic pipeline byte-identical.
    """
    if relax_steps:
        mesh = relax(mesh, relax_params or RelaxParams(), iterations=relax_steps)
    if rim_steps:
        mesh = smooth_boundary(mesh, iterations=rim_steps)
    if levels:
        mesh = catmull_clark(mesh, levels)
    if fair_steps:
        mesh = minimal_surface(mesh, strength=fair_strength, iterations=fair_steps)
    for _ in range(int(groom_cycles)):
        mesh, _info = groom_mesh(mesh, **(groom_kwargs or {}))
        if groom_fair_steps:
            mesh = minimal_surface(mesh, strength=fair_strength,
                                   iterations=groom_fair_steps)
    return mesh


def surface(
    braid: Braid | str,
    state: State | StateSurface | None = None,
    params: SurfaceParams | None = None,
    **kwargs,
) -> Mesh:
    """Build and finish in one call.

    >>> surface("AAA").info().genus
    1
    """
    return finish(state_surface(braid, state, params), **kwargs)


def _selftest():
    ok = True
    mesh = seifert_surface("AAA")
    base = finish(mesh, relax_steps=20, levels=1, fair_steps=3)
    info0 = base.info()
    groomed, ginfo = groom_mesh(base)
    info1 = groomed.info()
    tri = groomed.triangulated()
    T = np.asarray(tri.faces, dtype=np.int64)
    x = (_solver_groom.crossing_count(tri.vertices, T)
         if _solver_groom is not None else -1)
    changed = ginfo.get("splits", 0) + ginfo.get("collapses", 0) > 0
    good = (
        info1.euler_characteristic == info0.euler_characteristic
        and info1.n_boundaries == info0.n_boundaries
        and info1.orientable == info0.orientable
        and info1.genus == info0.genus == 1
        and x == 0
        and changed
        and ginfo.get("cycles_reverted", 0) == 0
        and all(len(f) == 3 for f in groomed.faces)
    )
    ok &= good
    print(
        f"pipeline: groomed AAA keeps (chi={info1.euler_characteristic}, "
        f"b={info1.n_boundaries}, genus={info1.genus}), selfx={x}, "
        f"{ginfo.get('splits', 0)} splits / {ginfo.get('collapses', 0)} "
        f"collapses {'OK' if good else 'FAIL'}"
    )

    # default path must not groom: finish() with groom_cycles=0 keeps
    # the historic quad faces
    good = any(len(f) == 4 for f in base.faces)
    ok &= good
    print(f"pipeline: default finish keeps quads "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("seifert.pipeline self-test failed")
