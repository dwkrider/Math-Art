"""polyhedra -- Platonic seeds, face recovery and convex hulls.

The Blender-free engine behind the Math Art polyhedron generators.
Python and numpy only, so the package imports and self-tests headlessly;
the registered operators stay in their flat generator modules.  Layout
follows `math_art/seifert/`, `minsurf/`, `knots/` and `patterns/`.

    seeds   the five Platonic solids as vertex/face tables, plus the
            reconstruction of an icosahedron's triangles from its twelve
            vertices.  Six generators each carried their own copy.
    hull    faces of the convex hull of a point set.
    fit     the 2 m cube convention: centred on the origin, largest
            extent 2.0, which every generator in the extension follows.
    flags   the flag complex -- (vertex, edge, face) triples with the
            three involutions -- and ALL EIGHT Conway operators as flag
            rewrites rather than bespoke vertex surgery.  Each is checked
            against the hand-written version it replaces, on every
            Platonic solid.

This is Pass A of the polyhedra extraction: data and pure geometry only.
Pass B -- the plane-arrangement/stellation engine, point groups, and the
flag complex that would re-found the Conway operators -- is a separate
piece of work with real behavioural risk, and is not started.

References are in each submodule's header; the seed coordinates are
classical (Euclid, Elements XIII; Coxeter, "Regular Polytopes", 1973).
"""

from .flags import (FlagComplex, ambo, chamfer, dual, gyro, kis,
                    propellor, reflect, whirl)
from .fit import cube_fit_error, fit_cube
from .hull import hull_faces
from .seeds import PHI, icosa_faces, seed_poly

__all__ = ["seed_poly", "icosa_faces", "PHI", "hull_faces", "fit_cube",
           "cube_fit_error", "FlagComplex",
           "dual", "ambo", "kis", "gyro", "propellor", "whirl", "chamfer",
           "reflect"]

__version__ = "1.0.0"
