"""styles -- shared render styles for the polyhedron generators.

A "style" here is a way of turning an abstract polyhedron -- a vertex
list and a face list -- into something physical, and the same choice
applies to almost any solid.  These modules are shared by the generators
rather than owned by any one of them, which is why they live here and
not beside a particular seed family.

    ball_and_stick  every edge becomes a cylindrical strut and every
                    vertex a spherical node, the way physical polyhedron
                    models and geodesic domes are actually built.
    facet_style     the shell dissected into one segment per face, each
                    side wall mitred at half the dihedral angle so
                    neighbouring segments meet flush.

Unlike the engine packages (`ifs`, `knots`, `minsurf`, `patterns`,
`polyhedra`, `seifert`), these touch `bpy`: they build objects, not just
geometry.  So this package is a namespace, not a headless engine, and it
is imported only from Blender-facing code.

The style modules that register their own operators -- `leonardo_style`,
`strahler_style`, `curvature_color`, `organic_wireframe` -- stay at the
top level with the other operator modules.
"""
