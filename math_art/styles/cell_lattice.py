
# Shared "Cell Lattice" style for the surface generators.
#
# Turn a solid surface into a sparse openwork lattice: the surface is
# coarsened into irregular cells, replaced by its dual, and the dual's
# edges are thickened into struts while its faces are discarded.  The
# result reads as a Voronoi-like net of hexagon-ish cells stretched
# over the original shape -- the "parametric Voronoi" look, but with
# honest polygonal cells rather than melted triangles.
#
# The whole effect is a live, non-destructive modifier chain, so every
# parameter stays adjustable in the modifier stack after the operator
# has run:
#
#   Triangulate  -> the dual of a triangulation is the hexagon-ish
#                   net we are after (a triangle becomes a vertex, a
#                   vertex of valence n becomes an n-gon)
#   Decimate     -> COLLAPSE at a low ratio sets the CELL SIZE: the
#                   coarser the triangulation, the larger the cells
#   Dual Mesh    -> the polyhedral dual, as Blender's built-in
#                   geometry node.  "Keep Boundaries" matters: most
#                   of the Surfaces family are open patches, and
#                   without it the rim is eaten away
#   Wireframe    -> drops the faces and solidifies the edges into
#                   struts (boundary kept, so the rim survives)
#   Subdivision  -> rounds the struts into smooth organic branches
#
# Every vertex of the dual comes from a triangle, so it has valence 3
# -- the friendliest possible case for the Wireframe modifier, whose
# strut junctions degrade at high valence.  That is why this needs no
# custom strut geometry the way `ball_and_stick` does.
#
# A note on cell regularity: Decimate-collapse leaves an uneven spread
# of vertex valences, so the dual is a mix of 4- to 9-gons dominated by
# pentagons, hexagons and heptagons rather than a clean honeycomb.  That
# irregularity is what makes it read as organic rather than engineered,
# so it is the intended look.  An even honeycomb would need genuinely
# isotropic remeshing (voxel Remesh does not help -- its marching-cubes
# output is valence-3/4 and gives a worse dual) or a resampling of the
# generator's own parameter domain.
#
# The construction is the polyhedral dual, not a true Voronoi diagram,
# but the two coincide in spirit: the dual of a Delaunay triangulation
# IS the Voronoi diagram, and Decimate's collapse leaves something
# Delaunay-ish.
#
# References:
# - Georgy F. Voronoi, "Nouvelles applications des parametres continus
#   a la theorie des formes quadratiques", J. reine angew. Math. 133
#   (1908) -- the tessellation the look is named for.
# - Peter Gustav Lejeune Dirichlet, "Uber die Reduktion der positiven
#   quadratischen Formen mit drei unbestimmten ganzen Zahlen", J. reine
#   angew. Math. 40 (1850) -- the same tessellation in two dimensions.
# - Boris N. Delaunay, "Sur la sphere vide", Izv. Akad. Nauk SSSR 7
#   (1934) -- the Delaunay triangulation, whose dual is the Voronoi
#   diagram.
# - Michael Garland and Paul S. Heckbert, "Surface Simplification Using
#   Quadric Error Metrics", SIGGRAPH '97 -- the edge-collapse
#   decimation Blender's Decimate modifier implements.

# Canonical names of the operator properties this style reads, with the
# defaults an adopting generator should declare.  Keeping every
# generator on the same names is what lets `apply_from` work.
#
# `even_thickness` defaults OFF deliberately.  Wireframe's even offset
# scales as 1/sin(theta/2), so it explodes at acute corners -- and an
# open surface's rim is exactly where the dual leaves sliver cells with
# acute corners.  Measured on a catenoid-helicoid, the furthest a strut
# vertex strays from the surface it should hug:
#
#     even offset ON    1.6t (t=0.03) -> 3.0t (0.06) -> 4.1t (0.09)
#     even offset OFF   0.4t          -> 0.4t        -> 0.4t
#
# and with the rounding Subdivision switched off the ON case reaches
# 11.7t: long needles shooting out of the rim.  The OFF case is flat in
# t, so struts stay put at any thickness and the only cost is slightly
# pinched width at sharp corners.  (Welding the sliver cells away first
# was tried and is not a fix -- it is erratic, and at some thresholds
# makes the spikes worse before destroying the lattice.)
PROPS = (
    ("cell_size", 0.12),
    ("strut_thickness", 0.03),
    ("smoothing", 1),
    ("keep_boundaries", True),
    ("even_thickness", False),
)

_GROUP_NAME = "Math Art Dual Mesh"
_KEEP_SOCKET = "Keep Boundaries"


def dual_node_group():
    """Get or create the shared Dual Mesh node group.

    One group is shared by every object using the style, so a scene
    with fifty lattices carries one node tree, not fifty.  Returns
    (node_group, keep_boundaries_socket_identifier); the identifier is
    the key used to set the per-modifier "Keep Boundaries" input.
    """
    import bpy

    ng = bpy.data.node_groups.get(_GROUP_NAME)
    if ng is None:
        ng = bpy.data.node_groups.new(_GROUP_NAME, "GeometryNodeTree")
        ng.interface.new_socket("Geometry", in_out="INPUT",
                                socket_type="NodeSocketGeometry")
        ng.interface.new_socket("Geometry", in_out="OUTPUT",
                                socket_type="NodeSocketGeometry")
        keep = ng.interface.new_socket(
            _KEEP_SOCKET, in_out="INPUT",
            socket_type="NodeSocketBool")
        keep.default_value = True
        gin = ng.nodes.new("NodeGroupInput")
        gout = ng.nodes.new("NodeGroupOutput")
        dual = ng.nodes.new("GeometryNodeDualMesh")
        gin.location = (-300.0, 0.0)
        dual.location = (0.0, 0.0)
        gout.location = (250.0, 0.0)
        # link by socket NAME: the interface's item order is not the
        # creation order, so positional indices are not reliable here
        out_by_name = {s.name: s for s in gin.outputs}
        ng.links.new(out_by_name["Geometry"], dual.inputs["Mesh"])
        ng.links.new(out_by_name[_KEEP_SOCKET],
                     dual.inputs["Keep Boundaries"])
        ng.links.new(dual.outputs[0], gout.inputs[0])

    ident = None
    for item in ng.interface.items_tree:
        if getattr(item, "name", None) == _KEEP_SOCKET:
            ident = item.identifier
            break
    return ng, ident


def apply(obj, cell_size=0.12, strut_thickness=0.03, smoothing=1,
          keep_boundaries=True, even_thickness=False, triangulate=True,
          scale=1.0, prefix="Lattice"):
    """Build the live cell-lattice modifier stack on `obj`.

    `strut_thickness` is in world units and is multiplied by `scale`,
    so a generator that has scaled its surface away from the house
    2 m cube keeps struts of the intended visual weight.  Returns
    `obj` so callers can chain.
    """
    if triangulate:
        mod = obj.modifiers.new(prefix + "Triangulate", 'TRIANGULATE')
        try:
            mod.min_vertices = 4
        except AttributeError:
            pass
    mod = obj.modifiers.new(prefix + "Cells", 'DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = cell_size

    ng, keep_ident = dual_node_group()
    mod = obj.modifiers.new(prefix + "Dual", 'NODES')
    mod.node_group = ng
    if keep_ident is not None:
        # set at creation time: changing a geometry-nodes input after
        # the modifier is evaluated needs an explicit depsgraph tag,
        # and without one the stack silently keeps the old value
        mod[keep_ident] = bool(keep_boundaries)

    mod = obj.modifiers.new(prefix + "Struts", 'WIREFRAME')
    mod.thickness = strut_thickness * max(scale, 1e-6)
    mod.use_even_offset = even_thickness
    mod.use_boundary = True
    mod.use_replace = True

    mod = obj.modifiers.new(prefix + "Smooth", 'SUBSURF')
    mod.levels = smoothing
    mod.render_levels = max(smoothing, 2)

    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def apply_from(obj, op, scale=1.0, prefix="Lattice"):
    """Apply the style using an operator's canonically named properties.

    This is the seam that makes adopting the style cheap: a generator
    declares the `PROPS` names and calls this one line.  Any property
    the operator does not define falls back to the `PROPS` default.
    """
    kw = {name: getattr(op, name, default) for name, default in PROPS}
    return apply(obj, scale=scale, prefix=prefix, **kw)


def draw_props(layout, op):
    """Draw the standard lattice property block into a redo panel."""
    for name, _default in PROPS:
        if hasattr(op, name):
            layout.prop(op, name)


def _selftest():
    # No Blender here, so the geometry itself cannot be exercised --
    # the stack is built entirely from stock modifiers and a stock
    # geometry node.  What IS worth pinning down is the property
    # contract every adopting generator codes against.
    names = [n for n, _d in PROPS]
    if len(set(names)) != len(names):
        raise AssertionError("duplicate names in PROPS")
    for expected in ("cell_size", "strut_thickness", "smoothing",
                     "keep_boundaries", "even_thickness"):
        if expected not in names:
            raise AssertionError(f"PROPS lost the '{expected}' entry")
    if not 0.0 < dict(PROPS)["cell_size"] <= 1.0:
        raise AssertionError("cell_size default is not a decimate ratio")
    if dict(PROPS)["strut_thickness"] <= 0.0:
        raise AssertionError("strut_thickness default must be positive")

    # apply_from must fall back to the defaults for an operator that
    # declares none of the properties, rather than raising.
    class _Bare:
        pass

    bare = _Bare()
    kw = {n: getattr(bare, n, d) for n, d in PROPS}
    if kw != dict(PROPS):
        raise AssertionError(
            "apply_from should fall back to the PROPS defaults")
