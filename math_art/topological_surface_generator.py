
# Topological Surface Generator for Blender
#
# The classic topology menagerie, after chapter 6 of Henry Segerman's
# "Visualizing Mathematics with 3D Printing" (figs 6-1..6-7) plus the
# Klein bottle from the project TODO:
#
#   * Klein bottle -- the iconic "bottle" shape (the standard smooth
#     closed-form immersion) and the figure-8 / twisted-torus form.
#   * Cross-cap and Steiner's Roman surface -- the two classical
#     immersions of the real projective plane RP^2.
#   * The Steiner family -- the shadows in 3-space of the VERONESE
#     surface, which embeds RP^2 in R^4 with no self-intersection at
#     all.  Turning the projection direction sweeps continuously from
#     the Roman surface (angle 0) to the cross-cap (angle 90 degrees),
#     with a Steiner surface at every angle in between; that no shadow
#     is ever free of singularities is the theorem, not a limitation of
#     the parametrisation.
#   * Boy's surface via the Bryant-Kusner parametrization (an RP^2
#     immersion with no pinch points).
#   * Orientable genus-g handlebody surfaces, meshed implicitly with
#     the marching-tetrahedra kernel from the minsurf package.
#   * Solid closed strips with n half-twists (n = 1 is a printable
#     Mobius band, fig 6-1), swept as a watertight solid directly.
#   * The Sudanese Mobius band -- Lawson's minimal Mobius band in S^3,
#     stereographically projected to R^3 (embedded, with a round
#     great-circle boundary).
#
#   * Morin's surface -- the halfway model of turning a sphere inside
#     out: the moment in Morin's eversion when the surface is exactly
#     half turned through, so that a quarter turn about the axis carries
#     it onto itself while exchanging its two sides.  Apery's
#     parametrization puts it in one family with Boy's surface, indexed
#     by an order n, and the family's PARITY decides the topology: even n
#     is an immersed sphere (n = 2 is Morin's), odd n an immersed
#     projective plane (n = 3 is Boy's).  Both are exact identities in
#     the formula -- see minsurf/topology.build_morin.
#
#   * Dyck's surface -- the closed one-sided surface of non-orientable
#     genus 3, in BOTH of the forms von Dyck proved equal in 1888: a
#     torus carrying one cross-cap (the default, and the form that
#     makes the theorem a picture) and a sphere carrying three.  The
#     cross-caps are attached by exact surgery, so chi = -1 by
#     construction -- see minsurf/topology.build_dyck.
#
#   * The Klein quartic -- the most symmetric genus-3 surface there is,
#     defined by x^3 y + y^3 z + z^3 x = 0 in the complex projective
#     plane and so not directly meshable; what is built is its regular
#     tiling, the map {3,7}_8 with automorphism group PSL(2,7) of order
#     168, carried through to the mesh: 56 triangles or the dual 24
#     heptagons on the Schulte-Wills genus-3 polyhedron.  The self-test
#     counts all 336 symmetries (with reflections) flag by flag -- see
#     minsurf/topology.build_klein_quartic.
#
# Non-orientable surfaces cannot embed in 3-space, so KLEIN / KLEIN8 /
# CROSSCAP / ROMAN / BOY are immersions with self-intersections. The
# parametric grids are closed combinatorially -- boundary
# identifications are made by vertex index, not by coordinate welding
# -- so each mesh carries the Euler characteristic of the abstract
# surface: 0 for the Klein bottles, 1 for the RP^2 models. Parameter
# grids are offset by fractional steps where needed so that no two
# grid samples land exactly on a self-intersection curve.
#
# References:
# - Klein bottle: F. Klein (1882).  The default classical bottle shape
#   is built by the tube scheme of G. Franzoni, "The Klein bottle in
#   its classical shape: a further step towards a good
#   parametrization", arXiv:0909.5354 (2009): a tube of varying radius
#   swept along a plane directrix, with the dumbbell-curve directrix of
#   the paper's section 4 (which closes) as the default and its
#   section-3 piriform directrix and the older polynomial immersion as
#   alternatives.  A converted copy is in research/papers/
#   surfaces-and-immersions/franzoni-2009-klein-bottle-classical-shape/.
# - Mobius band (the plain ruled one-sided strip): A. F. Mobius (1858)
#   and J. B. Listing (1858), as the standard half-twist ruled
#   parametrization.
# - Boy's surface: W. Boy, Math. Ann.
#   57 (1903), here via the R. Bryant - R. Kusner parametrization.
# - Cross-cap and Roman surface: two immersions of RP^2 due to
#   J. Steiner (Rome, 1844).
# - Veronese surface: G. Veronese (1854-1917); see M. Berger,
#   "Geometry Revealed", Springer 2010, p. 47.  The two named
#   projections used as the endpoints of the Steiner family here are
#   from R. Ferreol, "Encyclopedie des formes mathematiques
#   remarquables", mathcurve.com, chapter "surface de Veronese"; a
#   converted copy is in research/books/
#   mathcurve_encyclopedie_formes_mathematiques/. Mobius band: A. F. Mobius (1858).
# - Sudanese Mobius band: H. B. Lawson, "Complete Minimal Surfaces in
#   S^3", Ann. of Math. 92 (1970), 335-374; named for Sue Goodman
#   and Daniel Asimov (cf. G. Francis, "A Topological Picturebook",
#   Springer 1987).
# - Morin's surface: Bernard Morin (1933-2018); B. Morin and J.-P. Petit
#   on turning a sphere inside out.  Parametrization by Francois Apery,
#   "Models of the Real Projective Plane" (Vieweg, 1987), p. 104; via
#   R. Ferreol, "Encyclopedie des formes mathematiques remarquables"
#   (mathcurve.com), chapter "surface de Morin".
# - Sphere eversion exists at all: Stephen Smale, "A classification of
#   immersions of the two-sphere", Trans. AMS 90 (1959), 281-290 -- a
#   proof that gave no picture, which is what Morin's model supplies.
# - Dyck's surface: W. von Dyck, "Beitraege zur Analysis situs",
#   Math. Ann. 32 (1888), 457-512 -- the proof that a sphere with
#   three cross-caps is the same closed surface as a torus with one.
# - R. Ferreol, "Encyclopedie des formes mathematiques remarquables"
#   (mathcurve.com), chapter "surface de Dyck" -- both forms of the
#   surface, and Christoph Soland's octagon presentation realized in
#   his wire sculpture "Janus bifrons" (Gymnase du Bugnon, Lausanne).
# - Klein quartic: F. Klein, "Ueber die Transformation siebenter
#   Ordnung der elliptischen Functionen", Math. Ann. 14 (1878) -- the
#   quartic curve and its regular map {3,7}_8 of genus 3 with 168
#   orientation-preserving symmetries.
# - E. Schulte and J. M. Wills, "A polyhedral realization of Felix
#   Klein's map {3,7}_8 on a Riemann surface of genus 3", J. London
#   Math. Soc. (2) 32 (1985), 539-547 -- the embedded 56-triangle
#   genus-3 polyhedron on two homothetic truncated tetrahedra that
#   carries the tiling here.
# - S. Levy (ed.), "The Eightfold Way: The Beauty of Klein's Quartic
#   Curve", MSRI Publications 35, Cambridge University Press (1999) --
#   the volume around Helaman Ferguson's sculpture of the 24-heptagon
#   tiling, which the dual view shows.
# - J. C. Baez, "Klein's Quartic Curve",
#   math.ucr.edu/home/baez/klein.html, and G. Egan, "Klein's Quartic
#   Curve", gregegan.net/SCIENCE/KleinQuartic/KleinQuartic.html --
#   the symmetry story and tetrahedral realizations of the tilings.
# - Menagerie after ch. 6 of H. Segerman, "Visualizing Mathematics
#   with 3D Printing" (2016).

bl_info = {
    "name": "Topological Surface Generator",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Topological Surface",
    "description": "Klein bottles, cross-cap, Roman and Boy surfaces, "
                   "genus-g handlebodies, solid twisted strips",
    "category": "Add Mesh",
}
import math

import numpy as np

# The mathematics lives in the sibling `minsurf` engine package;
# this module is the Blender layer over it.
try:
    from .minsurf.topology import (build_boy, build_crosscap, build_morin,
                                   build_ovalesque, ovalesque_point,
                                   build_steiner,
                                   build_dyck, build_klein_quartic,
                                   build_genus, build_klein_bottle,
                                   build_klein_franzoni,
                                   build_mobius_band,
                                   build_nonorientable,
                                   build_klein_figure8, build_roman,
                                   build_sudanese_mobius,
                                   build_twist_strip, edge_face_counts,
                                   winding_conflict_edges)
except ImportError:  # flat import outside the package
    from minsurf.topology import (build_boy, build_crosscap, build_morin,
                                  build_ovalesque, ovalesque_point,
                                  build_steiner,
                                  build_dyck, build_klein_quartic,
                                  build_genus, build_klein_bottle,
                                  build_klein_franzoni,
                                  build_mobius_band,
                                  build_nonorientable,
                                  build_klein_figure8, build_roman,
                                  build_sudanese_mobius,
                                  build_twist_strip, edge_face_counts,
                                  winding_conflict_edges)
try:
    from .sharp_creases import mark_sharp
except ImportError:  # flat import outside the package
    try:
        from sharp_creases import mark_sharp
    except ImportError:               # headless numeric self-test only
        mark_sharp = None






# ==========================================================================
# Mesh bookkeeping helpers (used by the standalone self-tests too)
# ==========================================================================





# ==========================================================================
# Klein bottles
# ==========================================================================





# ==========================================================================
# Sudanese Mobius band (Lawson's minimal Mobius band in S^3)
# ==========================================================================
# Half of Lawson's minimally-immersed Klein bottle in the unit 3-sphere
#   x(t, v) = (cos t cos v, sin t cos v, cos 2t sin v, sin 2t sin v),
# taking v in [0, pi], is an EMBEDDED Mobius band whose single boundary
# is a great circle.  Stereographically projecting to R^3 keeps that
# boundary a round circle.  We project from the point of S^3 farthest
# from the band, p = (-1, 0, -1, 0)/sqrt2 (the band's dot with p peaks
# at 1/sqrt2, so the denominator 1 - x.p never drops below ~0.293 and
# the image stays bounded).  In the basis e1 = (0,1,0,0),
# e2 = (0,0,0,1), e3 = (1,0,-1,0)/sqrt2 of p^perp this reduces to
#   X = x2/s,  Y = x4/s,  Z = (x1 - x3)/(sqrt2 s),  s = 1 + (x1+x3)/sqrt2.
# The nickname "Sudanese" honours topologists Sue Goodman and Daniel
# Asimov (Sue + Dan), not the country.



# ==========================================================================
# RP^2 immersions: cross-cap, Roman surface, Boy's surface
# ==========================================================================













# ==========================================================================
# Genus-g handlebody surface (implicit, marching tetrahedra)
# ==========================================================================
# q(x, y) = product over g+1 overlapping circles in a row of the
# NORMALIZED factors (rho_i^2 - r^2) / (rho_i^2 + r^2) in (-1, 1),
# rho_i = distance to center i. Normalizing keeps distant circles'
# factors near +1 instead of growing without bound, so the width of
# the saddle channel where adjacent circle interiors connect (at the
# circle-circle crossing points q has a saddle of value 0) stays
# O(sqrt(eps)) independent of g -- with raw factors the channel
# shrinks below the marching-grid cell size for g >= 3 and the solid
# falls apart. The planar region {q <= eps} is then the blob around
# the circle union minus the g lens cores (inside exactly two circles
# the product is positive; its peak is ~0.034 even for the middle
# lens at g = 5, comfortably above eps). Adding k z^2 keeps every
# vertical fiber of the solid an interval, so {q + k z^2 <= eps} is a
# genus-g handlebody and its boundary has Euler char 2 - 2g.





# ==========================================================================
# Solid closed strip with n half-twists (Mobius band and friends)
# ==========================================================================



# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


PRESET_ITEMS = [
    ('KLEIN', "Klein Bottle",
     "The Klein bottle in its classical bottle shape, built by "
     "Franzoni's tube scheme: a tube of varying radius swept along a "
     "plane directrix.  The default dumbbell directrix closes the "
     "surface exactly (no boundary); the paper's piriform directrix "
     "and the older polynomial immersion are offered as alternative "
     "renditions"),
    ('KLEIN8', "Klein Bottle (Figure-8)",
     "Figure-8 / twisted-torus Klein bottle immersion"),
    ('ETRUSCAN_VENUS', "Etruscan Venus",
     "Cox, Francis and Idaszak's Etruscan Venus (1989): the Roman "
     "surface's curve family uncurled to the unit circle -- a singular "
     "Klein bottle, the connected sum of two Roman surfaces, with 12 "
     "pinch points.  Francis's ovalesque sweep F(1, 0)"),
    ('IDA', "Ida Surface",
     "The immersed Klein bottle at the far corner F(1, 1) of "
     "Francis's ovalesque family: the Venus with its 12 pinch points "
     "cancelled by the Romboy homotopy parameter -- a smooth "
     "immersion, named for its programmer Ray Idaszak"),
    ('KLEIN_NESTED', "Nested Klein Bottles",
     "Klein bottles nested one inside another, in the manner of Alan "
     "Bennett's glass series and Jos Leys's renders: tubes of "
     "decreasing radius around the shared dumbbell directrix, so each "
     "shell genuinely encloses the next"),
    ('MOBIUS', "Moebius Strip",
     "The canonical one-sided band (Moebius and Listing, 1858): a "
     "strip closed up after a half twist, as the standard ruled "
     "parametrization, seam glued so the mesh has the band's true "
     "topology -- one side, one boundary edge"),
    ('SUDANESE', "Sudanese Mobius Band",
     "Lawson's minimal Mobius band in S^3, stereographically "
     "projected to R^3 (embedded, round boundary circle)"),
    ('CROSSCAP', "Cross-Cap",
     "Standard cross-cap immersion of the projective plane"),
    ('ROMAN', "Roman Surface",
     "Steiner's Roman surface (projective plane)"),
    ('STEINER', "Steiner Surface (Veronese shadow)",
     "The shadow in 3-space of the Veronese surface, which embeds the "
     "projective plane in R^4. Turning the projection sweeps from the "
     "Roman surface at 0 degrees to the cross-cap at 90"),
    ('BOY', "Boy's Surface",
     "Boy's surface, Bryant-Kusner parametrization"),
    ('MORIN', "Morin's Surface",
     "The halfway model of turning a sphere inside out: the moment in "
     "Morin's eversion when the surface is exactly half turned through "
     "and a quarter turn exchanges its two sides.  Apery's order-n "
     "family, in which EVEN n gives an immersed sphere (n = 2 is "
     "Morin's) and ODD n an immersed projective plane (n = 3 is Boy's)"),
    ('NONORIENT', "Non-Orientable Genus-k",
     "The closed non-orientable surface N_k: a sphere with k "
     "cross-caps. k = 1 is the projective plane, k = 2 the Klein "
     "bottle, k = 3 Dyck's surface. Immersed, with a segment of "
     "double points per cross-cap -- none of them embeds in 3-space"),
    ('DYCK', "Dyck's Surface",
     "The closed one-sided surface of non-orientable genus 3, in "
     "either of the two forms von Dyck proved equal in 1888: a torus "
     "carrying one cross-cap, or a sphere carrying three.  The torus "
     "form is the default -- a handle and a cross-cap on one surface "
     "is the theorem made visible"),
    ('GENUS', "Genus-g Surface",
     "Orientable genus-g handlebody surface (implicit)"),
    ('KLEIN_QUARTIC', "Klein Quartic",
     "The most symmetric surface of genus 3, carrying its regular "
     "tiling through to the mesh: 24 heptagons (the Eightfold Way "
     "view) or the dual 56 triangles, on the Schulte-Wills genus-3 "
     "polyhedron.  Its symmetry group has order 168 -- the largest "
     "any genus-3 surface allows"),
    ('TWIST_STRIP', "Twisted Strip (solid)",
     "Solid closed strip with n half-twists; n = 1 is a Mobius band"),
]

_IMMERSIONS = {'KLEIN', 'KLEIN8', 'MOBIUS', 'SUDANESE', 'CROSSCAP',
               'ROMAN', 'BOY',
               'MORIN',
               'STEINER'}


if _IN_BLENDER:

    def _new_object(context, name, verts, faces, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(map(float, v)) for v in np.asarray(verts)],
                       [], [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth',
                                [smooth] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    class MESH_OT_topological_surface_add(bpy.types.Operator):
        """Add a classic topological surface (Klein bottle, projective
        plane immersions, genus-g handlebody, solid twisted strip)"""
        bl_idname = "mesh.topological_surface_add"
        bl_label = "Topological Surface"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(name="Surface", items=PRESET_ITEMS,
                             default='KLEIN',
                             description="Which topological surface to build")
        klein_form: EnumProperty(
            name="Rendition", default='DUMBBELL',
            description="How the classical bottle shape is built "
                        "(Klein Bottle preset only)",
            items=[('DUMBBELL', "Dumbbell Tube (closed)",
                    "Franzoni's dumbbell-curve directrix: the one "
                    "rendition whose image genuinely closes, so the "
                    "mesh has no boundary at all"),
                   ('PIRIFORM', "Piriform Tube (open at the cusp)",
                    "Franzoni's re-parametrized piriform directrix "
                    "(his section 3).  The directrix's speed vanishes "
                    "at the cusp, so -- as the paper itself notes -- "
                    "the image misses a circle there and the tube is "
                    "left honestly open (two rim circles)"),
                   ('POLYNOMIAL', "Polynomial Immersion",
                    "The older closed-form polynomial immersion; "
                    "squatter than the classical shape, seam left "
                    "split as before")])
        venus_waist: FloatProperty(
            name="Waist", default=1.0, min=0.3, max=2.0,
            description="Waist radius r1 of the ovalesque sweep; the "
                        "height stays 2, so the classic figure has "
                        "waist-to-height ratio near one half")
        nest_shells: IntProperty(
            name="Shells", default=3, min=2, max=5,
            description="How many bottles to nest (Nested Klein "
                        "Bottles preset only)")
        nest_ratio: FloatProperty(
            name="Shell Ratio", default=0.62, min=0.3, max=0.85,
            description="Tube-radius ratio between one shell and the "
                        "next: each inner bottle's tube radius is "
                        "this fraction of the previous one's")
        klein_length: FloatProperty(
            name="Length", default=20.0, min=4.0, max=60.0,
            description="Length of the directrix the tube is swept "
                        "along -- the bottle's height (the paper's a, "
                        "default 20)")
        klein_width: FloatProperty(
            name="Width", default=8.0, min=0.5, max=40.0,
            description="Sideways spread of the directrix -- how far "
                        "the neck swings out before diving back "
                        "through the wall (the paper's b, default 8)")
        klein_radius: FloatProperty(
            name="Tube Radius", default=5.5, min=0.5, max=20.0,
            description="Overall radius of the swept tube, before the "
                        "taper varies it (the paper's c, default 11/2)")
        klein_taper: FloatProperty(
            name="Taper", default=0.4, min=0.0, max=1.2,
            description="Spread between the tube's minimum and maximum "
                        "radius: 0 keeps the tube uniform, larger "
                        "values fatten the bulb and tighten the neck "
                        "(the paper's d, default 2/5)")
        res_u: IntProperty(
            name="Resolution U", default=96, min=8, max=512,
            description="Samples along u (around); for the genus "
                        "surface, implicit grid density")
        res_v: IntProperty(
            name="Resolution V", default=48, min=4, max=512,
            description="Samples along v (across / radial)")
        morin_order: IntProperty(
            name="Order", default=2, min=2, max=12,
            description="Order n of Apery's family, and the surface's "
                        "rotational symmetry.  EVEN n gives an immersed "
                        "sphere -- 2 is Morin's own surface -- and ODD n "
                        "an immersed projective plane, 3 being Boy's "
                        "(Morin's surface only)")
        morin_k: FloatProperty(
            name="Pinch", default=1.0, min=0.0, max=1.35,
            description="Apery's k.  It only enters the denominator "
                        "sqrt2 - k sin 2u sin nv, so it deepens the "
                        "surface's lobes without touching any of its "
                        "symmetries; 0 flattens it to a round shape "
                        "(Morin's surface only)")
        steiner_angle: FloatProperty(
            name="Projection Angle", default=0.0, min=-180.0, max=180.0,
            description="Direction, in degrees, from which the "
                        "Veronese surface's four-dimensional embedding "
                        "is projected into 3-space: 0 gives Steiner's "
                        "Roman surface, 90 the cross-cap, and every "
                        "angle between gives another Steiner surface")
        genus: IntProperty(
            name="Genus", default=2, min=1, max=5,
            description="Number of handles (verified for 1-5)")
        dyck_form: EnumProperty(
            name="Form", default='TORUS',
            description="Which of the two equal forms of the surface "
                        "to build (Dyck's Surface preset only)",
            items=[('TORUS', "Torus With Cross-Cap",
                    "A torus with one disk cut away and the rim of "
                    "the hole glued to itself antipodally: the "
                    "connected sum of a torus and a projective plane"),
                   ('SPHERE', "Sphere With Three Cross-Caps",
                    "A sphere carrying three cross-caps: the "
                    "connected sum of three projective planes -- the "
                    "same surface, by von Dyck's theorem")])
        quartic_tiling: EnumProperty(
            name="Tiling", default='HEPTAGONS',
            description="Which of the two dual regular tilings of the "
                        "quartic to carry as the mesh faces (Klein "
                        "Quartic preset only)",
            items=[('HEPTAGONS', "Heptagons",
                    "The tiling by 24 heptagons, three around every "
                    "corner -- the view realized in Helaman "
                    "Ferguson's sculpture The Eightfold Way"),
                   ('TRIANGLES', "Triangles",
                    "The dual tiling by 56 triangles, seven around "
                    "every corner -- the Schulte-Wills polyhedron "
                    "with flat faces")])
        quartic_rounding: IntProperty(
            name="Rounding", default=2, min=0, max=4,
            description="Subdivision levels that round the tiled "
                        "polyhedron toward a smooth form, keeping the "
                        "tiling as the control cage; 0 keeps the flat "
                        "faces")
        cross_caps: IntProperty(
            name="Cross-Caps k", default=3, min=1, max=8,
            description="Number of cross-caps: N_k has Euler "
                        "characteristic 2 - k. 1 = projective plane, "
                        "2 = Klein bottle, 3 = Dyck's surface")
        cap_size: FloatProperty(
            name="Cross-Cap Size", default=0.0, min=0.0, max=1.2,
            description="Radius of the disk each cross-cap replaces; "
                        "0 sizes it from k, large for one cap and "
                        "small enough to keep several clear of one "
                        "another")
        cap_pinch: FloatProperty(
            name="Cross-Cap Pinch", default=0.55, min=0.0, max=1.5,
            description="How far each cross-cap is lifted over its "
                        "double-point segment; 0 leaves the two sheets "
                        "coincident and unreadable")
        twists: IntProperty(
            name="Half-Twists", default=1, min=0, max=12,
            description="Half-twists per revolution; 1 = Mobius band")
        strip_width: FloatProperty(
            name="Strip Width", default=0.6, min=0.05, max=2.0,
            description="Width of the twisted strip's band")
        strip_thickness: FloatProperty(
            name="Strip Thickness", default=0.18, min=0.01, max=1.0,
            description="Thickness of the solid strip")
        ridge: BoolProperty(
            name="Center Ridge", default=False,
            description="Raised ridge along the strip center line")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size of the result")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="Immersed surfaces only: 0 = raw surface, "
                        "> 0 = Solidify modifier of this thickness")
        smooth: BoolProperty(name="Smooth Shading", default=True,
                             description="Shade the surface smooth "
                                         "rather than faceted")

        def execute(self, context):
            p = self.preset
            seam_sharp = False
            if p == 'KLEIN':
                if self.klein_form == 'POLYNOMIAL':
                    V, F = build_klein_bottle(self.res_u, self.res_v)
                else:
                    V, F = build_klein_franzoni(
                        self.res_u, self.res_v, self.klein_length,
                        self.klein_width, self.klein_radius,
                        self.klein_taper, self.klein_form)
                    # the closed dumbbell mesh carries the unavoidable
                    # winding-flip ring on its seam circle; sharp-split
                    # the normals there instead of splitting vertices
                    seam_sharp = self.klein_form == 'DUMBBELL'
                name = "Klein Bottle"
            elif p == 'MOBIUS':
                V, F = build_mobius_band(self.res_u, self.res_v,
                                         width=self.strip_width)
                seam_sharp = True
                name = "Moebius Strip"
            elif p == 'KLEIN8':
                V, F = build_klein_figure8(self.res_u, self.res_v)
                name = "Klein Bottle 8"
            elif p == 'ETRUSCAN_VENUS':
                V, F = build_ovalesque(self.res_u, self.res_v, 1.0,
                                       0.0, self.venus_waist, 2.0)
                seam_sharp = True
                name = "Etruscan Venus"
            elif p == 'IDA':
                V, F = build_ovalesque(self.res_u, self.res_v, 1.0,
                                       1.0, self.venus_waist, 2.0)
                seam_sharp = True
                name = "Ida Surface"
            elif p == 'KLEIN_NESTED':
                V, F = build_nested_klein(
                    self.res_u, self.res_v, self.nest_shells,
                    self.nest_ratio, self.klein_length,
                    self.klein_width, self.klein_radius,
                    self.klein_taper)
                seam_sharp = True
                name = "Nested Klein Bottles"
            elif p == 'SUDANESE':
                V, F = build_sudanese_mobius(self.res_u, self.res_v)
                name = "Sudanese Mobius Band"
            elif p == 'CROSSCAP':
                V, F = build_crosscap(self.res_u, self.res_v)
                name = "Cross-Cap"
            elif p == 'ROMAN':
                V, F = build_roman(self.res_u, self.res_v)
                name = "Roman Surface"
            elif p == 'STEINER':
                V, F = build_steiner(self.res_u, self.res_v,
                                     math.radians(self.steiner_angle))
                name = "Steiner Surface"
            elif p == 'BOY':
                V, F = build_boy(self.res_u, self.res_v)
                name = "Boy Surface"
            elif p == 'MORIN':
                V, F = build_morin(max(8, self.res_v), max(8, self.res_u),
                                   self.morin_order, self.morin_k)
                name = ("Morin Surface" if self.morin_order % 2 == 0
                        else "Boy Surface (Apery n=%d)" % self.morin_order)
            elif p == 'NONORIENT':
                V, F = build_nonorientable(
                    self.cross_caps, max(16, self.res_u),
                    max(8, self.res_v // 2), hole=self.cap_size,
                    pinch=self.cap_pinch)
                name = f"Non-Orientable N{self.cross_caps}"
            elif p == 'DYCK':
                if self.dyck_form == 'SPHERE':
                    V, F = build_nonorientable(
                        3, max(16, self.res_u),
                        max(8, self.res_v // 2), hole=self.cap_size,
                        pinch=self.cap_pinch)
                else:
                    V, F = build_dyck(
                        max(16, self.res_u), max(8, self.res_v),
                        hole=self.cap_size, pinch=self.cap_pinch)
                name = "Dyck Surface"
            elif p == 'KLEIN_QUARTIC':
                V, F = build_klein_quartic(
                    dual=(self.quartic_tiling == 'HEPTAGONS'))
                name = "Klein Quartic"
            elif p == 'GENUS':
                cell = 8.0 / max(self.res_u, 16)
                V, F = build_genus(self.genus, cell)
                name = f"Genus-{self.genus} Surface"
            else:  # TWIST_STRIP
                V, F = build_twist_strip(
                    self.twists, self.res_u, self.strip_width,
                    self.strip_thickness, self.ridge)
                name = ("Mobius Band" if self.twists == 1
                        else f"Twisted Strip ({self.twists})")
            if len(F) == 0:
                self.report({'ERROR'}, "Empty mesh")
                return {'CANCELLED'}
            # center on the origin and fit within a 2 m cube, then scale
            V = np.asarray(V, float)
            lo, hi = V.min(axis=0), V.max(axis=0)
            ext = float((hi - lo).max())
            V = (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)
            obj = _new_object(context, name, V * self.scale, F,
                              smooth=self.smooth)
            if seam_sharp and mark_sharp is not None:
                # A closed non-orientable mesh cannot wind consistently:
                # one ring of edges is traversed the same way by both of
                # its faces, and averaged smooth normals degenerate
                # there into a dark crease.  Marking exactly that ring
                # sharp splits the normals at the seam -- each side
                # shades smoothly and the renderer's double-sided flip
                # hides the sign -- which is what the old split-vertex
                # seam achieved, but on a genuinely closed mesh.  No
                # crease weight: the surface through the seam is smooth
                # geometry, not a fold a subdivider should keep.
                mark_sharp(obj.data, winding_conflict_edges(F),
                           crease=False)
            if p == 'KLEIN_QUARTIC' and self.quartic_rounding > 0:
                # round the tiled polyhedron toward the smooth genus-3
                # form; the exact combinatorics stay in the base mesh,
                # which remains the modifier's control cage
                mod = obj.modifiers.new("Rounding", 'SUBSURF')
                mod.levels = self.quartic_rounding
                mod.render_levels = self.quartic_rounding
            if p in _IMMERSIONS and self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            p = self.preset
            if p == 'NONORIENT':
                lay.prop(self, 'cross_caps')
                lay.prop(self, 'cap_size')
                lay.prop(self, 'cap_pinch')
                lay.prop(self, 'res_u')
                lay.prop(self, 'res_v')
            elif p == 'DYCK':
                lay.prop(self, 'dyck_form')
                lay.prop(self, 'cap_size')
                lay.prop(self, 'cap_pinch')
                lay.prop(self, 'res_u')
                lay.prop(self, 'res_v')
            elif p == 'KLEIN_QUARTIC':
                lay.prop(self, 'quartic_tiling')
                lay.prop(self, 'quartic_rounding')
            elif p == 'GENUS':
                lay.prop(self, 'genus')
                lay.prop(self, 'res_u')
            elif p == 'TWIST_STRIP':
                lay.prop(self, 'twists')
                lay.prop(self, 'res_u')
                lay.prop(self, 'strip_width')
                lay.prop(self, 'strip_thickness')
                lay.prop(self, 'ridge')
            else:
                if p in ('ETRUSCAN_VENUS', 'IDA'):
                    lay.prop(self, 'venus_waist')
                if p == 'KLEIN_NESTED':
                    for k in ('nest_shells', 'nest_ratio',
                              'klein_length', 'klein_width',
                              'klein_radius', 'klein_taper'):
                        lay.prop(self, k)
                if p == 'KLEIN':
                    lay.prop(self, 'klein_form')
                    if self.klein_form != 'POLYNOMIAL':
                        for k in ('klein_length', 'klein_width',
                                  'klein_radius', 'klein_taper'):
                            lay.prop(self, k)
                if p == 'MOBIUS':
                    lay.prop(self, 'strip_width')
                if p == 'STEINER':
                    lay.prop(self, 'steiner_angle')
                if p == 'MORIN':
                    lay.prop(self, 'morin_order')
                    lay.prop(self, 'morin_k')
                    lay.label(text=("immersed sphere (Morin)"
                                    if self.morin_order % 2 == 0
                                    else "projective plane (Boy)"))
                lay.prop(self, 'res_u')
                lay.prop(self, 'res_v')
                lay.prop(self, 'thickness')
                if self.thickness > 0:
                    col = lay.column(align=True)
                    col.label(text="Immersed non-orientable surfaces")
                    col.label(text="thicken into their orientable")
                    col.label(text="double where they self-intersect")
                    col.label(text="(fine for viewing)")
            lay.prop(self, 'scale')
            lay.prop(self, 'smooth')

    def _menu_func(self, context):
        self.layout.operator_menu_enum(
            "mesh.topological_surface_add", "preset",
            text="Topological Surface", icon='MESH_TORUS')

    _classes = (MESH_OT_topological_surface_add,)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


def build_nested_klein(nu, nv, shells=3, ratio=0.62, a=20.0,
                       b=8.0, c=5.5, d=0.4):
    """(verts, faces) of SHELLS Franzoni dumbbell Klein bottles nested
    one inside another -- Alan Bennett's glass series (Science Museum,
    1995) and Jos Leys's renders, built as a composition of the
    shipped bottle: every shell is the SAME closed dumbbell tube, with
    the tube radius scaled by `ratio` from shell to shell around the
    shared directrix.  Tubes of strictly decreasing radius around one
    directrix are strictly nested, and the self-test measures exactly
    that: every vertex of shell i+1 has |generalized winding number|
    >= 1/2 with respect to shell i's closed mesh -- genuinely
    enclosed, not merely concentric -- while each shell individually
    stays closed, chi = 0 and one-sided."""
    shells = max(2, int(shells))
    ratio = min(0.9, max(0.1, float(ratio)))
    verts, faces = [], []
    for i in range(shells):
        Vi, Fi = build_klein_franzoni(nu, nv, a, b,
                                      c * ratio ** i,
                                      d * ratio ** i)
        off = len(verts)
        verts.extend(tuple(v) for v in Vi)
        faces.extend(tuple(off + k for k in f) for f in Fi)
    return verts, faces


def _selftest():
    # standalone smoke tests of the numeric core
    def stats(name, V, F, chi_want, nbound_want=0):
        cnt = edge_face_counts(F)
        chi = len(V) - len(cnt) + len(F)
        nbound = sum(1 for c in cnt.values() if c == 1)
        print(f"{name:10s}: {len(V):6d} verts {len(F):6d} faces "
              f"chi = {chi:3d} (want {chi_want:3d}) "
              f"boundary edges = {nbound}")
        assert chi == chi_want and nbound == nbound_want, name

    # the DEFAULT Klein bottle (Franzoni's dumbbell tube) is index-glued
    # and genuinely closed: chi = 0 with NO boundary edges
    V, F = build_klein_franzoni(64, 32)
    stats("klein", V, F, 0, nbound_want=0)
    # the piriform tube cannot close (the paper's own caveat): its two
    # rim circles near the cusp are honest boundary
    V, F = build_klein_franzoni(64, 32, directrix='PIRIFORM')
    stats("kleinpiri", V, F, 0, nbound_want=64)
    # the legacy polynomial immersion keeps its split seam (2
    # coincident rims of nv edges each), so cut open it is an
    # orientable cylinder: chi = 0 with 2*nv boundary edges
    V, F = build_klein_bottle(64, 32)
    stats("kleinpoly", V, F, 0, nbound_want=64)
    V, F = build_klein_figure8(64, 32)
    stats("klein8", V, F, 0, nbound_want=64)
    # the plain Mobius band: chi = 0 and its single boundary edge,
    # 2*nu edges long
    V, F = build_mobius_band(64, 8)
    stats("mobius", V, F, 0, nbound_want=128)
    # split-seam Sudanese band: cut open it is a disk (chi 1); its
    # boundary is the full grid perimeter, 2*nu + 2*nv edges.
    V, F = build_sudanese_mobius(64, 32)
    stats("sudanese", V, F, 1, nbound_want=2 * 64 + 2 * 32)
    V, F = build_crosscap(64, 24)
    stats("crosscap", V, F, 1)
    V, F = build_roman(64, 24)
    stats("roman", V, F, 1)
    V, F = build_boy(64, 24)
    stats("boy", V, F, 1)
    # every Steiner shadow of the Veronese surface is a projective
    # plane, so chi = 1 at every projection angle -- not just at the
    # two named ones
    for deg in (0, 30, 45, 90, 135, 180):
        V, F = build_steiner(64, 24, math.radians(deg))
        stats("steiner@%d" % deg, V, F, 1)
    for g in (1, 2, 3):
        V, F = build_genus(g, cell=0.125)
        stats(f"genus-{g}", V, [tuple(t) for t in F], 2 - 2 * g)
    # Dyck's surface: both of von Dyck's forms are the same closed
    # one-sided surface with chi = -1 (deep gates -- one-sidedness,
    # regularity of the Klein map -- live in minsurf.topology's own
    # self-test; these are the operator-facing counts)
    V, F = build_dyck(64, 32)
    stats("dyck-torus", V, F, -1)
    V, F = build_nonorientable(3, 64, 16)
    stats("dyck-sphere", V, F, -1)
    # the Klein quartic in both tilings: 24/84/56 one way, 56/84/24
    # the other, chi = -4 either way
    V, F = build_klein_quartic(dual=False)
    stats("klein37", V, F, -4)
    assert len(V) == 24 and len(F) == 56
    V, F = build_klein_quartic(dual=True)
    stats("klein73", V, F, -4)
    assert len(V) == 56 and len(F) == 24
    for n in (0, 1, 2, 3):
        V, F = build_twist_strip(n, 96, ridge=(n == 1))
        cnt = edge_face_counts(F)
        ok = all(c == 2 for c in cnt.values())
        print(f"twist n={n}: {len(V)} verts, watertight = {ok}")
        assert ok
    # ---- nested Klein bottles ------------------------------------
    # (1) each shell individually: closed, chi = 0, and ONE-SIDED --
    #     orienting the faces by spanning-tree propagation must hit
    #     conflicts (the Klein identification's winding flip), which a
    #     torus (also chi = 0) would not; (2) the nesting is real, not
    #     just concentric: every sampled vertex of shell i+1 has
    #     |generalized winding number| >= 1/2 with respect to shell
    #     i's closed mesh, measured by per-triangle solid angles.
    import math as _m
    shells = []
    for i in range(3):
        Vi, Fi = build_klein_franzoni(48, 24, 20.0, 8.0,
                                      5.5 * 0.62 ** i, 0.4 * 0.62 ** i)
        shells.append((Vi, Fi))
        cnt = edge_face_counts(Fi)
        chi = len(Vi) - len(cnt) + len(Fi)
        nb = sum(1 for c in cnt.values() if c == 1)
        assert chi == 0 and nb == 0, (i, chi, nb)
        # orientation propagation: build face adjacency with the
        # relative sense (same directed edge = same winding sense =
        # conflict when both are to be consistently oriented)
        owner = {}
        adj = {}
        for fi, f in enumerate(Fi):
            for k in range(len(f)):
                e = (f[k], f[(k + 1) % len(f)])
                key = frozenset(e)
                if key in owner:
                    fj, ej = owner[key]
                    same = (e == ej)   # traversed in the SAME direction
                    adj.setdefault(fi, []).append((fj, same))
                    adj.setdefault(fj, []).append((fi, same))
                else:
                    owner[key] = (fi, e)
        orient = [0] * len(Fi)
        orient[0] = 1
        stack = [0]
        conflicts = 0
        while stack:
            f1 = stack.pop()
            for f2, same in adj.get(f1, []):
                want = -orient[f1] if same else orient[f1]
                if orient[f2] == 0:
                    orient[f2] = want
                    stack.append(f2)
                elif orient[f2] != want:
                    conflicts += 1
        assert conflicts > 0, "shell %d oriented consistently: it "             "would be a torus, not a Klein bottle" % i
    print("nested klein: each of 3 shells closed, chi = 0, one-sided "
          "(orientation propagation conflicts > 0)")

    def _wind(V, F, P):
        tot = 0.0
        for f in F:
            for i in range(1, len(f) - 1):
                ax, ay, az = (V[f[0]][0] - P[0], V[f[0]][1] - P[1],
                              V[f[0]][2] - P[2])
                bx, by, bz = (V[f[i]][0] - P[0], V[f[i]][1] - P[1],
                              V[f[i]][2] - P[2])
                cx, cy, cz = (V[f[i + 1]][0] - P[0],
                              V[f[i + 1]][1] - P[1],
                              V[f[i + 1]][2] - P[2])
                la = _m.sqrt(ax * ax + ay * ay + az * az)
                lb = _m.sqrt(bx * bx + by * by + bz * bz)
                lc = _m.sqrt(cx * cx + cy * cy + cz * cz)
                num = (ax * (by * cz - bz * cy)
                       - ay * (bx * cz - bz * cx)
                       + az * (bx * cy - by * cx))
                den = (la * lb * lc
                       + (ax * bx + ay * by + az * bz) * lc
                       + (ax * cx + ay * cy + az * cz) * lb
                       + (bx * cx + by * cy + bz * cz) * la)
                tot += 2.0 * _m.atan2(num, den)
        return tot / (4.0 * _m.pi)
    for k in range(2):
        Vout, Fout = shells[k]
        Vin, _Fin = shells[k + 1]
        step = max(1, len(Vin) // 25)
        wmin = 1e9
        for idx in range(0, len(Vin), step):
            P = (Vin[idx][0] + 1e-7, Vin[idx][1] - 1e-7,
                 Vin[idx][2] + 1e-7)
            wmin = min(wmin, abs(_wind(Vout, Fout, P)))
        assert wmin >= 0.5, (k, wmin)
        print("nested klein: shell %d encloses shell %d -- min "
              "|winding| %.3f over sampled vertices" % (k, k + 1, wmin))

    # the assembled composition builds and keeps the shells' counts
    V, F = build_nested_klein(32, 16, 3, 0.62)
    cnt = edge_face_counts(F)
    assert sum(1 for c in cnt.values() if c == 1) == 0
    assert len(V) - len(cnt) + len(F) == 0   # 3 x chi(Klein) = 0
    print("nested klein: assembled 3-shell mesh closed, chi = 0")

    # ---- the ovalesque family (Etruscan Venus / Ida) --------------
    # gates on what the source (Francis, Picturebook pp. 96/178-179)
    # claims: (1) the closure identity P(theta+pi, -tau) = P(theta,
    # tau) that glues the mesh; (2) the corner F(0,1) IS Apery's
    # published Boy immersion, reproduced to 1e-12 with r1 = 1/sqrt 2,
    # r2 = 3/2; (3) the l = 0 members double-cover (P(theta, tau+pi) =
    # P(theta, tau)) -- projective planes -- while l = 1 members do
    # not; (4) Venus and Ida mesh closed, chi = 0 and ONE-SIDED; (5)
    # the Venus is SINGULAR (12 pinch points: the normal collapses,
    # min |N| small) while Ida is a genuine immersion (min |N| bounded
    # away) -- measured on the same grid.
    import math as _m
    rng = np.random.default_rng(7)
    thr = rng.uniform(0.0, _m.pi, 40)
    tar = rng.uniform(0.0, 2.0 * _m.pi, 40)
    for lv, bv in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
        err = float(np.max(np.abs(
            ovalesque_point(thr + _m.pi, -tar, lv, bv)
            - ovalesque_point(thr, tar, lv, bv))))
        assert err < 1e-12, (lv, bv, err)
    s2 = _m.sqrt(2.0)
    D = 2.0 - s2 * np.sin(3.0 * thr) * np.sin(2.0 * tar)
    boy = np.stack([
        (s2 * np.cos(2 * thr) * np.cos(tar) ** 2
         + np.cos(thr) * np.sin(2 * tar)) / D,
        (s2 * np.sin(2 * thr) * np.cos(tar) ** 2
         - np.sin(thr) * np.sin(2 * tar)) / D,
        3.0 * np.cos(tar) ** 2 / D], axis=-1)
    err = float(np.max(np.abs(
        ovalesque_point(thr, tar, 0.0, 1.0, 1.0 / s2, 1.5) - boy)))
    assert err < 1e-12, err
    err0 = float(np.max(np.abs(
        ovalesque_point(thr, tar + _m.pi, 0.0, 0.7)
        - ovalesque_point(thr, tar, 0.0, 0.7))))
    err1 = float(np.min(np.abs(
        ovalesque_point(thr, tar + _m.pi, 1.0, 0.7)
        - ovalesque_point(thr, tar, 1.0, 0.7))))
    assert err0 < 1e-12 and err1 > 1e-3, (err0, err1)
    print("ovalesque: closure identity to 1e-12 at all four corners; "
          "F(0,1) reproduces Apery's Boy immersion to 1e-12; l = 0 "
          "double-covers (RP2), l = 1 does not (Klein)")

    for nick, bv in (("venus", 0.0), ("ida", 1.0)):
        V, F = build_ovalesque(48, 96, 1.0, bv)
        cnt = edge_face_counts(F)
        chi = len(V) - len(cnt) + len(F)
        nb = sum(1 for c in cnt.values() if c == 1)
        assert chi == 0 and nb == 0, (nick, chi, nb)
        owner, adj = {}, {}
        for fi, f in enumerate(F):
            for k in range(len(f)):
                e = (f[k], f[(k + 1) % len(f)])
                key = frozenset(e)
                if key in owner:
                    fj, ej = owner[key]
                    adj.setdefault(fi, []).append((fj, e == ej))
                    adj.setdefault(fj, []).append((fi, e == ej))
                else:
                    owner[key] = (fi, e)
        orient = [0] * len(F)
        orient[0] = 1
        stack = [0]
        conflicts = 0
        while stack:
            f1 = stack.pop()
            for f2, same in adj.get(f1, []):
                want = -orient[f1] if same else orient[f1]
                if orient[f2] == 0:
                    orient[f2] = want
                    stack.append(f2)
                elif orient[f2] != want:
                    conflicts += 1
        assert conflicts > 0, nick
        print(f"{nick:10s}: closed, chi = 0, one-sided "
              f"(orientation conflicts)")

    def _minN(bv):
        h = 1e-5
        best = 1e9
        for thv in np.linspace(0.001, _m.pi - 0.001, 60):
            tav = np.linspace(0.0, 2.0 * _m.pi, 120, endpoint=False)
            dth = (ovalesque_point(thv + h, tav, 1.0, bv)
                   - ovalesque_point(thv - h, tav, 1.0, bv)) / (2 * h)
            dta = (ovalesque_point(thv, tav + h, 1.0, bv)
                   - ovalesque_point(thv, tav - h, 1.0, bv)) / (2 * h)
            nn = np.linalg.norm(np.cross(dth, dta), axis=-1)
            best = min(best, float(nn.min()))
        return best
    mn_v = _minN(0.0)
    mn_i = _minN(1.0)
    assert mn_v < 0.08 and mn_i > 0.5, (mn_v, mn_i)
    print("ovalesque: Venus is singular as the source says (normal "
          "collapses at pinch points, min |N| %.3f) while Ida is a "
          "genuine immersion (min |N| %.3f)" % (mn_v, mn_i))

    print("standalone tests passed")
