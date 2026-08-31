# The classical crease patterns, and folding them.
#
# Two operators over `math_art/crease/`:
#
#   mesh.crease_pattern_add   builds a classical pattern flat
#   object.fold_solve         folds the active pattern, caching the whole
#                             fold path as shape keys so it animates
#
# ONE OPERATOR, ONE DIAL.  The seven classical patterns are one entry
# with a Pattern selector rather than seven menu entries, matching the
# way CMC surfaces and the Hopf/Willmore tori are each a single entry:
# they are one construction with one dial, and seven entries for seven
# parameterisations of "a periodic crease pattern" would be clutter.
#
# WHY SHIP PATTERNS AT ALL, when the add-on does not design crease
# patterns?  Because the scope rule is about substitutability, and these
# are folk mathematics -- no author to credit, no copyright to infringe,
# nothing to download.  Without them the add-on could open every crease
# pattern in the world and produce none.
#
# HOW THE ANIMATION WORKS.  A fold is not a cheap function of a
# parameter: each state costs a Newton solve.  So the path is solved
# ONCE by continuation, every state is stored as a shape key, and a
# single custom property `fold_t` drives them through hat-function
# drivers.  Keyframe `fold_t` and the model folds; nothing re-solves at
# render time.
#
# References:
#   K. Miura, "Method of Packaging and Deployment of Large Membranes in
#       Space," ISAS report 618, 1985.
#   M. Schenk, S. D. Guest, "Geometry of Miura-folded metamaterials,"
#       PNAS 110(9), 2013.
#   T. Tachi, "Simulation of Rigid Origami," Origami^4, 2009.
#   E. D. Demaine, M. L. Demaine, V. Hart, G. N. Price, T. Tachi,
#       "(Non)existence of Pleated Folds," Graphs and Combinatorics
#       27(3), 2011 -- why the hypar folds only faceted.

import os as _os

import numpy as np

import bpy
# Imported HERE, not inside the functions that use it.  `import
# bpy.utils.previews` binds the name `bpy` in whatever scope it runs in,
# so doing it inside unregister() shadowed the module-level `bpy` for
# that whole function and the class-unregister loop below it then raised
# UnboundLocalError -- leaving every operator registered.
import bpy.utils.previews
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from mathutils import Vector

try:
    from . import crease
except ImportError:                                   # headless import
    import crease

_ASSIGN_CODE = {"M": 0, "V": 1, "F": 2, "U": 3, "B": 4}

_PATTERN_ITEMS = (
    ('MIURA', "Miura-ori",
     "Rigid-foldable parallelogram corrugation; flat-foldable at any panel angle"),
    ('ACCORDION', "Accordion Pleat",
     "Parallel pleats: no interior vertices, so it folds unconditionally"),
    ('WATERBOMB', "Waterbomb",
     "The magic-ball tessellation: squares split by both diagonals and a mid-line"),
    ('YOSHIMURA', "Yoshimura Diamond",
     "The diamond buckle pattern of a cylinder"),
    ('HYPAR', "Pleated Hypar",
     "Four sectors of concentric pleats: the classical saddle, two "
     "corners up and two down"),
    # NAME AS REQUESTED, DESCRIPTION AS MEASURED.  A true monkey saddle
    # (z = r^3 cos 3(theta)) alternates three up and three down around
    # the rim.  What the six-sector pleated hypar actually folds to here
    # is 3-fold periodic but 2 up and 4 down (+0.39, -0.24, -0.29,
    # repeated), so the description says what it does rather than
    # claiming the textbook surface.  Whether a 3-up/3-down branch also
    # exists is open -- this pattern has many degrees of freedom and the
    # solver settles on one of them.
    ('MONKEY', "Monkey Saddle",
     "Six sectors of concentric pleats: the hexagonal hypar, whose rim "
     "repeats every third sector instead of simply alternating"),
)

# --------------------------------------------------------------------
# The pattern gallery
# --------------------------------------------------------------------
# A thumbnail selector rather than a dropdown, the same way Saddle
# Polyhedron picks its solid: these five are told apart by what they
# LOOK like, and "Yoshimura Diamond" tells a reader nothing that a
# picture of the folded tube does not tell them faster.
#
# THE THUMBNAILS SHOW THE FOLDED STATE, not the flat crease pattern.
# Flat, all five are grids of lines and three of them are near enough
# identical at 128 px to be useless as a choice; folded, they are a
# corrugation, a pleat, a ball, a tube and a saddle -- which is the
# distinction the user is actually making.  `tools/bake_fold_icons.py`
# renders them through the same studio rig as the menu icons and the
# documentation figures, so the three cannot drift apart.
_ICON_DIR = _os.path.join(_os.path.dirname(__file__), "icons", "folds")
_fold_previews = None
#: Blender does not own the strings a dynamic enum callback returns, so
#: the built items must be kept alive in a module global or the labels
#: garble.  Same reason as the saddle gallery's cache.
_pattern_items_cache = []


def _load_fold_icons():
    """One preview per pattern, for the gallery selector."""
    global _fold_previews
    if _fold_previews is not None:
        return
    try:
        _fold_previews = bpy.utils.previews.new()
    except Exception:
        _fold_previews = None
        return
    for key, _label, _desc in _PATTERN_ITEMS:
        path = _os.path.join(_ICON_DIR, "%s.png" % key)
        if not _os.path.exists(path):
            continue
        try:
            _fold_previews.load(key, path, 'IMAGE')
        except Exception:
            pass                     # a missing icon is non-fatal


#: The fold angle at which each pattern becomes the thing it is named
#: after.  These are NOT one shared default: the classical patterns
#: reach their characteristic shape at genuinely different angles, and
#: at the wrong one a pattern reads as a half-folded sheet rather than
#: as itself.  The waterbomb and the Yoshimura both close into a
#: cylinder -- the shape each is actually known for -- at roughly 43 and
#: 37 degrees, and pushed past that they collapse again.  The hypar has
#: no proper planar-facet folding at all (Demaine et al. 2011), so it is
#: driven hard enough for its saddle to be unmistakable.
#:
#: Switching pattern in the redo panel resets Fold Angle to the new
#: pattern's value, so the number on screen is always the one being
#: folded to; adjust it afterwards and it stays until you switch again.
_NATURAL_FOLD = {
    'MIURA': 70.0,
    'ACCORDION': 70.0,
    'WATERBOMB': 43.0,
    'YOSHIMURA': 37.0,
    'HYPAR': 150.0,
    'MONKEY': 150.0,
}

#: How many rows -- rings, for the concentric-pleat patterns -- each
#: pattern wants by default.  Same reasoning as `_NATURAL_FOLD`: one
#: shared number cannot suit all six.  Four rows is a readable sheet for
#: the tessellations, but a hypar at four rings is a few coarse steps
#: rather than a surface, and Demaine, Demaine and Lubiw say why --
#: "the more concentric squares one folds, the closer the pleated hypar
#: is to a true hypar surface".  Sixteen is where it reads as one.
#:
#: `tools/bake_fold_icons.py` takes its ring count from HERE, so the
#: thumbnail and the default output cannot drift apart.
_NATURAL_ROWS = {
    'MIURA': 4,
    'ACCORDION': 4,
    'WATERBOMB': 4,
    'YOSHIMURA': 4,
    'HYPAR': 16,
    'MONKEY': 16,
}

#: Patterns whose sector count is part of their identity rather than a
#: free parameter.  The hypar and the monkey saddle are the SAME
#: construction at four and six sectors, and the difference is a
#: different surface -- so the count is pinned here and Columns is
#: hidden for them, instead of being a number the user has to know.
_PINNED_SIDES = {'HYPAR': 4, 'MONKEY': 6}


def _pattern_changed(self, context):
    """Snap the per-pattern settings to the newly chosen pattern.

    Both directions matter, which is why every pattern is in the tables
    rather than only the ones that differ: switching TO the hypar has to
    raise the ring count to 16, and switching AWAY from it has to put it
    back, or the next Miura is built at sixteen rows because of a choice
    made for a different pattern.
    """
    nat = _NATURAL_FOLD.get(self.pattern)
    if nat is not None:
        self.fold_angle = np.deg2rad(nat)
    rows = _NATURAL_ROWS.get(self.pattern)
    if rows is not None:
        self.rows = rows


def _pattern_label(key):
    for k, label, _desc in _PATTERN_ITEMS:
        if k == key:
            return label
    return key


def _pattern_items(self, context):
    """The five patterns, each with its folded thumbnail if one is baked.

    An un-baked pattern still appears, with icon 0 -- a partial bake is
    a valid build, exactly as it is for the menu icons.
    """
    _load_fold_icons()
    out = []
    for i, (key, label, desc) in enumerate(_PATTERN_ITEMS):
        icon = 0
        if _fold_previews is not None and key in _fold_previews:
            icon = _fold_previews[key].icon_id
        out.append((key, label, desc, icon, i))
    _pattern_items_cache[:] = out
    return out


def _frame_to_mesh(name, frame, positions, size):
    """Build a mesh from a flat frame, optionally placed in 3-D."""
    xy = frame.verts[:, :2]
    extent = float(max(xy.max(axis=0) - xy.min(axis=0))) or 1.0
    scale = size / extent
    centre = np.append(0.5 * (xy.max(axis=0) + xy.min(axis=0)), 0.0)

    src = positions if positions is not None else np.hstack(
        [xy, np.zeros((len(xy), 1))])
    co = [tuple((p - centre) * scale) for p in src]

    faces = [list(f) for f in (frame.faces or [])]
    edges = [] if faces else [tuple(int(i) for i in e) for e in frame.edges]

    me = bpy.data.meshes.new(name)
    me.from_pydata(co, edges, faces)
    me.update()

    if frame.assignment is not None:
        # from_pydata may reorder edges when faces are supplied, so map
        # by vertex pair rather than trusting index order.
        want = {}
        for k, (a, b) in enumerate(frame.edges):
            want[(int(a), int(b))] = str(frame.assignment[k])
            want[(int(b), int(a))] = str(frame.assignment[k])
        attr = me.attributes.new("crease_assignment", 'INT', 'EDGE')
        vals = []
        for e in me.edges:
            a, b = e.vertices
            vals.append(_ASSIGN_CODE.get(want.get((a, b), "U"), 3))
        attr.data.foreach_set("value", vals)
    me.update()
    return me, scale, centre


def _frame_from_object(obj):
    """Rebuild a crease.Frame from a mesh and its edge attributes."""
    me = obj.data
    verts = np.array([(v.co.x, v.co.y, v.co.z) for v in me.vertices],
                     dtype=float)
    edges = np.array([tuple(e.vertices) for e in me.edges],
                     dtype=np.int64).reshape(-1, 2)
    code_to_char = {0: "M", 1: "V", 2: "F", 3: "U", 4: "B"}
    attr = me.attributes.get("crease_assignment")
    if attr is None:
        raise crease.FoldError(
            "this mesh carries no crease_assignment attribute, so "
            "there is nothing to fold; build or import a crease "
            "pattern first")
    assign = np.array(
        [code_to_char.get(int(d.value), "U") for d in attr.data],
        dtype="<U1")
    faces = [list(p.vertices) for p in me.polygons]
    return crease.Frame(verts=verts, edges=edges, assignment=assign,
                        faces=faces or None)


def _fold_object(obj, fold_angle, steps, animate):
    """Fold `obj` in place; return a report string.

    ONE implementation, called both by Fold Pattern and by Crease
    Pattern's own Fold checkbox.  The checkbox exists to save a second
    operator call, not to be a second folder -- if it re-derived the
    fold it would be free to disagree with the button, which is the
    class of bug this module has already paid for once between
    `residual` and `place`.
    """
    frame = _frame_from_object(obj)
    if frame.faces is None:
        frame.faces = crease.build_faces(frame.verts, frame.edges)
    folder = crease.rigid.RigidFolder(frame)
    if not folder.n_vars:
        raise crease.rigid.FoldFailure(
            "no foldable creases: every edge is boundary or flat")
    path = folder.fold_path(float(fold_angle), steps=int(steps))
    states = [folder.place(r) for r in path]

    # `fold_path` stops at the last state that genuinely satisfies the
    # constraint, so a pattern asked to fold further than it can simply
    # returns a shorter path.  Say so rather than let the user wonder
    # why the slider stopped short.
    want = abs(np.rad2deg(float(fold_angle)))
    got = float(np.rad2deg(np.abs(path[-1]).max())) if len(path) else 0.0
    short = "" if got >= want - 1.0 else (
        f" (asked for {want:.0f} deg; this pattern stops folding rigidly "
        f"at about {got:.0f})")

    me = obj.data
    if not animate:
        for v, p in zip(me.vertices, states[-1]):
            v.co = Vector(p)
        me.update()
        obj["fold_is_flat"] = False
        return (f"folded to {got:.1f} deg{short}; "
                f"{folder.dof(path[-1])} degree(s) of freedom")

    # Cache the path: one shape key per solved state, blended by a hat
    # function of a single property so the fold is one dial.
    if obj.data.shape_keys:
        obj.shape_key_clear()
    obj.shape_key_add(name="Flat", from_mix=False)
    for i, p in enumerate(states):
        key = obj.shape_key_add(name=f"Fold {i:02d}", from_mix=False)
        for kv, co in zip(key.data, p):
            kv.co = Vector(co)
        key.slider_min, key.slider_max = 0.0, 1.0

    obj["fold_t"] = 0.0
    obj.id_properties_ui("fold_t").update(
        min=0.0, max=1.0, description="Fold progress, 0 flat to 1 folded")
    n = len(states) - 1
    for i, key in enumerate(obj.data.shape_keys.key_blocks[1:]):
        fc = key.driver_add("value")
        drv = fc.driver
        drv.type = 'SCRIPTED'
        var = drv.variables.new()
        var.name = "t"
        var.type = 'SINGLE_PROP'
        var.targets[0].id = obj
        var.targets[0].data_path = '["fold_t"]'
        drv.expression = f"max(0.0, 1.0 - abs(t*{n} - {i}))"

    obj["fold_is_flat"] = False
    return (f"folded to {got:.1f} deg{short} over {n} steps; "
            f"keyframe the Fold T property to animate; "
            f"{folder.dof(path[-1])} degree(s) of freedom")


class MESH_OT_crease_pattern_add(bpy.types.Operator):
    """Add a classical origami crease pattern, flat"""

    bl_idname = "mesh.crease_pattern_add"
    bl_label = "Crease Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    # A dynamic items callback, so the thumbnails can be attached.  It
    # costs the `default=` keyword -- Blender rejects one on a dynamic
    # enum -- and the default becomes "the first item", which is Miura
    # either way.
    pattern: EnumProperty(
        name="Pattern", items=_pattern_items, update=_pattern_changed,
        description="Which classical pattern to build")
    rows: IntProperty(
        name="Rows", default=4, min=1, max=64,
        description="Panels across the sheet, one way")
    cols: IntProperty(
        name="Columns", default=6, min=1, max=64,
        description="Panels across the sheet, the other way")
    panel_angle: FloatProperty(
        name="Panel Angle", default=np.deg2rad(60.0),
        min=np.deg2rad(5.0), max=np.deg2rad(85.0), subtype='ANGLE',
        description="Acute angle of the parallelogram (Miura only)")
    size: FloatProperty(
        name="Sheet Size", default=2.0, min=0.001, max=1000.0,
        unit='LENGTH', description="Longest side of the flat sheet")
    check: BoolProperty(
        name="Report Checks", default=True,
        description="Check Maekawa and Kawasaki after building and "
                    "report any vertices that fail")
    auto_fold: BoolProperty(
        name="Fold", default=True,
        description="Fold the pattern as soon as it is built, so Fold "
                    "Pattern need not be run separately. Turn this off "
                    "to keep the flat crease pattern")
    fold_angle: FloatProperty(
        name="Fold Angle", default=np.deg2rad(70.0),
        min=np.deg2rad(-179.0), max=np.deg2rad(179.0), subtype='ANGLE',
        description="Dihedral angle to drive the pattern to")
    steps: IntProperty(
        name="Steps", default=12, min=1, max=120,
        description="States solved along the fold path; each becomes a "
                    "shape key, so this is also the animation resolution")
    animate: BoolProperty(
        name="Animate", default=True,
        description="Cache the whole path as shape keys driven by a "
                    "single Fold property, instead of only the end state")

    def draw(self, context):
        L = self.layout
        # a gallery, not a dropdown: the patterns are told apart by
        # shape.  Drawn before use_property_split so the thumbnails get
        # the full panel width rather than the right-hand column.
        L.template_icon_view(self, "pattern", show_labels=True,
                             scale=6.0, scale_popup=6.0)
        # NAME THE THING.  `show_labels` only labels the cells inside
        # the popup grid -- once it closes, the widget is a bare
        # picture, so the panel never says which pattern is built.
        # Centred under the thumbnail, where the popup's own label was.
        row = L.row()
        row.alignment = 'CENTER'
        row.label(text=_pattern_label(self.pattern))

        L.use_property_split = True
        # Rows and Columns each get their own line.  Paired in one
        # `row(align=True)` under use_property_split, Blender prints the
        # first property's label and suppresses the second -- so it read
        # "Rows [4] [6]", with the column count unlabelled.
        # "Rows" is the ring count for the concentric-pleat patterns,
        # and Columns means nothing there -- their sector count is
        # pinned, so showing a control that does nothing is worse than
        # showing none.
        L.prop(self, "rows",
               text="Rings" if self.pattern in _PINNED_SIDES else "Rows")
        if self.pattern not in _PINNED_SIDES:
            L.prop(self, "cols")
        if self.pattern == 'MIURA':
            L.prop(self, "panel_angle")
        L.prop(self, "size")
        L.prop(self, "check")
        L.separator()
        L.prop(self, "auto_fold")
        if self.auto_fold:
            L.prop(self, "fold_angle")
            L.prop(self, "steps")
            L.prop(self, "animate")

    def execute(self, context):
        kw = dict(rows=self.rows, cols=self.cols, alpha=self.panel_angle,
                  count=max(2, self.cols),
                  sides=_PINNED_SIDES.get(self.pattern,
                                          max(3, min(8, self.cols))),
                  rings=max(2, self.rows))
        try:
            frame = crease.patterns.build(self.pattern, **kw)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        frame.faces = crease.build_faces(frame.verts, frame.edges)
        me, _scale, _c = _frame_to_mesh(self.pattern.title(), frame,
                                        None, self.size)
        obj = bpy.data.objects.new(me.name, me)
        context.collection.objects.link(obj)
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        obj["fold_is_flat"] = True

        msg = f"{frame.n_verts} vertices, {frame.n_edges} creases, " \
              f"{frame.n_faces} panels"
        level = {'INFO'}
        if self.check:
            rep = crease.check(frame)
            msg += "; " + rep.summary()
            if not rep:
                level = {'WARNING'}
        if self.pattern == 'HYPAR':
            msg += "; note the pleated hypar has no planar-facet folding " \
                   "(Demaine et al. 2011) -- it folds only faceted"

        # A fold that fails is NOT a failed build.  The flat pattern is
        # already correct and on screen, and cancelling here would throw
        # it away and leave the user with nothing -- so report the
        # reason and keep the paper.
        if self.auto_fold:
            try:
                msg += "; " + _fold_object(obj, self.fold_angle,
                                           self.steps, self.animate)
            except (crease.FoldError, crease.rigid.FoldFailure) as exc:
                msg += f"; left flat -- {exc}"
                level = {'WARNING'}

        self.report(level, msg)
        return {'FINISHED'}


class OBJECT_OT_fold_solve(bpy.types.Operator):
    """Rigidly fold the active crease pattern, caching the path to shape keys"""

    bl_idname = "object.fold_solve"
    bl_label = "Fold Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    fold_angle: FloatProperty(
        name="Fold Angle", default=np.deg2rad(70.0),
        min=np.deg2rad(-179.0), max=np.deg2rad(179.0), subtype='ANGLE',
        description="Dihedral angle to drive the pattern to")
    steps: IntProperty(
        name="Steps", default=12, min=1, max=120,
        description="States solved along the fold path; each becomes a "
                    "shape key, so this is also the animation resolution")
    animate: BoolProperty(
        name="Animate", default=True,
        description="Cache the whole path as shape keys driven by a "
                    "single Fold property, instead of only the end state")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        try:
            msg = _fold_object(obj, self.fold_angle, self.steps,
                               self.animate)
        except (crease.FoldError, crease.rigid.FoldFailure) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, msg)
        return {'FINISHED'}


_CLASSES = (MESH_OT_crease_pattern_add, OBJECT_OT_fold_solve)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    global _fold_previews
    # Blender leak-checks preview collections at shutdown, so the
    # gallery's must be handed back or unregistering warns.
    if _fold_previews is not None:
        try:
            bpy.utils.previews.remove(_fold_previews)
        except Exception:
            pass
        _fold_previews = None
    _pattern_items_cache.clear()
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
