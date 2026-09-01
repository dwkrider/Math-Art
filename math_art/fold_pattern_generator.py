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
    # TWO KRESLINGS, because Kresling gives two.  Belts may be stacked
    # all the same way or "inclined alternately to left and right", and
    # the two are one construction apart -- but they look nothing alike
    # flat, and they do different things: one turns as it deploys, the
    # other pumps straight along its axis.  Two thumbnails tell that
    # apart instantly where a Stacking dropdown would hide it, which is
    # the same reasoning that gives the hypar and the monkey saddle
    # separate entries.
    ('KRESZIG', "Kresling Zigzag",
     "Kresling with its bands inclined alternately left and right, so "
     "the creases zigzag and the diagonals mirror into Vs. The band "
     "twists cancel, so this tube pumps along its axis without turning"),
    ('TWIST', "Square Twist",
     "A central square with a pleat off each side; folding the pleats "
     "swings the square through a right angle and hides the surplus "
     "underneath. One molecule, not a tessellation, and the rigidly "
     "foldable variant -- its creases are not symmetric, which is what "
     "lets it fold"),
    ('RESCH', "Resch Triangular",
     "Ron Resch's triangular tessellation: every grid vertex inflated "
     "into a small triangular face, whose tucks hide surplus material "
     "and leave a stiff sheet that takes curvature both ways. Rows and "
     "Columns count inflated vertices. It does NOT fold flat -- a hub "
     "carries three mountains and three valleys, so Maekawa fails there "
     "by construction, and Report Checks will say so"),
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
    'KRESLING': 47.0,
    'KRESZIG': 47.0,
    'RESCH': 70.0,
    'TWIST': 90.0,
}

#: How many rows -- rings, for the concentric-pleat patterns -- each
#: pattern wants by default.  Same reasoning as `_NATURAL_FOLD`: one
#: shared number cannot suit all six.  Four rows is a readable sheet for
#: the tessellations, but a hypar at four rings is a few coarse steps
#: rather than a surface, and Demaine, Demaine and Lubiw say why --
#: "the more concentric squares one folds, the closer the pleated hypar
#: is to a true hypar surface".  Twelve reads as one while staying
#: quick to solve; the count is a straight legibility-against-cost
#: trade, so raise it when a render wants a finer pleat.
#:
#: `tools/bake_fold_icons.py` takes its ring count from HERE, so the
#: thumbnail and the default output cannot drift apart.
_NATURAL_ROWS = {
    'MIURA': 4,
    'ACCORDION': 4,
    'WATERBOMB': 4,
    'YOSHIMURA': 4,
    'HYPAR': 12,
    'MONKEY': 12,
    'KRESLING': 4,
    'KRESZIG': 4,
    'RESCH': 3,
    'TWIST': 1,
}

#: The panel angle each pattern wants, for the patterns the angle means
#: anything to -- and only those, since the control is hidden for the
#: rest.  The Kresling is the reason this table exists: its angle is not
#: a free shape knob but the one parameter of its closure condition, and
#: a cell too shallow for the chosen side count has NO deployed state at
#: all.  Sixty degrees suits the Miura and is right on the Kresling's
#: degenerate edge at six sides, so switching pattern has to move it.
_NATURAL_ANGLE = {
    'MIURA': 60.0,
    'KRESLING': 72.0,
    'KRESZIG': 72.0,
}

#: Which solver a pattern needs to become itself.  Rigid Panels is the
#: right default and is exact where it applies, but it drives every
#: crease along ONE continuation parameter -- so a pattern whose creases
#: want genuinely different angles can only get near its shape, never to
#: it.  The Kresling is that case: its targets run from 42 to 103
#: degrees, and under Rigid Panels the tube stalls half open (widest
#: extent 7.65 -> 4.95) where Bending Paper closes it (-> 3.71, seam
#: zero).  So the choice is per pattern rather than one global default.
_NATURAL_SOLVER = {'KRESLING': 'COMPLIANT', 'KRESZIG': 'COMPLIANT'}

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
    ang = _NATURAL_ANGLE.get(self.pattern)
    if ang is not None:
        self.panel_angle = np.deg2rad(ang)
    if hasattr(self, "solver"):
        self.solver = _NATURAL_SOLVER.get(self.pattern, 'RIGID')


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

    # THE ANGLES TRAVEL WITH THE MESH TOO, when the pattern knows them.
    # Writing the assignment and not the angle loses the difference
    # between "this crease is a valley" and "this crease is a valley of
    # 43 degrees", and the second is what tells the solver which of the
    # branches meeting at the flat state to take.  The Kresling folded
    # to a shallow crimp rather than its tube for exactly this reason:
    # the maker measured every angle off the target cylinder and the
    # mesh then dropped all of them on the floor.  Mapped by vertex pair
    # for the same reason as the assignment above -- `from_pydata`
    # reorders edges once faces are supplied.
    if frame.fold_angle is not None:
        ang = {}
        for k, (a, b) in enumerate(frame.edges):
            val = float(frame.fold_angle[k])
            val = 0.0 if val != val else val            # NaN means unknown
            ang[(int(a), int(b))] = ang[(int(b), int(a))] = val
        fa = me.attributes.new("fold_angle", 'FLOAT', 'EDGE')
        fa.data.foreach_set("value", [ang.get(tuple(e.vertices), 0.0)
                                      for e in me.edges])
    me.update()
    return me, scale, centre


def flat_source(obj):
    """The FLAT coordinates to fold from, or None if there are none.

    Folding starts from the flat crease pattern, always -- so an object
    that has already been folded needs its flat state back, not its
    current one.  Two cases, and the difference matters:

      * folded WITH animation: the fold lives in shape keys and the base
        mesh is still flat, so re-folding at another angle just works;
      * folded WITHOUT animation, or a mesh that was never flat at all
        (the corrugation's pleated form): the coordinates are genuinely
        3-D and there is nothing to fold from.

    The first used to fail as loudly as the second because this read the
    base mesh blindly.  It now prefers the "Flat" basis key.
    """
    me = obj.data
    keys = me.shape_keys
    if keys and keys.key_blocks:
        basis = keys.key_blocks[0]
        co = np.array([(p.co.x, p.co.y, p.co.z) for p in basis.data],
                      dtype=float)
        if float(np.ptp(co[:, 2])) < 1e-9:
            co[:, 2] = 0.0
            return co
    co = np.array([(v.co.x, v.co.y, v.co.z) for v in me.vertices],
                  dtype=float)
    if float(np.ptp(co[:, 2])) >= 1e-9:
        return None
    # A sheet lying in a plane OTHER than z = 0 is still flat; the
    # solver only needs it in the plane it works in.  Dropping the
    # constant offset is an isometry, so nothing about the pattern
    # changes -- and not doing it made a crease pattern that had merely
    # been MOVED look folded.
    co[:, 2] = 0.0
    return co


def _frame_from_object(obj):
    """Rebuild a crease.Frame from a mesh and its edge attributes."""
    me = obj.data
    flat = flat_source(obj)
    verts = (flat if flat is not None
             else np.array([(v.co.x, v.co.y, v.co.z) for v in me.vertices],
                           dtype=float))
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
    # Read the fold angles back too, when the mesh carries them.  A
    # pattern fitted to a surface knows the angle every crease must
    # reach; without them the solver can only guess a uniform one and
    # folds a different object.
    fa = me.attributes.get("fold_angle")
    angles = None
    if fa is not None:
        angles = np.array([float(d.value) for d in fa.data], dtype=float)
        if not np.any(np.abs(angles) > 1e-9):
            angles = None              # all zero means "not recorded"

    # A RECORDED ANGLE IS ALSO A SEED, and the seed has to be rebuilt
    # here or it is lost: the mesh carries per-edge attributes, not the
    # frame's `meta`, so a pattern that told the solver which branch to
    # leave the flat state along has that advice thrown away the moment
    # it becomes an object.  The rigid continuation then falls back to
    # the mountain/valley signs, which fold every crease at one rate --
    # for the Kresling that is a shallow crimp (widest extent 9.04 to
    # 7.06) instead of the tube (3.47), and the bug is invisible because
    # the crimp looks like a perfectly plausible fold.
    meta = {} if angles is None else {"fold_seed": np.nan_to_num(angles)}
    return crease.Frame(verts=verts, edges=edges, assignment=assign,
                        fold_angle=angles, meta=meta,
                        faces=faces or None)


#: Vertex-colour attribute the strain display writes.  A colour
#: attribute rather than a float one so it shows in the viewport without
#: the user having to build a material first -- same choice, and the
#: same name discipline, as `curvature_color.py`.
_STRAIN_ATTR = "Fold Strain"


def _paint_strain(obj, strain, clamp=None):
    """Write per-vertex strain as a colour ramp; return the range used.

    WHITE is unstrained, RED is stretched, BLUE is compressed.  Paper
    does not stretch, so on a rigid-foldable pattern this should come
    out essentially white everywhere -- and a hot band is the model
    saying where the sheet is fighting itself: a jam, an
    over-constrained vertex, or a target angle it cannot reach.
    """
    import numpy as _np
    me = obj.data
    s = _np.asarray(strain, dtype=float)
    if len(s) != len(me.vertices):
        return None
    lim = float(clamp) if clamp else float(_np.abs(s).max())
    if not _np.isfinite(lim) or lim <= 0.0:
        lim = 1e-9
    t = _np.clip(s / lim, -1.0, 1.0)
    cols = _np.ones((len(s), 4))
    tp = _np.clip(t, 0.0, None)[:, None]
    tn = _np.clip(-t, 0.0, None)[:, None]
    hot = _np.array([0.85, 0.12, 0.10])
    cold = _np.array([0.12, 0.30, 0.85])
    cols[:, :3] = 1.0 + tp * (hot - 1.0) + tn * (cold - 1.0)

    attr = me.color_attributes.get(_STRAIN_ATTR)
    if attr is not None and (attr.domain != 'POINT'
                             or attr.data_type != 'FLOAT_COLOR'):
        me.color_attributes.remove(attr)
        attr = None
    if attr is None:
        attr = me.color_attributes.new(_STRAIN_ATTR, 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set("color", cols.ravel())
    me.color_attributes.active_color = attr
    me.update()
    return lim


def _compliant_setup(obj):
    """Build the compliant folder for `obj`, triangulating if needed.

    Split out from `_compliant_fold` so a modal operator can own the
    stepping: the solve has to be interruptible to show progress, and it
    cannot be interruptible while it lives inside one blocking call.
    """
    import numpy as _np
    frame = _frame_from_object(obj)
    if frame.faces is None:
        frame.faces = crease.build_faces(frame.verts, frame.edges)
    added = 0
    if any(len(f) != 3 for f in frame.faces):
        tris, diags = crease.triangulate(frame.verts, frame.faces)
        frame.faces = tris
        if diags:
            frame.edges = _np.vstack(
                [frame.edges, _np.array(diags, dtype=_np.int64)])
            frame.assignment = _np.concatenate(
                [frame.assignment,
                 _np.array([crease.UNASSIGNED] * len(diags), dtype="<U1")])
            added = len(diags)
    return crease.compliant.CompliantFolder(frame), added


def _compliant_finish(obj, cf, states, added, drive, colour_strain):
    """Apply a finished compliant solve to the object; return the report."""
    import numpy as _np
    if states:
        _apply_states(obj, states)
    else:
        for v, p in zip(obj.data.vertices, cf.pos):
            v.co = Vector(p)
        obj.data.update()
    obj["fold_is_flat"] = False
    msg = (f"compliant fold to {drive * 100:.0f}% of target; "
           f"max strain {float(_np.abs(cf.edge_strain()).max()):.2e}")
    if added:
        msg += f"; {added} panel diagonal(s) added to allow bending"
    if colour_strain:
        lim = _paint_strain(obj, cf.vertex_strain())
        if lim is not None:
            msg += f"; strain shown to +/-{lim:.2e}"
    return msg


def _compliant_fold(obj, drive, steps, animate, colour_strain,
                    progress=None):
    """Fold `obj` with the compliant solver.

    Triangulates first when it has to.  A quad panel has no interior
    bending freedom, so a compliant model would hold it just as flat as
    the rigid one does -- which would make choosing this solver look
    like it had done nothing.
    """
    import numpy as _np
    frame = _frame_from_object(obj)
    if frame.faces is None:
        frame.faces = crease.build_faces(frame.verts, frame.edges)
    added = 0
    if any(len(f) != 3 for f in frame.faces):
        tris, diags = crease.triangulate(frame.verts, frame.faces)
        frame.faces = tris
        if diags:
            frame.edges = _np.vstack(
                [frame.edges, _np.array(diags, dtype=_np.int64)])
            frame.assignment = _np.concatenate(
                [frame.assignment,
                 _np.array([crease.UNASSIGNED] * len(diags), dtype="<U1")])
            added = len(diags)

    cf = crease.compliant.CompliantFolder(frame)
    if animate:
        # One shape key per sampled state, as the rigid path does, so
        # the two solvers animate through the same `fold_t` dial.
        n = max(2, int(steps))
        states = []
        per = max(200, cf.settle_steps // n)
        for i in range(n):
            d = drive * (i + 1) / n
            for _ in range(per):
                cf.step(d)
            states.append(cf.pos.copy())
            if progress is not None:
                progress((i + 1) / n)
        _apply_states(obj, states)
    else:
        cf.run(drive=drive, progress=progress)
        for v, p in zip(obj.data.vertices, cf.pos):
            v.co = Vector(p)
        obj.data.update()

    obj["fold_is_flat"] = False
    msg = (f"compliant fold to {drive * 100:.0f}% of target; "
           f"max strain {float(_np.abs(cf.edge_strain()).max()):.2e}")
    if added:
        msg += f"; {added} panel diagonal(s) added to allow bending"
    if colour_strain:
        lim = _paint_strain(obj, cf.vertex_strain())
        if lim is not None:
            msg += f"; strain shown to +/-{lim:.2e}"
    return msg


def _apply_states(obj, states):
    """Cache a list of vertex-position arrays as driven shape keys."""
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
    solver: EnumProperty(
        name="Solver", default='RIGID',
        items=[
            ('RIGID', "Rigid Panels",
             "Panels stay perfectly flat and only the creases bend. "
             "Exact for rigid-foldable patterns like the Miura, and the "
             "right choice when you want clean faceted geometry"),
            ('COMPLIANT', "Bending Paper",
             "Let the paper bend between creases, as real paper does. "
             "Slower, and the only way to reach a pattern whose creases "
             "want genuinely different angles -- the Kresling's tube "
             "closes under this and not under Rigid Panels"),
        ],
        description="How the sheet is allowed to deform while folding")
    fold_angle: FloatProperty(
        name="Fold Angle", default=np.deg2rad(70.0),
        min=np.deg2rad(-179.0), max=np.deg2rad(179.0), subtype='ANGLE',
        description="Dihedral angle to drive the pattern to")
    drive: FloatProperty(
        name="Fold Amount", default=1.0, min=0.0, max=1.0, subtype='FACTOR',
        description="How far toward each crease's target angle to drive "
                    "the sheet, when the solver is Bending Paper")
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
        if self.pattern in _NATURAL_ANGLE:
            L.prop(self, "panel_angle")
        L.prop(self, "size")
        L.prop(self, "check")
        L.separator()
        L.prop(self, "auto_fold")
        if self.auto_fold:
            L.prop(self, "solver")
            L.prop(self, "drive" if self.solver == 'COMPLIANT'
                   else "fold_angle")
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

        # A maker that already knows its faces keeps them.  The Kresling
        # measures its fold angles against that exact triangulation, so
        # re-deriving the faces here would be re-deriving the thing the
        # targets were computed from.
        if frame.faces is None:
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
                if self.solver == 'COMPLIANT':
                    msg += "; " + _compliant_fold(
                        obj, self.drive, self.steps, self.animate, False)
                else:
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
    solver: EnumProperty(
        name="Solver", default='RIGID',
        items=[
            ('RIGID', "Rigid Panels",
             "Panels stay perfectly flat and only the creases bend. "
             "Exact for rigid-foldable patterns like the Miura, and the "
             "right choice when you want clean faceted geometry"),
            ('COMPLIANT', "Bending Paper",
             "Let the paper bend between creases, as real paper does. "
             "Slower and approximate, but it is the only way to fold "
             "patterns that have no rigid folding at all -- the pleated "
             "hypar is proved to be one (Demaine et al. 2011)"),
        ],
        description="How the sheet is allowed to deform while folding")
    drive: FloatProperty(
        name="Fold Amount", default=0.8, min=0.0, max=1.0, subtype='FACTOR',
        description="How far toward each crease's target angle to drive "
                    "the sheet, for the bending-paper solver")
    colour_strain: BoolProperty(
        name="Colour by Strain", default=False,
        description="Paint each vertex by how much the paper is stretched "
                    "there -- white none, red stretched, blue compressed. "
                    "Paper does not stretch, so a hot band shows where the "
                    "pattern is fighting itself")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def draw(self, context):
        L = self.layout
        L.use_property_split = True
        L.prop(self, "solver")
        if self.solver == 'RIGID':
            L.prop(self, "fold_angle")
        else:
            L.prop(self, "drive")
        L.prop(self, "steps")
        L.prop(self, "animate")
        L.prop(self, "colour_strain")


    def execute(self, context):
        obj = context.active_object
        try:
            if self.solver == 'COMPLIANT':
                msg = _compliant_fold(obj, self.drive, self.steps,
                                      self.animate, self.colour_strain)
            else:
                msg = _fold_object(obj, self.fold_angle, self.steps,
                                   self.animate)
                if self.colour_strain:
                    # Strain is a compliant-solver quantity: the rigid
                    # model asserts the panels do not deform, so there
                    # is nothing to colour and pretending otherwise
                    # would draw a field of exact zeros.
                    msg += ("; strain colouring needs the bending-paper "
                            "solver -- the rigid one holds panels exactly "
                            "rigid, so its strain is zero by definition")
        except (crease.FoldError, crease.rigid.FoldFailure,
                crease.compliant.CompliantFailure) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, msg)
        return {'FINISHED'}



# --------------------------------------------------------------------
# The fold panel
# --------------------------------------------------------------------
# WHY THE SETTINGS LEFT THE OPERATOR.  An Add-menu operator runs the
# moment it is chosen, and Blender's redo panel then re-runs it on every
# property change.  For the rigid solver that is fine -- a fifth of a
# second -- but the compliant one takes eight, so choosing "Fold
# Pattern" started an eight-second solve before the user had touched a
# setting, and every subsequent tweak started another.  The redo panel
# is the wrong control surface for anything expensive: it assumes
# re-running is cheap.
#
# So the settings live on the scene, in a sidebar panel, and nothing
# folds until the button is pressed.  `object.fold_solve` keeps its own
# properties and stays scriptable -- the tests and the icon baker call
# it directly -- but it is no longer in the Add menu, because an entry
# there is a promise that choosing it is safe.


class MathArtFoldSettings(bpy.types.PropertyGroup):
    solver: EnumProperty(
        name="Solver", default='RIGID',
        items=[
            ('RIGID', "Rigid Panels",
             "Panels stay perfectly flat and only the creases bend. "
             "Exact for rigid-foldable patterns like the Miura, and fast"),
            ('COMPLIANT', "Bending Paper",
             "Let the paper bend between creases, as real paper does. "
             "Slower, and the only way to fold patterns that have no "
             "rigid folding at all -- the pleated hypar is proved to be "
             "one (Demaine et al. 2011)"),
        ],
        description="How the sheet is allowed to deform while folding")
    fold_angle: FloatProperty(
        name="Fold Angle", default=np.deg2rad(70.0),
        min=np.deg2rad(-179.0), max=np.deg2rad(179.0), subtype='ANGLE',
        description="Dihedral angle to drive the pattern to")
    drive: FloatProperty(
        name="Fold Amount", default=0.8, min=0.0, max=1.0, subtype='FACTOR',
        description="How far toward each crease's target angle to drive "
                    "the sheet, for the bending-paper solver")
    steps: IntProperty(
        name="Steps", default=12, min=1, max=120,
        description="States solved along the fold path; each becomes a "
                    "shape key, so this is also the animation resolution")
    animate: BoolProperty(
        name="Animate", default=True,
        description="Cache the whole path as shape keys driven by a "
                    "single Fold property, instead of only the end state")
    colour_strain: BoolProperty(
        name="Colour by Strain", default=False,
        description="Paint each vertex by how much the paper is stretched "
                    "there. Needs the bending-paper solver")


class VIEW3D_PT_math_art_fold(bpy.types.Panel):
    bl_label = "Origami Fold"
    bl_idname = "VIEW3D_PT_math_art_fold"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Math Art"
    bl_order = 4

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and obj.data.attributes.get("crease_assignment") is not None)

    def draw_header(self, context):
        self.layout.label(icon='MOD_SIMPLEDEFORM')

    def draw(self, context):
        st = context.scene.math_art_fold
        lay = self.layout
        lay.use_property_split = True

        lay.prop(st, "solver")
        if st.solver == 'RIGID':
            lay.prop(st, "fold_angle")
        else:
            lay.prop(st, "drive")
        lay.prop(st, "steps")
        lay.prop(st, "animate")
        lay.prop(st, "colour_strain")
        if st.colour_strain and st.solver == 'RIGID':
            # Say it here rather than after an eight-second solve.
            lay.label(text="Strain needs Bending Paper", icon='INFO')

        lay.separator()
        obj = context.active_object
        foldable = obj is not None and flat_source(obj) is not None
        row = lay.row()
        row.scale_y = 1.5
        row.enabled = foldable
        row.operator("object.fold_run", icon='PLAY',
                     text="Fold" if st.solver == 'RIGID'
                     else "Fold (Esc cancels)")
        if not foldable:
            # Say it here rather than letting the button fail.  The
            # corrugation's pleated form lands in exactly this state --
            # it carries creases, so this panel appears for it, but it
            # is a folded shape and folding starts from the flat pattern.
            col = lay.column(align=True)
            col.label(text="This mesh is already folded", icon='INFO')
            col.label(text="Fold the flat crease pattern instead")

        if obj is not None and "fold_t" in obj:
            lay.separator()
            lay.prop(obj, '["fold_t"]', text="Fold Progress", slider=True)


class OBJECT_OT_fold_run(bpy.types.Operator):
    """Fold the active crease pattern using the panel's settings"""

    bl_idname = "object.fold_run"
    bl_label = "Fold"
    # NO properties of its own, deliberately: an operator with none has
    # an empty redo panel, so Blender has nothing to re-run it for.
    bl_options = {'REGISTER', 'UNDO'}

    # THE MODAL LOOP LIVES HERE, in the operator the button actually
    # invokes.  It used to live in `fold_solve`, with this one calling
    # that one via INVOKE_DEFAULT and passing its {'RUNNING_MODAL'}
    # straight back -- which does nothing, because a modal handler
    # belongs to the operator that registered it and THIS operator had
    # registered none.  Blender simply ended the operator, the timer
    # never ticked, and no progress appeared.  An operator cannot
    # delegate being modal.
    #
    # PROGRESS IS REPORTED IN THE STATUS BAR, NOT ON THE CURSOR.
    # `wm.progress_begin/update` replaces the mouse pointer with a
    # spinning number, which is unpleasant to work under and hides the
    # pointer just when you might want to click Esc.  The status text
    # says the same thing without commandeering the cursor, and it needs
    # the same modal loop to be visible at all.

    _timer = None
    _cf = None
    _states = None
    _obj = None
    _i = 0
    _n = 0
    _per = 0
    _added = 0
    _drive = 1.0
    _colour = False

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def _blocking(self, context, st):
        """Everything except the compliant solve, which is fast."""
        obj = context.active_object
        try:
            if st.solver == 'COMPLIANT':
                msg = _compliant_fold(obj, st.drive, st.steps, st.animate,
                                      st.colour_strain)
            else:
                msg = _fold_object(obj, st.fold_angle, st.steps, st.animate)
                if st.colour_strain:
                    msg += ("; strain colouring needs the bending-paper "
                            "solver -- the rigid one holds panels exactly "
                            "rigid, so its strain is zero by definition")
        except (crease.FoldError, crease.rigid.FoldFailure,
                crease.compliant.CompliantFailure) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, msg)
        return {'FINISHED'}

    def execute(self, context):
        # Scripted path: no event loop, so do the whole thing at once.
        return self._blocking(context, context.scene.math_art_fold)

    def invoke(self, context, event):
        st = context.scene.math_art_fold
        if st.solver != 'COMPLIANT':
            return self._blocking(context, st)     # rigid is sub-second

        obj = context.active_object
        try:
            self._cf, self._added = _compliant_setup(obj)
        except (crease.FoldError, crease.compliant.CompliantFailure) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        self._obj = obj
        self._drive = st.drive
        self._colour = st.colour_strain
        self._states = [] if st.animate else None
        self._n = max(2, int(st.steps)) if st.animate else 40
        self._per = max(200, self._cf.settle_steps // self._n)
        self._i = 0

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        context.workspace.status_text_set("Folding 0%  |  Esc to cancel")
        return {'RUNNING_MODAL'}

    def _cleanup(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            self._cleanup(context)
            self.report({'WARNING'},
                        "fold cancelled; the sheet is left part-folded")
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'RUNNING_MODAL'}

        d = self._drive * (self._i + 1) / self._n
        for _ in range(self._per):
            self._cf.step(d)
        if self._states is not None:
            self._states.append(self._cf.pos.copy())
        self._i += 1

        frac = self._i / self._n
        context.workspace.status_text_set(
            f"Folding {frac * 100:.0f}%  |  Esc to cancel")

        if self._i < self._n:
            return {'RUNNING_MODAL'}

        self._cleanup(context)
        try:
            msg = _compliant_finish(self._obj, self._cf, self._states,
                                    self._added, self._drive, self._colour)
        except Exception as exc:                      # noqa: BLE001
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, msg)
        return {'FINISHED'}


_CLASSES = (MESH_OT_crease_pattern_add, OBJECT_OT_fold_solve,
            OBJECT_OT_fold_run, MathArtFoldSettings,
            VIEW3D_PT_math_art_fold)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.math_art_fold = bpy.props.PointerProperty(
        type=MathArtFoldSettings)


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
    if hasattr(bpy.types.Scene, "math_art_fold"):
        del bpy.types.Scene.math_art_fold
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
