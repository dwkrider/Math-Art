
# The Math Art tab: edit the selected generated object.
#
# The panel does not lay out any controls of its own.  It hands its
# layout to the settings group -- which carries the generator's own
# `draw()` -- so what appears in the sidebar is the same UI, in the same
# order, with the same tooltips and the same show-when-relevant logic as
# the redo panel that follows Add > Math Art.  Parity is structural
# rather than maintained.
#
# Around that it adds only what editing an existing object needs and
# creating a new one does not: whether to rebuild as you drag, whether
# the geometry is behind the settings, and a way out.

try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

try:
    from . import build, clone
    from .registry import root_for_object, settings_for_object
except ImportError:                     # flat import outside the package
    import build
    import clone
    from registry import root_for_object, settings_for_object


def target(context):
    """(root, info, settings) for whatever is selected, or (None,)*3.

    Selecting any piece of a multi-object build edits that build, so
    everything the panel offers resolves to the group's root first.
    """
    root = root_for_object(getattr(context, 'active_object', None))
    if root is None:
        return None, None, None
    info, pg = settings_for_object(root)
    if info is None or pg is None:
        return None, None, None
    return root, info, pg


if _IN_BLENDER:

    class MATHART_OT_rebuild(bpy.types.Operator):
        """Rebuild this object from its Math Art settings"""
        bl_idname = "math_art.rebuild"
        bl_label = "Rebuild"
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            return target(context)[0] is not None

        def execute(self, context):
            obj, _info, _pg = target(context)
            try:
                built = build.rebuild(obj, context)
            except build.LiveError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            elements, faces = build._geometry_counts(built)
            self.report({'INFO'}, "%s: %d elements, %d faces, %.2fs"
                        % (built.name, elements, faces,
                           built.math_art.last_seconds))
            return {'FINISHED'}

    class MATHART_OT_reset_settings(bpy.types.Operator):
        """Put every setting back to this generator's defaults"""
        bl_idname = "math_art.reset_settings"
        bl_label = "Reset to Defaults"
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            return target(context)[0] is not None

        def execute(self, context):
            obj, _info, pg = target(context)
            # One rebuild at the end, not one per property: resetting a
            # hundred settings would otherwise be a hundred builds.
            build._BUILDING[0] = True
            try:
                clone.reset_to_defaults(pg)
            finally:
                build._BUILDING[0] = False
            try:
                build.rebuild(obj, context)
            except build.LiveError as exc:
                self.report({'WARNING'}, str(exc))
            return {'FINISHED'}

    class MATHART_OT_detach(bpy.types.Operator):
        """Forget this object's generator, keeping the geometry

        The Math Art panel stops offering to rebuild it, so the mesh can
        be edited by hand without a later settings change overwriting the
        work."""
        bl_idname = "math_art.detach"
        bl_label = "Detach from Generator"
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            return target(context)[0] is not None

        def execute(self, context):
            obj = target(context)[0]
            settings = obj.math_art
            # The settings themselves are left in place: re-attaching is
            # then a matter of restoring one name, and nothing about the
            # object the user can see has changed.
            settings.generator = ''
            for entry in settings.members:
                if entry.obj is not None:
                    entry.obj.math_art.group_root = None
            settings.members.clear()
            self.report({'INFO'}, "%s is no longer linked to a generator"
                        % obj.name)
            return {'FINISHED'}

    class VIEW3D_PT_math_art_object(bpy.types.Panel):
        bl_label = "Generator"
        bl_idname = "VIEW3D_PT_math_art_object"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "Math Art"
        bl_order = 0

        @classmethod
        def poll(cls, context):
            return target(context)[0] is not None

        def draw_header(self, context):
            self.layout.label(icon='MATSHADERBALL')

        def draw(self, context):
            lay = self.layout
            obj, info, pg = target(context)
            settings = obj.math_art

            title = lay.row()
            title.label(text=info.label)

            # Selecting a companion edits the assembly it belongs to, so
            # say which object is actually being edited rather than
            # leaving the panel looking like it belongs to the piece.
            if obj is not context.active_object:
                lay.label(text="Part of %s" % obj.name, icon='LINKED')

            stale = build.is_stale(obj)

            row = lay.row(align=True)
            row.prop(settings, 'autobuild', toggle=True,
                     icon='AUTO' if settings.autobuild else 'FILE_REFRESH')
            act = row.row(align=True)
            act.alert = stale
            act.scale_y = 1.3 if stale else 1.0
            act.operator("math_art.rebuild", icon='FILE_REFRESH',
                         text="Rebuild *" if stale else "Rebuild")

            if settings.n_created > 1:
                lay.label(text="Builds %d objects" % settings.n_created,
                          icon='OUTLINER_OB_GROUP_INSTANCE')
            if stale and not settings.autobuild:
                lay.label(text="Settings changed since the last build",
                          icon='INFO')

            if build.is_hand_edited(obj):
                warn = lay.column(align=True)
                warn.alert = True
                warn.label(text="Geometry was edited by hand",
                           icon='ERROR')
                warn.label(text="A rebuild will discard those edits")
                warn.operator("math_art.detach", icon='UNLINKED')

            if settings.last_seconds:
                lay.label(text="Last build %.2fs" % settings.last_seconds,
                          icon='TIME')

            lay.separator()
            self._draw_settings(context, lay, info, pg)

            lay.separator()
            lay.operator("math_art.reset_settings", icon='LOOP_BACK')
            if not build.is_hand_edited(obj):
                lay.operator("math_art.detach", icon='UNLINKED')

        def _draw_settings(self, context, lay, info, pg):
            """Draw the generator's own creation UI, or say why not.

            The generator's `draw` is written for a redo panel, where a
            failure is a traceback in the console and a missing box.  In
            a sidebar it would be a tab that will not open, so a failure
            falls back to the plain property list rather than taking the
            panel down.
            """
            box = lay.column()
            if 'draw' in info.pg_cls.__dict__:
                info.pg_cls.layout = box
                try:
                    info.pg_cls.draw(pg, context)
                    return
                except Exception as exc:
                    box.alert = True
                    box.label(text="%s: %s" % (type(exc).__name__, exc),
                              icon='ERROR')
                finally:
                    info.pg_cls.layout = None
            column = box.column()
            column.use_property_split = True
            for prop in clone.editable_props(info.pg_cls):
                column.prop(pg, prop.identifier)

    CLASSES = (MATHART_OT_rebuild, MATHART_OT_reset_settings,
               MATHART_OT_detach, VIEW3D_PT_math_art_object)


def _selftest():
    """The panel is pure Blender UI; what is checkable here is that the
    module imports headlessly, so the extension's own test runner and
    any flat import keep working."""
    ok = not _IN_BLENDER or bool(CLASSES)
    print("live.panel: module imports without Blender present")
    print("RESULT:", "OK" if ok else "BAD")
    assert ok
