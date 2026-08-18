# Gem materials as Blender shader node groups.
#
# The bpy half of the gem engine: the optics live in `gems/optics.py` and
# `gems/materials.py`, which are numpy-and-Python only and testable
# without a renderer; this module turns their numbers into nodes.
#
# WHY A NODE GROUP AND NOT AN IOR.  Blender 5.1 has no dispersion.  The
# Glass BSDF's inputs are Color, Roughness, IOR, Normal, Weight and the
# two Thin Film sockets, and the Principled BSDF has no Abbe number
# either -- both were checked on the installed build, not assumed.  A
# gemstone with one IOR has no fire at all, and fire is most of what a
# cut stone is for.
#
# So dispersion is built rather than switched on: N Glass BSDFs, each at
# the index its own band sees, each masked to that band's share of the
# spectrum, summed.  Three bands is the classic RGB split; more bands
# reduce the colour banding that shows on a strongly dispersive stone,
# at a proportional cost in shading.  The band weights are normalised to
# sum to white, so raising the band count refines the fire without
# changing the exposure.
#
# ABSORPTION goes on the Volume socket, not into a surface tint, and that
# is the difference between colour that responds to the cut and colour
# painted on it.  Beer-Lambert attenuation depends on the path a ray
# takes through the stone, so a deep pavilion reads richer than a shallow
# one and a large stone darker than a small one of the same design --
# which is exactly the interaction between cut and colour that a
# gemstone generator exists to show.  The density is scaled by the
# stone's real size in millimetres against the 2 m proxy the mesh is
# fitted to.
#
# References:
#   Stephane Guy & Cyril Soler, "Graphics gems revisited: fast and
#     physically-based rendering of gemstones", ACM TOG 23(3), 2004 --
#     the three-channel treatment of dispersion and absorption this
#     node group is a shader-graph rendering of.
#   Alexander Wilkie et al., "Hero Wavelength Spectral Sampling",
#     Computer Graphics Forum 33(4), 2014 -- the spectrally correct
#     method, which is a renderer feature rather than something a node
#     group can provide; the band split here is the approximation it
#     would replace.

try:
    from ..gems import materials as gem_materials
    from ..gems import optics
except (ImportError, ValueError):        # flat import outside the package
    try:
        from gems import materials as gem_materials
        from gems import optics
    except ImportError:
        import materials as gem_materials
        import optics

try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

GROUP_PREFIX = "Gem Dispersion"


def band_plan(material_key, bands=3):
    """(band colour, index) pairs for a material -- the shader's recipe.

    Pure: this is the whole numeric content of the node group, so it can
    be checked without Blender.
    """
    m = gem_materials.get(material_key)
    idx = optics.band_indices(m.n_d, m.dispersion, bands)
    wts = optics.band_weights(bands)
    return [(w, n) for w, (_, n) in zip(wts, idx)]


if _IN_BLENDER:

    def _group_name(material_key, bands):
        return f"{GROUP_PREFIX} {material_key} x{bands}"

    def dispersion_group(material_key, bands=3):
        """Build (or reuse) the node group that splits a stone's fire."""
        name = _group_name(material_key, bands)
        if name in bpy.data.node_groups:
            return bpy.data.node_groups[name]

        g = bpy.data.node_groups.new(name, 'ShaderNodeTree')
        g.interface.new_socket("Color", in_out='INPUT',
                               socket_type='NodeSocketColor')
        g.interface.new_socket("Roughness", in_out='INPUT',
                               socket_type='NodeSocketFloat')
        g.interface.new_socket("BSDF", in_out='OUTPUT',
                               socket_type='NodeSocketShader')
        gin = g.nodes.new('NodeGroupInput')
        gin.location = (-500, 0)
        gout = g.nodes.new('NodeGroupOutput')

        plan = band_plan(material_key, bands)
        prev = None
        for k, (weight, n) in enumerate(plan):
            glass = g.nodes.new('ShaderNodeBsdfGlass')
            glass.location = (-260, 220 - k * 190)
            glass.inputs['IOR'].default_value = float(n)
            g.links.new(gin.outputs['Roughness'], glass.inputs['Roughness'])
            # the band's share of the spectrum, times the user's colour
            mul = g.nodes.new('ShaderNodeMixRGB')
            mul.blend_type = 'MULTIPLY'
            mul.location = (-440, 220 - k * 190)
            mul.inputs['Fac'].default_value = 1.0
            mul.inputs['Color2'].default_value = (*weight, 1.0)
            g.links.new(gin.outputs['Color'], mul.inputs['Color1'])
            g.links.new(mul.outputs['Color'], glass.inputs['Color'])
            if prev is None:
                prev = glass.outputs['BSDF']
            else:
                add = g.nodes.new('ShaderNodeAddShader')
                add.location = (-40, 220 - k * 190)
                g.links.new(prev, add.inputs[0])
                g.links.new(glass.outputs['BSDF'], add.inputs[1])
                prev = add.outputs['Shader']
        gout.location = (180, 0)
        g.links.new(prev, gout.inputs['BSDF'])
        return g

    def gem_material(material_key=None, size_mm=6.5, bands=3, color=None,
                     roughness=0.0, simple=False):
        """A Blender material for a gem species.

        `simple` returns a single Principled BSDF at the sodium-D index --
        no dispersion, but cheap and EEVEE-friendly for viewport work.
        The name says so, so a preview material cannot be mistaken for
        the real one in a render.
        """
        key = material_key or gem_materials.DEFAULT
        m = gem_materials.get(key)
        label = (f"Gem {m.label}" if not simple
                 else f"Gem {m.label} (preview, no dispersion)")
        mat = bpy.data.materials.new(label)
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        out.location = (300, 0)

        base = color if color is not None else (1.0, 1.0, 1.0, 1.0)
        if simple:
            p = nt.nodes.new('ShaderNodeBsdfPrincipled')
            p.location = (0, 0)
            p.inputs['Base Color'].default_value = base
            p.inputs['IOR'].default_value = m.n_d
            p.inputs['Roughness'].default_value = roughness
            p.inputs['Transmission Weight'].default_value = 1.0
            nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])
        else:
            grp = nt.nodes.new('ShaderNodeGroup')
            grp.node_tree = dispersion_group(key, bands)
            grp.location = (0, 0)
            grp.inputs['Color'].default_value = base
            grp.inputs['Roughness'].default_value = roughness
            nt.links.new(grp.outputs['BSDF'], out.inputs['Surface'])

        # Colour as absorption along the path, not as a surface tint.
        if any(a > 0.0 for a in m.alpha_rgb):
            vol = nt.nodes.new('ShaderNodeVolumeAbsorption')
            vol.location = (0, -260)
            dens = optics.absorption_density(m.alpha_rgb, size_mm)
            peak = max(dens) or 1.0
            # Blender's node takes one density and a colour; carry the
            # per-channel ratios in the colour and the magnitude in the
            # density.  exp(-a_i x) is then reproduced per channel.
            vol.inputs['Color'].default_value = (
                *[1.0 - (d / peak) * 0.999 for d in dens], 1.0)
            vol.inputs['Density'].default_value = float(peak)
            nt.links.new(vol.outputs['Volume'], out.inputs['Volume'])

        mat["gem_material"] = key
        mat["gem_bands"] = 0 if simple else bands
        mat["gem_size_mm"] = size_mm
        return mat

    def register():
        pass

    def unregister():
        pass

else:                                   # importable outside Blender
    def register():
        pass

    def unregister():
        pass


def _selftest():
    ok = True

    # --- the recipe, which is the whole numeric content -------------------
    plan = band_plan("DIAMOND", 3)
    good = len(plan) == 3 and plan[0][1] < plan[2][1]
    ok &= good
    print(f"styles.gem_material: diamond's three bands carry rising indices "
          f"{plan[0][1]:.4f} (red) to {plan[2][1]:.4f} (blue) "
          f"{'OK' if good else 'wrong'}")

    # the spread across the bands IS the fire, so it must track dispersion
    spread = {}
    for key in ("FLUORITE", "DIAMOND", "MOISSANITE"):
        p = band_plan(key, 3)
        spread[key] = p[2][1] - p[0][1]
    good = spread["FLUORITE"] < spread["DIAMOND"] < spread["MOISSANITE"]
    ok &= good
    print(f"styles.gem_material: index spread orders fluorite "
          f"{spread['FLUORITE']:.4f} < diamond {spread['DIAMOND']:.4f} < "
          f"moissanite {spread['MOISSANITE']:.4f} "
          f"{'OK' if good else 'wrong'}")

    # band weights must always sum to white, so band count is not exposure
    bad = []
    for nb in (1, 3, 6, 9):
        p = band_plan("DIAMOND", nb)
        for i in range(3):
            if abs(sum(w[i] for w, _ in p) - 1.0) > 1e-9:
                bad.append(nb)
                break
    good = not bad
    ok &= good
    print(f"styles.gem_material: 1, 3, 6 and 9 band recipes all sum to white "
          f"{'OK' if good else f'not at {bad}'}")

    # more bands must refine, not shift: the extremes stay put
    p3, p9 = band_plan("DIAMOND", 3), band_plan("DIAMOND", 9)
    good = abs(p3[0][1] - p9[0][1]) < 1e-9 and abs(p3[2][1] - p9[-1][1]) < 1e-9
    ok &= good
    print(f"styles.gem_material: raising the band count keeps the red and "
          f"violet ends fixed {'OK' if good else 'shifted the range'}")

    # --- absorption scales with the stone, which is the whole point -------
    a = gem_materials.get("SAPPHIRE").alpha_rgb
    d_small = optics.absorption_density(a, 3.0)
    d_large = optics.absorption_density(a, 12.0)
    good = all(abs(y - 4.0 * x) < 1e-12 for x, y in zip(d_small, d_large))
    ok &= good
    print(f"styles.gem_material: a 12 mm sapphire absorbs 4x a 3 mm one "
          f"{'OK' if good else 'does not scale'}")

    # a colourless species must not get a volume at all
    good = all(a == 0.0 for a in gem_materials.get("ROCK_CRYSTAL").alpha_rgb)
    ok &= good
    print(f"styles.gem_material: rock crystal is colourless, so it takes no "
          f"absorption {'OK' if good else 'tinted'}")

    if not _IN_BLENDER:
        print("styles.gem_material: node construction needs Blender -- "
              "covered by tests/test_gems.py")
    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("styles.gem_material self-test failed")
