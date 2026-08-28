"""slicing -- the fabrication engine behind Slice for Fabrication.

Mesh in, cuttable sheets out.  Everything here is numpy-and-plain-Python
with no `bpy`, so the whole pipeline self-tests headlessly; the Blender
layer over it is `math_art/fabrication_slicer.py`.

    polyclip   2-D polygon toolkit: area, containment, the
               Weiler-Atherton difference that cuts a slot, miter
               offsetting for kerf, and dog-bone corner relief.
    sections   mesh intersect plane -> closed outlines, welded by EDGE
               IDENTITY rather than by coordinate, so the chaining is
               exact and an open surface is reported instead of being
               chained into nonsense.
    parts      outlines grouped with the holes nested inside them -- a
               torus slice is ONE part with a hole, not two parts.
    slots      the interlock: crossing lines, the span both pieces
               share, and the complementary half-slots that let them
               pass through one another and stop flush.
    glyphs     a stroke font, so an engraved part label is a path to
               follow rather than a shape to fill.
    layout     kerf compensation, nesting onto sheets, and the
               chirality convention.
    drawing    the device-independent layered drawing, and the cut
               order both exporters obey.
    svg, dxf   the two serializers, which know nothing about slicing.
    build      the orchestrator: scale to a real size, pick a
               technique, and report every joint that will not work.

The order above is the dependency order.
"""
