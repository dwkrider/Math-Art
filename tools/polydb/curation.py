# Curated per-solid facts that cannot be computed from vertices and faces.
#
# Everything the geometry can decide is computed elsewhere; this module holds
# only what must come from the literature: the classical notation, the
# cross-reference acronyms, the family tags a human assigns, and the
# space-filling data.
#
# SPACE FILLING. Whether a polyhedron tiles space is not something to read off
# a vertex table, so it is curated -- but it is curated with PARAMETERS, and
# with a check the validator can run. Three distinctions matter and a plain
# boolean loses all of them:
#
#   * filling space ALONE versus in combination. The regular octahedron does
#     not tile space by itself; with tetrahedra it forms the alternated cubic
#     honeycomb. A boolean either lies about the octahedron or hides the
#     honeycomb.
#   * filling by TRANSLATIONS alone -- a parallelohedron. Fedorov proved there
#     are exactly five: cube, hexagonal prism, rhombic dodecahedron, elongated
#     dodecahedron and truncated octahedron. This is a much stronger property
#     than tiling, and it is the one relevant to lattices and packing.
#   * the LATTICE itself. Recording the translation basis makes the claim
#     falsifiable: for a parallelohedron the basis must span a cell of exactly
#     the solid's own volume, which tools/polydb_validate.py checks.
#
# References:
#   E. S. Fedorov (1885), "Nachala Ucheniya o Figurah" -- the five
#     parallelohedra.
#   P. R. Cromwell, "Polyhedra", Cambridge (1997), ch. 7 (space-filling).
#   H. S. M. Coxeter, "Regular Polytopes", 3rd ed. (1973), ch. 4 (honeycombs).
#   Wenninger; and the standard crystallographic space-group symbols.

import math

SQ2 = math.sqrt(2.0)

# -- classical notation, acronyms and family tags ---------------------------
#
# Keyed by slug. Only fields present are applied; the builder never overwrites
# a computed field with one of these.

CURATION = {
    # ---- Platonic -------------------------------------------------------
    "tetrahedron": {
        "families+": ["platonic", "regular", "deltahedron"],
        "notation": {"schlafli": "{3,3}", "conway": "T", "coxeter_diagram": "x3o3o"},
        "ids": {"bowers": "tet", "wikipedia": "Tetrahedron", "wolfram": "Tetrahedron"},
        "alternate_names": ["Triangular pyramid", "Regular tetrahedron"],
        "self_dual": True,
    },
    "octahedron": {
        "families+": ["platonic", "regular", "deltahedron"],
        "notation": {"schlafli": "{3,4}", "conway": "O", "coxeter_diagram": "x3o4o"},
        "ids": {"bowers": "oct", "wikipedia": "Octahedron", "wolfram": "Octahedron"},
        "alternate_names": ["Regular octahedron", "Square bipyramid", "Triangular antiprism"],
    },
    "cube": {
        "families+": ["platonic", "regular", "zonohedron", "space-filling", "prism"],
        "notation": {"schlafli": "{4,3}", "conway": "C", "coxeter_diagram": "x4o3o"},
        "ids": {"bowers": "cube", "wikipedia": "Cube", "wolfram": "Cube"},
        "alternate_names": ["Hexahedron", "Regular hexahedron", "Square prism"],
    },
    "icosahedron": {
        "families+": ["platonic", "regular", "deltahedron"],
        "notation": {"schlafli": "{3,5}", "conway": "I", "coxeter_diagram": "x3o5o"},
        "ids": {"bowers": "ike", "wikipedia": "Regular_icosahedron",
                "wolfram": "Icosahedron"},
        "alternate_names": ["Regular icosahedron"],
    },
    "dodecahedron": {
        "families+": ["platonic", "regular"],
        "notation": {"schlafli": "{5,3}", "conway": "D", "coxeter_diagram": "x5o3o"},
        "ids": {"bowers": "doe", "wikipedia": "Regular_dodecahedron",
                "wolfram": "Dodecahedron"},
        "alternate_names": ["Regular dodecahedron", "Pentagonal dodecahedron"],
    },
    # ---- Archimedean ----------------------------------------------------
    "truncated-tetrahedron": {
        "families+": ["archimedean"],
        "notation": {"schlafli": "t{3,3}", "conway": "tT", "coxeter_diagram": "x3x3o"},
        "ids": {"bowers": "tut", "wikipedia": "Truncated_tetrahedron",
                "wolfram": "TruncatedTetrahedron"},
    },
    "cuboctahedron": {
        "families+": ["archimedean", "quasiregular"],
        "notation": {"schlafli": "r{4,3}", "conway": "aC", "coxeter_diagram": "o4x3o"},
        "ids": {"bowers": "co", "wikipedia": "Cuboctahedron",
                "wolfram": "Cuboctahedron"},
        "alternate_names": ["Vector equilibrium"],
    },
    "truncated-cube": {
        "families+": ["archimedean"],
        "notation": {"schlafli": "t{4,3}", "conway": "tC", "coxeter_diagram": "x4x3o"},
        "ids": {"bowers": "tic", "wikipedia": "Truncated_cube",
                "wolfram": "TruncatedCube"},
        "alternate_names": ["Truncated hexahedron"],
    },
    "truncated-octahedron": {
        "families+": ["archimedean", "zonohedron", "space-filling", "permutohedron"],
        "notation": {"schlafli": "t{3,4}", "conway": "tO", "coxeter_diagram": "x3x4o"},
        "ids": {"bowers": "toe", "wikipedia": "Truncated_octahedron",
                "wolfram": "TruncatedOctahedron"},
        "alternate_names": ["Mecon", "Permutohedron of order 4"],
    },
    "rhombicuboctahedron": {
        "families+": ["archimedean"],
        "notation": {"schlafli": "rr{4,3}", "conway": "eC", "coxeter_diagram": "x4o3x"},
        "ids": {"bowers": "sirco", "wikipedia": "Rhombicuboctahedron",
                "wolfram": "SmallRhombicuboctahedron"},
        "alternate_names": ["Small rhombicuboctahedron"],
    },
    "truncated-cuboctahedron": {
        "families+": ["archimedean", "zonohedron"],
        "notation": {"schlafli": "tr{4,3}", "conway": "bC", "coxeter_diagram": "x4x3x"},
        "ids": {"bowers": "girco", "wikipedia": "Truncated_cuboctahedron",
                "wolfram": "GreatRhombicuboctahedron"},
        "alternate_names": ["Great rhombicuboctahedron", "Rhombitruncated cuboctahedron"],
    },
    "snub-cube": {
        "families+": ["archimedean", "chiral"],
        "notation": {"schlafli": "sr{4,3}", "conway": "sC", "coxeter_diagram": "s4s3s"},
        "ids": {"bowers": "snic", "wikipedia": "Snub_cube", "wolfram": "SnubCube"},
        "alternate_names": ["Snub cuboctahedron"],
    },
    "icosidodecahedron": {
        "families+": ["archimedean", "quasiregular"],
        "notation": {"schlafli": "r{5,3}", "conway": "aD", "coxeter_diagram": "o5x3o"},
        "ids": {"bowers": "id", "wikipedia": "Icosidodecahedron",
                "wolfram": "Icosidodecahedron"},
    },
    "truncated-dodecahedron": {
        "families+": ["archimedean"],
        "notation": {"schlafli": "t{5,3}", "conway": "tD", "coxeter_diagram": "x5x3o"},
        "ids": {"bowers": "tid", "wikipedia": "Truncated_dodecahedron",
                "wolfram": "TruncatedDodecahedron"},
    },
    "truncated-icosahedron": {
        "families+": ["archimedean", "goldberg", "fullerene"],
        "notation": {"schlafli": "t{3,5}", "conway": "tI", "coxeter_diagram": "x3x5o"},
        "ids": {"bowers": "ti", "wikipedia": "Truncated_icosahedron",
                "wolfram": "TruncatedIcosahedron"},
        "alternate_names": ["Buckyball", "Football", "Soccer ball",
                            "Goldberg polyhedron GP(1,1)"],
    },
    "rhombicosidodecahedron": {
        "families+": ["archimedean"],
        "notation": {"schlafli": "rr{5,3}", "conway": "eD", "coxeter_diagram": "x5o3x"},
        "ids": {"bowers": "srid", "wikipedia": "Rhombicosidodecahedron",
                "wolfram": "SmallRhombicosidodecahedron"},
        "alternate_names": ["Small rhombicosidodecahedron"],
    },
    "truncated-icosidodecahedron": {
        "families+": ["archimedean", "zonohedron"],
        "notation": {"schlafli": "tr{5,3}", "conway": "bD", "coxeter_diagram": "x5x3x"},
        "ids": {"bowers": "grid", "wikipedia": "Truncated_icosidodecahedron",
                "wolfram": "GreatRhombicosidodecahedron"},
        "alternate_names": ["Great rhombicosidodecahedron",
                            "Rhombitruncated icosidodecahedron"],
    },
    "snub-dodecahedron": {
        "families+": ["archimedean", "chiral"],
        "notation": {"schlafli": "sr{5,3}", "conway": "sD", "coxeter_diagram": "s5s3s"},
        "ids": {"bowers": "snid", "wikipedia": "Snub_dodecahedron",
                "wolfram": "SnubDodecahedron"},
        "alternate_names": ["Snub icosidodecahedron"],
    },
    # ---- Kepler-Poinsot -------------------------------------------------
    "small-stellated-dodecahedron": {
        "families+": ["kepler-poinsot", "regular", "star", "stellation"],
        "notation": {"schlafli": "{5/2,5}", "coxeter_diagram": "x5o5/2o"},
        "ids": {"bowers": "sissid", "wikipedia": "Small_stellated_dodecahedron",
                "wolfram": "SmallStellatedDodecahedron"},
    },
    "great-dodecahedron": {
        "families+": ["kepler-poinsot", "regular", "star"],
        "notation": {"schlafli": "{5,5/2}", "coxeter_diagram": "x5/2o5o"},
        "ids": {"bowers": "gad", "wikipedia": "Great_dodecahedron",
                "wolfram": "GreatDodecahedron"},
    },
    "great-stellated-dodecahedron": {
        "families+": ["kepler-poinsot", "regular", "star", "stellation"],
        "notation": {"schlafli": "{5/2,3}", "coxeter_diagram": "x3o5/2o"},
        "ids": {"bowers": "gissid", "wikipedia": "Great_stellated_dodecahedron",
                "wolfram": "GreatStellatedDodecahedron"},
    },
    "great-icosahedron": {
        "families+": ["kepler-poinsot", "regular", "star", "deltahedron"],
        "notation": {"schlafli": "{3,5/2}", "coxeter_diagram": "x5/2o3o"},
        "ids": {"bowers": "gike", "wikipedia": "Great_icosahedron",
                "wolfram": "GreatIcosahedron"},
    },
}


# -- space filling ----------------------------------------------------------
#
# `basis` is the translation lattice, given AT THIS DATABASE'S NORMALISATION
# (edge length 1 for these solids). For a parallelohedron the parallelepiped
# it spans has exactly the volume of one cell, which the validator checks:
#   |det(basis)| == volume * cells_per_lattice_point.

SPACE_FILLING = {
    "cube": {
        "fills_space_alone": True,
        "parallelohedron": True,
        "honeycomb": {"name": "Cubic honeycomb", "schlafli": "{4,3,4}",
                      "cells_per_edge": 4, "space_group": "Pm-3m",
                      "space_group_number": 221},
        "lattice": {"type": "primitive cubic",
                    "basis": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
        "cells_per_lattice_point": 1,
        "orientations": 1,
        "requires_reflections": False,
        "note": "One of Fedorov's five parallelohedra.",
    },
    "truncated-octahedron": {
        "fills_space_alone": True,
        "parallelohedron": True,
        "honeycomb": {"name": "Bitruncated cubic honeycomb", "schlafli": "2t{4,3,4}",
                      "cells_per_edge": 3, "space_group": "Im-3m",
                      "space_group_number": 229},
        # Voronoi cell of the body-centred cubic lattice. At edge 1 the
        # conventional cube side is a = 2*sqrt(2); the primitive bcc vectors
        # are a/2 * (-1,1,1) and cyclic.
        "lattice": {"type": "body-centred cubic",
                    "basis": [[-SQ2, SQ2, SQ2], [SQ2, -SQ2, SQ2], [SQ2, SQ2, -SQ2]]},
        "cells_per_lattice_point": 1,
        "orientations": 1,
        "requires_reflections": False,
        "note": "One of Fedorov's five parallelohedra; the Voronoi cell of the "
                "body-centred cubic lattice, and the permutohedron of order 4.",
    },
    "rhombic-dodecahedron": {
        "fills_space_alone": True,
        "parallelohedron": True,
        "honeycomb": {"name": "Rhombic dodecahedral honeycomb", "schlafli": None,
                      "cells_per_edge": None, "space_group": "Fm-3m",
                      "space_group_number": 225},
        "lattice": {"type": "face-centred cubic", "basis": None},
        "cells_per_lattice_point": 1,
        "orientations": 1,
        "requires_reflections": False,
        "note": "One of Fedorov's five parallelohedra; the Voronoi cell of the "
                "face-centred cubic lattice. The dual honeycomb is the "
                "tetrahedral-octahedral honeycomb.",
    },
    "octahedron": {
        "fills_space_alone": False,
        "parallelohedron": False,
        "combination_fillings": [
            {"with": ["tetrahedron"], "ratio": "1 octahedron : 2 tetrahedra",
             "honeycomb": "Tetrahedral-octahedral honeycomb",
             "schlafli": "h{4,3,4}"}],
        "note": "Does NOT tile space by itself -- a common error. With regular "
                "tetrahedra in the ratio 1:2 it forms the alternated cubic "
                "(tetrahedral-octahedral) honeycomb.",
    },
    "tetrahedron": {
        "fills_space_alone": False,
        "parallelohedron": False,
        "combination_fillings": [
            {"with": ["octahedron"], "ratio": "2 tetrahedra : 1 octahedron",
             "honeycomb": "Tetrahedral-octahedral honeycomb",
             "schlafli": "h{4,3,4}"}],
        "note": "Aristotle's claim that regular tetrahedra fill space is false; "
                "the error stood for centuries. See Cromwell, 'Polyhedra' (1997), "
                "ch. 7, and Lagarias & Zong on the packing history.",
    },
    "gyrobifastigium": {
        "fills_space_alone": True,
        "parallelohedron": False,
        "honeycomb": {"name": "Gyrobifastigium honeycomb", "schlafli": None,
                      "cells_per_edge": None, "space_group": None,
                      "space_group_number": None},
        "cells_per_lattice_point": None,
        "orientations": 2,
        "requires_reflections": False,
        "note": "The only Johnson solid that tiles three-dimensional space.",
    },
    "hexagonal-prism": {
        "fills_space_alone": True,
        "parallelohedron": True,
        "honeycomb": {"name": "Hexagonal prismatic honeycomb", "schlafli": "{6,3}x{}",
                      "cells_per_edge": None, "space_group": "P6/mmm",
                      "space_group_number": 191},
        "cells_per_lattice_point": 1,
        "orientations": 1,
        "requires_reflections": False,
        "note": "One of Fedorov's five parallelohedra.",
    },
    "triangular-prism": {
        "fills_space_alone": True,
        "parallelohedron": False,
        "honeycomb": {"name": "Triangular prismatic honeycomb",
                      "schlafli": "{3,6}x{}", "cells_per_edge": None,
                      "space_group": "P6/mmm", "space_group_number": 191},
        "orientations": 1,
        "requires_reflections": False,
        "note": "Tiles space because its triangular base tiles the plane; not a "
                "parallelohedron, since translations alone do not suffice.",
    },
}

FEDOROV_PARALLELOHEDRA = ("cube", "hexagonal-prism", "rhombic-dodecahedron",
                          "elongated-dodecahedron", "truncated-octahedron")


def for_slug(slug):
    return CURATION.get(slug, {})


def space_filling_for(slug):
    return SPACE_FILLING.get(slug)
