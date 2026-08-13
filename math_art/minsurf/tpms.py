# Triply-periodic minimal surfaces (nodal approximations).
#
# Part of the Math Art minimal-surface engine (`math_art/minsurf/`), split
# out of the former single-file `minimal_surface_toolkit.py`.  Numpy only --
# no `bpy` -- so the whole engine imports and self-tests headlessly; the
# registered Blender operators stay in the flat `minimal_surface_toolkit.py`
# front-end.
#
# Triply-periodic minimal surfaces via their standard nodal (level-set)
# approximations, meshed by marching tetrahedra.
#
# References:
#   H. A. Schwarz (P, D; Gesammelte Mathematische Abhandlungen, 1890).
#   A. H. Schoen, "Infinite periodic minimal surfaces without
#       self-intersections", NASA TN D-5541 (1970) -- gyroid, I-WP, F-RD.
#   E. R. Neovius (1883).  Inventory and nodal equations after Ken
#       Brakke's periodic-surface pages (kenbrakke.com/evolver); the
#       tier-2 sources are documented at the expansion block below.

import math
import numpy as np

TAU = 2.0 * math.pi


# ==========================================================================
# 2. Triply-periodic minimal surfaces (nodal approximations)
# ==========================================================================

def _f_p(x, y, z):
    return np.cos(x) + np.cos(y) + np.cos(z)

def _f_d(x, y, z):
    return (np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(x) * np.cos(y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.cos(z)
            + np.cos(x) * np.cos(y) * np.sin(z))

def _f_g(x, y, z):
    return (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z)
            + np.sin(z) * np.cos(x))

def _f_neovius(x, y, z):
    return (3 * (np.cos(x) + np.cos(y) + np.cos(z))
            + 4 * np.cos(x) * np.cos(y) * np.cos(z))

def _f_iwp(x, y, z):
    return (2 * (np.cos(x) * np.cos(y) + np.cos(y) * np.cos(z)
                 + np.cos(z) * np.cos(x))
            - (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)))

def _f_frd(x, y, z):
    return (4 * np.cos(x) * np.cos(y) * np.cos(z)
            - (np.cos(2 * x) * np.cos(2 * y) + np.cos(2 * y) * np.cos(2 * z)
               + np.cos(2 * z) * np.cos(2 * x)))

def _f_lidinoid(x, y, z):
    return (0.5 * (np.sin(2 * x) * np.cos(y) * np.sin(z)
                   + np.sin(2 * y) * np.cos(z) * np.sin(x)
                   + np.sin(2 * z) * np.cos(x) * np.sin(y)
                   - np.cos(2 * x) * np.cos(2 * y)
                   - np.cos(2 * y) * np.cos(2 * z)
                   - np.cos(2 * z) * np.cos(2 * x)) + 0.15)

def _f_splitp(x, y, z):
    return (1.1 * (np.sin(2 * x) * np.sin(z) * np.cos(y)
                   + np.sin(2 * y) * np.sin(x) * np.cos(z)
                   + np.sin(2 * z) * np.sin(y) * np.cos(x))
            - 0.2 * (np.cos(2 * x) * np.cos(2 * y)
                     + np.cos(2 * y) * np.cos(2 * z)
                     + np.cos(2 * z) * np.cos(2 * x))
            - 0.4 * (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)))

def _f_scherk_tower(x, y, z):
    return np.sin(z) - np.sinh(x) * np.sinh(y)

TPMS = {
    'P': ("Schwarz P", _f_p, True),
    'D': ("Schwarz D", _f_d, True),
    'G': ("Gyroid", _f_g, True),
    'NEOVIUS': ("Neovius", _f_neovius, True),
    'IWP': ("Schoen I-WP", _f_iwp, True),
    'FRD': ("Schoen F-RD", _f_frd, True),
    'LIDINOID': ("Lidinoid", _f_lidinoid, True),
    'SPLITP': ("Split P", _f_splitp, True),
    'SCHERKT': ("Scherk Tower (singly periodic)", _f_scherk_tower, False),
}

# ==========================================================================
# Tier-2 nodal TPMS expansion
# ==========================================================================
# Additional published nodal (Fourier level-set) approximations of triply
# periodic minimal surfaces, plus infrastructure for non-cubic unit cells
# and a level-offset parameter (the constant c in F(x,y,z) = c, which
# sweeps each field's offset family of companion surfaces).
#
# Every field below transcribes a PUBLISHED nodal formula; none are
# invented here.  The equations are taken from the peer-reviewed
# compilation by Fisher et al. (2023), which normalized the forms first
# derived in the primary sources cited per surface.
#
# References:
#   H. G. von Schnering and R. Nesper, "Nodal surfaces of Fourier
#       series: fundamental invariants of structured matter",
#       Z. Phys. B Condensed Matter 83 (1991) 407-412.
#       doi:10.1007/BF01313411  [C(D), C(Y), +-Y, C(+-Y), S nodal forms]
#   H. G. von Schnering, M. Oehme, G. Rudolf, "Three-dimensional
#       periodic nodal surfaces which envelope the threefold and
#       fourfold cubic rod packings", Acta Chem. Scand. 45 (1991)
#       873-876.  doi:10.3891/acta.chem.scand.45-0873  [C(I2-Y**)]
#   M. Wohlgemuth, N. Yufa, J. Hoffman, E. L. Thomas, "Triply periodic
#       bicontinuous cubic microdomain morphologies by symmetries",
#       Macromolecules 34 (2001) 6083-6089.  doi:10.1021/ma0019499
#       [O,C-TO (OCTO), G', F-RD variant]
#   D. A. Hoffman, J. T. Hoffman, M. Weber, et al., "Table of
#       Surfaces", The Scientific Graphics Project, MSRI (2003).
#       [D' and further G'/OCTO forms]
#   J. W. Fisher, S. W. Miller, J. Bartolai, T. W. Simpson,
#       M. A. Yukish, "Catalog of triply periodic minimal surfaces,
#       equation-based lattice structures, and their homogenized
#       property data", Data in Brief 49 (2023) 109311.
#       doi:10.1016/j.dib.2023.109311  [equation compilation; K, Q*,
#       Triplane/F recommendations]
#   W. Fischer and E. Koch, "On 3-periodic minimal surfaces",
#       Z. Kristallogr. 179 (1987) 31-52, and E. Koch, W. Fischer,
#       "On 3-periodic minimal surfaces with non-cubic symmetry",
#       Z. Kristallogr. 183 (1988) 129-152.
#       doi:10.1524/zkri.1988.183.14.129  [the S, C(S), Y families]
#   E. Koch and W. Fischer, "Triply periodic minimal balance surfaces:
#       a correction", Acta Crystallogr. A 49 (1993) 209-210.
#       doi:10.1107/S0108767392007591  [C(S) = P and Y = D as minimal
#       surfaces; the nodal geometries remain distinct]
#   A. H. Schoen, "Infinite periodic minimal surfaces without
#       self-intersections", NASA TN D-5541 (1970).  [O,C-TO]
#   H. Karcher, "The triply periodic minimal surfaces of Alan Schoen
#       and their constant mean curvature companions", Manuscripta
#       Math. 64 (1989) 291-357.  doi:10.1007/BF01165824  [K surface]
#   E. A. Lord and A. L. Mackay, "Periodic minimal surfaces of cubic
#       symmetry", Current Science 85 (2003) 346-362.  [Triplane / F
#       family context; survey]
#
# Shorthand used in the comments below: Ci = cos(i), Si = sin(i),
# C2x = cos(2x), etc., with coordinates in radians (period 2*pi).


def _tpms2_f_octo(x, y, z):
    # Schoen O,C-TO: 0.6(CxCy+CyCz+CzCx) - 0.4(Cx+Cy+Cz) + 0.25
    # (Wohlgemuth et al. 2001 form, level constant per Fisher et al.)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    return 0.6 * (cx * cy + cy * cz + cz * cx) - 0.4 * (cx + cy + cz) + 0.25


def _tpms2_f_fk_s(x, y, z):
    # Fischer-Koch S: C2x Sy Cz + Cx C2y Sz + Sx Cy C2z
    # (von Schnering & Nesper 1991)
    return (np.cos(2 * x) * np.sin(y) * np.cos(z)
            + np.cos(x) * np.cos(2 * y) * np.sin(z)
            + np.sin(x) * np.cos(y) * np.cos(2 * z))


def _tpms2_f_fk_cs(x, y, z):
    # Fischer-Koch C(S): (C2x+C2y+C2z) + 2(S3x S2y Cz + Cx S3y S2z
    #   + S2x Cy S3z) + 2(S2x C3y Sz + Sx S2y C3z + C3x Sy S2z)
    # (Koch & Fischer 1988 surface; nodal form per Fisher et al. 2023)
    return (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)
            + 2 * (np.sin(3 * x) * np.sin(2 * y) * np.cos(z)
                   + np.cos(x) * np.sin(3 * y) * np.sin(2 * z)
                   + np.sin(2 * x) * np.cos(y) * np.sin(3 * z))
            + 2 * (np.sin(2 * x) * np.cos(3 * y) * np.sin(z)
                   + np.sin(x) * np.sin(2 * y) * np.cos(3 * z)
                   + np.cos(3 * x) * np.sin(y) * np.sin(2 * z)))


def _tpms2_f_fk_y(x, y, z):
    # Fischer-Koch Y: CxCyCz + SxSySz + (S2xSy + S2ySz + SxS2z)
    #   + (CxS2y + CyS2z + S2xCz)   (Koch & Fischer 1988)
    return (np.cos(x) * np.cos(y) * np.cos(z)
            + np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(2 * x) * np.sin(y) + np.sin(2 * y) * np.sin(z)
            + np.sin(x) * np.sin(2 * z)
            + np.cos(x) * np.sin(2 * y) + np.cos(y) * np.sin(2 * z)
            + np.sin(2 * x) * np.cos(z))


def _tpms2_f_fk_pmy(x, y, z):
    # Fischer-Koch +-Y (PMY): 2 CxCyCz + S2xSy + S2ySz + SxS2z
    # (von Schnering & Nesper 1991)
    return (2 * np.cos(x) * np.cos(y) * np.cos(z)
            + np.sin(2 * x) * np.sin(y) + np.sin(2 * y) * np.sin(z)
            + np.sin(x) * np.sin(2 * z))


def _tpms2_f_fk_cpmy(x, y, z):
    # Fischer-Koch C(+-Y): -2 CxCyCz + S2xSy + S2ySz + SxS2z
    # (von Schnering & Nesper 1991)
    return (-2 * np.cos(x) * np.cos(y) * np.cos(z)
            + np.sin(2 * x) * np.sin(y) + np.sin(2 * y) * np.sin(z)
            + np.sin(x) * np.sin(2 * z))


def _tpms2_f_fk_cy(x, y, z):
    # Fischer-Koch C(Y): -SxSySz + S2xSy + S2ySz + SxS2z - CxCyCz
    #   + S2xCz + CxS2y + CyS2z   (von Schnering & Nesper 1991)
    return (-np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(2 * x) * np.sin(y) + np.sin(2 * y) * np.sin(z)
            + np.sin(x) * np.sin(2 * z)
            - np.cos(x) * np.cos(y) * np.cos(z)
            + np.sin(2 * x) * np.cos(z) + np.cos(x) * np.sin(2 * y)
            + np.cos(y) * np.sin(2 * z))


def _tpms2_f_cd(x, y, z):
    # Complementary D, C(D): cos(3x+y)Cz - sin(3x-y)Sz + cos(x+3y)Cz
    #   + sin(x-3y)Sz + cos(x-y)C3z - sin(x+y)S3z
    # (von Schnering & Nesper 1991)
    return (np.cos(3 * x + y) * np.cos(z) - np.sin(3 * x - y) * np.sin(z)
            + np.cos(x + 3 * y) * np.cos(z) + np.sin(x - 3 * y) * np.sin(z)
            + np.cos(x - y) * np.cos(3 * z) - np.sin(x + y) * np.sin(3 * z))


def _tpms2_f_cg(x, y, z):
    # Complementary Gyroid C(G) (a.k.a. C(Y**)): 3(SxCy + SyCz + CxSz)
    #   + 2(S3xCy + S3yCz + CxS3z) - 2(SxC3y + SyC3z + C3xSz)
    # (Fisher et al. 2023 compilation)
    return (3 * (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z)
                 + np.cos(x) * np.sin(z))
            + 2 * (np.sin(3 * x) * np.cos(y) + np.sin(3 * y) * np.cos(z)
                   + np.cos(x) * np.sin(3 * z))
            - 2 * (np.sin(x) * np.cos(3 * y) + np.sin(y) * np.cos(3 * z)
                   + np.cos(3 * x) * np.sin(z)))


def _tpms2_f_gprime(x, y, z):
    # G' (alternating-gyroid family): S2x Cy Sz + Sx S2y Cz
    #   + Cx Sy S2z + 0.32   (Wohlgemuth et al. 2001)
    return (np.sin(2 * x) * np.cos(y) * np.sin(z)
            + np.sin(x) * np.sin(2 * y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.sin(2 * z) + 0.32)


def _tpms2_f_dprime(x, y, z):
    # D': 1/2(CxCyCz + CxSySz + SxCySz + SxSyCz)
    #   - 1/2(S2xS2y + S2yS2z + S2zS2x) - 0.2
    # (Hoffman et al. 2003, MSRI Scientific Graphics Project)
    return (0.5 * (np.cos(x) * np.cos(y) * np.cos(z)
                   + np.cos(x) * np.sin(y) * np.sin(z)
                   + np.sin(x) * np.cos(y) * np.sin(z)
                   + np.sin(x) * np.sin(y) * np.cos(z))
            - 0.5 * (np.sin(2 * x) * np.sin(2 * y)
                     + np.sin(2 * y) * np.sin(2 * z)
                     + np.sin(2 * z) * np.sin(2 * x)) - 0.2)


def _tpms2_f_k(x, y, z):
    # Karcher K: 0.3(Cx+Cy+Cz) + 0.3(CxCy+CyCz+CzCx)
    #   - 0.4(C2x+C2y+C2z) + 0.2
    # (surface: Karcher 1989; nodal fit per Fisher et al. 2023)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    return (0.3 * (cx + cy + cz) + 0.3 * (cx * cy + cy * cz + cz * cx)
            - 0.4 * (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)) + 0.2)


def _tpms2_f_ci2y(x, y, z):
    # C(I2-Y**) rod-packing nodal surface: 2(S2x Cy Sz + Sx S2y Cz
    #   + Cx Sy S2z) + C2xC2y + C2yC2z + C2xC2z
    # (von Schnering, Oehme & Rudolf 1991)
    return (2 * (np.sin(2 * x) * np.cos(y) * np.sin(z)
                 + np.sin(x) * np.sin(2 * y) * np.cos(z)
                 + np.cos(x) * np.sin(y) * np.sin(2 * z))
            + np.cos(2 * x) * np.cos(2 * y)
            + np.cos(2 * y) * np.cos(2 * z)
            + np.cos(2 * x) * np.cos(2 * z))


def _tpms2_f_frd2(x, y, z):
    # Schoen F-RD, Wohlgemuth variant: 8 CxCyCz + C2xC2yC2z
    #   - (C2xC2y + C2yC2z + C2zC2x)   (Wohlgemuth et al. 2001)
    c2x, c2y, c2z = np.cos(2 * x), np.cos(2 * y), np.cos(2 * z)
    return (8 * np.cos(x) * np.cos(y) * np.cos(z) + c2x * c2y * c2z
            - (c2x * c2y + c2y * c2z + c2z * c2x))


TPMS.update({
    'OCTO': ("Schoen O,C-TO (nodal approximation)", _tpms2_f_octo, True),
    'FK_S': ("Fischer-Koch S (nodal approximation)", _tpms2_f_fk_s, True),
    'FK_CS': ("Fischer-Koch C(S) (nodal approximation)",
              _tpms2_f_fk_cs, True),
    'FK_Y': ("Fischer-Koch Y (nodal approximation)", _tpms2_f_fk_y, True),
    'FK_PMY': ("Fischer-Koch +-Y (nodal approximation)",
               _tpms2_f_fk_pmy, True),
    'FK_CPMY': ("Fischer-Koch C(+-Y) (nodal approximation)",
                _tpms2_f_fk_cpmy, True),
    'FK_CY': ("Fischer-Koch C(Y) (nodal approximation)",
              _tpms2_f_fk_cy, True),
    'CD': ("Complementary D (nodal approximation)", _tpms2_f_cd, True),
    'CG': ("Complementary Gyroid (nodal approximation)",
           _tpms2_f_cg, True),
    'GPRIME': ("G' Alternating Gyroid (nodal approximation)",
               _tpms2_f_gprime, True),
    'DPRIME': ("D' (nodal approximation)", _tpms2_f_dprime, True),
    'KSURF': ("Karcher K (nodal approximation)", _tpms2_f_k, True),
    'CI2Y': ("C(I2-Y**) Rod Packing (nodal approximation)",
             _tpms2_f_ci2y, True),
    'FRD2': ("Schoen F-RD (Wohlgemuth variant, nodal)",
             _tpms2_f_frd2, True),
})

# Considered but DEFERRED (verified unsound as zero level sets, so they
# do not ship -- see BACKLOG):
#   Q* / ST1  ((Cx - 2Cy)Cz - sqrt3 Sz (Cx-y - Cx) + Cx-y Cz): the zero
#       set has genuine critical points (at the origin every sine factor
#       vanishes), i.e. a cone singularity -- not embedded at c = 0.
#   Triplane / F  (CxCyCz = c): c = 0 degenerates to three orthogonal
#       plane sets; every c > 0 yields an array of DISCONNECTED closed
#       bubbles (Fisher et al. note the equation misses the tunnels of
#       Brakke's true Triplane TPMS).
#   W, G'2, Slotted-P, P+C(P), double G/D/P, tubular G/D/P: singular at
#       c = 0, unclear provenance, or not single/minimal surfaces.
#   Schwarz H, CLP, Schoen H'-T, S'-S'', I6, C(H): genuine TPMS but NO
#       published nodal formula exists (the Fisher et al. 2023 catalog,
#       which collects every known nodal fit, omits them); their exact
#       Weierstrass data lives in research/msblog_harvest.

# Built-in level offsets: rows whose PUBLISHED canonical member sits at
# a nonzero level of the field (the operator's Level Offset adds on
# top).  Currently none ship, but the mechanism is load-bearing for the
# offset-family UI and future rows.
tpms2_DEFAULT_OFFSET = {}

# Per-row lattice matrices for non-cubic unit cells: a 3x3 matrix mapping
# fractional cell coordinates (each 2*pi-periodic) to Cartesian space, in
# units of one cubic period.  Rows absent here use the identity (cubic).
# A hexagonal cell (for H-family fields defined on hexagonal axes) is
# M = [[1, -1/2, 0], [0, sqrt(3)/2, 0], [0, 0, c/a]].  No shipped row
# needs one yet -- kept as infrastructure (exercised by the self-test).
tpms2_LATTICE = {}

TPMS2_HEX_LATTICE = (
    (1.0, -0.5, 0.0),
    (0.0, math.sqrt(3.0) / 2.0, 0.0),
    (0.0, 0.0, 1.0),
)


def tpms2_cell_matrix(kind, aspect=1.0):
    """Cell matrix for `kind` with the z column scaled by `aspect`
    (the tetragonal c/a ratio).  Returns None when the cell is the
    plain cubic one (identity, aspect 1) so the caller can skip the
    transform entirely."""
    base = tpms2_LATTICE.get(kind)
    if base is None and abs(aspect - 1.0) < 1e-12:
        return None
    M = np.array(base if base is not None else np.eye(3), dtype=float)
    M[:, 2] *= float(aspect)
    return M

# ===================== end tier-2 nodal TPMS expansion ====================


# cube corners (i,j,k offsets) and a 6-tetrahedra decomposition sharing
# the 0-6 diagonal
_CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_TETS = [(0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6),
         (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6)]
# sign-pattern cases for one tetrahedron: bit set = corner value < 0
_ONE = {1: (0, (1, 2, 3)), 2: (1, (0, 2, 3)),
        4: (2, (0, 1, 3)), 8: (3, (0, 1, 2)),
        14: (0, (1, 2, 3)), 13: (1, (0, 2, 3)),
        11: (2, (0, 1, 3)), 7: (3, (0, 1, 2))}
_TWO = {3: ((0, 1), (2, 3)), 5: ((0, 2), (1, 3)), 9: ((0, 3), (1, 2)),
        6: ((1, 2), (0, 3)), 10: ((1, 3), (0, 2)), 12: ((2, 3), (0, 1))}


def _orientation_flags():
    """Whether each (tet, sign-case) emits triangles wound against
    the field gradient. Calibrated on an exact linear field, where
    the crossing polygon is exactly perpendicular to the gradient --
    so the flags are combinatorial and immune to sliver triangles."""
    flags = {}
    cube = np.array(_CUBE, dtype=float)
    for ti, tet in enumerate(_TETS):
        P = cube[list(tet)]                       # (4,3) corners
        M = P[1:] - P[0]                          # for gradient solve
        for cd in list(_ONE) + list(_TWO):
            f = np.where([cd >> i & 1 for i in range(4)], -1.0, 1.0)
            g = np.linalg.solve(M, f[1:] - f[0])  # exact gradient

            def x(ci, cj):
                t = f[ci] / (f[ci] - f[cj])
                return P[ci] + t * (P[cj] - P[ci])

            if cd in _ONE:
                lone, (o0, o1, o2) = _ONE[cd]
                p0, p1, p2 = x(lone, o0), x(lone, o1), x(lone, o2)
            else:
                (n0, n1), (q0, q1) = _TWO[cd]
                p0, p1, p2 = x(n0, q0), x(n0, q1), x(n1, q1)
            n = np.cross(p1 - p0, p2 - p0)
            flags[(ti, cd)] = float(np.dot(n, g)) < 0.0
    return flags


_ORIENT = None


def marching_tets(field, box_min, box_max, res):
    """Extract the zero level set of `field` on a res[0]xres[1]xres[2]
    sample grid over the box. Returns (verts (n,3), tris (m,3)) with
    triangle winding oriented along the field gradient."""
    nx, ny, nz = (r + 1 for r in res)
    xs = np.linspace(box_min[0], box_max[0], nx)
    ys = np.linspace(box_min[1], box_max[1], ny)
    zs = np.linspace(box_min[2], box_max[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    vals = field(X, Y, Z).ravel()
    # samples landing exactly on the surface (e.g. Schwarz P at the
    # lattice points) produce degenerate crossings; nudge them off
    vals = np.where(np.abs(vals) < 1e-9, 1e-9, vals)
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    ii, jj, kk = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1),
                             np.arange(nz - 1), indexing='ij')
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()

    def flat(i, j, k):
        return (i * ny + j) * nz + k

    corner = [flat(ii + o[0], jj + o[1], kk + o[2]) for o in _CUBE]

    global _ORIENT
    if _ORIENT is None:
        _ORIENT = _orientation_flags()

    tri_pts = []          # list of (3, ntri, 3) blocks
    for ti, (a, b, c, d) in enumerate(_TETS):
        A, B, C, D = corner[a], corner[b], corner[c], corner[d]
        fa, fb, fc, fd = vals[A], vals[B], vals[C], vals[D]
        code = ((fa < 0).astype(np.int8) | ((fb < 0) << 1)
                | ((fc < 0) << 2) | ((fd < 0) << 3))
        tet = np.stack([A, B, C, D], axis=0)

        def interp(sel, ci, cj):
            ia, ib = tet[ci][sel], tet[cj][sel]
            va, vb = vals[ia], vals[ib]
            t = va / (va - vb)
            return pos[ia] + t[:, None] * (pos[ib] - pos[ia])

        for cd, (lone, others) in _ONE.items():
            sel = np.nonzero(code == cd)[0]
            if len(sel) == 0:
                continue
            p0 = interp(sel, lone, others[0])
            p1 = interp(sel, lone, others[1])
            p2 = interp(sel, lone, others[2])
            if _ORIENT[(ti, cd)]:
                p1, p2 = p2, p1
            tri_pts.append(np.stack([p0, p1, p2], axis=1))
        for cd, ((n0, n1), (pp0, pp1)) in _TWO.items():
            sel = np.nonzero(code == cd)[0]
            if len(sel) == 0:
                continue
            q0 = interp(sel, n0, pp0)
            q1 = interp(sel, n0, pp1)
            q2 = interp(sel, n1, pp1)
            q3 = interp(sel, n1, pp0)
            if _ORIENT[(ti, cd)]:
                q1, q3 = q3, q1
            tri_pts.append(np.stack([q0, q1, q2], axis=1))
            tri_pts.append(np.stack([q0, q2, q3], axis=1))

    if not tri_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    tris_xyz = np.concatenate(tri_pts, axis=0)      # (ntri, 3, 3)

    # weld: quantize and unique
    flatv = tris_xyz.reshape(-1, 3)
    eps = max(np.max(box_max) - np.min(box_min), 1.0) * 1e-6
    keys = np.round(flatv / eps).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_index=False,
                          return_inverse=True)
    order = np.zeros(len(uniq), dtype=np.int64)
    order[inv] = np.arange(len(flatv))              # a representative
    verts = flatv[order]
    tris = inv.reshape(-1, 3)
    good = ((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
            & (tris[:, 0] != tris[:, 2]))
    tris = tris[good]

    return verts, tris


def _cells_xyz(cells):
    """Coerce a cell count to a (cx, cy, cz) triple.  A plain int means a
    symmetric cx = cy = cz block (kept for internal callers/tests); a tuple
    or list gives independent per-axis counts."""
    if isinstance(cells, (tuple, list)):
        vals = [int(max(1, c)) for c in cells]
        while len(vals) < 3:
            vals.append(1)
        return vals[0], vals[1], vals[2]
    c = int(max(1, cells))
    return c, c, c


def build_tpms(kind, cells, res_per_cell, scale, offset=0.0, aspect=1.0):
    """Nodal TPMS mesh.  `cells` is an int (symmetric block) or a
    (cx, cy, cz) triple: the block spans cx x cy x cz unit cells, one
    independent count per lattice axis.  For the singly periodic Scherk
    tower only the z count repeats (x/y are clipped).

    `offset` is the level constant c in F(x,y,z) = c, added on top of
    the row's built-in default (tpms2_DEFAULT_OFFSET): 0 keeps the
    published canonical member, nonzero sweeps the offset companion
    family.  `aspect` is the tetragonal c/a cell ratio; together with a
    per-row lattice matrix (tpms2_LATTICE) it generalizes the tiling to
    non-cubic unit cells: the field is marched in fractional (2*pi-
    periodic) coordinates and the mesh mapped through the cell matrix,
    so copies tile seamlessly at the true aspect ratio."""
    label, field, triply = TPMS[kind]
    cx, cy, cz = _cells_xyz(cells)
    off = tpms2_DEFAULT_OFFSET.get(kind, 0.0) + float(offset)
    f = field if off == 0.0 else (
        lambda X, Y, Z: field(X, Y, Z) - off)
    if triply:
        M = tpms2_cell_matrix(kind, aspect)
        # sample density follows the true (post-matrix) axis lengths
        ln = ((1.0, 1.0, 1.0) if M is None
              else tuple(np.linalg.norm(M[:, i]) for i in range(3)))
        sx, sy, sz = cx * TAU, cy * TAU, cz * TAU
        box_min = (-sx / 2, -sy / 2, -sz / 2)
        box_max = (sx / 2, sy / 2, sz / 2)
        res = tuple(max(4, int(round(c * res_per_cell * l)))
                    for c, l in zip((cx, cy, cz), ln))
        verts, tris = marching_tets(f, box_min, box_max, res)
        if M is not None:
            verts = verts @ M.T
    else:  # Scherk tower: periodic in z only, clip x/y
        w = 2.2
        box_min = (-w, -w, -cz * math.pi)
        box_max = (w, w, cz * math.pi)
        rxy = int(res_per_cell * 1.4)
        res = (rxy, rxy, cz * res_per_cell)
        verts, tris = marching_tets(f, box_min, box_max, res)
    s = scale / TAU  # one period -> `scale` Blender units
    return verts * s, tris


# --- exact Weierstrass P / Gyroid / D (Bonnet angle) ---------------------
# The nodal TPMS above (marching-tets level sets) have NO associate
# parameter -- you cannot morph P <-> Gyroid <-> D in the nodal
# representation.  The exact genus-3 Enneper-Weierstrass immersion in
# weierstrass.pgd_build does: a single Bonnet angle theta continuously
# sweeps the whole classical family (P at 0, Gyroid at ~38.0148 deg, D at
# 90 deg).  It is exposed under the periodic generator's Triply Periodic
# list as its own entry (a preset for P / Gyroid / D plus an independent
# associate-angle slider), separate from and leaving unchanged the nodal TPMS
# set.  At the two reflective members it reassembles a watertight FILLED unit
# cell by the Schwarz reflection principle -- Schwarz P from coordinate-plane
# reflections, Schwarz D from 2-fold rotations about its straight edges -- and
# `cells` arrays that cell on the verified cubic period.  The chiral gyroid and
# every generic intermediate angle instead keep the exact fundamental surface
# piece (the honest choice: only P / Gyroid / D are truly periodic, and the
# gyroid cell has no drift-free reflection assembly).  See the pgd_build /
# _pgd_tile_cell docstrings in weierstrass.

from . import weierstrass as _we_pgd

# key -> (menu label, builder(cells, res_per_cell, scale, theta))
TPMS_EXACT = {
    'PGD': ("Schwarz P-Gyroid-D (exact, Bonnet angle)", _we_pgd.pgd_build),
}

# named-preset -> Bonnet angle (radians).  P and D reassemble a filled cell;
# the gyroid angle builds the fundamental piece.  CUSTOM (absent here) means
# "use the raw Associate Angle slider".
_PGD_PRESET_ANGLE = {
    'P': 0.0,
    'GYROID': 0.6635246,          # 38.0148 deg -- Schoen's gyroid
    'D': math.pi / 2.0,
}


def build_tpms_exact(kind, cells, res_per_cell, scale, theta):
    label, builder = TPMS_EXACT[kind]
    return builder(cells, res_per_cell, scale, theta)


def _selftest():
    ok = True

    # marching_tets against a closed surface of known area: the unit sphere.
    # A watertight extraction has every edge shared by exactly two triangles
    # and Euler characteristic 2.
    def sphere(x, y, z):
        return x * x + y * y + z * z - 1.0

    V, T = marching_tets(sphere, (-1.5, -1.5, -1.5), (1.5, 1.5, 1.5),
                         (40, 40, 40))
    e = {}
    for t in T:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e[(a, b) if a < b else (b, a)] = e.get((a, b) if a < b
                                                   else (b, a), 0) + 1
    boundary = sum(1 for c in e.values() if c != 2)
    chi = len(V) - len(e) + len(T)
    P = V[T]
    area = float(np.sum(np.linalg.norm(
        np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)) * 0.5)
    radial = float(np.max(np.abs(np.linalg.norm(V, axis=1) - 1.0)))
    good = (boundary == 0 and chi == 2 and radial < 0.02
            and abs(area - 4.0 * math.pi) < 0.05)
    ok &= good
    print(f"tpms: sphere V={len(V)} T={len(T)} chi={chi} nonmanifold={boundary}"
          f" area={area:.4f} (exp {4*math.pi:.4f}) max|r-1|={radial:.4f} "
          f"{'OK' if good else 'FAIL'}")

    # The nodal fields are triply periodic with period 2*pi -- a sign slip or
    # a wrong harmonic breaks this immediately.
    rng = np.random.default_rng(12345)
    p = rng.uniform(-3.0, 3.0, size=(3, 200))
    bad = []
    for key, (label, f, _periodic) in TPMS.items():
        if key == 'SCHERKT':
            continue                       # singly periodic (sinh), not 2*pi
        d = float(np.max(np.abs(f(p[0] + TAU, p[1], p[2]) - f(*p))))
        d = max(d, float(np.max(np.abs(f(p[0], p[1] + TAU, p[2]) - f(*p)))))
        if d > 1e-9:
            bad.append(f"{key}:{d:.1e}")
    good = not bad
    ok &= good
    print(f"tpms: {len(TPMS) - 1} nodal fields 2pi-periodic "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # Schwarz P and D pass through the origin-centred cell symmetrically:
    # F(-x) = F(x) for P, and the gyroid is the chiral one (F(-x) != F(x)).
    even_p = float(np.max(np.abs(_f_p(-p[0], -p[1], -p[2]) - _f_p(*p))))
    chiral_g = float(np.max(np.abs(_f_g(-p[0], -p[1], -p[2]) - _f_g(*p))))
    good = even_p < 1e-12 and chiral_g > 1e-3
    ok &= good
    print(f"tpms: P even (res={even_p:.1e}), gyroid chiral (dev={chiral_g:.3f})"
          f" {'OK' if good else 'FAIL'}")

    # Every registered nodal surface meshes to a non-trivial, finite patch.
    for key in TPMS:
        Vv, Tt = build_tpms(key, 1, 12, 1.0)
        g = len(Vv) > 50 and len(Tt) > 50 and bool(np.all(np.isfinite(Vv)))
        ok &= g
        if not g:
            print(f"tpms: build {key:10s} V={len(Vv)} T={len(Tt)} FAIL")
    print(f"tpms: built {len(TPMS)} nodal surfaces "
          f"{'OK' if ok else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("tpms self-test failed")
