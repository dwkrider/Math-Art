
# Prime Knot Generator for Blender
#
# All 249 prime knots with up to 10 crossings (Rolfsen table), built
# from the minimum braid words of Thomas Gittings ("Minimum braids: a
# complete invariant of knots and links", arXiv:math/0401051, Table
# 1). The braid closure is laid out around a circle -- strands at
# radii by braid level, crossings as over/under bumps -- and then
# relaxed with curve smoothing plus self-repulsion into a rounded,
# KnotPlot-like presentation.
#
# Every braid word in the table is verified programmatically (run
# this file with plain python): the closure permutation must be a
# single cycle, and the Alexander polynomial computed via the reduced
# Burau representation must match Gittings' published value at t=10.
#
# Output styles follow the classic Torus Knot + add-on: a Bezier /
# Poly / NURBS curve with bevel radius, or a mesh tube.
#
# References:
# - Dale Rolfsen, "Knots and Links", Publish or Perish, 1976 (the
#   Rolfsen table and its knot numbering).
# - Knot nomenclature n_k after J. W. Alexander & G. B. Briggs
#   (1926/27), extending P. G. Tait's 19th-century enumeration.
# - Thomas A. Gittings, "Minimum braids: a complete invariant of
#   knots and links", arXiv:math/0401051, 2004 (Table 1 braid words).
# - Verification via the Alexander polynomial (J. W. Alexander, 1928)
#   computed from the reduced Burau representation (Werner Burau, 1935).

bl_info = {
    "name": "Prime Knots",
    "author": "Math Art project (braid table after Thomas "
              "Gittings)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Prime Knot",
    "description": "All prime knots to 10 crossings from verified "
                   "minimum braids",
    "category": "Add Curve",
}

import math

KNOTS = (
    ('3_1', 'AAA', 91),
    ('4_1', 'AbAb', 71),
    ('5_1', 'AAAAA', 9091),
    ('5_2', 'AAABaB', 172),
    ('6_1', 'AABacBc', 152),
    ('6_2', 'AAAbAb', 7271),
    ('6_3', 'AAbAbb', 7471),
    ('7_1', 'AAAAAAA', 909091),
    ('7_2', 'AAABaBCbC', 253),
    ('7_3', 'AAAAABaB', 17272),
    ('7_4', 'AABaBBCbC', 334),
    ('7_5', 'AAAABaBB', 16462),
    ('7_6', 'AAbACbC', 5651),
    ('7_7', 'AbAbCbC', 5851),
    ('8_1', 'AABaBCbdCd', 233),
    ('8_2', 'AAAAAbAb', 727271),
    ('8_3', 'AABacBcdCd', 314),
    ('8_4', 'AAAbAbcBc', 15452),
    ('8_5', 'AAAbAAAb', 735371),
    ('8_6', 'AAAABacBc', 14642),
    ('8_7', 'AAAAbAbb', 745471),
    ('8_8', 'AAABacBcc', 14842),
    ('8_9', 'AAAbAbbb', 743471),
    ('8_10', 'AAAbAAbb', 753571),
    ('8_11', 'AABaBBcBc', 13832),
    ('8_12', 'AbACbdCd', 4231),
    ('8_13', 'AAbAbbcBc', 14032),
    ('8_14', 'AAABaBcBc', 13022),
    ('8_15', 'AAbACBBBC', 23023),
    ('8_16', 'AAbAAbAb', 671761),
    ('8_17', 'AAbAbAbb', 669761),
    ('8_18', 'AbAbAbAb', 587951),
    ('8_19', 'AAABAAAB', 900991),
    ('8_20', 'AAAbaaab', 8281),
    ('8_21', 'AAABaaBB', 6461),
    ('9_1', 'AAAAAAAAA', 90909091),
    ('9_2', 'AAABaBCbCDcD', 334),
    ('9_3', 'AAAAAAABaB', 1727272),
    ('9_4', 'AAAAABaBCbC', 25453),
    ('9_5', 'AABaBBCbCDcD', 496),
    ('9_6', 'AAAAAABaBB', 1645462),
    ('9_7', 'AAAABaBCbCC', 23833),
    ('9_8', 'AAbAbcBDcD', 13022),
    ('9_9', 'AAAAABaBBB', 1653562),
    ('9_10', 'AABaBBBBCbC', 32824),
    ('9_11', 'AAAAbACbC', 563651),
    ('9_12', 'AAbACbCDcD', 12212),
    ('9_13', 'AAAABaBBCbC', 32014),
    ('9_14', 'AABacBcDcD', 12412),
    ('9_15', 'AAABacBDcD', 11402),
    ('9_16', 'AAAABBaBBB', 1571752),
    ('9_17', 'AbAbbbCbC', 581851),
    ('9_18', 'AAABaBBBCbC', 31204),
    ('9_19', 'AbAbbcBDcD', 11602),
    ('9_20', 'AAAbACbCC', 579851),
    ('9_21', 'AABaBcBDcD', 10592),
    ('9_22', 'AbAbCbbbC', 589951),
    ('9_23', 'AAABaBBCbCC', 30394),
    ('9_24', 'AAbACbbbC', 587951),
    ('9_25', 'AAbACBBdCd', 19583),
    ('9_26', 'AAAbAbCbC', 598051),
    ('9_27', 'AAbAbbCbC', 596051),
    ('9_28', 'AAbACbbCC', 606151),
    ('9_29', 'AbbCbAbCb', 606151),
    ('9_30', 'AAbbAbCbC', 604151),
    ('9_31', 'AAbAbCbCC', 614251),
    ('9_32', 'AAbAbACbC', 524341),
    ('9_33', 'AbAbbACbC', 522341),
    ('9_34', 'AbAbCbAbC', 538541),
    ('9_35', 'AABaBBCbbDcBDC', 577),
    ('9_36', 'AAAbAACbC', 571751),
    ('9_37', 'AAbACbadCbCd', 10792),
    ('9_38', 'AABBcBaBCCB', 37765),
    ('9_39', 'AABacbADCbCD', 17963),
    ('9_40', 'AbACbACbC', 458731),
    ('9_41', 'AABacbbDCbCD', 19783),
    ('9_42', 'AAAbaaCbC', 8081),
    ('9_43', 'AAABAAcBc', 719171),
    ('9_44', 'AAABaacBc', 6661),
    ('9_45', 'AABaBACbC', 4841),
    ('9_46', 'AbAbCBaBC', 152),
    ('9_47', 'AbAbcbAbc', 655561),
    ('9_48', 'AABaBAcBaBc', 4031),
    ('9_49', 'AABAAcBaBCC', 24643),
    ('10_1', 'AABaBCbCDceDe', 314),
    ('10_2', 'AAAAAAAbAb', 72727271),
    ('10_3', 'AABaBCbdCdeDe', 476),
    ('10_4', 'AAAbAbcBcdCd', 23633),
    ('10_5', 'AAAAAAbAbb', 74545471),
    ('10_6', 'AAAAAABacBc', 1463642),
    ('10_7', 'AABaBCbCCdCd', 20393),
    ('10_8', 'AAAAAbAbcBc', 1545452),
    ('10_9', 'AAAAAbAbbb', 74363471),
    ('10_10', 'AAbAbbcBcdCd', 20593),
    ('10_11', 'AAAABacBcdCd', 30194),
    ('10_12', 'AAAAABacBcc', 1489942),
    ('10_13', 'AABacBDceDe', 9172),
    ('10_14', 'AAAAABaBcBc', 1308122),
    ('10_15', 'AAAAbAbcBcc', 1481842),
    ('10_16', 'AABaBBcBcdCd', 29384),
    ('10_17', 'AAAAbAbbbb', 74383471),
    ('10_18', 'AAABaBcBcdCd', 27764),
    ('10_19', 'AAAAbAbbcBc', 1400032),
    ('10_20', 'AAAABaBCbdCd', 22013),
    ('10_21', 'AABaBBBBcBc', 1381832),
    ('10_22', 'AAAABacBccc', 1487942),
    ('10_23', 'AAbAbbbbcBc', 1416232),
    ('10_24', 'AABaBBBCbdCd', 27764),
    ('10_25', 'AAAABaBBcBc', 1324322),
    ('10_26', 'AAAbAbbbcBc', 1414232),
    ('10_27', 'AAAABaBcBcc', 1342522),
    ('10_28', 'AABaBBCbdCdd', 28774),
    ('10_29', 'AAAbACbdCd', 434431),
    ('10_30', 'AABaBBCbCdCd', 25334),
    ('10_31', 'AAABacBccdCd', 27964),
    ('10_32', 'AAAbAbbcBcc', 1332422),
    ('10_33', 'AABaBcBccdCd', 26344),
    ('10_34', 'AAABaBCbdCdd', 22213),
    ('10_35', 'AbAbcBDceDe', 9982),
    ('10_36', 'AAABaBCbCdCd', 18773),
    ('10_37', 'AAABacBcdCdd', 28774),
    ('10_38', 'AAABaBBCbdCd', 26954),
    ('10_39', 'AAABaBBBcBc', 1316222),
    ('10_40', 'AAABaBBcBcc', 1350622),
    ('10_41', 'AbAbbCbdCd', 450631),
    ('10_42', 'AAbAbCbdCd', 464831),
    ('10_43', 'AAbACbdCdd', 448631),
    ('10_44', 'AAbACbCdCd', 466831),
    ('10_45', 'AbAbCbCdCd', 481031),
    ('10_46', 'AAAAAbAAAb', 73545371),
    ('10_47', 'AAAAAbAAbb', 75363571),
    ('10_48', 'AAAAbbAbbb', 75201571),
    ('10_49', 'AAAAbACBBBC', 2308123),
    ('10_50', 'AABaBBcBBBc', 1398032),
    ('10_51', 'AABaBBcBBcc', 1432432),
    ('10_52', 'AAAbAAbbcBc', 1416232),
    ('10_53', 'AABaBcBDCCCD', 44326),
    ('10_54', 'AAAbAAbcBcc', 1489942),
    ('10_55', 'AAABacBDCCCD', 36955),
    ('10_56', 'AAABaBcBBBc', 1324322),
    ('10_57', 'AAABaBcBBcc', 1358722),
    ('10_58', 'AbACbdccEdE', 16543),
    ('10_59', 'AbAbCbbdCd', 458731),
    ('10_60', 'AbAbbCbCBDcD', 472931),
    ('10_61', 'AAAbAAAbcBc', 1553552),
    ('10_62', 'AAAAbAAAbb', 75282571),
    ('10_63', 'AAbACBBBCDcD', 37765),
    ('10_64', 'AAAbAAAbbb', 75100571),
    ('10_65', 'AABaBcBBBcc', 1424332),
    ('10_66', 'AAAbACBBBCC', 2242513),
    ('10_67', 'AAABaBCbbdCBdC', 26144),
    ('10_68', 'AAbAbbcBBdCbdc', 27964),
    ('10_69', 'AABacBADcBcD', 483031),
    ('10_70', 'AbACbbbdCd', 442531),
    ('10_71', 'AAbACbbdCd', 456731),
    ('10_72', 'AAAABBaBcBc', 1242512),
    ('10_73', 'AABaBAcBcDcD', 474931),
    ('10_74', 'AABaBBCbbdCBdC', 26144),
    ('10_75', 'AbAbCbbDcBDC', 464831),
    ('10_76', 'AAAABacBBBc', 1406132),
    ('10_77', 'AAAABacBBcc', 1424332),
    ('10_78', 'AABaBAcBDcDD', 440531),
    ('10_79', 'AAAbbAAbbb', 75938671),
    ('10_80', 'AAAbAACBBBC', 2234413),
    ('10_81', 'AAbACBBdcccd', 374921),
    ('10_82', 'AAAAbAbAbb', 66918761),
    ('10_83', 'AABaBcBBcBc', 1268812),
    ('10_84', 'AAABacBBcBc', 1276912),
    ('10_85', 'AAAAbAAbAb', 67100761),
    ('10_86', 'AAbAbAbbcBc', 1266812),
    ('10_87', 'AAABacBcBcc', 1258712),
    ('10_88', 'AbACbCbdCd', 407321),
    ('10_89', 'AbAbcBADCbCD', 409321),
    ('10_90', 'AAbAbcBAcbb', 1348622),
    ('10_91', 'AAAbAbbAbb', 67756861),
    ('10_92', 'AAABBcBaBcB', 1176902),
    ('10_93', 'AAbAAbAbcBc', 1334422),
    ('10_94', 'AAAbAAbbAb', 67736861),
    ('10_95', 'AAbbCbAbccb', 1285012),
    ('10_96', 'AbaCbaCdCbCd', 489131),
    ('10_97', 'AABaBAcBaBCdCd', 31085),
    ('10_98', 'AABBcBaBBcB', 1258712),
    ('10_99', 'AAbAAbbAbb', 68574961),
    ('10_100', 'AAAbAAbAAb', 67918861),
    ('10_101', 'AAABaCbACBBDcD', 51697),
    ('10_102', 'AAbACbAbbcc', 1340522),
    ('10_103', 'AABacBBcBBc', 1350622),
    ('10_104', 'AAAbbAbAbb', 67675861),
    ('10_105', 'AAbACBBdcBcd', 393121),
    ('10_106', 'AAAbAbAAbb', 67655861),
    ('10_107', 'AAbAcbbDcBcD', 391121),
    ('10_108', 'AAbAACbAbcc', 1326322),
    ('10_109', 'AAbAbbAAbb', 68493961),
    ('10_110', 'AbACBBBdcBcd', 376921),
    ('10_111', 'AABBcBBaBcB', 1250612),
    ('10_112', 'AAAbAbAbAb', 59474051),
    ('10_113', 'AAABcBaBcBc', 1129492),
    ('10_114', 'AABacBcBcBc', 1185002),
    ('10_115', 'AbACBBdcBccd', 325511),
    ('10_116', 'AAbAAbAbAb', 60292151),
    ('10_117', 'AABBcBaBcBc', 1211302),
    ('10_118', 'AAbAbAbbAb', 60312151),
    ('10_119', 'AAbACbAbccb', 1201202),
    ('10_120', 'AABacbADCBBCCD', 57448),
    ('10_121', 'AABcBaBcBcB', 1137592),
    ('10_122', 'AABcBacBcBc', 1111292),
    ('10_123', 'AbAbAbAbAb', 52867441),
    ('10_124', 'AAAAABAAAB', 90090991),
    ('10_125', 'AAAAAbaaab', 819181),
    ('10_126', 'AAAAABaaaB', 835381),
    ('10_127', 'AAAAABaaBB', 653561),
    ('10_128', 'AAABAABBCbC', 1711072),
    ('10_129', 'AAAbaaCbaCb', 14842),
    ('10_130', 'AAAbaabbcBc', 16462),
    ('10_131', 'AAABaaBBCbC', 13022),
    ('10_132', 'AAAbaabcBcc', 9091),
    ('10_133', 'AAABaaBCbCC', 5651),
    ('10_134', 'AAABAABCbCC', 1637362),
    ('10_135', 'AAABaBcbbbc', 22213),
    ('10_136', 'AbAbcBBDcD', 6461),
    ('10_137', 'AbAbCBBdCd', 5041),
    ('10_138', 'AbAbcbbDcD', 573751),
    ('10_139', 'AAAABAAABB', 90171991),
    ('10_140', 'AAAbaaabcBc', 8281),
    ('10_141', 'AAAAbaaabb', 735371),
    ('10_142', 'AAABAAABCbC', 1719172),
    ('10_143', 'AAAABaaaBB', 753571),
    ('10_144', 'AABaBAcBAcb', 21203),
    ('10_145', 'AABaBACBaBC', 10711),
    ('10_146', 'AAbAbaCbAbC', 13222),
    ('10_147', 'AAAbAbcBaBc', 13832),
    ('10_148', 'AAAABaaBaB', 761671),
    ('10_149', 'AAAABaBaBB', 579851),
    ('10_150', 'AAAbAACbaCB', 653561),
    ('10_151', 'AAABaaCbACb', 687961),
    ('10_152', 'AAABBAABBB', 89353891),
    ('10_153', 'AAABAAcbbbc', 892891),
    ('10_154', 'AABaBACBBBC', 966601),
    ('10_155', 'AAABaaBaaB', 743471),
    ('10_156', 'AAAbaaCBaBC', 671761),
    ('10_157', 'AAABBaBaBB', 498041),
    ('10_158', 'AAABaacbAbc', 685961),
    ('10_159', 'AAABaBaaBB', 679861),
    ('10_160', 'AAABAAcBaBc', 637361),
    ('10_161', 'AAABaBAABB', 982801),
    ('10_162', 'AABaaBBAcBc', 22013),
    ('10_163', 'AAbaaCBaBBC', 606151),
    ('10_164', 'AAbAbbcBaBc', 20593),
    ('10_165', 'AABacBaBCCB', 11402),
)


def parse_letters(s):
    """Gittings letter notation: a..z = generators 1.., A..Z their
    inverses."""
    word = []
    for ch in s:
        if ch.islower():
            word.append(ord(ch) - ord('a') + 1)
        elif ch.isupper():
            word.append(-(ord(ch) - ord('A') + 1))
        else:
            raise ValueError(f"bad braid character {ch!r}")
    return word


def closure_components(word):
    """Number of components of the braid closure."""
    n = max(abs(g) for g in word) + 1
    perm = list(range(n))
    for g in word:
        i = abs(g) - 1
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
    seen = set()
    comps = 0
    for s0 in range(n):
        if s0 in seen:
            continue
        comps += 1
        cur = s0
        while cur not in seen:
            seen.add(cur)
            cur = perm[cur]
    return comps


# ---- Alexander polynomial via reduced Burau (verification) -------------

def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
            for i in range(n)]


def _pneg(a):
    return [-x for x in a]


def _pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def _ptrim(a):
    while a and a[-1] == 0:
        a.pop()
    return a or [0]


def _pdivexact(a, b):
    a = list(a)
    b = _ptrim(list(b))
    out = [0] * (len(a) - len(b) + 1)
    for k in range(len(out) - 1, -1, -1):
        c, r = divmod(a[len(b) - 1 + k], b[-1])
        if r:
            raise ValueError("inexact division")
        out[k] = c
        for j, y in enumerate(b):
            a[j + k] -= c * y
    if _ptrim(a) != [0]:
        raise ValueError("nonzero remainder")
    return _ptrim(out)


def alexander_at(word, x=10):
    """|Alexander polynomial| of the braid closure at t=x, from the
    reduced Burau representation (quotient of the unreduced one by
    its fixed vector; inverse generators are scaled by t, giving
    t^k * rho, handled by det(P - t^k I))."""
    n = max(abs(g) for g in word) + 1
    one, t = [1], [0, 1]

    def fold(col):
        last = col[n - 1]
        return [_ptrim(_padd(col[i], _pneg(last)))
                for i in range(n - 1)]

    def gen(i, inv):
        diag = t if inv else one
        M = [[diag if r == c else [0] for c in range(n)]
             for r in range(n)]
        if not inv:
            M[i][i] = [1, -1]
            M[i][i + 1] = t
            M[i + 1][i] = one
            M[i + 1][i + 1] = [0]
        else:
            M[i][i] = [0]
            M[i][i + 1] = t
            M[i + 1][i] = one
            M[i + 1][i + 1] = [-1, 1]
        cols = [fold([M[r][j] for r in range(n)])
                for j in range(n - 1)]
        return [[cols[j][r] for j in range(n - 1)]
                for r in range(n - 1)]

    m = n - 1
    R = [[one if i == j else [0] for j in range(m)] for i in range(m)]
    for g in word:
        G = gen(abs(g) - 1, g < 0)
        R = [[_ptrim(_padd([0], _prowcol(G, R, r, c, m)))
              for c in range(m)] for r in range(m)]
    k = sum(1 for g in word if g < 0)
    tk = [0] * k + [1]
    for i in range(m):
        R[i][i] = _ptrim(_padd(R[i][i], _pneg(tk)))
    q = _det(R)
    r = _pdivexact(q, [1] * n)
    while r[0] == 0:
        r = r[1:]
    return abs(sum(c * x ** i for i, c in enumerate(r)))


def _prowcol(A, B, r, c, m):
    acc = [0]
    for k in range(m):
        acc = _padd(acc, _pmul(A[r][k], B[k][c]))
    return acc


def _det(M):
    m = len(M)
    if m == 1:
        return M[0][0]
    det = [0]
    for c in range(m):
        minor = [[M[r][cc] for cc in range(m) if cc != c]
                 for r in range(1, m)]
        term = _pmul(M[0][c], _det(minor))
        det = _padd(det, term if c % 2 == 0 else _pneg(term))
    return _ptrim(det)


# ---- geometry: braid closure embedding + relaxation --------------------

def braid_closure_points(word, ring=1.6, spacing=0.3, bump=0.26):
    """Closed 3D polyline of the braid closure: strands at radii by
    braid level around a circle, crossings as +-z bumps."""
    n = max(abs(g) for g in word) + 1
    L = len(word)

    def pt(a, lev, z=0.0):
        r = ring + (lev - (n - 1) / 2.0) * spacing
        return (r * math.cos(a), r * math.sin(a), z)

    paths = [[] for _ in range(n)]        # per strand id
    strands = list(range(n))              # level -> strand id
    for lev in range(n):
        paths[lev].append(pt(0.0, lev))
    for s, g in enumerate(word):
        a0 = 2 * math.pi * s / L
        a1 = 2 * math.pi * (s + 1) / L
        am = (a0 + a1) / 2
        i = abs(g) - 1
        for lev in range(n):
            if lev != i and lev != i + 1:
                paths[strands[lev]].append(pt(a1, lev))
        z = bump if g > 0 else -bump
        sid_i, sid_j = strands[i], strands[i + 1]
        paths[sid_i].append(pt(am, i + 0.5, z))
        paths[sid_i].append(pt(a1, i + 1))
        paths[sid_j].append(pt(am, i + 0.5, -z))
        paths[sid_j].append(pt(a1, i))
        strands[i], strands[i + 1] = sid_j, sid_i
    end_level = {sid: lev for lev, sid in enumerate(strands)}
    loop = [paths[0][0]]
    sid = 0
    while True:
        loop.extend(paths[sid][1:])
        sid = end_level[sid]
        if sid == 0:
            break
    loop.pop()                            # closing duplicate
    return loop


def resample_closed(pts, m):
    """Resample a closed polyline to m evenly spaced points."""
    import numpy as np
    P = np.asarray(pts, dtype=float)
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    want = np.linspace(0.0, total, m, endpoint=False)
    out = np.empty((m, 3))
    j = 0
    for i, s in enumerate(want):
        while cum[j + 1] < s:
            j += 1
        t = (s - cum[j]) / (seg[j] or 1.0)
        out[i] = P[j] + t * (P[(j + 1) % len(P)] - P[j])
    return out


def relax_knot(P, iters=150, repel=0.35, step=0.25, smooth=0.5,
               excl=6):
    """Rounds the closure into a smooth knot: curve smoothing plus
    self-repulsion below `repel`, size renormalized each step so the
    strand cannot pass through itself for conservative settings."""
    import numpy as np
    P = np.asarray(P, dtype=float).copy()
    m = len(P)
    target = np.linalg.norm(P - P.mean(0), axis=1).mean()
    idx = np.arange(m)
    ring_d = np.abs(idx[:, None] - idx[None, :])
    ring_d = np.minimum(ring_d, m - ring_d)
    near = ring_d <= excl
    for _ in range(iters):
        Pn = (np.roll(P, 1, 0) + np.roll(P, -1, 0)) / 2
        P = P + smooth * (Pn - P)
        D = P[:, None, :] - P[None, :, :]
        d = np.linalg.norm(D, axis=-1)
        np.fill_diagonal(d, 1.0)
        mask = (d < repel) & ~near
        if mask.any():
            w = np.where(mask, (repel - d) / d, 0.0)
            P = P + step * (D * w[:, :, None]).sum(axis=1)
        c = P.mean(0)
        P -= c
        r = np.linalg.norm(P, axis=1).mean()
        if r > 1e-9:
            P *= target / r
    return P


def build_knot(braid_text, samples=240, iters=150, repel=0.35,
               scale=1.0, mirror=False):
    word = parse_letters(braid_text)
    if closure_components(word) != 1:
        raise ValueError("braid closure is a link, not a knot")
    pts = braid_closure_points(word)
    P = resample_closed(pts, max(samples, 4 * len(pts) // 3))
    P = relax_knot(P, iters=iters, repel=repel)
    P = resample_closed(P, samples)
    if mirror:
        P[:, 2] = -P[:, 2]
    # Center on the origin and fit within a 2 m cube (max extent
    # 2.0 * scale) so the knot lands framed by default.
    if len(P):
        lo = P.min(axis=0)
        hi = P.max(axis=0)
        cen = 0.5 * (lo + hi)
        ext = float((hi - lo).max())
        s = (2.0 * scale / ext) if ext > 1e-9 else scale
        P = (P - cen) * s
    return P


# ---- Blender layer -----------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           StringProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _ITEMS = [('CUSTOM', "Custom braid",
               "Closure of the braid word entered below")]
    _ITEMS += [(name, name.replace('_', '.'),
                f"{name.split('_')[0]} crossings, minimum braid "
                f"{braid}")
               for (name, braid, _ap) in KNOTS]
    _BRAIDS = {name: braid for (name, braid, _ap) in KNOTS}

    class CURVE_OT_prime_knot_add(bpy.types.Operator):
        """Add a prime knot (Rolfsen table, up to 10 crossings) from
        its verified minimum braid, relaxed into a smooth curve"""
        bl_idname = "curve.prime_knot_add"
        bl_label = "Prime Knot"
        bl_options = {'REGISTER', 'UNDO'}

        knot: EnumProperty(name="Knot", items=_ITEMS, default='3_1')
        braid: StringProperty(
            name="Braid Word", default="AAA",
            description="Letters a..z are braid generators, A..Z "
                        "their inverses (used for Custom)")
        samples: IntProperty(name="Curve Samples", default=240,
                             min=60, max=1000)
        iters: IntProperty(
            name="Relax Iterations", default=150, min=0, max=600,
            description="Smoothing + self-repulsion rounds (0 shows "
                        "the raw braid closure)")
        repel: FloatProperty(
            name="Strand Clearance", default=0.35, min=0.05, max=1.0,
            description="Self-repulsion distance while relaxing")
        mirror: BoolProperty(
            name="Mirror", default=False,
            description="Mirror image of the knot")
        output: EnumProperty(
            name="Output",
            items=[('BEZIER', "Bezier Curve", "auto-smoothed"),
                   ('POLY', "Poly Curve", ""),
                   ('NURBS', "NURBS Curve", ""),
                   ('MESH', "Mesh Tube", "swept tube mesh")],
            default='BEZIER')
        radius: FloatProperty(
            name="Tube Radius", default=0.08, min=0.0, max=1.0,
            step=1, precision=3,
            description="Curve bevel depth / tube radius")
        resolution: IntProperty(name="Bevel Resolution", default=6,
                                min=1, max=16)
        tube_sides: IntProperty(name="Tube Sides", default=12,
                                min=3, max=32)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            braid = (self.braid if self.knot == 'CUSTOM'
                     else _BRAIDS[self.knot])
            try:
                P = build_knot(braid, self.samples, self.iters,
                               self.repel, self.scale, self.mirror)
            except (ValueError, KeyError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            name = ("Knot " + self.knot.replace('_', '.')
                    if self.knot != 'CUSTOM' else "Knot custom")
            if self.output == 'MESH':
                verts, faces = self._tube(P, self.radius,
                                          self.tube_sides)
                me = bpy.data.meshes.new(name)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
                me.update()
                obj = bpy.data.objects.new(name, me)
            else:
                cu = bpy.data.curves.new(name, 'CURVE')
                cu.dimensions = '3D'
                if self.output == 'BEZIER':
                    sp = cu.splines.new('BEZIER')
                    sp.bezier_points.add(len(P) - 1)
                    for i, p in enumerate(P):
                        bp = sp.bezier_points[i]
                        bp.co = p
                        bp.handle_left_type = 'AUTO'
                        bp.handle_right_type = 'AUTO'
                else:
                    sp = cu.splines.new(self.output)
                    sp.points.add(len(P) - 1)
                    for i, p in enumerate(P):
                        sp.points[i].co = (p[0], p[1], p[2], 1.0)
                    if self.output == 'NURBS':
                        sp.order_u = 4
                sp.use_cyclic_u = True
                cu.bevel_depth = self.radius
                cu.bevel_resolution = self.resolution
                obj = bpy.data.objects.new(name, cu)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: braid {braid}, {len(P)} samples")
            return {'FINISHED'}

        @staticmethod
        def _tube(P, radius, sides):
            """Closed tube with parallel-transport frames; the frame
            holonomy is distributed so the seam matches."""
            import numpy as np
            m = len(P)
            T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
            T /= np.linalg.norm(T, axis=1)[:, None]
            N = np.empty_like(P)
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ref, T[0])) > 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            N[0] = np.cross(T[0], ref)
            N[0] /= np.linalg.norm(N[0])
            for i in range(1, m):
                v = N[i - 1] - T[i] * np.dot(N[i - 1], T[i])
                N[i] = v / (np.linalg.norm(v) or 1.0)
            # transport once more to measure the closure twist
            v = N[m - 1] - T[0] * np.dot(N[m - 1], T[0])
            v /= (np.linalg.norm(v) or 1.0)
            B0 = np.cross(T[0], N[0])
            ang = math.atan2(np.dot(v, B0), np.dot(v, N[0]))
            verts = []
            faces = []
            for i in range(m):
                corr = -ang * i / m
                B = np.cross(T[i], N[i])
                for j in range(sides):
                    a = 2 * math.pi * j / sides + corr
                    p = (P[i] + radius
                         * (math.cos(a) * N[i] + math.sin(a) * B))
                    verts.append(tuple(p))
            for i in range(m):
                i2 = (i + 1) % m
                for j in range(sides):
                    j2 = (j + 1) % sides
                    faces.append([i * sides + j, i * sides + j2,
                                  i2 * sides + j2, i2 * sides + j])
            return verts, faces

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'knot')
            if self.knot == 'CUSTOM':
                lay.prop(self, 'braid')
            for k in ('samples', 'iters', 'repel', 'mirror',
                      'output', 'radius'):
                lay.prop(self, k)
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            elif self.radius > 0:
                lay.prop(self, 'resolution')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.prime_knot_add",
                             icon='CURVE_DATA')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_prime_knot_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_prime_knot_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        fails = []
        counts = {}
        for (name, braid, ap) in KNOTS:
            word = parse_letters(braid)
            cr = int(name.split('_')[0])
            counts[cr] = counts.get(cr, 0) + 1
            ok = closure_components(word) == 1 \
                and alexander_at(word) == ap
            if not ok:
                fails.append(name)
                print(f"FAIL {name} {braid}")
        print(f"{len(KNOTS)} knots, per crossings: "
              f"{sorted(counts.items())}")
        print("braid table:",
              "ALL VERIFIED" if not fails else f"FAILURES {fails}")
        # embedding smoke test (no numpy relaxation here)
        for name in ('3_1', '4_1', '8_18', '10_124'):
            braid = dict((n, b) for (n, b, a) in KNOTS)[name]
            pts = braid_closure_points(parse_letters(braid))
            print(f"{name}: closure polyline {len(pts)} points")
