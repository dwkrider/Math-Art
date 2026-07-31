
# Notable / Other Polyhedra for Blender
#
# A gallery of famous individual polyhedra that don't fit the parametric
# families: the final stellation of the icosahedron (the "echidnahedron", the
# outermost of the 59 stellations), the Schonhardt polyhedron (the twisted
# octahedron that cannot be triangulated without extra vertices), Jessen's
# orthogonal icosahedron (a shaky polyhedron whose dihedral angles are all
# right angles), Durer's solid (the truncated rhombohedron from Melencolia I),
# the Bilinski dodecahedron (the "other" rhombic dodecahedron, of golden
# rhombi), Escher's Solid (the stellated rhombic dodecahedron of M. C.
# Escher's "Waterfall"), and a polyhedral realization of Felix Klein's
# regular map {3,7}_8 on a genus-3 surface.
#
# References:
# - M. C. Escher, "Waterfall" (1961) and "Study for Stars" (1948); the
#   solid is the first stellation of the rhombic dodecahedron.
# - J. C. P. Miller / H. S. M. Coxeter, P. Du Val, H. T. Flather &
#   J. F. Petrie, "The Fifty-Nine Icosahedra" (1938); the final stellation
#   is Wenninger model W42. Echidnahedron connectivity from A. Hume's netlib
#   polyhedra database (file 141).
# - E. Schonhardt, "Ueber die Zerlegung von Dreieckspolyedern in
#   Tetraeder", Math. Ann. 98 (1928), 309-312.
# - B. Jessen, "Orthogonal icosahedra", Nordisk Mat. Tidskr. 15 (1967).
# - A. Durer, "Melencolia I" (1514); analysis: see Weitzel, Schreiber.
# - S. Bilinski, "Ueber die Rhombenisoeder", Glasnik Mat.-Fiz. Astr. 15
#   (1960), 251-263.
# - F. Klein, "Ueber die Transformation siebenter Ordnung der elliptischen
#   Functionen", Math. Ann. 14 (1878) (the quartic / map {3,7}_8);
#   polyhedral realization: E. Schulte & J. M. Wills, "A polyhedral
#   realization of Felix Klein's map {3,7}_8 on a Riemann surface of genus
#   3", J. London Math. Soc. 32 (1985), 539-547.  (The Klein coordinates
#   here are a construction on two homothetic truncated tetrahedra,
#   verified to carry the full 168 automorphisms of {3,7}_8.)

bl_info = {
    "name": "Notable Polyhedra",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Final stellation of the icosahedron (echidnahedron), "
                   "Schonhardt, Jessen's orthogonal icosahedron, Durer's "
                   "solid, Bilinski dodecahedron, Klein's regular map",
    "category": "Add Mesh",
}

import math

PHI = (1 + 5 ** 0.5) / 2


def schonhardt(twist_deg=40.0):
    """Schonhardt polyhedron: two parallel triangles, the top twisted so the
    three long side diagonals become concave (non-convex; cannot be split
    into tetrahedra without a new vertex)."""
    tw = math.radians(twist_deg)
    A = [(math.cos(2 * math.pi * k / 3), math.sin(2 * math.pi * k / 3), -0.5)
         for k in range(3)]
    B = [(math.cos(2 * math.pi * k / 3 + tw),
          math.sin(2 * math.pi * k / 3 + tw), 0.5) for k in range(3)]
    V = A + B
    F = [[2, 1, 0], [3, 4, 5]]
    for i in range(3):
        a0, a1, b0, b1 = i, (i + 1) % 3, 3 + i, 3 + (i + 1) % 3
        F += [[a0, a1, b0], [a1, b1, b0]]      # concave split
    return V, F


def escher():
    """Escher's Solid: the (first) stellation of the rhombic dodecahedron --
    a pyramid raised on each of the 12 rhombic faces, its apex where the four
    neighbouring face-planes meet (= twice the face centre).  The star solid
    of M. C. Escher's "Waterfall" (1961) and "Study for Stars".  V26/E72/F48."""
    cube = [(float(sx), float(sy), float(sz))
            for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
    octa = [(2., 0., 0.), (-2., 0., 0.), (0., 2., 0.), (0., -2., 0.),
            (0., 0., 2.), (0., 0., -2.)]
    V = cube + octa
    idx = {tuple(p): i for i, p in enumerate(V)}
    normals = []
    for a in (1, -1):
        for b in (1, -1):
            normals += [(a, b, 0), (a, 0, b), (0, a, b)]
    F = []
    for n in normals:
        octv = [o for o in octa if sum(o[i] * n[i] for i in range(3)) == 2]
        cubv = [c for c in cube if sum(c[i] * n[i] for i in range(3)) == 2]
        cen = tuple(sum(p[i] for p in octv + cubv) / 4.0 for i in range(3))
        u = tuple(octv[0][i] - cen[i] for i in range(3))
        un = math.sqrt(sum(t * t for t in u)) or 1.0
        u = tuple(t / un for t in u)
        nn = math.sqrt(sum(t * t for t in n)) or 1.0
        nu = tuple(t / nn for t in n)
        w = (nu[1] * u[2] - nu[2] * u[1], nu[2] * u[0] - nu[0] * u[2],
             nu[0] * u[1] - nu[1] * u[0])

        def ang(p, cen=cen, u=u, w=w):
            d = tuple(p[i] - cen[i] for i in range(3))
            return math.atan2(sum(d[i] * w[i] for i in range(3)),
                              sum(d[i] * u[i] for i in range(3)))
        ring = sorted(octv + cubv, key=ang)
        apex = tuple(2.0 * c for c in n)
        ai = idx.get(apex)
        if ai is None:
            ai = len(V)
            V.append(apex)
            idx[apex] = ai
        ri = [idx[tuple(p)] for p in ring]
        for k in range(4):
            F.append([ri[k], ri[(k + 1) % 4], ai])
    return V, F


# --- Final stellation of the icosahedron (echidnahedron, Wenninger W42) ---
# 92 vertices in three concentric shells -- inner regular dodecahedron (20),
# middle regular icosahedron (12), outer nonuniform truncated icosahedron (60,
# the spike tips) -- at radius ratios
#   sqrt(3/2(3+r5)) : sqrt(1/2(25+11 r5)) : sqrt(1/2(97+43 r5)),  r5=sqrt(5).
# 180 triangles lying in the 20 icosahedral face-planes (9 per plane), each
# plane carrying a 9/4 enneagram face. Strongly non-convex: only the 60 outer
# tips are on the convex hull, the 32 inner vertices are interior. Connectivity
# from Andrew Hume's netlib polyhedra database (file 141, the file that coined
# the name "echidnahedron"), recentred and rescaled to max|coord|=1.
_ECHI_V = [
    [-1, -0.111544831, -0.0168070986], [-0.96830656, 0.1303580473, -0.2410682751],
    [-0.9651189231, 0.0226388584, 0.2841789475], [-0.91383786, 0.4140459375, -0.0786832584],
    [-0.9118677921, 0.3474718175, 0.2459373776], [-0.800207652, -0.5987353418, -0.1179340154],
    [-0.7172331489, 0.0345746156, -0.7050573977], [-0.7088878072, -0.2474378823, 0.6700576834],
    [-0.6450357503, -0.6579327582, -0.4046950635], [-0.6418481134, -0.7656519472, 0.120552159],
    [-0.5937546871, -0.2665256791, -0.7675572694], [-0.5854093455, -0.548538177, 0.6075578117],
    [-0.574632241, 0.7772791543, -0.2799279048], [-0.5694745362, 0.6029858453, 0.5699399538],
    [-0.5075925471, 0.2590650894, -0.8294334292], [-0.5024348423, 0.0847717805, 0.0204344294],
    [-0.4972771376, -0.0895215285, 0.8703022879], [-0.4194603392, 0.7180817379, -0.5666889529],
    [-0.4111149976, 0.43606924, 0.8084261281], [-0.3907747023, -0.8614353789, -0.3434369636],
    [-0.3888046343, -0.9280094989, -0.0188163276], [-0.3630215713, 0.9351955081, -0.0796833003],
    [-0.3598339344, 0.8274763191, 0.4455639222], [-0.3078001991, -0.2281254214, -0.9305603459],
    [-0.3026424944, -0.4024187304, -0.0806924874], [-0.2974847896, -0.5767120393, 0.7691753711],
    [-0.2545490681, 0.0967075377, -0.9688019158], [-0.2513614312, -0.0110116513, -0.4435546932],
    [-0.2493913633, -0.0775857713, -0.1189340572], [-0.2481737944, -0.1187308402, 0.0816925293],
    [-0.2462037264, -0.1853049602, 0.4063131653], [-0.2430160896, -0.2930241492, 0.9315603878],
    [-0.2164803544, 0.1231720381, -0.1425686472], [-0.2145102864, 0.0565979181, 0.1820519888],
    [-0.1949226633, 0.2061021189, 0.0434509594], [-0.1632292233, 0.4480049972, -0.180810217],
    [-0.1600415864, 0.3402858083, 0.3444370055], [-0.1119481602, 0.8394120764, -0.5436724229],
    [-0.1036028185, 0.5573995785, 0.8314426582], [-0.0930018926, -0.1779282566, -0.2050685189],
    [-0.0910318247, -0.2445023766, 0.1195521171], [-0.0770670833, 0.9735957657, -0.2426863768],
    [-0.0719093785, 0.7993024568, 0.6071814817], [-0.0397507616, 0.1469047025, -0.2433100887],
    [-0.0365631247, 0.0391855136, 0.2819371338], [-0.0048696847, 0.2810883919, 0.0576759573],
    [0.0048696847, -0.2810883919, -0.0576759573], [0.0365631247, -0.0391855136, -0.2819371338],
    [0.0397507616, -0.1469047025, 0.2433100887], [0.0719093785, -0.7993024568, -0.6071814817],
    [0.0770670833, -0.9735957657, 0.2426863768], [0.0910318247, 0.2445023766, -0.1195521171],
    [0.0930018926, 0.1779282566, 0.2050685189], [0.1036028185, -0.5573995784, -0.8314426582],
    [0.1119481602, -0.8394120764, 0.5436724229], [0.1600415864, -0.3402858083, -0.3444370055],
    [0.1632292233, -0.4480049972, 0.180810217], [0.1949226633, -0.2061021189, -0.0434509594],
    [0.2145102864, -0.0565979181, -0.1820519888], [0.2164803544, -0.1231720381, 0.1425686472],
    [0.2430160896, 0.2930241492, -0.9315603878], [0.2462037264, 0.1853049602, -0.4063131653],
    [0.2481737944, 0.1187308402, -0.0816925293], [0.2493913633, 0.0775857713, 0.1189340572],
    [0.2513614312, 0.0110116513, 0.4435546932], [0.2545490681, -0.0967075377, 0.9688019158],
    [0.2974847896, 0.5767120393, -0.7691753711], [0.3026424944, 0.4024187304, 0.0806924874],
    [0.3078001991, 0.2281254214, 0.9305603459], [0.3598339344, -0.8274763191, -0.4455639222],
    [0.3630215713, -0.9351955081, 0.0796833003], [0.3888046343, 0.9280094989, 0.0188163276],
    [0.3907747023, 0.8614353789, 0.3434369636], [0.4111149976, -0.43606924, -0.8084261281],
    [0.4194603392, -0.7180817379, 0.5666889529], [0.4972771376, 0.0895215285, -0.8703022879],
    [0.5024348423, -0.0847717805, -0.0204344294], [0.5075925471, -0.2590650894, 0.8294334292],
    [0.5694745362, -0.6029858453, -0.5699399538], [0.574632241, -0.7772791543, 0.2799279048],
    [0.5854093455, 0.548538177, -0.6075578117], [0.5937546871, 0.2665256791, 0.7675572694],
    [0.6418481134, 0.7656519472, -0.120552159], [0.6450357503, 0.6579327582, 0.4046950635],
    [0.7088878072, 0.2474378823, -0.6700576834], [0.7172331489, -0.0345746156, 0.7050573977],
    [0.800207652, 0.5987353418, 0.1179340154], [0.9118677921, -0.3474718175, -0.2459373776],
    [0.91383786, -0.4140459375, 0.0786832584], [0.9651189231, -0.0226388584, -0.2841789475],
    [0.96830656, -0.1303580473, 0.2410682751], [1, 0.111544831, 0.0168070986],
]
_ECHI_F = [
    [65, 64, 44], [65, 44, 48], [65, 48, 64], [77, 64, 48], [77, 48, 59],
    [77, 59, 64], [85, 64, 59], [85, 59, 63], [85, 63, 64], [81, 64, 63],
    [81, 63, 52], [81, 52, 64], [68, 64, 52], [68, 52, 44], [68, 44, 64],
    [31, 30, 48], [31, 48, 44], [31, 44, 30], [16, 30, 44], [16, 44, 33],
    [16, 33, 30], [7, 30, 33], [7, 33, 29], [7, 29, 30], [11, 30, 29],
    [11, 29, 40], [11, 40, 30], [25, 30, 40], [25, 40, 48], [25, 48, 30],
    [74, 56, 59], [74, 59, 48], [74, 48, 56], [54, 56, 48], [54, 48, 40],
    [54, 40, 56], [50, 56, 40], [50, 40, 46], [50, 46, 56], [70, 56, 46],
    [70, 46, 57], [70, 57, 56], [79, 56, 57], [79, 57, 59], [79, 59, 56],
    [90, 76, 63], [90, 63, 59], [90, 59, 76], [88, 76, 59], [88, 59, 57],
    [88, 57, 76], [87, 76, 57], [87, 57, 58], [87, 58, 76], [89, 76, 58],
    [89, 58, 62], [89, 62, 76], [91, 76, 62], [91, 62, 63], [91, 63, 76],
    [83, 67, 52], [83, 52, 63], [83, 63, 67], [86, 67, 63], [86, 63, 62],
    [86, 62, 67], [82, 67, 62], [82, 62, 51], [82, 51, 67], [71, 67, 51],
    [71, 51, 45], [71, 45, 67], [72, 67, 45], [72, 45, 52], [72, 52, 67],
    [38, 36, 44], [38, 44, 52], [38, 52, 36], [42, 36, 52], [42, 52, 45],
    [42, 45, 36], [22, 36, 45], [22, 45, 34], [22, 34, 36], [13, 36, 34],
    [13, 34, 33], [13, 33, 36], [18, 36, 33], [18, 33, 44], [18, 44, 36],
    [6, 27, 28], [6, 28, 32], [6, 32, 27], [14, 27, 32], [14, 32, 43],
    [14, 43, 27], [26, 27, 43], [26, 43, 47], [26, 47, 27], [23, 27, 47],
    [23, 47, 39], [23, 39, 27], [10, 27, 39], [10, 39, 28], [10, 28, 27],
    [1, 15, 32], [1, 32, 28], [1, 28, 15], [0, 15, 28], [0, 28, 29],
    [0, 29, 15], [2, 15, 29], [2, 29, 33], [2, 33, 15], [4, 15, 33],
    [4, 33, 34], [4, 34, 15], [3, 15, 34], [3, 34, 32], [3, 32, 15],
    [17, 35, 43], [17, 43, 32], [17, 32, 35], [12, 35, 32], [12, 32, 34],
    [12, 34, 35], [21, 35, 34], [21, 34, 45], [21, 45, 35], [41, 35, 45],
    [41, 45, 51], [41, 51, 35], [37, 35, 51], [37, 51, 43], [37, 43, 35],
    [60, 61, 47], [60, 47, 43], [60, 43, 61], [66, 61, 43], [66, 43, 51],
    [66, 51, 61], [80, 61, 51], [80, 51, 62], [80, 62, 61], [84, 61, 62],
    [84, 62, 58], [84, 58, 61], [75, 61, 58], [75, 58, 47], [75, 47, 61],
    [53, 55, 39], [53, 39, 47], [53, 47, 55], [73, 55, 47], [73, 47, 58],
    [73, 58, 55], [78, 55, 58], [78, 58, 57], [78, 57, 55], [69, 55, 57],
    [69, 57, 46], [69, 46, 55], [49, 55, 46], [49, 46, 39], [49, 39, 55],
    [8, 24, 28], [8, 28, 39], [8, 39, 24], [19, 24, 39], [19, 39, 46],
    [19, 46, 24], [20, 24, 46], [20, 46, 40], [20, 40, 24], [9, 24, 40],
    [9, 40, 29], [9, 29, 24], [5, 24, 29], [5, 29, 28], [5, 28, 24],
]

GALLERY = {
    "ECHIDNAHEDRON": {"name": "Final Stellation of Icosahedron",
                      "V": _ECHI_V, "F": _ECHI_F},
    "JESSEN": {"name": "Jessen's Orthogonal Icosahedron",
               "V": [[-2, -1, 0], [-2, 1, 0], [-1, 0, -2], [-1, 0, 2],
                     [0, -2, -1], [0, -2, 1], [0, 2, -1], [0, 2, 1],
                     [1, 0, -2], [1, 0, 2], [2, -1, 0], [2, 1, 0]],
               "F": [[0, 2, 4], [5, 3, 0], [6, 2, 1], [1, 3, 7], [4, 8, 10],
                     [10, 9, 5], [11, 8, 6], [7, 9, 11], [3, 2, 0],
                     [0, 4, 10], [10, 5, 0], [1, 2, 3], [11, 6, 1],
                     [1, 7, 11], [6, 4, 2], [3, 5, 7], [4, 6, 8], [9, 7, 5],
                     [8, 9, 10], [11, 9, 8]]},
    "DURER": {"name": "Durer's Solid (Melencolia I)",
              "V": [[0.0, 0.419469524122, -0.64771662102],
                    [-0.363271264003, -0.209734762061, -0.64771662102],
                    [0.363271264003, -0.209734762061, -0.64771662102],
                    [0.0, 0.678715947274, -0.367200443531],
                    [-0.587785252292, -0.339357973637, -0.367200443531],
                    [0.587785252292, -0.339357973637, -0.367200443531],
                    [0.0, -0.678715947274, 0.367200443531],
                    [0.587785252292, 0.339357973637, 0.367200443531],
                    [-0.587785252292, 0.339357973637, 0.367200443531],
                    [0.0, -0.419469524122, 0.64771662102],
                    [0.363271264003, 0.209734762061, 0.64771662102],
                    [-0.363271264003, 0.209734762061, 0.64771662102]],
              "F": [[2, 1, 0], [9, 10, 11], [1, 4, 8, 3, 0], [9, 6, 5, 7, 10],
                    [2, 5, 6, 4, 1], [10, 7, 3, 8, 11], [0, 3, 7, 5, 2],
                    [11, 8, 4, 6, 9]]},
    "BILINSKI": {"name": "Bilinski Dodecahedron",
                 "V": [[0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                       [PHI, 0, PHI], [PHI, 0, -PHI], [-PHI, 0, PHI],
                       [-PHI, 0, -PHI], [PHI, 1, 0], [PHI, -1, 0],
                       [-PHI, 1, 0], [-PHI, -1, 0], [0, 0, PHI * PHI],
                       [0, 0, -PHI * PHI]],
                 "F": [[10, 0, 8, 1], [8, 0, 12, 4], [12, 0, 10, 6],
                       [13, 1, 8, 5], [10, 1, 13, 7], [9, 2, 11, 3],
                       [12, 2, 9, 4], [11, 2, 12, 6], [9, 3, 13, 5],
                       [13, 3, 11, 7], [8, 4, 9, 5], [11, 6, 10, 7]]},
    "KLEIN": {"name": "Klein Regular Map {3,7}_8 (genus 3)",
              "V": [[1, 1, 3], [1, -3, -1], [-3, 1, -1], [1, 3, 1],
                    [-1.5, -0.5, 0.5], [-1, -1, 3], [1.5, 0.5, 0.5],
                    [1.5, -0.5, -0.5], [0.5, -1.5, -0.5], [3, 1, 1],
                    [1, -1, -3], [-0.5, -0.5, 1.5], [0.5, -0.5, -1.5],
                    [-1.5, 0.5, -0.5], [3, -1, -1], [-1, -3, 1],
                    [-0.5, 0.5, -1.5], [-3, -1, 1], [0.5, 0.5, 1.5],
                    [-0.5, -1.5, 0.5], [0.5, 1.5, 0.5], [-1, 1, -3],
                    [-0.5, 1.5, -0.5], [-1, 3, -1]],
              "F": [[0, 9, 3], [0, 3, 4], [0, 4, 5], [0, 5, 6], [0, 6, 7],
                    [0, 7, 8], [0, 8, 9], [1, 13, 10], [1, 14, 11],
                    [1, 15, 12], [1, 16, 13], [1, 10, 14], [1, 11, 15],
                    [1, 12, 16], [2, 19, 17], [2, 20, 18], [2, 21, 19],
                    [2, 22, 20], [2, 23, 21], [2, 17, 22], [2, 18, 23],
                    [3, 11, 4], [4, 13, 5], [5, 15, 6], [6, 10, 7],
                    [7, 12, 8], [8, 14, 9], [9, 16, 3], [3, 18, 11],
                    [4, 21, 13], [5, 17, 15], [6, 20, 10], [7, 23, 12],
                    [8, 19, 14], [9, 22, 16], [3, 23, 18], [4, 19, 21],
                    [5, 22, 17], [6, 18, 20], [7, 21, 23], [8, 17, 19],
                    [9, 20, 22], [3, 16, 23], [4, 11, 19], [5, 13, 22],
                    [6, 15, 18], [7, 10, 21], [8, 12, 17], [9, 14, 20],
                    [10, 20, 14], [11, 18, 15], [12, 23, 16], [13, 21, 10],
                    [14, 19, 11], [15, 17, 12], [16, 22, 13]]},
}

ITEMS = [("ECHIDNAHEDRON", "Final Stellation of Icosahedron",
          "echidnahedron -- the outermost stellation (W42)"),
         ("SCHONHARDT", "Schonhardt Polyhedron", "twisted octahedron"),
         ("JESSEN", "Jessen's Orthogonal Icosahedron", "all right angles"),
         ("DURER", "Durer's Solid", "Melencolia I"),
         ("BILINSKI", "Bilinski Dodecahedron", "golden rhombi"),
         ("ESCHER", "Escher's Solid",
          "stellated rhombic dodecahedron (Escher's Waterfall)"),
         ("KLEIN", "Klein Regular Map {3,7} (genus 3)", "")]


def build(kind):
    if kind == 'SCHONHARDT':
        V, F = schonhardt()
    elif kind == 'ESCHER':
        V, F = escher()
    else:
        S = GALLERY[kind]
        V = [tuple(float(c) for c in v) for v in S["V"]]
        F = [list(f) for f in S["F"]]
    cen = [sum(v[i] for v in V) / len(V) for i in range(3)]
    V = [tuple(v[i] - cen[i] for i in range(3)) for v in V]
    mx = max((abs(c) for v in V for c in v), default=1.0) or 1.0
    return [tuple(c / mx for c in v) for v in V], F


def _self_test():
    want = {'ECHIDNAHEDRON': (92, 270, 180, 2),
            'SCHONHARDT': (6, 12, 8, 2), 'JESSEN': (12, 30, 20, 2),
            'DURER': (12, 18, 8, 2), 'BILINSKI': (14, 24, 12, 2),
            'ESCHER': (26, 72, 48, 2), 'KLEIN': (24, 84, 56, -4)}
    for kind, _lbl, _d in ITEMS:
        V, F = build(kind)
        E = {}
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                k = (min(a, b), max(a, b))
                E[k] = E.get(k, 0) + 1
        chi = len(V) - len(E) + len(F)
        e2 = all(v == 2 for v in E.values())
        print(f"{kind:11s} V={len(V):2d} E={len(E):2d} F={len(F):2d} "
              f"chi={chi:3d} edge-in-2={e2}  want{want[kind]}")


try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_notable_polyhedron_add(bpy.types.Operator):
        """Add a notable individual polyhedron (the echidnahedron / final
        stellation of the icosahedron, Schonhardt, Jessen, Durer, Bilinski,
        or Klein's genus-3 regular map)"""
        bl_idname = "mesh.notable_polyhedron_add"
        bl_label = "Notable Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(name="Solid", items=ITEMS)
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", ""),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo Style "
                    "modifier (as in Leonardo's drawings for Pacioli's "
                    "De divina proportione)"),
                   ('WIRE', "Wireframe", "Wireframe modifier")],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.3, min=0.02, max=0.95,
            description="Leonardo face frame width (fraction of the face)")
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.001, max=1.0,
            description="Panel / strut thickness")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            V, F = build(self.solid)
            label = dict((i[0], i[1]) for i in ITEMS)[self.solid]
            me = bpy.data.meshes.new(label)
            me.from_pydata([tuple(c * self.scale for c in v) for v in V],
                           [], [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            me.update()
            obj = bpy.data.objects.new(label, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.style == 'LEONARDO':
                try:
                    from . import leonardo_style
                except ImportError:
                    import leonardo_style
                leonardo_style.add_modifier(obj, self.border, self.thickness)
            elif self.style == 'WIRE':
                mod = obj.modifiers.new("Wireframe", 'WIREFRAME')
                mod.thickness = self.thickness
                mod.use_even_offset = False
            self.report({'INFO'}, f"{label}: V={len(V)} F={len(F)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'solid')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style != 'SOLID':
                lay.prop(self, 'thickness')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.notable_polyhedron_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_notable_polyhedron_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_notable_polyhedron_add)


if __name__ == "__main__" and not _IN_BLENDER:
    _self_test()
