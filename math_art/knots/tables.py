# Knot table: minimum braid words.
#
# Part of the Math Art knot engine (`math_art/knots/`), extracted from the
# generators that had accumulated it.  NumPy/stdlib only -- no `bpy` -- so
# the engine imports and self-tests headlessly; the registered operators
# stay in their flat generator modules and import this package.
#
# The 249 minimum braid words are Table 1 of:
#   Thomas A. Gittings, "Minimum braids: a complete invariant of knots
#   and links", arXiv:math/0401051 (2004).
# Each row is (name, braid word, alexander).  The third column is Gittings'
# published Alexander value at t = 10, and it is what makes the table
# self-verifying: `alexander.alexander_at(word)` -- whose default
# evaluation point is x = 10 -- must reproduce it.
#
# NOT the knot determinant, which is |Delta(-1)|: the trefoil's determinant
# is 3, while its entry here is 91.

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


def _selftest():
    # Structural gates on the table itself.  The MATHEMATICAL gate -- that
    # every row's Alexander value reproduces its recorded column -- runs in
    # alexander.py (spot checks) and over all 249 rows in
    # prime_knot_generator._selftest.
    ok = True
    names = [n for n, _w, _a in KNOTS]
    good = len(names) == len(set(names))
    ok &= good
    print(f"tables: {len(KNOTS)} rows, names unique {'OK' if good else 'FAIL'}")

    # Every word is drawn from the letter alphabet, and a knot named N_k
    # must have a braid word with at least N letters -- a word shorter than
    # the crossing number cannot represent it.
    bad = []
    for name, word, det in KNOTS:
        cr = int(name.split('_')[0])
        if not word or not all(c.isalpha() for c in word):
            bad.append(f"{name}:alphabet")
        elif len(word) < cr:
            bad.append(f"{name}:{len(word)}<{cr}")
        elif not isinstance(det, int):
            bad.append(f"{name}:det")
    good = not bad
    ok &= good
    print(f"tables: words well-formed and >= crossing number "
          f"{'OK' if good else 'FAIL ' + ','.join(bad[:5])}")

    # The table is ordered by crossing number, and the counts match the
    # classical census (1 trefoil, 1 figure-eight, 2 five-crossing, 3 six,
    # 7 seven, 21 eight, 49 nine, 165 ten).
    counts = {}
    for name, _w, _a in KNOTS:
        cr = int(name.split('_')[0])
        counts[cr] = counts.get(cr, 0) + 1
    census = {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165}
    good = counts == census
    ok &= good
    print(f"tables: census {sorted(counts.items())} "
          f"{'OK' if good else 'FAIL exp ' + str(sorted(census.items()))}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("tables self-test failed")
