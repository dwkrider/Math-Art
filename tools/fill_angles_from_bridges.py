"""Complete pearce_table.py's truncated angle lists from Bridges 2009.

Pearce prints several angle sequences with a trailing "etc.", so the
transcription recorded only the printed prefix and set
`angles_truncated`.  Table 1 of van Ballegooijen, Gailiunas & Erdely's
"Spidronised Space-fillers" (Bridges 2009, pp. 271-278) lists the same
faces as 34 named NESTS with their full angle sequences, and maps each
nest to the Table 8.1 entries that use it -- so the gaps can be filled
from an independent publication rather than guessed.

The fill is conservative: a face is completed only when exactly one
nest of that polygon size is listed for the solid AND the recorded
prefix is a sub-multiset of the nest's full sequence.  Anything else is
left alone and reported.

References:
- Walt van Ballegooijen, Paul Gailiunas & Daniel Erdely, "Spidronised
  Space-fillers", Bridges 2009 Conference Proceedings, pp. 271-278 --
  Table 1, the 34 nests, their G-angles and the polyhedra using them.
"""
import collections
import re
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "math_art"))

MD = (r"C:\Users\dkrid\Projects\2026_07_21_Math_Art\research\journals"
      r"\bridges\2009\bridges2009-271\bridges2009-271.md")


def parse_angles(s):
    s = s.strip()
    m = re.fullmatch(r'(\d+)x\(([^)]*)\)', s)
    if m:
        return [float(x) for x in m.group(2).split(';')] * int(m.group(1))
    m = re.fullmatch(r'(\d+)x([\d.]+)', s)
    if m:
        return [float(m.group(2))] * int(m.group(1))
    if ';' in s:
        return [float(x) for x in s.split(';')]
    try:
        return [float(s)]
    except ValueError:
        return None


def nests(path=MD):
    out = {}
    for line in open(path, encoding='utf-8'):
        if not line.startswith('| n'):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) < 7:
            continue
        m = re.match(r'(\d+)-gon', c[3])
        if not m:
            continue
        out[c[0]] = dict(n=int(m.group(1)),
                         angles=parse_angles(c[6]),
                         solids=[int(x) for x in re.findall(r'\b(\d{2})\b', c[1])])
    return out


#: the ten Universal Node angles, exactly.  Bridges rounds to one
#: decimal (70.5) where Pearce prints arcminutes (70d32'), so every
#: Bridges value is SNAPPED to the nearest exact angle before being
#: written back.  That every one of them snaps to within a twentieth of
#: a degree is itself a check that the two tables describe the same
#: geometry.
import math as _math
LEGAL = (_math.degrees(_math.acos(_math.sqrt(2.0 / 3.0))),   # 35d16'
         45.0,
         _math.degrees(_math.acos(1.0 / _math.sqrt(3.0))),   # 54d44'
         60.0,
         _math.degrees(_math.acos(1.0 / 3.0)),               # 70d32'
         90.0,
         _math.degrees(_math.acos(-1.0 / 3.0)),              # 109d28'
         120.0,
         _math.degrees(_math.acos(-_math.sqrt(2.0 / 3.0))),  # 144d44'
         180.0)


def snap(v, tol=0.05):
    best = min(LEGAL, key=lambda a: abs(a - v))
    if abs(best - v) > tol:
        raise ValueError("%.3f is not a Universal Node angle" % v)
    return best


def label(v):
    v = snap(v)
    d = int(v)
    mnt = int(round((v - d) * 60.0))
    if mnt == 60:
        d, mnt = d + 1, 0
    return "%dd%02d'" % (d, mnt) if mnt else "%dd" % d


def report():
    import pearce_table as pt
    N = nests()
    by_solid = collections.defaultdict(list)
    for code, d in N.items():
        for s in d['solids']:
            by_solid[s].append(code)

    filled = skipped = 0
    for r in pt.TABLE:
        codes = by_solid.get(r['number'], [])
        for f in r['faces']:
            have = list(f['angles'])
            if len(have) >= f['n']:
                continue                      # already complete
            cand = [N[c] for c in codes
                    if N[c]['n'] == f['n'] and N[c]['angles']]
            if len(cand) != 1:
                skipped += 1
                print("  #%-3d %2d-gon: %d candidate nests -- left alone"
                      % (r['number'], f['n'], len(cand)))
                continue
            full = [label(x) for x in cand[0]['angles']]
            pre = collections.Counter(have)
            allc = collections.Counter(full)
            if any(allc[k] < v for k, v in pre.items()):
                skipped += 1
                print("  #%-3d %2d-gon: prefix %s not in nest %s -- left alone"
                      % (r['number'], f['n'], have, full))
                continue
            filled += 1
            print("  #%-3d %2d-gon: %s -> %s"
                  % (r['number'], f['n'], ','.join(have), ','.join(full)))
    print()
    print("fillable: %d   left alone: %d" % (filled, skipped))


if __name__ == "__main__":
    report()
