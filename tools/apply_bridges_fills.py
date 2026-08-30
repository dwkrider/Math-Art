"""Apply the Bridges angle fills to pearce_table.py, in place."""
import re, sys, collections, importlib
sys.path.insert(0, 'math_art'); sys.path.insert(0, 'tools')
import pearce_table as pt
import fill_angles_from_bridges as fb

N = fb.nests()
by_solid = collections.defaultdict(list)
for code, d in N.items():
    for s in d['solids']:
        by_solid[s].append(code)

# what to fill: (number, face-index) -> new angle tuple
fills = {}
for r in pt.TABLE:
    codes = by_solid.get(r['number'], [])
    for fi, f in enumerate(r['faces']):
        have = list(f['angles'])
        if len(have) >= f['n']:
            continue
        cand = [N[c] for c in codes if N[c]['n'] == f['n'] and N[c]['angles']]
        if len(cand) != 1:
            continue
        try:
            full = [fb.label(x) for x in cand[0]['angles']]
        except ValueError:
            continue
        pre, allc = collections.Counter(have), collections.Counter(full)
        if any(allc[k] < v for k, v in pre.items()):
            continue
        fills[(r['number'], fi)] = full
print("fills to apply: %d" % len(fills))

src = open('math_art/pearce_table.py', encoding='utf-8').read()
# split into entry blocks
starts = [(m.start(), int(m.group(1)))
          for m in re.finditer(r'dict\(number=(\d+),', src)]
starts.append((len(src), None))
out = []
prev = 0
applied = 0
for i, (pos, num) in enumerate(starts[:-1]):
    end = starts[i + 1][0]
    block = src[pos:end]
    spans = list(re.finditer(r'angles=\(([^)]*)\)', block))
    newblock = block
    for fi in range(len(spans) - 1, -1, -1):    # back to front
        key = (num, fi)
        if key not in fills:
            continue
        rep = 'angles=(' + ', '.join('"%s"' % a if "'" in a else "'%s'" % a
                                     for a in fills[key]) + ')'
        m = spans[fi]
        newblock = newblock[:m.start()] + rep + newblock[m.end():]
        applied += 1
    out.append(src[prev:pos]); out.append(newblock); prev = end
out.append(src[prev:])
open('math_art/pearce_table.py', 'w', encoding='utf-8').write(''.join(out))
print("applied: %d" % applied)
