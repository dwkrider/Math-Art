"""Reader for the subset of Surface Evolver datafile syntax that Ken
Brakke's triply-periodic collection uses.

WHY THIS EXISTS.  Every surface on
`kenbrakke.com/evolver/examples/periodic/periodic.html` is fully
specified by its `.fe` file: the boundary contour, the symmetry
generators as explicit 4x4 matrices, and the word that assembles the
cell.  Transcribing those by hand is what went wrong repeatedly while
this subsystem was being built -- a Gauss map taken from the wrong cell
of a notebook, three hybrids built from one contour when their contours
differ, a screw's determinant sign copied wrong.  Each of those passed
every topology check and produced the wrong surface.

Reading the datafile removes the transcription step entirely.  What is
left to get wrong is this parser, which is one thing to test rather than
forty.

WHAT IT HANDLES, and nothing more:

  parameter NAME = expr        #define NAME expr
  vertices                     id x y z [attributes]
  edges                        id v1 v2 [attributes]
  faces                        id e1 e2 e3 ...   (negative = reversed)
  view_transform_generators N  followed by N 4x4 matrices
  transform_expr "word"        wherever it appears, keyed by command name

Expressions may use the parameters, the four arithmetic operators,
parentheses, unary minus, `sqrt(...)` and `pi`.  They are evaluated by
`_expr`, a small recursive-descent parser -- NOT by `eval`, because
these files come off a website.

WHAT IT DOES NOT HANDLE: constraints, energies, boundary
parameterisations, torus/period declarations, or anything in the command
language beyond picking `transform_expr` strings out of it.  Files using
those are still read; the unsupported parts are ignored, so a caller
must check `fe.torus` before trusting the generator list, since a torus
model appends its period translations implicitly.
"""

import math
import os
import re

import numpy as np

_NUM = re.compile(r'(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')
_NAME = re.compile(r'[A-Za-z_]\w*')

# Where the datafiles live.
#
# In the REPO, under `research/data/evolver-datafiles` -- which is
# gitignored, so the files are on hand for authoring without being
# committed.  Only `tools/bake_fe_cells.py` and this module's self-test
# read them; the shipped extension carries `minsurf/fecells.py`, which is
# generated from the surface database, and never needs the datafiles or
# this path at all.  Their absence is therefore a SKIP, not a failure.
#
# `MATH_ART_FE_DIR` overrides, for a checkout that keeps them elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))       # math_art/minsurf -> repo

MIRROR_DOWNLOADS = os.environ.get(
    'MATH_ART_FE_DIR',
    os.path.join(_REPO, 'research', 'data', 'evolver-datafiles'))


class FEError(ValueError):
    pass


def _expr(text, env):
    """Evaluate one Evolver arithmetic expression against `env`."""
    s = text.strip()
    pos = [0]

    def peek():
        while pos[0] < len(s) and s[pos[0]] in ' \t':
            pos[0] += 1
        return s[pos[0]] if pos[0] < len(s) else ''

    def atom():
        c = peek()
        if c == '(':
            pos[0] += 1
            v = expr()
            if peek() != ')':
                raise FEError("unbalanced ( in %r" % text)
            pos[0] += 1
            return v
        if c == '-':
            pos[0] += 1
            return -atom()
        if c == '+':
            pos[0] += 1
            return atom()
        m = _NUM.match(s, pos[0])
        if m:
            pos[0] = m.end()
            return float(m.group(0))
        m = _NAME.match(s, pos[0])
        if m:
            pos[0] = m.end()
            name = m.group(0)
            if peek() == '(':
                pos[0] += 1
                v = expr()
                if peek() != ')':
                    raise FEError("unbalanced ( in %r" % text)
                pos[0] += 1
                fn = {'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
                      'tan': math.tan, 'exp': math.exp, 'log': math.log,
                      'atan': math.atan, 'asin': math.asin,
                      'acos': math.acos, 'abs': abs}.get(name.lower())
                if fn is None:
                    raise FEError("unknown function %r" % name)
                return fn(v)
            if name.lower() == 'pi':
                return math.pi
            if name in env:
                return float(env[name])
            raise FEError("unknown name %r in %r" % (name, text))
        raise FEError("cannot parse %r at %d" % (text, pos[0]))

    def power():
        v = atom()
        if peek() == '^':
            pos[0] += 1
            return v ** power()
        return v

    def term():
        v = power()
        while True:
            c = peek()
            if c == '*':
                pos[0] += 1
                v *= power()
            elif c == '/':
                pos[0] += 1
                v /= power()
            else:
                return v

    def expr():
        v = term()
        while True:
            c = peek()
            if c == '+':
                pos[0] += 1
                v += term()
            elif c == '-':
                pos[0] += 1
                v -= term()
            else:
                return v

    v = expr()
    if peek():
        raise FEError("trailing %r in %r" % (s[pos[0]:], text))
    return v


def _strip(src):
    """Remove // and /* */ comments."""
    src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
    return re.sub(r'//[^\n]*', '', src)


class FEFile(object):
    """A parsed datafile.

    Attributes: `params`, `vertices` (id -> xyz), `edges` (id -> (a, b)),
    `faces` (list of signed edge-id lists), `generators` (list of 4x4),
    `gen_names` (their datafile comments, in order), `words` (command
    name -> transform_expr string), `torus` (bool).
    """

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        src = _strip(open(path, encoding='utf-8', errors='replace').read())
        self.params = {}
        for m in re.finditer(r'^\s*(?:parameter|#define)\s+(\w+)\s*='
                             r'?\s*([^\n]+)$', src, re.M):
            key, rhs = m.group(1), m.group(2).strip()
            try:
                self.params[key] = _expr(rhs, self.params)
            except FEError:
                pass                      # forward reference or non-numeric
        # a second pass resolves anything that referred to a later name
        for m in re.finditer(r'^\s*(?:parameter|#define)\s+(\w+)\s*='
                             r'?\s*([^\n]+)$', src, re.M):
            key, rhs = m.group(1), m.group(2).strip()
            if key not in self.params:
                try:
                    self.params[key] = _expr(rhs, self.params)
                except FEError:
                    pass
        self.torus = bool(re.search(r'^\s*(torus|periods)\b', src, re.M))
        self.vertices = self._vertices(src)
        self.edges = self._edges(src)
        self.faces = self._faces(src)
        self.generators, self.gen_names = self._generators(src)
        self.words = self._words(src)

    # -- sections ---------------------------------------------------
    def _section(self, src, name, stop):
        m = re.search(r'^\s*%s\s*$' % name, src, re.M | re.I)
        if not m:
            return ''
        tail = src[m.end():]
        m2 = re.search(r'^\s*(%s)\b' % '|'.join(stop), tail, re.M | re.I)
        return tail[:m2.start()] if m2 else tail

    def _vertices(self, src):
        body = self._section(src, 'vertices',
                             ['edges', 'faces', 'facets', 'bodies', 'read'])
        out = {}
        for line in body.splitlines():
            t = line.split()
            if len(t) < 4 or not t[0].isdigit():
                continue
            try:
                out[int(t[0])] = np.array(
                    [_expr(t[1], self.params), _expr(t[2], self.params),
                     _expr(t[3], self.params)], dtype=float)
            except FEError:
                continue
        return out

    def _edges(self, src):
        body = self._section(src, 'edges',
                             ['faces', 'facets', 'bodies', 'read'])
        out = {}
        for line in body.splitlines():
            t = line.split()
            if len(t) < 3 or not t[0].isdigit():
                continue
            try:
                out[int(t[0])] = (int(t[1]), int(t[2]))
            except ValueError:
                continue
        return out

    def _faces(self, src):
        # Both spellings occur, and roughly a quarter of the collection
        # uses `facets`; reading only `faces` silently yields a surface
        # with no faces at all rather than an error.
        body = (self._section(src, 'faces', ['bodies', 'read'])
                or self._section(src, 'facets', ['bodies', 'read']))
        out = []
        for line in body.splitlines():
            t = line.split()
            if len(t) < 2 or not t[0].isdigit():
                continue
            ids = []
            for tok in t[1:]:
                if re.fullmatch(r'[+-]?\d+', tok):
                    ids.append(int(tok))
                else:
                    break
            if ids:
                out.append(ids)
        return out

    def _generators(self, src):
        m = re.search(r'view_transform_generators\s+(\d+)', src)
        if not m:
            return [], []
        n = int(m.group(1))
        tail = src[m.end():]
        # numbers and expressions, 16 per matrix, `swap_colors` ignored
        tail = re.sub(r'\bswap_colors\b', ' ', tail)
        # Split on commas as well as whitespace: about a third of these
        # files write their matrices as `1,0,0,0  0,1,0,0 ...` rather
        # than space separated, and a whitespace-only tokeniser swallows
        # each row as one unparsable token.
        toks = re.findall(r'[^\s,]+', tail)
        vals, names, cur = [], [], []
        i = 0
        while i < len(toks) and len(vals) < n * 16:
            tok = toks[i]
            if re.fullmatch(r'[A-Za-z_]\w*', tok) and tok not in self.params \
                    and tok.lower() not in ('sqrt', 'pi'):
                break                      # ran into the next section
            cur.append(tok)
            try:
                v = _expr(''.join(cur), self.params)
            except FEError:
                i += 1
                continue
            vals.append(v)
            cur = []
            i += 1
        if len(vals) < n * 16:
            raise FEError("%s: wanted %d generator numbers, parsed %d"
                          % (self.name, n * 16, len(vals)))
        mats = [np.asarray(vals[k * 16:(k + 1) * 16],
                           dtype=float).reshape(4, 4) for k in range(n)]
        for k in range(n):
            names.append(chr(ord('a') + k))
        return mats, names

    def _words(self, src):
        out = {}
        for m in re.finditer(r'(\w+)\s*:=\s*\{[^}]*?transform_expr\s+'
                             r'"([A-Za-z]+)"', src, re.S):
            out[m.group(1)] = m.group(2)
        return out

    # -- derived ----------------------------------------------------
    def contour(self):
        """The face's boundary as an ordered list of points.

        Evolver faces list SIGNED edge ids; a negative id traverses that
        edge backwards.  Following the signs is what keeps the polygon
        from coming out scrambled.
        """
        if not self.faces:
            return None
        pts = []
        for eid in self.faces[0]:
            e = self.edges.get(abs(eid))
            if e is None:
                return None
            a, b = e if eid > 0 else (e[1], e[0])
            if a not in self.vertices:
                return None
            pts.append(self.vertices[a])
        return np.asarray(pts, dtype=float)

    def boundary_loops(self):
        """Every closed chain of once-used edges, as point loops.

        An edge used by exactly one face is on the outside; the rest are
        interior seams.  The number of chains says what the complex IS:

          1 loop   a disk, to be spanned by `build_disk_grid`
          2 loops  an ANNULUS, for `build_annulus_grid`

        The distinction is not academic.  I-6's surface is the side wall
        of a prism -- eight faces between a bottom rosette and a top one
        -- so its boundary is two loops, and reading only the first face
        gives a four-point contour that relaxes to a flat degenerate
        patch.  Several of the pinned datafiles are annuli this way.
        """
        use = {}
        for face in self.faces:
            for eid in face:
                use[abs(eid)] = use.get(abs(eid), 0) + 1
        rim = [e for e, k in use.items() if k == 1 and e in self.edges]
        if len(rim) < 3:
            return []
        adj = {}
        for e in rim:
            a, b = self.edges[e]
            adj.setdefault(a, []).append((b, e))
            adj.setdefault(b, []).append((a, e))
        # A rim vertex of degree 4 is normal, not a defect: a rosette
        # contour pinches at the origin, where two petals meet, and
        # I-6's rim has exactly two such vertices among twelve.  Chain
        # greedily through them rather than refusing the whole complex.
        loops = []
        unused = set(rim)
        while unused:
            e0 = next(iter(unused))
            unused.discard(e0)
            v0, v1 = self.edges[e0]
            order = [v0, v1]
            while True:
                cur = order[-1]
                step = [(w, e) for w, e in adj.get(cur, []) if e in unused]
                if not step:
                    break
                w, e = step[0]
                unused.discard(e)
                if w == order[0] and not [1 for _v, _e in adj.get(w, [])
                                          if _e in unused]:
                    break
                order.append(w)
            if len(order) >= 3 and all(v in self.vertices for v in order):
                loops.append(np.asarray([self.vertices[v] for v in order],
                                        dtype=float))
        return loops

    def boundary_loop(self):
        """The single outer boundary, when there is exactly one."""
        loops = self.boundary_loops()
        return loops[0] if len(loops) == 1 else None

    def span_loop(self):
        """`boundary_loop` if the complex has one, else `contour`."""
        loop = self.boundary_loop()
        return loop if loop is not None else self.contour()

    def letters(self):
        """{letter: 4x4} for the declared generators."""
        return {nm: M for nm, M in zip(self.gen_names, self.generators)}

    def __repr__(self):
        return ("<FEFile %s: %d verts, %d edges, %d faces, %d gens, "
                "words %s%s>" % (self.name, len(self.vertices),
                                 len(self.edges), len(self.faces),
                                 len(self.generators),
                                 sorted(self.words),
                                 ", torus" if self.torus else ""))


def read(path):
    return FEFile(path)


def _selftest():
    ok = True
    # The expression parser is the part everything else rests on.
    env = {'a': 2.0, 'HT': 0.5}
    cases = [("1", 1.0), ("-3", -3.0), ("2*a", 4.0), ("a/4", 0.5),
             ("sqrt(3)/2", math.sqrt(3.0) / 2.0),
             ("3*sqrt(3)/2", 3.0 * math.sqrt(3.0) / 2.0),
             ("-sqrt(.75)", -math.sqrt(0.75)), ("2*HT", 1.0),
             ("(1+a)*2", 6.0), ("-.5", -0.5), ("2^3", 8.0)]
    worst = 0.0
    for txt, want in cases:
        worst = max(worst, abs(_expr(txt, env) - want))
    good = worst < 1e-12
    ok &= good
    print("fedata: expression parser %d cases, worst error %.1e %s"
          % (len(cases), worst, 'OK' if good else 'FAIL'))

    # `eval` must never be reachable: an unknown name is an error, not a
    # lookup into the interpreter.
    bad = 0
    for txt in ("__import__('os')", "open('x')", "a.__class__"):
        try:
            _expr(txt, env)
        except (FEError, ZeroDivisionError):
            bad += 1
    good = bad == 3
    ok &= good
    print("fedata: hostile expressions rejected %d/3 %s"
          % (bad, 'OK' if good else 'FAIL'))

    # Against the real datafiles, where they are on this machine.  The
    # mirror is a local archive, so its absence is a SKIP rather than a
    # failure -- but when it is present the parser must reproduce the
    # generator matrices that were transcribed BY HAND into
    # `plateau.RING_CELLS`, exactly.  That is the check that matters:
    # hand transcription is what this module exists to replace, and the
    # two agreeing to 0.0 is what says the replacement is faithful.
    mirror = MIRROR_DOWNLOADS
    if os.path.isdir(mirror):
        n = bad = 0
        for fn in sorted(os.listdir(mirror)):
            if not fn.endswith('.fe'):
                continue
            n += 1
            try:
                read(os.path.join(mirror, fn)).contour()
            except Exception:
                bad += 1
        good = n > 0 and bad == 0
        ok &= good
        print("fedata: parsed %d datafiles, %d failures %s"
              % (n, bad, 'OK' if good else 'FAIL'))

        try:
            from . import plateau as _pl
        except ImportError:
            _pl = None
        if _pl is not None:
            worst = 0.0
            checked = 0
            for fn, key in (('I-8.fe', 'I8'), ('I-9.fe', 'I9'),
                            ('RIII.fe', 'R3')):
                path = os.path.join(mirror, fn)
                if not os.path.exists(path) or key not in _pl.RING_CELLS:
                    continue
                auto = read(path).letters()
                for letter, Mh in _pl.RING_CELLS[key][0]().items():
                    Ma = auto.get(letter)
                    if Ma is None:
                        worst = float('inf')
                        continue
                    worst = max(worst, float(np.max(np.abs(Mh - Ma))))
                    checked += 1
            good = checked > 0 and worst < 1e-12
            ok &= good
            print("fedata: %d hand-transcribed generators reproduced, "
                  "worst |diff| %.1e %s"
                  % (checked, worst, 'OK' if good else 'FAIL'))
    else:
        print("fedata: datafile mirror absent, parser checked on "
              "expressions only SKIP")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("fedata self-test failed")
