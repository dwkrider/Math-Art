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


# Two backends for the SAME grammar.  A datafile's `frame` command works
# on every vertex at once (`set vertex y y-aa`), so the identical
# expression has to evaluate against scalars in one place and against
# whole coordinate columns in another; swapping the function table is
# all that differs.
_MATH_FNS = {'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
             'tan': math.tan, 'exp': math.exp, 'log': math.log,
             'atan': math.atan, 'asin': math.asin, 'acos': math.acos,
             'abs': abs}
_ARRAY_FNS = {'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos,
              'tan': np.tan, 'exp': np.exp, 'log': np.log,
              'atan': np.arctan, 'asin': np.arcsin, 'acos': np.arccos,
              'abs': np.abs}


def _expr(text, env, fns=None):
    """Evaluate one Evolver arithmetic expression against `env`."""
    fns = _MATH_FNS if fns is None else fns
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
                fn = fns.get(name.lower())
                if fn is None:
                    raise FEError("unknown function %r" % name)
                return fn(v)
            if name.lower() == 'pi':
                return math.pi
            if name in env:
                v = env[name]
                return v if isinstance(v, np.ndarray) else float(v)
            raise FEError("unknown name %r in %r" % (name, text))
        raise FEError("cannot parse %r at %d" % (text, pos[0]))

    # `^` BINDS TIGHTER THAN UNARY MINUS, as it does in Evolver's own
    # grammar, where `'^'` is declared after `UMINUS_TOK`.  Folding the
    # minus into `atom()` instead -- which is what this did -- makes
    # `-x^2` parse as `(-x)^2`, and the sign of the result is simply
    # wrong.  Schwarz P's fourth constraint carries the content
    # integrand `c2: -x^2/3`, so its boundary integral came out
    # +1/24 instead of -1/24 and the enclosed volume with it.
    def unary():
        c = peek()
        if c == '-':
            pos[0] += 1
            return -unary()
        if c == '+':
            pos[0] += 1
            return unary()
        return power()

    def power():
        v = atom()
        if peek() == '^':
            pos[0] += 1
            return v ** unary()      # right-assoc, and `x^-2` still parses
        return v

    # The accumulators below are deliberately NOT `+=` / `-=` / `*=`.
    # With the array backend a name lookup returns a numpy VIEW of a
    # coordinate column, and an in-place operator would then write
    # through it: evaluating the harmless-looking `min(vertex, z-x)`
    # would subtract x from z across the whole mesh as a side effect,
    # shearing the surface before any statement asked it to move.  That
    # silently put two boundary arcs of half this collection into planes
    # the datafile never mentions, on a patch that still looked minimal.
    def term():
        v = unary()
        while True:
            c = peek()
            if c == '*':
                pos[0] += 1
                v = v * unary()
            elif c == '/':
                pos[0] += 1
                v = v / unary()
            else:
                return v

    def expr():
        v = term()
        while True:
            c = peek()
            if c == '+':
                pos[0] += 1
                v = v + term()
            elif c == '-':
                pos[0] += 1
                v = v - term()
            else:
                return v

    v = expr()
    if peek():
        raise FEError("trailing %r in %r" % (s[pos[0]:], text))
    return v


def _vexpr(text, env):
    """`_expr` over numpy columns -- `env` may hold arrays."""
    return _expr(text, env, _ARRAY_FNS)


def _close_paren(s, i):
    """Index of the `)` matching the `(` at `i`."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == '(':
            depth += 1
        elif s[j] == ')':
            depth -= 1
            if depth == 0:
                return j
    raise FEError("unbalanced ( in %r" % s)


def _split_commas(s):
    """Split on commas that are not inside parentheses or brackets."""
    out, cur, depth = [], [], 0
    for ch in s:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append(''.join(cur))
    return out


def _statements(body):
    """Split a command body into statements on top-level semicolons."""
    body = re.sub(r'//[^\n]*', '', body)
    out, cur, depth = [], [], 0
    for ch in body:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        if ch == ';' and depth == 0:
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append(''.join(cur))
    return [' '.join(s.split()) for s in out if s.strip()]


def _strip(src):
    """Remove // and /* */ comments and join continued lines."""
    src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
    src = re.sub(r'//[^\n]*', '', src)
    # A trailing backslash continues the line.  The vertex and edge
    # sections are read line by line, so a continued vertex is simply
    # not seen -- and a datafile missing three of its seven corners
    # yields a boundary that will not close, reported as "no loops"
    # rather than as the parse failure it is.
    return re.sub(r'\\[ \t]*\r?\n[ \t]*', ' ', src)


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
        self.src = src
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
        self.constraints = self._constraints(src)

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
        self.vertex_fixed = set()
        self.vertex_constraints = {}
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
            # As with edges, the attributes decide whether this point is
            # given or found.  `1 ... constraints 1,7` is a CORNER that
            # slides along the line where two planes cross while the
            # surface minimises; pinning it holds the free arcs either
            # side of it by their ends.
            rest = ' '.join(t[4:])
            vid = int(t[0])
            if re.search(r'\bfixed\b', rest):
                self.vertex_fixed.add(vid)
            cons = [int(v) for m in re.finditer(
                r'\bconstraints?\s+([\d,\s]+)', rest)
                for v in re.findall(r'\d+', m.group(1))]
            if cons:
                self.vertex_constraints[vid] = cons
        return out

    def _edges(self, src):
        """Endpoints per edge, and the attributes that follow them.

        The attributes matter as much as the endpoints.  An edge written
        `4  3 2 constraint 1` is NOT part of a pinned polygon: it is a
        FREE boundary that slides on the plane `x + z = 1` while the
        surface minimises, and its final shape is an output of the solve,
        not an input to it.  Twenty-four of Brakke's forty-three adjoint
        datafiles have such an arc, so reading the endpoints alone spans
        the wrong contour -- silently, and in most cases still producing
        one clean assembled sheet.
        """
        body = self._section(src, 'edges',
                             ['faces', 'facets', 'bodies', 'read'])
        out = {}
        self.edge_fixed = set()
        self.edge_constraints = {}
        for line in body.splitlines():
            t = line.split()
            if len(t) < 3 or not t[0].isdigit():
                continue
            try:
                eid, a, b = int(t[0]), int(t[1]), int(t[2])
            except ValueError:
                continue
            out[eid] = (a, b)
            rest = ' '.join(t[3:])
            if re.search(r'\bfixed\b', rest):
                self.edge_fixed.add(eid)
            cons = [int(v) for m in re.finditer(
                r'\bconstraints?\s+([\d,\s]+)', rest)
                for v in re.findall(r'\d+', m.group(1))]
            if cons:
                self.edge_constraints[eid] = cons
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
        vals, names, cur, exprs = [], [], [], []
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
            exprs.append(''.join(cur))
            cur = []
            i += 1
        if len(vals) < n * 16:
            raise FEError("%s: wanted %d generator numbers, parsed %d"
                          % (self.name, n * 16, len(vals)))
        mats = [np.asarray(vals[k * 16:(k + 1) * 16],
                           dtype=float).reshape(4, 4) for k in range(n)]
        # The EXPRESSIONS are kept beside the values, because the adjoint
        # files write their generators in terms of `rhs1`..`rhs6` --
        # offsets Evolver only learns after conjugating.  Parsed with
        # those still at their declared 0 the matrices are placeholders;
        # `generators_with` re-evaluates them once the offsets have been
        # measured off the conjugate.
        self.generator_exprs = [exprs[k * 16:(k + 1) * 16]
                                for k in range(n)]
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

    def _constraints(self, src):
        """{n: (lhs, rhs)} from the `constraint` blocks, both sides kept.

        Both sides, because both carry geometry.  A constraint is written
        `formula: <lhs> = <rhs>`, and across this collection the right
        side is a NUMBER (`z = 1`), a COORDINATE (`x = z`, a diagonal
        mirror) or a NAME (`sqrt(3)*x - y = rhs1`, an offset Evolver only
        learns after conjugating) in roughly equal measure.  An earlier
        reading of this section demanded an identifier on the right, so
        `x = z` came back as the plane `x = <the value of z>` and `z = 1`
        was dropped without a word -- which is why several cells landed
        their boundary on planes the datafile never declared.

        The declaration also straddles lines: about a quarter of the
        collection puts `formula:` on the line after `constraint N`.
        """
        out = {}
        # `formula` and `function` are the same keyword to Evolver, and
        # the collection uses both spellings in both cases -- Neovius'
        # datafile writes `CONSTRAINT 1` / `FUNCTION:  X3  = 1`.  Reading
        # only lowercase `formula` left that file with no constraints at
        # all, and its whole boundary is constrained.
        pat = (r'^[ \t]*constraint[ \t]+(\d+)\b[^\n]*(?:\n[ \t]*)?'
               r'(?:formula|function)[ \t]*:?[ \t]*([^\n]+)')
        for m in re.finditer(pat, src, re.M | re.I):
            body = m.group(2).strip().rstrip(';')
            if '=' not in body:
                continue
            lhs, _sep, rhs = body.partition('=')
            if lhs.strip() and rhs.strip():
                out[int(m.group(1))] = (lhs.strip(), rhs.strip())
        return out

    def _linear_form(self, form, env):
        """(gradient, value at the origin) if `form` is linear in x,y,z.

        `x1`/`x2`/`x3` are Evolver's other names for the coordinates and
        are bound alongside them, because the same collection uses both
        spellings -- Neovius' constraints are written `X3 = 1`.
        """
        base = dict(self.params)
        base.update(env or {})
        base.update({'x': 0.0, 'y': 0.0, 'z': 0.0,
                     'x1': 0.0, 'x2': 0.0, 'x3': 0.0,
                     'X1': 0.0, 'X2': 0.0, 'X3': 0.0})
        alias = {'x': ('x', 'x1', 'X1'), 'y': ('y', 'x2', 'X2'),
                 'z': ('z', 'x3', 'X3')}
        try:
            f0 = _expr(form, base)
            grad = []
            for ax in ('x', 'y', 'z'):
                e = dict(base)
                for nm in alias[ax]:
                    e[nm] = 1.0
                grad.append(_expr(form, e) - f0)
            # A quadratic form differences to something plausible at 1
            # and wrong at 2; I-WP's datafile really does constrain a
            # boundary to a SPHERE, and reading that as a plane puts an
            # arc somewhere it never goes.
            for k, ax in enumerate(('x', 'y', 'z')):
                e = dict(base)
                for nm in alias[ax]:
                    e[nm] = 2.0
                if abs(_expr(form, e) - f0 - 2.0 * grad[k]) > 1e-9:
                    return None
        except FEError:
            return None
        return np.asarray(grad, dtype=float), float(f0)

    def constraint_plane(self, n, env=None):
        """Constraint `n` as a plane `v . p = c`, or None.

        Returns `(v, c, name)`.  `v` is UNNORMALISED, deliberately: the
        forms are written as Evolver stores them (`sqrt(3)*x - y` has
        length 2), and the offset it records is the form's value, not a
        distance.  When the offset is left to be measured the datafile
        names it, and `name` is that name -- the plane is then
        `v . p = name + c`.
        """
        c = self.constraints.get(n)
        if c is None:
            return None
        lhs, rhs = c
        L = self._linear_form(lhs, env)
        if L is None:
            return None
        R = self._linear_form(rhs, env)
        if R is not None:
            v, const, name = L[0] - R[0], R[1] - L[1], None
        else:
            if not re.fullmatch(r'[A-Za-z_]\w*', rhs):
                return None
            v, const, name = L[0], -L[1], rhs
        if float(np.linalg.norm(v)) <= 1e-12:
            return None
        return v, const, name

    def body_volume(self):
        """The volume a `bodies` section fixes, or None.

        Four of these surfaces cannot be found by minimising area at all.
        Schwarz P's fundamental piece has its whole boundary free on four
        mirror planes, and area alone would slide it into a corner and
        collapse it -- so the datafile pins the volume instead, with
        Brakke's own note saying why: "Surface is stabilized with a
        volume constraint, since we know the P-surface equipartitions
        volume."  Schwarz D, Neovius and Schoen's I-WP are the same.
        """
        m = re.search(r'^\s*bodies\s*$(.*?)(?=^\s*\w+\s*$|\Z)',
                      self.src, re.M | re.S | re.I)
        if m:
            # Stop at the next keyword.  Schoen's I-WP writes
            # `volume 1/(1+sqrt(.75))/6  volconst -1/3`, and swallowing
            # the `volconst` clause into the expression made the whole
            # target unparseable -- so I-WP had no volume constraint at
            # all, and with a fully free boundary its area then descends
            # to a collapse (measured: 0.72 of Evolver's once the
            # iteration cap stopped hiding it).
            v = re.search(r'\bvolume\s+(.+?)(?=\bvolconst\b|\n|$)',
                          m.group(1), re.I | re.S)
            if v:
                try:
                    return _expr(v.group(1).strip().rstrip(';'), self.params)
                except FEError:
                    pass
        # Neovius states the same thing as a named quantity instead.
        q = re.search(r'^\s*quantity\s+\w+\s+fixed\s*=\s*([^\n]+?)\s*'
                      r'(?:method|global_method|$)', self.src, re.M | re.I)
        if q:
            try:
                return _expr(q.group(1).strip(), self.params)
            except FEError:
                pass
        return None

    def body_volconst(self):
        """The constant Evolver seeds a body's volume sum with.

        `volconst` is not decoration: the body's volume is this plus the
        facet and content integrals, so leaving it out shifts the target
        by a fixed amount and the constraint holds the surface in the
        wrong place.
        """
        m = re.search(r'^\s*bodies\s*$(.*?)(?=^\s*\w+\s*$|\Z)',
                      self.src, re.M | re.S | re.I)
        if not m:
            return 0.0
        v = re.search(r'\bvolconst\s+([^\n]+)', m.group(1), re.I)
        if not v:
            return 0.0
        try:
            return _expr(v.group(1).strip().rstrip(';'), self.params)
        except FEError:
            return 0.0

    def constraint_content(self, n):
        """A constraint's `content` integrand `(c1, c2, c3)`, or None.

        Evolver closes an open body over its constraint planes with a
        line integral round the boundary, and this is the vector field it
        integrates.  Without it the enclosed volume is only the cone from
        the origin over the surface, which is not what the datafile
        fixes.
        """
        blocks = re.split(r'^\s*constraint\s+(\d+)\b', self.src,
                          flags=re.M | re.I)
        for i in range(1, len(blocks) - 1, 2):
            if int(blocks[i]) != n:
                continue
            body = blocks[i + 1]
            got = []
            for k in (1, 2, 3):
                mm = re.search(r'^\s*c%d\s*:\s*([^\n]+)' % k, body, re.M | re.I)
                if not mm:
                    return None
                got.append(mm.group(1).strip().rstrip(';'))
            return tuple(got)
        return None

    def constraint_normal(self, n, env=None):
        """The unit normal of constraint `n`'s plane."""
        pl = self.constraint_plane(n, env)
        if pl is None:
            return None
        return pl[0] / float(np.linalg.norm(pl[0]))

    def constraint_value(self, n, pts):
        """Evaluate constraint `n`'s left-hand form at points.

        The forms are UNNORMALISED -- `sqrt(3)*x - y` has length 2 -- so
        the `rhs` Evolver stores is this value, not the distance along a
        unit normal.  Measuring the offset with a normalised normal and
        substituting it as `rhs` puts the mirror in the wrong place by
        that factor.
        """
        c = self.constraints.get(n)
        if c is None:
            return None
        form = c[0]
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        base = dict(self.params)
        out = []
        for p in pts:
            e = dict(base)
            e['x'], e['y'], e['z'] = float(p[0]), float(p[1]), float(p[2])
            try:
                out.append(_expr(form, e))
            except FEError:
                return None
        return np.asarray(out, dtype=float)

    def generators_with(self, env):
        """Re-evaluate the generator matrices with extra names bound.

        Used to substitute the measured `rhs` offsets into an adjoint
        file's generators.
        """
        if not getattr(self, 'generator_exprs', None):
            return list(self.generators)
        full = dict(self.params)
        full.update(env or {})
        out = []
        for k, rows in enumerate(self.generator_exprs):
            declared = np.ravel(self.generators[k])
            vals = []
            for j, e in enumerate(rows):
                try:
                    vals.append(_expr(e, full))
                except FEError:
                    # Fall back to the number the datafile DECLARED, not
                    # to zero.  Most of these files have eight generators
                    # and two constraints: six of the matrices are fixed
                    # and want no substitution at all, and zeroing an
                    # entry whose name happens to be unmeasured turns a
                    # rotation into a singular matrix -- which is what
                    # made most of the collection refuse to assemble.
                    vals.append(float(declared[j]))
            out.append(np.asarray(vals, dtype=float).reshape(4, 4))
        return out

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
        return [np.asarray([self.vertices[v] for v in vids], dtype=float)
                for vids, _eids in self.boundary_chains()]

    def boundary_chains(self):
        """Every boundary loop as `(vertex ids, edge ids)`, in step.

        The ids are what makes the loop usable.  A datafile's `frame`
        command talks about `vertex[3]` and `edge where original == 5`,
        so a bare list of points cannot be told what to do with itself:
        arc *k* of the loop is edge `eids[k]`, running from corner
        `vids[k]` to `vids[k+1]`, and that mapping is the only way the
        declared constraints reach the right part of the boundary.
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
            e0 = min(unused)
            unused.discard(e0)
            v0, v1 = self.edges[e0]
            order, eord = [v0, v1], [e0]
            while True:
                cur = order[-1]
                step = [(w, e) for w, e in adj.get(cur, []) if e in unused]
                if not step:
                    break
                w, e = step[0]
                unused.discard(e)
                eord.append(e)
                if w == order[0] and not [1 for _v, _e in adj.get(w, [])
                                          if _e in unused]:
                    break
                order.append(w)
            if len(order) >= 3 and all(v in self.vertices for v in order):
                loops.append((order, eord[:len(order)]))
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

    # -- the `frame` command ----------------------------------------
    def command(self, name):
        """The raw body of `name := { ... }`, brace-matched."""
        m = re.search(r'^[ \t]*%s[ \t]*:=[ \t]*\{' % re.escape(name),
                      self.src, re.M)
        if not m:
            return None
        i = self.src.index('{', m.start())
        depth = 0
        for j in range(i, len(self.src)):
            if self.src[j] == '{':
                depth += 1
            elif self.src[j] == '}':
                depth -= 1
                if depth == 0:
                    return self.src[i + 1:j]
        return None

    def frame_program(self):
        """The datafile's `frame` command, parsed into statements.

        This is the step the whole adjoint pipeline turns on, and it is
        not a set of equations -- it is a short imperative script.  After
        conjugating, the surface is minimal but sits at an arbitrary
        place and scale, so `frame` moves it: it reads a few distances
        off the conjugate, translates and rescales until the boundary
        lands on the declared planes, and only then says which arc
        belongs to which constraint.  Guessing that placement, or
        matching arcs to constraints by position, is what made most of
        this collection come out as a heap of disjoint sheets.

        Statements, all of which really occur across the collection:

            ('set', name, expr)              a := vertex[2].y
            ('vertex', coord, expr)          set vertex y y-aa
            ('constraint', edge, n)          arc `edge` lies on `n`
            ('ifgt', a, b, name, expr)       if p > q then m := p
            ('while', cond, subprogram)      while (k <= 6) do { ... }

        The edge and constraint of a `('constraint', ...)` are kept as
        EXPRESSIONS rather than numbers, because a third of these files
        assign their constraints from inside a `while` loop -- `set ee
        constraint connum` -- so the pair is only known once the loop is
        running.

        Where a file has no `frame` at all the framing is inlined into
        `adj` beside the `adjoint` call itself; both are read, and in
        that order, which is the order `gogo` invokes them.  Anything
        that cannot change a static reconstruction -- `unfix`, `recalc`,
        a stray evolution step -- is dropped.
        """
        prog = []
        for cmd in ('adj', 'frame', 'newframe'):
            body = self.command(cmd)
            if body is not None:
                prog += self._parse_frame(body)
            if cmd == 'frame' and body is not None:
                break
        return prog

    def _parse_frame(self, body):
        prog = []
        for s in _statements(body):
            m = re.match(r'^while\s*\((.+?)\)\s*do\s*\{(.*)\}$', s, re.S)
            if m:
                prog.append(('while', m.group(1),
                             self._parse_frame(m.group(2))))
                continue
            m = re.match(r'^foreach\s+edges?\s+(\w+)\s+where\s+original\s*'
                         r'==\s*(\w+)\s+do\s*\{(.*)\}$', s, re.S)
            if m:
                seen = []
                for c in re.finditer(r'constraint\s*\(?\s*(\w+)', m.group(3)):
                    if c.group(1) not in seen:
                        seen.append(c.group(1))
                        prog.append(('constraint', m.group(2), c.group(1)))
                continue
            m = re.match(r'^set\s+edges?\s+constraint\s+\(?(\w+)\)?\s+where'
                         r'\s+original\s*==\s*(\w+)$', s)
            if m:
                prog.append(('constraint', m.group(2), m.group(1)))
                continue
            m = re.match(r'^set\s+vertex\s+([xyz])\s+(.+)$', s)
            if m:
                prog.append(('vertex', m.group(1), m.group(2)))
                continue
            m = re.match(r'^if\s+(.+?)\s*>\s*(.+?)\s+then\s+(\w+)\s*:=\s*(.+)$',
                         s)
            if m:
                prog.append(('ifgt', m.group(1), m.group(2),
                             m.group(3), m.group(4)))
                continue
            m = re.match(r'^(\w+)\s*:=\s*(.+)$', s)
            if m:
                prog.append(('set', m.group(1), m.group(2)))
                continue
            m = re.match(r'^(\w+)\s*([-+*/])=\s*(.+)$', s)
            if m:
                prog.append(('set', m.group(1), '(%s) %s (%s)'
                             % (m.group(1), m.group(2), m.group(3))))
        return prog

    def _reduce(self, text, P, corners, arc_of, env):
        """Rewrite an Evolver frame expression down to plain arithmetic.

        The aggregates are folded to numbers here rather than taught to
        the arithmetic evaluator, which stays a small, closed grammar
        over names and operators.  `min(vertex, z-x)` needs the whole
        point set; `vertex[3].x` needs to know which mesh row the
        datafile's third corner became; `avg(edge ee where original==2,
        avg(ee.vertex, x))` needs the boundary labelling.  All three are
        answered here and substituted as literals.
        """
        def corner(m):
            i = corners.get(int(m.group(1)))
            if i is None:
                raise FEError("no corner for vertex[%s]" % m.group(1))
            return '(%.17g)' % float(P[i]['xyz'.index(m.group(2))])

        text = re.sub(r'vertex\s*\[\s*(\d+)\s*\]\s*\.\s*([xyz])', corner, text)
        cols = {'x': P[:, 0], 'y': P[:, 1], 'z': P[:, 2]}

        def rec(t):
            m = re.search(r'\b(minimum|maximum|min|max|avg|sum)\s*\(', t)
            if not m:
                return t
            i = t.index('(', m.end() - 1)
            j = _close_paren(t, i)
            head, args = m.group(1), _split_commas(t[i + 1:j])
            if head in ('minimum', 'maximum') and len(args) == 2:
                a = _expr(rec(args[0]), env)
                b = _expr(rec(args[1]), env)
                v = min(a, b) if head == 'minimum' else max(a, b)
            elif len(args) == 2 and args[0].strip() == 'vertex':
                e = dict(env)
                e.update(cols)
                col = np.atleast_1d(_vexpr(rec(args[1]), e))
                v = {'min': np.min, 'max': np.max,
                     'avg': np.mean, 'sum': np.sum}[head](col)
            elif len(args) == 2 and re.match(r'^edges?\b', args[0].strip()):
                w = re.search(r'original\s*==\s*(\d+)', args[0])
                if w is None or arc_of is None:
                    raise FEError("cannot select %r" % args[0].strip())
                sel = np.nonzero(arc_of == int(w.group(1)))[0]
                if not len(sel):
                    raise FEError("no boundary on edge %s" % w.group(1))
                # `avg(edge ee where ..., avg(ee.vertex, x))` -- the
                # inner aggregate is over the two ends of each edge, so
                # for a resampled arc it is just the form itself.  Peel
                # it off rather than recursing, which would try to read
                # `ee.vertex` as a point set.
                inner = args[1].strip()
                mm = re.match(r'^(?:avg|sum)\s*\(\s*\w+\s*\.\s*'
                              r'vertex(?:es)?\s*,(.*)\)$', inner, re.S)
                if mm:
                    inner = mm.group(1)
                e = dict(env)
                e.update({k: c[sel] for k, c in cols.items()})
                col = np.atleast_1d(_vexpr(inner.strip(), e))
                v = {'min': np.min, 'max': np.max,
                     'avg': np.mean, 'sum': np.sum}[head](col)
            else:
                raise FEError("unsupported %s(%s)" % (head, args[0][:20]))
            return t[:m.start()] + ('(%.17g)' % float(v)) + rec(t[j + 1:])

        return rec(text)

    def run_frame(self, P, corners, arc_of=None, prog=None):
        """Execute `frame` on a conjugated patch.

        `P` is the conjugate's points, `corners` maps a datafile vertex
        id to its row in `P`, and `arc_of` labels each row with the
        original edge it lies on (-1 off the boundary).  Returns the
        repositioned points, `{original edge: [constraint numbers]}` and
        the environment of names the script computed.
        """
        P = np.array(P, dtype=float)
        prog = self.frame_program() if prog is None else prog
        env = dict(self.params)
        econ = {}

        def run(steps):
            for st in steps:
                if st[0] == 'constraint':
                    eid = int(round(_expr(st[1], env)))
                    n = int(round(_expr(st[2], env)))
                    econ.setdefault(eid, [])
                    if n not in econ[eid]:
                        econ[eid].append(n)
                elif st[0] == 'set':
                    env[st[1]] = _expr(
                        self._reduce(st[2], P, corners, arc_of, env), env)
                elif st[0] == 'ifgt':
                    a = _expr(self._reduce(st[1], P, corners, arc_of, env),
                              env)
                    b = _expr(self._reduce(st[2], P, corners, arc_of, env),
                              env)
                    if a > b:
                        env[st[3]] = _expr(
                            self._reduce(st[4], P, corners, arc_of, env), env)
                elif st[0] == 'while':
                    # Bounded, because a mis-parsed condition would
                    # otherwise spin forever on a file read off a website.
                    for _ in range(256):
                        if not self._cond(st[1], env):
                            break
                        run(st[2])
                elif st[0] == 'vertex':
                    e = dict(env)
                    e.update({'x': P[:, 0], 'y': P[:, 1], 'z': P[:, 2]})
                    col = _vexpr(
                        self._reduce(st[2], P, corners, arc_of, env), e)
                    P[:, 'xyz'.index(st[1])] = np.broadcast_to(
                        np.asarray(col, dtype=float), (len(P),))

        run(prog)
        return P, econ, env

    def _cond(self, text, env):
        """Evaluate a `while` condition: one comparison, no booleans."""
        m = re.match(r'^\s*(.+?)\s*(<=|>=|==|!=|<|>)\s*(.+?)\s*$', text)
        if not m:
            raise FEError("cannot read condition %r" % text)
        a, b = _expr(m.group(1), env), _expr(m.group(3), env)
        return {'<=': a <= b, '>=': a >= b, '==': a == b, '!=': a != b,
                '<': a < b, '>': a > b}[m.group(2)]

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
             ("(1+a)*2", 6.0), ("-.5", -0.5), ("2^3", 8.0),
             # `^` binds tighter than unary minus, as in Evolver's own
             # grammar.  Getting this backwards silently flips the sign
             # of a content integrand and with it an enclosed volume.
             ("-2^2", -4.0), ("2^-1", 0.5), ("2^3^2", 512.0),
             ("-a^2/3", -4.0 / 3.0), ("-a*2", -4.0)]
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
