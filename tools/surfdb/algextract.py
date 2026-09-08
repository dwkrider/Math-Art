"""Read the algebraic presets' Python bodies back into their equations.

`surfaces/algebraic.py` holds 148 presets.  63 of them carry a
`HAUSER_EQUATION` string -- the equation as the gallery prints it -- and
22 more are Goursat family members whose polynomial is assembled from
coefficients.  The remaining 63 are defined SOLELY by the body of a
Python function, so the database stored `polynomial: null` for them and
a reader with the records could not reproduce the surface.

But those bodies are readable.  They are straight-line arithmetic on
x, y, z: a few local assignments, maybe a helper call or a small
unrolled loop, and a `return`.  This module inlines that arithmetic
back into a single expression in the project's exact language.

WHAT THIS IS NOT.  It is not a decompiler and not a simplifier.  It
substitutes each local's expression at every use and folds arithmetic
on numbers, and that is all.  The author's own structure survives --
`core * core - lam * p * q * r * s` comes back out with `core`, `lam`,
`p`, `q`, `r` and `s` spelled in place -- which keeps the result
readable and, more importantly, keeps it honest: nothing here rewrites
the mathematics into a form the author did not write.

THE ORACLE IS NOT OPTIONAL.  An extracted string is a CANDIDATE.  The
caller must put it through `polynomial.verify_against` with the shipped
function as the oracle, and store it only if it matches at every sample
point.  This is the house rule that has already caught two wrong
polynomials in this project: a plausible-but-wrong equation does not
error, it silently defines a different surface.  Extraction lowers the
transcription risk to nearly nothing -- the equation comes from the
code that draws the picture, not from a person reading a paper -- but
"nearly nothing" is not "nothing", and the gate costs one call.

Numbers vs. symbols.  The inliner evaluates to a Python float wherever
every operand is numeric, so `math.cos(2 * math.pi * j / 5)` inside an
unrolled loop becomes a literal, and module constants like `_SQRT2`
fold away.  Parameters (the `mu` of `PRESETS[key][1]`) are substituted
at the operator's own default, which is the member of the family the
record describes and the one its figure shows.
"""

import ast
import inspect
import math
import textwrap


class Unextractable(Exception):
    """The body is not straight-line arithmetic we will inline."""


# The exact language's whitelist (expr.FUNCTIONS), by the names they are
# reached under in the generator: bare, `math.` or `np.`.
_FUNCS = {
    "sqrt": "sqrt", "abs": "abs", "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan", "arcsin": "asin",
    "arccos": "acos", "arctan": "atan", "sinh": "sinh", "cosh": "cosh",
    "tanh": "tanh", "exp": "exp", "log": "log", "cbrt": "cbrt",
    "fabs": "abs", "absolute": "abs",
}

# Calls that are casts or shape plumbing, not mathematics.
_IDENTITY = {"asarray", "array", "float", "float64", "atleast_1d"}
_ONES = {"ones_like", "ones"}
_ZEROS = {"zeros_like", "zeros"}

_MODULES = {"np", "numpy", "math", "cmath"}

# Guard against an inliner that "succeeds" by emitting something no one
# can read.  Hit only by genuinely huge expansions, and a refusal with a
# stated reason beats a 400 kB one-liner nobody will ever check.
_MAX_LEN = 200000


# Binding strength, used only to decide where brackets are NEEDED.  An
# inliner that brackets every subexpression is correct and unreadable --
# `((x) * (x))` for `x * x` -- and these strings are meant to be read by
# a person reconstructing the surface, so the brackets have to be earned.
_P_ADD, _P_MUL, _P_UNARY, _P_POW, _P_ATOM = 0, 1, 2, 3, 4

_PREC = {"+": _P_ADD, "-": _P_ADD, "*": _P_MUL, "/": _P_MUL, "**": _P_POW}


class _Sym:
    """A built expression, with the binding strength of its top node."""

    __slots__ = ("text", "prec")

    def __init__(self, text, prec=_P_ATOM):
        self.text = text
        self.prec = prec

    def __str__(self):
        return self.text

    def paren(self, need):
        """The text, bracketed only if it binds more loosely than `need`."""
        return "(%s)" % self.text if self.prec < need else self.text


def _num(v):
    """Format a Python number as a literal in the exact language."""
    if isinstance(v, int):
        return str(v)
    if v != v or v in (float("inf"), float("-inf")):
        raise Unextractable("non-finite constant %r" % v)
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


class _Inliner:
    """Fold a function body down to one expression string.

    Every `go()` returns either a float/int (the subtree is numeric and
    was folded) or a string (the subtree mentions x, y or z).  Keeping
    the two apart is what lets `math.cos(2*math.pi*j/5)` collapse to a
    literal inside an unrolled loop while `cos(x)` stays symbolic.
    """

    def __init__(self, mod, coords=("x", "y", "z")):
        self.mod = mod
        self.coords = coords
        self.depth = 0

    # -- helpers ----------------------------------------------------
    @staticmethod
    def _sym(v, need):
        """Render a value as an operand that binds at least as tightly.

        Anything that is neither a number nor an already-built
        expression is a bug caught here rather than stringified: a
        `None` or a tuple reaching this point would otherwise be
        pasted into the output as the text "(None)", which parses as a
        free name and would fail far from its cause.
        """
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            t = _num(v)
            # A negative literal binds like a unary minus, not an atom.
            return "(%s)" % t if t.startswith("-") and need > _P_ADD else t
        if isinstance(v, _Sym):
            return v.paren(need)
        raise Unextractable("arithmetic on %s" % type(v).__name__)

    def _join(self, a, op, b):
        """Combine two operands, bracketing only where precedence needs it.

        `+` and `*` associate freely, so the right operand may share the
        level.  `-`, `/` and `**` do not: `a - (b - c)` and `a / (b * c)`
        need their brackets, and `**` is right-associative so it is the
        LEFT side that must be protected.
        """
        p = _PREC[op]
        if op == "**":
            return _Sym("%s ** %s" % (self._sym(a, _P_ATOM),
                                      self._sym(b, _P_UNARY)), _P_POW)
        right = p if op in ("+", "*") else p + 1
        return _Sym("%s %s %s" % (self._sym(a, p), op, self._sym(b, right)), p)

    def _lookup_global(self, name):
        """A module-level constant: a number, or a sequence of them.

        Sequences matter because the published coefficient sets live as
        module tuples -- `_LABS_A`, `_ENDRASS_PARAMS_168` -- and are
        unpacked or splatted at the top of the body they belong to.
        """
        if not hasattr(self.mod, name):
            raise Unextractable("free name %r" % name)
        return self._as_value(getattr(self.mod, name), name)

    def _as_value(self, v, name):
        if isinstance(v, bool):
            raise Unextractable("global %r is a flag, not a number" % name)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, (tuple, list)):
            return tuple(self._as_value(e, name) for e in v)
        # A numpy array of published coordinates -- the tetrahedron's
        # four circle centres, say -- is a sequence like any other.
        if hasattr(v, "tolist") and hasattr(v, "shape"):
            return self._as_value(v.tolist(), name)
        raise Unextractable("global %r is not a number or a sequence of "
                            "numbers" % name)

    # -- expressions ------------------------------------------------
    def go(self, node, env):
        if isinstance(node, ast.Constant):
            v = node.value
            if v is None:
                # `None` is not a number, but it IS a decidable value:
                # the presets use `a=None` defaults with an
                # `a0 if a is None else float(a)` line to mean "take the
                # published coefficient".  Carry it so that test can be
                # settled; any attempt to do arithmetic on it still
                # fails, in _binop, where it should.
                return None
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise Unextractable("constant %r" % (v,))
            return v

        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            return self._lookup_global(node.id)

        if isinstance(node, ast.Subscript):
            base = self.go(node.value, env)
            if not isinstance(base, tuple):
                raise Unextractable("subscript of a non-sequence")
            idx = self.go(node.slice, env)
            if not (isinstance(idx, (int, float)) and float(idx).is_integer()):
                raise Unextractable("non-constant index")
            try:
                return base[int(idx)]
            except IndexError:
                raise Unextractable("index %d out of range" % int(idx))

        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.go(e, env) for e in node.elts)

        if isinstance(node, ast.IfExp):
            # Only a test that does NOT depend on x, y or z may be
            # decided here.  A branch on a coordinate is a different
            # surface on each side and is not one expression at all.
            return self.go(node.body if self._decide(node.test, env)
                           else node.orelse, env)

        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) \
                    and node.value.id in _MODULES:
                if node.attr == "pi":
                    return math.pi
                if node.attr == "e":
                    return math.e
                if node.attr in ("inf", "Inf"):
                    raise Unextractable("infinity")
            raise Unextractable("attribute %s" % node.attr)

        if isinstance(node, ast.UnaryOp):
            v = self.go(node.operand, env)
            if isinstance(node.op, ast.UAdd):
                return v
            if isinstance(node.op, ast.USub):
                if isinstance(v, (int, float)):
                    return -v
                return _Sym("-" + self._sym(v, _P_UNARY), _P_UNARY)
            raise Unextractable("unary %s" % type(node.op).__name__)

        if isinstance(node, ast.BinOp):
            return self._binop(node, env)

        if isinstance(node, ast.Call):
            return self._call(node, env)

        raise Unextractable("syntax %s" % type(node).__name__)

    _OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.Pow: "**"}

    _CMP = {
        ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
        ast.Is: lambda a, b: a is b, ast.IsNot: lambda a, b: a is not b,
    }

    def _decide(self, node, env):
        """Settle a test that must not depend on the coordinates.

        This is the one place the inliner could go badly wrong: taking a
        branch whose condition depends on x, y or z would emit ONE
        expression for a function that is genuinely piecewise, and the
        result could still pass a sampling oracle if the other branch
        happens to be rare in the sample.  So a test is only settled
        when it evaluates to a plain bool from numbers, `None` and
        parameters -- never from anything carrying a coordinate, which
        is exactly what `go()` returns as a *string*.
        """
        v = self._value_of(node, env)
        if not isinstance(v, bool):
            raise Unextractable("condition does not settle to true or false")
        return v

    def _value_of(self, node, env):
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise Unextractable("chained comparison")
            fn = self._CMP.get(type(node.ops[0]))
            if fn is None:
                raise Unextractable("comparison %s"
                                    % type(node.ops[0]).__name__)
            a = self.go(node.left, env)
            b = self.go(node.comparators[0], env)
            if isinstance(a, _Sym) or isinstance(b, _Sym):
                raise Unextractable("branch on a coordinate")
            return bool(fn(a, b))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self._value_of(node.operand, env)
        if isinstance(node, ast.BoolOp):
            vals = [self._value_of(v, env) for v in node.values]
            return (all(vals) if isinstance(node.op, ast.And) else any(vals))
        v = self.go(node, env)
        if isinstance(v, _Sym):
            raise Unextractable("branch on a coordinate")
        return bool(v)

    def _binop(self, node, env):
        a = self.go(node.left, env)
        b = self.go(node.right, env)
        op = self._OPS.get(type(node.op))
        if op is None:
            raise Unextractable("operator %s" % type(node.op).__name__)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            # Fold, but never fold a division by zero or a power that
            # leaves the reals -- let those surface as a refusal rather
            # than as a nan buried in a coefficient.
            try:
                if op == "+":
                    return a + b
                if op == "-":
                    return a - b
                if op == "*":
                    return a * b
                if op == "/":
                    return a / b
                r = a ** b
            except (ZeroDivisionError, OverflowError, ValueError) as exc:
                raise Unextractable("constant folding failed: %s" % exc)
            if isinstance(r, complex):
                raise Unextractable("constant power leaves the reals")
            return r
        return self._join(a, op, b)

    def _call(self, node, env):
        f = node.func
        # bound method / attribute call: np.cos(...), math.sqrt(...)
        if isinstance(f, ast.Attribute):
            if not (isinstance(f.value, ast.Name) and f.value.id in _MODULES):
                raise Unextractable("call on %s" % ast.dump(f)[:40])
            name = f.attr
        elif isinstance(f, ast.Name):
            name = f.id
        else:
            raise Unextractable("indirect call")

        if name in _IDENTITY:
            if not node.args:
                raise Unextractable("%s() with no argument" % name)
            return self.go(node.args[0], env)
        if name in _ONES:
            return 1
        if name in _ZEROS:
            return 0

        if name in _FUNCS:
            if len(node.args) != 1:
                raise Unextractable("%s() takes one argument here" % name)
            a = self.go(node.args[0], env)
            out = _FUNCS[name]
            if isinstance(a, (int, float)):
                fold = getattr(math, out if out != "abs" else "fabs", None)
                if fold is not None:
                    try:
                        return fold(a)
                    except ValueError as exc:
                        raise Unextractable("%s(%r): %s" % (out, a, exc))
            return _Sym("%s(%s)" % (out, self._sym(a, _P_ADD)))

        # a local closure (an inner def) or a module-level helper
        target = env.get(name)
        if isinstance(target, tuple) and target and target[0] == "fn":
            return self._apply(target[1], target[2], node, env)
        g = getattr(self.mod, name, None)
        if isinstance(g, ast.FunctionDef) or inspect.isfunction(g):
            return self._apply(g, {}, node, env)
        raise Unextractable("call to %s()" % name)

    def _apply(self, fdef, closure_env, call, env):
        """Inline a helper or nested def at its call site."""
        self.depth += 1
        if self.depth > 12:
            raise Unextractable("call nesting deeper than 12")
        try:
            if not isinstance(fdef, ast.FunctionDef):
                fdef = _parse_def(fdef)
            # `f(x, y, z, mu, *_ENDRASS_PARAMS_160)` -- splat a constant
            # coefficient tuple into the positional arguments.
            args = []
            for a in call.args:
                if isinstance(a, ast.Starred):
                    v = self.go(a.value, env)
                    if not isinstance(v, tuple):
                        raise Unextractable("splat of a non-sequence")
                    args.extend(v)
                else:
                    args.append(self.go(a, env))
            names = [a.arg for a in fdef.args.args]
            if len(args) > len(names):
                raise Unextractable("%s() got too many arguments" % fdef.name)
            inner = dict(closure_env)
            # defaults first, then positional actuals over the top
            defaults = fdef.args.defaults
            for nm, d in zip(names[len(names) - len(defaults):], defaults):
                inner[nm] = self.go(d, env)
            for nm, v in zip(names, args):
                inner[nm] = v
            for kw in call.keywords:
                if kw.arg is None:
                    raise Unextractable("**kwargs at a call site")
                inner[kw.arg] = self.go(kw.value, env)
            missing = [n for n in names if n not in inner]
            if missing:
                raise Unextractable("%s() missing %s"
                                    % (fdef.name, ", ".join(missing)))
            return self.run(fdef.body, inner)
        finally:
            self.depth -= 1

    # -- statements -------------------------------------------------
    def run(self, body, env):
        """Execute straight-line statements; return the `return` value."""
        for st in body:
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
                continue                            # docstring
            if isinstance(st, ast.Return):
                if st.value is None:
                    raise Unextractable("bare return")
                return self.go(st.value, env)
            if isinstance(st, ast.Assign):
                self._assign(st, env)
                continue
            if isinstance(st, ast.AugAssign):
                if not isinstance(st.target, ast.Name):
                    raise Unextractable("augmented assignment to non-name")
                cur = env.get(st.target.id)
                if cur is None:
                    raise Unextractable("augmented assignment before "
                                        "assignment: %s" % st.target.id)
                fake = ast.BinOp(left=ast.Constant(value=0), op=st.op,
                                 right=st.value)
                # evaluate `cur op rhs` through the normal path
                env[st.target.id] = self._binop_vals(cur, st.op, st.value, env)
                del fake
                continue
            if isinstance(st, ast.FunctionDef):
                env[st.name] = ("fn", st, dict(env))
                continue
            if isinstance(st, ast.Delete):
                for t in st.targets:                # `del mu` -- a no-op
                    if isinstance(t, ast.Name):
                        env.pop(t.id, None)
                continue
            if isinstance(st, ast.If):
                # Decidable without the coordinates, or not at all --
                # see _decide().  A taken branch may itself return.
                taken = st.body if self._decide(st.test, env) else st.orelse
                if any(isinstance(s, ast.Return) for s in taken):
                    return self.run(taken, env)
                for s in taken:
                    if isinstance(s, ast.Assign):
                        self._assign(s, env)
                    elif isinstance(s, ast.Pass):
                        continue
                    else:
                        raise Unextractable("statement %s in a branch"
                                            % type(s).__name__)
                continue
            if isinstance(st, ast.For):
                self._unroll(st, env)
                continue
            if isinstance(st, ast.Pass):
                continue
            raise Unextractable("statement %s" % type(st).__name__)
        raise Unextractable("function has no return")

    def _binop_vals(self, a, op, rhs_node, env):
        b = self.go(rhs_node, env)
        node = ast.BinOp(left=ast.Constant(value=0), op=op,
                         right=ast.Constant(value=0))
        opsym = self._OPS.get(type(node.op))
        if opsym is None:
            raise Unextractable("operator %s" % type(op).__name__)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return {"+": lambda: a + b, "-": lambda: a - b,
                    "*": lambda: a * b, "/": lambda: a / b,
                    "**": lambda: a ** b}[opsym]()
        return self._join(a, opsym, b)

    def _assign(self, st, env):
        # `a = b = expr` binds both names to the one value.
        value = self.go(st.value, env)
        for t in st.targets:
            self._bind(t, value, env)

    def _bind(self, t, value, env):
        if isinstance(t, ast.Name):
            env[t.id] = value
            return
        if isinstance(t, (ast.Tuple, ast.List)):
            # `a1, a2, a3 = _LABS_A` -- the right-hand side is a tuple
            # of published coefficients, which go() has already read.
            if not isinstance(value, tuple):
                raise Unextractable("unpacking a non-sequence")
            if len(t.elts) != len(value):
                raise Unextractable("unpacking %d names from %d values"
                                    % (len(t.elts), len(value)))
            for lhs, v in zip(t.elts, value):
                self._bind(lhs, v, env)
            return
        raise Unextractable("assignment to %s" % type(t).__name__)

    def _unroll(self, st, env):
        """Unroll `for j in range(n):` / over a literal sequence.

        Only bounded loops with a constant trip count, which is what the
        preset bodies use -- a product over five pentagon planes, a sum
        over three cyclic terms.  The loop variable is a NUMBER on each
        pass, so trig on it folds and the unrolled body is as clean as
        if the author had written the terms out.
        """
        if st.orelse:
            raise Unextractable("for/else")
        it = st.iter
        values = None
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) \
                and it.func.id == "range":
            bounds = [self.go(a, env) for a in it.args]
            if not all(isinstance(b, (int, float))
                       and float(b).is_integer() for b in bounds):
                raise Unextractable("range() with a non-constant bound")
            values = list(range(*[int(b) for b in bounds]))
        else:
            # A named constant table -- `for c, n, r in _DECO_TET_CIRCLES`
            # -- is as fixed as a literal, so it unrolls the same way.
            try:
                seq = self.go(it, env)
            except Unextractable:
                seq = None
            if isinstance(seq, tuple):
                values = list(seq)
        if values is None:
            raise Unextractable("loop over a non-constant iterable")
        if len(values) > 64:
            raise Unextractable("loop of %d passes" % len(values))
        for v in values:
            # The target may destructure -- `for (cx, cy, cz), n, r in ...`
            # over a table of circle centres, normals and radii.
            self._bind(st.target, v, env)
            for inner in st.body:
                if isinstance(inner, ast.Assign):
                    self._assign(inner, env)
                elif isinstance(inner, ast.AugAssign):
                    if not isinstance(inner.target, ast.Name):
                        raise Unextractable("augmented assign in loop")
                    cur = env.get(inner.target.id)
                    if cur is None:
                        raise Unextractable("loop accumulator %s is unset"
                                            % inner.target.id)
                    env[inner.target.id] = self._binop_vals(
                        cur, inner.op, inner.value, env)
                elif isinstance(inner, ast.Pass):
                    continue
                else:
                    raise Unextractable("statement %s in a loop"
                                        % type(inner).__name__)


def _parse_def(fn):
    """The ast.FunctionDef of a live function object.

    An already-parsed def passes straight through, which is how a
    caller can inline a function whose source `inspect` cannot reach.
    """
    if isinstance(fn, ast.FunctionDef):
        return fn
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        raise Unextractable("no source: %s" % exc)
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise Unextractable("source does not parse: %s" % exc)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise Unextractable("not a plain function (a lambda or a closure)")


def extract(fn, mod, params=None, coords=("x", "y", "z")):
    """Inline `fn`'s body to one expression string.

    `fn` is a shipped `PRESETS[key][1]`; `mod` the module it lives in
    (its globals supply the numeric constants); `params` the values to
    substitute for the arguments after x, y, z -- normally the
    operator's own defaults.

    Returns the string.  Raises `Unextractable` with a stated reason,
    which the caller should record: a refusal with a reason is data, a
    silent null is a bug that looks like data.
    """
    fdef = _parse_def(fn)
    names = [a.arg for a in fdef.args.args]
    if len(names) < len(coords):
        raise Unextractable("takes %d arguments, expected at least %d"
                            % (len(names), len(coords)))
    env = {}
    for want, got in zip(coords, names):
        env[got] = _Sym(want)
    supplied = dict(params or {})
    defaults = fdef.args.defaults
    for nm, d in zip(names[len(names) - len(defaults):], defaults):
        if nm not in supplied:
            try:
                supplied[nm] = ast.literal_eval(d)
            except ValueError:
                pass
    for nm in names[len(coords):]:
        if nm not in supplied:
            raise Unextractable("parameter %r has no value" % nm)
        v = supplied[nm]
        # `None` is allowed through: several presets default a
        # coefficient to None and select the published value with
        # `a0 if a is None else float(a)`.  It can only ever be used in
        # such a test -- arithmetic on it is refused in _sym().
        if v is None:
            env[nm] = None
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise Unextractable("parameter %r is %r, not a number"
                                % (nm, v))
        env[nm] = v

    out = _Inliner(mod, coords).run(fdef.body, env)
    if not isinstance(out, _Sym):
        raise Unextractable("body folds to the constant %r -- it does not "
                            "mention the coordinates" % out)
    if len(out.text) > _MAX_LEN:
        raise Unextractable("expansion is %d characters; the body reuses "
                            "locals too heavily to inline readably"
                            % len(out.text))
    return out.text


def preset_defaults(mod, key, fn):
    """The values the OPERATOR would pass for a preset's extra arguments.

    The preset functions take (x, y, z, mu, ...) with `mu` and `fold`
    declared with NO Python default, because they are shared operator
    properties rather than function arguments: `build_algebraic`
    supplies mu=1.3 and fold=3, and `PRESET_PARAMS` overrides them per
    preset (KUMMER at 1.3, NORM_ONE at 4.0).

    Reading `inspect.signature().default` therefore yields
    `inspect._empty` -- not a number -- for exactly the argument that
    matters.  Both the extraction and the oracle must use the same
    values, or a mismatch would read as a wrong equation when it is
    only a wrong parameter.
    """
    sig = inspect.signature(fn)
    names = list(sig.parameters)[3:]
    rows = getattr(mod, "PRESET_PARAMS", {}).get(key, ())
    declared = {}
    for row in rows:
        attr, kind, dflt = row[0], row[2], row[3]
        declared[attr] = (int(round(float(dflt))) if kind == "INT"
                          else float(dflt))
    shared = {"mu": float(getattr(mod, "MU_DEFAULT", 1.3)),
              "fold": int(getattr(mod, "FOLD_DEFAULT", 3))}
    out = {}
    for name in names:
        if name in declared:
            out[name] = declared[name]
            continue
        d = sig.parameters[name].default
        if d is not inspect.Parameter.empty:
            out[name] = d
            continue
        if name in shared:
            out[name] = shared[name]
            continue
        raise Unextractable("argument %r has no default and is not a "
                            "declared operator property" % name)
    return out


def _selftest():
    """Inline a handful of hand-checked bodies; raise on failure."""
    import ast as _ast

    src = textwrap.dedent('''
        import math
        _SQRT2 = math.sqrt(2.0)
        _COEFFS = (2.5, -1.5)

        def _t2(t):
            return 2.0 * t * t - 1.0

        def straight(x, y, z, mu=2.0):
            core = x * x + y * y + z * z - mu
            return core * core - 1.0

        def helper(x, y, z, mu=0.0):
            return _t2(x) + _t2(y) + _SQRT2 * z + mu

        def nested(x, y, z, mu=0.0):
            def sq(t):
                return t * t
            return sq(x) + sq(y) - sq(z)

        def looped(x, y, z, mu=0.0):
            prod = 1.0
            for j in range(3):
                a = 2.0 * math.pi * j / 3.0
                prod = prod * (math.cos(a) * x + math.sin(a) * y - z)
            return prod

        def unpacked(x, y, z, mu=0.0):
            a, b = x * y, y * z
            return a + b - z

        def folds(x, y, z, mu=3.0):
            return mu * 2.0 - 1.0

        def branches(x, y, z, mu=0.0):
            if x > 0:
                return x
            return y

        def decidable(x, y, z, mu=2.0):
            if mu > 1.0:
                return x * x - y * z
            return z

        def coefficients(x, y, z, mu=0.0):
            a1, a2 = _COEFFS
            return a1 * x * x + a2 * y - z

        def picks(x, y, z, mu=0.0, a=None):
            a = 4.0 if a is None else float(a)
            return a * x + y - z

        def precedence(x, y, z, mu=0.0):
            # Every place a dropped bracket would change the value:
            # non-associative - and /, right-associative **, a unary
            # minus under a power, and a sum used as a base.
            a = x - y
            b = y - z
            c = x + y
            return (a - b) / (b - a) + c ** 2 - (-x) ** 3 + x / (y * z) \
                - a * (b + c) + 2 ** (x + y)
    ''')
    ns = {}
    exec(compile(src, "<algextract-selftest>", "exec"), ns)   # noqa: S102

    # The stand-in module carries the PARSED defs, because functions made
    # by exec() have no source for inspect to find.  Numeric globals go
    # on as themselves, exactly as in the real module.
    class _Mod:
        pass
    mod = _Mod()
    tree = ast.parse(src)
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for k, v in ns.items():
        setattr(mod, k, defs.get(k, v))

    import random

    def same(text, fn, params=None, n=120):
        """Numeric agreement between the extracted text and the body."""
        rng = random.Random(7)
        env_extra = params or {}
        for _ in range(n):
            p = {"x": rng.uniform(-2, 2), "y": rng.uniform(-2, 2),
                 "z": rng.uniform(-2, 2)}
            got = eval(text, {"__builtins__": {}}, dict(       # noqa: S307
                p, **{k: getattr(math, k) for k in
                      ("sqrt", "sin", "cos", "tan", "exp", "log", "sinh",
                       "cosh", "tanh")}, abs=abs, **env_extra))
            want = fn(p["x"], p["y"], p["z"], **(params or {}))
            if abs(got - want) > 1e-9 * max(1.0, abs(want)):
                raise AssertionError("%s: %r vs %r\n  %s"
                                     % (fn.__name__, got, want, text))

    for name in ("straight", "helper", "nested", "looped", "unpacked",
                 "decidable", "coefficients", "picks", "precedence"):
        fn = ns[name]
        text = extract(defs[name], mod)
        _ast.parse(text, mode="eval")            # it must be an expression
        same(text, fn)

    # a parameter override reaches the body
    text = extract(defs["straight"], mod, params={"mu": 3.5})
    same(text, ns["straight"], params={"mu": 3.5})
    if "3.5" not in text:
        raise AssertionError("parameter default was not substituted: %s" % text)

    # NEGATIVE CONTROLS -- each must refuse, not guess.
    # A branch on a COORDINATE is the one this must never take: the
    # function is genuinely piecewise, so no single expression is it,
    # and a sampling oracle could still pass if one side is rare.
    for name, why in (("folds", "coordinate-free"),
                      ("branches", "branch on x")):
        try:
            extract(defs[name], mod)
        except Unextractable:
            pass
        else:
            raise AssertionError("%s should have been refused (%s)"
                                 % (name, why))

    # a missing parameter is a refusal, never a guessed zero
    noparam = ast.parse(textwrap.dedent("""
        def noparam(x, y, z, mu):
            return x + y + z + mu
    """)).body[0]
    try:
        extract(noparam, mod)
    except Unextractable:
        pass
    else:
        raise AssertionError("a parameter with no value must be refused")

    print("RESULT: OK  (surfdb.algextract, 10 bodies inlined incl. a "
          "precedence adversary, 3 refusals incl. a branch on a "
          "coordinate)")
