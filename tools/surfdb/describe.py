"""Reader-facing descriptions: templated prose, and typeset formulae.

Every record gets a `description` block. It has three parts and they come
from three different places on purpose:

    summary   assembled here from the record's own categorical fields
    curated   written by a person, in curation.py, for surfaces that
              deserve more than a template can give
    formulas  DERIVED from the expressions the record already stores

WHY THE SUMMARY IS TEMPLATED AND NOT WRITTEN.  A record's
`definition.note` is written for someone who already knows the subject
and is carefully hedged about what is and is not being claimed -- "no
elementary (g, dh) pair on a plane domain is stored in the code", "the
shipped implementation is authoritative". Summarising that automatically
would produce confident sentences the database does not support. The
template only ever restates controlled vocabularies whose values are
known in advance, so it can be wrong about style but not about fact.

WHY THE FORMULAE ARE DERIVED AND NEVER TYPED.  This is the rule the
project has already paid for twice. `weextract.py` records that a
hand-typed pass using `inspect.getsource` silently truncated eight of
sixteen entries into "plausible-looking rational functions of the wrong
value", and `polynomial.verify_against` exists because a wrong equation
does not raise -- it quietly describes a different surface. A formula
typed by hand into a description would fail exactly that way, and it
would fail on the page a reader trusts most. So the formula shown is
compiled from the same string the mesher integrates.

LATEX AND MATHML, BOTH FROM THE AST.  The page needs MathML to render
without a library -- the site vendors nothing and fetches nothing -- and
LaTeX is what a reader wants to copy. Writing a LaTeX parser to get the
MathML would put a second transcription step in the middle, which is the
failure this module exists to avoid, so both are emitted from the same
parse tree in one pass. The LaTeX is carried inside the MathML as an
`<annotation>`, which is where a copy-paste-aware browser looks for it.

NUMBERS ARE NOT PRETTIFIED.  A decic's coefficients really are things
like 6.854101966249686, and rounding them for display would put a
different polynomial on the page. Long constants are ugly; a wrong
surface is worse. Trailing zeros go, nothing else does.
"""
import ast

from . import expr

# Names that should set as Greek. Everything else sets upright-italic as
# written, which is right for x, y, z, u, v and for parameters like a, k.
GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "rho", "sigma", "tau",
    "upsilon", "phi", "chi", "psi", "omega",
}

# The language's named constants (expr.CONSTANTS). These are NOT
# variables and must not be set as though they were: "pi" typeset as two
# italic letters reads as p times i, which is a different expression.
# `e` takes upright roman for the same reason -- an italic e is a
# variable, an upright one is Euler's number.
CONST_LATEX = {"pi": "\\pi", "e": "\\mathrm{e}",
               "phi": "\\varphi", "inf": "\\infty"}
CONST_MATHML = {"pi": "π", "e": "e", "phi": "φ", "inf": "∞"}

# Functions that get a name in roman type rather than a symbol.
NAMED = {
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh", "exp", "log", "min", "max",
}

# Binding strength, so parentheses appear where they change the meaning
# and nowhere else.
P_ADD, P_MUL, P_UNARY, P_POW, P_ATOM = 1, 2, 3, 4, 5


def _num(v):
    """A number as it should appear: exact, with no trailing noise."""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    s = repr(float(v))
    if s.endswith(".0"):
        s = s[:-2]
    if "e" in s or "E" in s:                       # 1e-05 -> 1 \times 10^{-5}
        mant, exp_ = s.lower().split("e")
        mant = mant.rstrip("0").rstrip(".") or "1"
        return (mant, int(exp_))
    return s


def _sym(name):
    if name in CONST_LATEX:
        return CONST_LATEX[name]
    if name in GREEK:
        return "\\" + name
    if name == "nn":            # the zoo's integer family index
        return "n"
    if len(name) > 1:
        return "\\mathit{%s}" % name
    return name


def _factors(n):
    """Flatten a product into its factors, left to right."""
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
        return _factors(n.left) + _factors(n.right)
    return [n]


def _grouped(n):
    """Factors of a product, with repeats collapsed into powers.

    The presets are written the way a programmer writes them -- `x * x`
    rather than `x ** 2`, because that is faster and reads fine in code --
    and typeset literally that comes out as "x x", which is nobody's idea
    of a quartic. Collapsing repeats is a change of NOTATION and not of
    value: it groups factors that are the same expression tree, so the
    product it denotes is identical term for term. That is the line this
    module draws. Rounding 1.6900000000000002 to 1.69 would cross it,
    because those are two different numbers, and it is not done.

    Returns [(node, exponent)] in order of first appearance.
    """
    out, seen = [], {}
    for f in _factors(n):
        key = ast.dump(f)
        if key in seen:
            out[seen[key]][1] += 1
        else:
            seen[key] = len(out)
            out.append([f, 1])
    return [(f, e) for f, e in out]


class _Latex:
    """AST -> LaTeX. Returns (text, precedence)."""

    def go(self, n):
        if isinstance(n, ast.Expression):
            return self.go(n.body)
        if isinstance(n, ast.Constant):
            v = _num(n.value)
            if isinstance(v, tuple):
                return "%s \\times 10^{%d}" % v, P_MUL
            return v, P_ATOM
        if isinstance(n, ast.Name):
            return _sym(n.id), P_ATOM
        if isinstance(n, ast.UnaryOp):
            t, _ = self.wrap(n.operand, P_UNARY)
            return ("-" if isinstance(n.op, ast.USub) else "+") + t, P_UNARY
        if isinstance(n, ast.Call):
            return self.call(n)
        if isinstance(n, ast.BinOp):
            return self.binop(n)
        raise expr.ExprError("cannot typeset %s" % type(n).__name__)

    def wrap(self, n, need):
        t, p = self.go(n)
        return ("\\left(%s\\right)" % t if p < need else t), p

    def call(self, n):
        f = n.func.id
        args = [self.go(a)[0] for a in n.args]
        if f == "sqrt":
            return "\\sqrt{%s}" % args[0], P_ATOM
        if f == "atan2":
            return "\\operatorname{atan2}\\left(%s\\right)" % ", ".join(args), P_ATOM
        if f == "pow" and len(args) == 2:
            return "{%s}^{%s}" % (args[0], args[1]), P_POW
        if f in NAMED:
            inner = ", ".join(args)
            return "\\%s\\left(%s\\right)" % (f, inner), P_ATOM
        return "\\operatorname{%s}\\left(%s\\right)" % (f, ", ".join(args)), P_ATOM

    def binop(self, n):
        op = n.op
        if isinstance(op, ast.Div):
            # \frac takes its own arguments, so neither side needs bracketing.
            return ("\\frac{%s}{%s}" % (self.go(n.left)[0], self.go(n.right)[0]),
                    P_ATOM)
        if isinstance(op, ast.Pow):
            base, _ = self.wrap(n.left, P_ATOM)
            return "%s^{%s}" % (base, self.go(n.right)[0]), P_POW
        if isinstance(op, ast.Mult):
            parts = []
            for f, e in _grouped(n):
                t, _ = self.wrap(f, P_ATOM if e > 1 else P_MUL)
                parts.append("%s^{%d}" % (t, e) if e > 1 else t)
            out = parts[0]
            for t in parts[1:]:
                # Juxtaposition where it reads cleanly, an explicit dot
                # where it would otherwise look like one longer name or
                # a numeral glued to a numeral.
                join = " " if (t[:1].isalpha() or t.startswith(chr(92))) else " " + chr(92) + "cdot "
                out += join + t
            return out, P_MUL
        if isinstance(op, (ast.Add, ast.Sub)):
            a, _ = self.wrap(n.left, P_ADD)
            b, _ = self.wrap(n.right, P_ADD)
            return "%s %s %s" % (a, "+" if isinstance(op, ast.Add) else "-", b), P_ADD
        raise expr.ExprError("cannot typeset operator %s" % type(op).__name__)


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


class _MathML:
    """AST -> MathML Core, from the same tree the LaTeX came from."""

    def go(self, n, need=0):
        if isinstance(n, ast.Expression):
            return self.go(n.body)
        if isinstance(n, ast.Constant):
            v = _num(n.value)
            if isinstance(v, tuple):
                return ("<mrow><mn>%s</mn><mo>&#xD7;</mo><msup><mn>10</mn>"
                        "<mn>%d</mn></msup></mrow>" % (_esc(v[0]), v[1]))
            return "<mn>%s</mn>" % _esc(v)
        if isinstance(n, ast.Name):
            return "<mi>%s</mi>" % _esc(self.ident(n.id))
        if isinstance(n, ast.UnaryOp):
            sign = "&#x2212;" if isinstance(n.op, ast.USub) else "+"
            return ("<mrow><mo>%s</mo>%s</mrow>"
                    % (sign, self.par(n.operand, P_UNARY)))
        if isinstance(n, ast.Call):
            return self.call(n)
        if isinstance(n, ast.BinOp):
            return self.binop(n)
        raise expr.ExprError("cannot typeset %s" % type(n).__name__)

    def ident(self, name):
        if name in CONST_MATHML:
            return CONST_MATHML[name]
        if name in GREEK:
            return _GREEK_CHAR.get(name, name)
        if name == "nn":
            return "n"
        return name

    def prec(self, n):
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, (ast.Add, ast.Sub)):
                return P_ADD
            if isinstance(n.op, ast.Mult):
                return P_MUL
            if isinstance(n.op, ast.Div):
                return P_ATOM
            if isinstance(n.op, ast.Pow):
                return P_POW
        if isinstance(n, ast.UnaryOp):
            return P_UNARY
        return P_ATOM

    def par(self, n, need):
        inner = self.go(n)
        if self.prec(n) < need:
            return ("<mrow><mo stretchy=\"false\">(</mo>%s"
                    "<mo stretchy=\"false\">)</mo></mrow>" % inner)
        return inner

    def call(self, n):
        f = n.func.id
        args = [self.go(a) for a in n.args]
        if f == "sqrt":
            return "<msqrt>%s</msqrt>" % args[0]
        sep = "<mo>,</mo>"
        body = sep.join(args)
        return ("<mrow><mi>%s</mi><mo>&#x2061;</mo><mo stretchy=\"false\">(</mo>"
                "%s<mo stretchy=\"false\">)</mo></mrow>" % (_esc(f), body))

    def binop(self, n):
        op = n.op
        if isinstance(op, ast.Div):
            return "<mfrac>%s%s</mfrac>" % (self.go(n.left), self.go(n.right))
        if isinstance(op, ast.Pow):
            return "<msup>%s%s</msup>" % (self.par(n.left, P_ATOM),
                                          self.go(n.right))
        if isinstance(op, ast.Mult):
            parts = []
            for f, e in _grouped(n):
                inner = self.par(f, P_ATOM if e > 1 else P_MUL)
                parts.append("<msup>%s<mn>%d</mn></msup>" % (inner, e)
                             if e > 1 else inner)
            return "<mrow>%s</mrow>" % "<mo>&#x2062;</mo>".join(parts)
        sign = "+" if isinstance(op, ast.Add) else "&#x2212;"
        return ("<mrow>%s<mo>%s</mo>%s</mrow>"
                % (self.par(n.left, P_ADD), sign, self.par(n.right, P_ADD)))


_GREEK_CHAR = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "upsilon": "υ", "phi": "φ", "chi": "χ",
    "psi": "ψ", "omega": "ω",
}


def typeset(text, relation=None):
    """One expression -> {'latex', 'mathml'}, or None if it is not typesettable.

    `relation` closes the statement, e.g. "= 0". It is typeset INSIDE the
    formula rather than left for the page to add afterwards, for two
    reasons. A polynomial on its own is an expression, not an equation --
    the implicit surface is the zero set, and the "= 0" is part of what
    is being said. And a sibling element beside display MathML has to be
    positioned against a box whose size the browser decides, which is how
    the relation ended up overlapping its own label.

    Returns None rather than raising: a record carrying an expression this
    cannot read should show no formula, never a wrong one.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        tree = expr.parse(text)
    except expr.ExprError:
        return None
    try:
        latex = _Latex().go(tree)[0]
        body = _MathML().go(tree)
    except (expr.ExprError, RecursionError):
        return None
    if relation:
        rel = relation.strip()
        latex = "%s %s" % (latex, rel)
        op, _, rhs = rel.partition(" ")
        body = ("%s<mo>%s</mo>%s"
                % (body, _esc(op), "<mn>%s</mn>" % _esc(rhs) if rhs else ""))
    mathml = ('<math xmlns="http://www.w3.org/1998/Math/MathML" '
              'display="block"><semantics><mrow>%s</mrow>'
              '<annotation encoding="application/x-tex">%s</annotation>'
              '</semantics></math>' % (body, _esc(latex)))
    return {"latex": latex, "mathml": mathml}


# --------------------------------------------------------------------
# The templated summary.
#
# Controlled vocabularies only. Every key is a value that occurs in the
# database; an unknown one falls through as its own token, so a newly
# added value reads awkwardly instead of disappearing.
# --------------------------------------------------------------------

FAMILY = {
    "minimal-periodic": "periodic minimal surface",
    "algebraic": "algebraic surface",
    "minimal": "minimal surface",
    "topological": "topological surface",
    "quadric": "quadric surface",
    "ruled": "ruled surface",
    "constant-curvature": "surface of constant curvature",
    "revolution": "surface of revolution",
    "misc": "surface",
    "swept": "swept surface",
    "cmc": "constant-mean-curvature surface",
    "physical": "physical surface model",
    "discrete": "discrete surface",
    "spectral": "spectral surface",
    "derived": "derived surface",
    "cyclide": "cyclide",
}

PERIODIC = {1: "singly periodic", 2: "doubly periodic", 3: "triply periodic"}

MODE = {
    "weierstrass": "given by a Weierstrass representation",
    "implicit": "defined by an implicit equation",
    "parametric": "given by a parametrisation",
    "nodal": "defined as a nodal algebraic surface",
    "derived": "derived from another surface in the catalogue",
}

CURVATURE = {
    "minimal": "of zero mean curvature",
    "cmc": "of constant mean curvature",
    "cmc1-bryant": "of constant mean curvature one in the Bryant sense",
    "flat": "of zero Gaussian curvature",
    "k-const-negative": "of constant negative Gaussian curvature",
    "k-const-positive": "of constant positive Gaussian curvature",
    "weingarten": "a Weingarten surface",
    "willmore": "a critical point of the Willmore energy",
}

EMBEDDING = {
    "embedded": "embedded",
    "immersed": "immersed",
    "singular": "immersed with singularities",
    "self-intersecting": "self-intersecting",
    "varies": "embedded or immersed depending on parameters",
}

SYMMETRY = {
    "space": "a crystallographic space group",
    "crystallographic": "a crystallographic group",
    "layer": "a layer group",
    "rod": "a rod group",
    "point": "a point group",
    "continuous": "a continuous symmetry group",
}


def _an(phrase):
    """Article by sound rather than spelling: "a uniform", "an ellipsoid"."""
    w = phrase.split()[0].lower().strip("(")
    if w[:3] in ("uni", "eul"):
        return "a " + phrase
    return ("an " if w[:1] in "aeiou" else "a ") + phrase


def summarise(rec):
    """One sentence about a surface, from its categorical fields only."""
    dfn = rec.get("definition") or {}
    topo = rec.get("topology") or {}
    sym = rec.get("symmetry") or {}
    fam = FAMILY.get(rec.get("primary_family"),
                     rec.get("primary_family") or "surface")
    cur = (rec.get("curvature") or {}).get("condition")
    rank = sym.get("periodicity_rank") or 0

    # The constant-curvature family names a condition the record then
    # states precisely; print the specific one, not both.
    if fam == "surface of constant curvature" and cur in CURVATURE:
        fam = "surface " + CURVATURE[cur]
        cur = None

    if rank in PERIODIC and fam == "periodic minimal surface":
        lead = "%s minimal surface" % PERIODIC[rank]
    elif rank in PERIODIC:
        lead = "%s %s" % (PERIODIC[rank], fam)
    else:
        lead = fam

    bits = ["%s is %s" % (rec["name"], _an(lead))]

    dup = ((cur == "minimal" and "minimal" in lead)
           or (cur == "cmc" and lead == "constant-mean-curvature surface"))
    if cur and cur != "none" and not dup:
        bits.append(CURVATURE.get(cur, cur))

    mode = MODE.get(dfn.get("mode"))
    if mode:
        bits.append(mode)

    facts = []
    if topo.get("genus_per_cell") is not None:
        facts.append("genus %s per unit cell" % topo["genus_per_cell"])
    elif topo.get("genus") is not None:
        facts.append("genus %s" % topo["genus"])
    ends = topo.get("ends") or []
    total = 0
    for e in ends:
        if isinstance(e.get("count"), int):
            total += e["count"]
        else:
            total = None
            break
    if total:
        facts.append("%d end%s" % (total, "" if total == 1 else "s"))
    emb = EMBEDDING.get((rec.get("embedding") or {}).get("quality"))
    if emb:
        facts.append(emb)
    if facts:
        bits.append(", ".join(facts))

    kind = SYMMETRY.get(sym.get("kind"))
    symbol = sym.get("hermann_mauguin") or sym.get("schoenflies")
    if kind and symbol:
        bits.append("with %s symmetry (%s)" % (symbol, kind))
    elif kind:
        bits.append("with %s" % kind)

    return ", ".join(bits) + "."


# Which stored expression is a defining formula, and what to call it.
FORMULA_FIELDS = (
    ("polynomial", "Implicit equation", "= 0"),
    ("level_function", "Level function", "= 0"),
    ("gauss_map", "Gauss map g", None),
    ("height_differential", "Height differential dh", None),
    ("x", "x(u, v)", None),
    ("y", "y(u, v)", None),
    ("z", "z(u, v)", None),
)


def formulas(rec):
    """Typeset every defining expression the record stores."""
    dfn = rec.get("definition") or {}
    out = []
    for key, label, tail in FORMULA_FIELDS:
        got = typeset(dfn.get(key), tail)
        if got:
            out.append({"field": key, "label": label,
                        "latex": got["latex"], "mathml": got["mathml"],
                        "relation": tail})
    return out


def describe(rec, curated=None):
    """The record's whole `description` block."""
    block = {"summary": summarise(rec), "source": "templated"}
    if curated:
        block["curated"] = curated
        block["source"] = "curated"
    f = formulas(rec)
    if f:
        block["formulas"] = f
    return block


def _selftest():
    cases = [
        ("x**2 + y**2 + z**2 - 1", "x^{2} + y^{2} + z^{2} - 1"),
        ("x*y", "x y"),
        ("2*x", "2 x"),              # juxtaposition, as it should be
        ("2*3", "2 \cdot 3"),        # but never numeral on numeral
        ("(x + y)*z", "\\left(x + y\\right) z"),
        ("x/(y + 1)", "\\frac{x}{y + 1}"),
        ("sqrt(x + 1)", "\\sqrt{x + 1}"),
        ("-x**2", "-x^{2}"),
        ("(x + y)**2", "\\left(x + y\\right)^{2}"),
        ("rho*z", "\\rho z"),
    ]
    for src, want in cases:
        got = typeset(src)
        if got is None:
            raise AssertionError("typeset(%r) returned None" % src)
        if got["latex"] != want:
            raise AssertionError("typeset(%r) latex = %r, want %r"
                                 % (src, got["latex"], want))
        if "<math" not in got["mathml"] or "</math>" not in got["mathml"]:
            raise AssertionError("typeset(%r) produced no MathML" % src)
        if "annotation" not in got["mathml"]:
            raise AssertionError("typeset(%r) MathML carries no LaTeX" % src)

    if "\\left(" in typeset("x*y*z")["latex"]:
        raise AssertionError("spurious parentheses in a plain product")

    # Repeated factors fold to a power. This is notation, not algebra:
    # the presets write x*x because that is faster to evaluate, and set
    # literally it reads "x x".
    for src, want in (("x*x", "x^{2}"), ("x*x*x", "x^{3}"),
                      ("2*x*x", "2 x^{2}"), ("x*y*x", "x^{2} y")):
        got = typeset(src)["latex"]
        if got != want:
            raise AssertionError("fold %r -> %r, want %r" % (src, got, want))
    if typeset("x*y*z")["latex"] != "x y z":
        raise AssertionError("distinct factors were folded together")
    if "msup" not in typeset("x*x")["mathml"]:
        raise AssertionError("MathML did not fold the repeated factor")

    # The named constants are constants, not variables: pi set as two
    # italic letters reads as p times i, a different expression, and an
    # italic e is a variable where an upright one is Euler's number.
    for src, want in (("pi", "\\pi"), ("phi", "\\varphi"),
                      ("e", "\\mathrm{e}"), ("2*pi*x", "2 \\pi x")):
        got = typeset(src)["latex"]
        if got != want:
            raise AssertionError("constant %r -> %r, want %r"
                                 % (src, got, want))
    if chr(0x3c0) not in typeset("pi")["mathml"]:
        raise AssertionError("MathML did not use the pi character")

    # Junk must yield nothing rather than something wrong.
    for bad in ("x +", "", None, 42):
        if typeset(bad) is not None:
            raise AssertionError("typeset(%r) should have refused" % (bad,))
    # An expression outside the language is refused by expr.parse.
    if typeset("__import__(chr(111))") is not None:
        raise AssertionError("a call outside the language was typeset")

    # A coefficient keeps every digit it was given: rounding one would
    # put a different surface on the page.
    lx = typeset("6.854101966249686*x")["latex"]
    if "6.854101966249686" not in lx:
        raise AssertionError("a coefficient was rounded: %r" % lx)

    rec = {"name": "Test", "primary_family": "algebraic",
           "definition": {"mode": "implicit", "polynomial": "x**2 - 1"},
           "curvature": {"condition": "none"},
           "symmetry": {"kind": "point", "schoenflies": "Ih"},
           "topology": {"genus": 0}, "embedding": {"quality": "embedded"}}
    d = describe(rec)
    if not d["summary"].startswith("Test is an algebraic surface"):
        raise AssertionError("summary: %r" % d["summary"])
    if not d.get("formulas"):
        raise AssertionError("no formula extracted from a stored polynomial")
    print("describe: OK (%d latex cases, mathml + annotation present)"
          % len(cases))
