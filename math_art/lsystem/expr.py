
# Arithmetic / logical expressions for parametric L-systems.
#
# Productions in a parametric L-system carry expressions in their
# conditions and in their successors' parameters:
#
#     A(x,y) : y <= 3 -> A(x*2, x+y)
#
# Those expressions have to be evaluated once per matching module per
# derivation step, so they are compiled ONCE into a reverse-Polish
# program and then evaluated against a parameter binding.  Nothing here
# uses eval() or exec(): the operator set is fixed and closed, which is
# both a safety property (a grammar may come from a user text block) and
# a requirement of Blender's extension review.
#
# The operator set and precedence follow the C-like syntax used by the
# plant-modelling literature (cpfg / L+C / L-Py):
#
#     ( )                     grouping
#     ^                       exponentiation, right-associative
#     unary - !               negation, logical not
#     * /                     multiplicative
#     + -                     additive
#     < <= > >= == !=         relational
#     &&                      logical and
#     ||                      logical or
#
# Relational and logical expressions evaluate to 0.0 for false and 1.0
# for true, and an EMPTY condition is true -- both conventions are stated
# explicitly in the sources below.
#
# One documented incompatibility is handled here.  "The Algorithmic
# Beauty of Plants" prints the equality relation as `=`; the 2003
# SIGGRAPH course notes and L+C use C syntax, where `=` is assignment and
# equality is `==`.  These are the same authors a decade apart.  We
# accept `==` and silently read a bare `=` inside an expression as
# equality, so listings transcribed from either source parse unchanged.
#
# References:
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990, section 1.10 (parametric
#   L-systems; the operator set and the empty-condition rule).
# - Przemyslaw Prusinkiewicz, Jim Hanan, Mark Hammel and Radomir Mech,
#   "L-systems: from the Theory to Visual Models of Plants", SIGGRAPH
#   2003 course notes "L-systems and Beyond", chapter 2-1, section 4 --
#   the C-like expression grammar reproduced here.
# - Radoslaw Karwowski and Przemyslaw Prusinkiewicz, "Design and
#   Implementation of the L+C Modeling Language", Electronic Notes in
#   Theoretical Computer Science 86(2), 2003.
# - Edsger W. Dijkstra's shunting-yard algorithm (1961) for the
#   infix-to-RPN conversion.

import math


class ExprError(ValueError):
    """A grammar's expression could not be parsed or evaluated."""


# --- token kinds -----------------------------------------------------
_NUM, _VAR, _OP, _FUN, _LPAR, _RPAR, _COMMA = range(7)

# (precedence, right_associative, arity)
_OPS = {
    '||': (1, False, 2),
    '&&': (2, False, 2),
    '==': (3, False, 2), '!=': (3, False, 2),
    '<': (4, False, 2), '<=': (4, False, 2),
    '>': (4, False, 2), '>=': (4, False, 2),
    '+': (5, False, 2), '-': (5, False, 2),
    '*': (6, False, 2), '/': (6, False, 2), '%': (6, False, 2),
    '^': (8, True, 2),
    'u-': (7, True, 1),        # unary minus
    'u!': (7, True, 1),        # logical not
}

# Longest first, so '<=' wins over '<' and '&&' over '&'.
_SYMBOLS = ('||', '&&', '==', '!=', '<=', '>=', '<', '>',
            '+', '-', '*', '/', '%', '^', '!', '=')

_FUNCS = {
    'sin': lambda a: math.sin(math.radians(a)),
    'cos': lambda a: math.cos(math.radians(a)),
    'tan': lambda a: math.tan(math.radians(a)),
    'asin': lambda a: math.degrees(math.asin(max(-1.0, min(1.0, a)))),
    'acos': lambda a: math.degrees(math.acos(max(-1.0, min(1.0, a)))),
    'atan': lambda a: math.degrees(math.atan(a)),
    'sqrt': lambda a: math.sqrt(max(0.0, a)),
    'exp': math.exp,
    'log': lambda a: math.log(a) if a > 0.0 else 0.0,
    'abs': abs,
    'sign': lambda a: (a > 0.0) - (a < 0.0),
    'floor': math.floor,
    'ceil': math.ceil,
    'round': lambda a: float(round(a)),
    'min': min,
    'max': max,
    'pow': lambda a, b: _safe_pow(a, b),
    'atan2': lambda a, b: math.degrees(math.atan2(a, b)),
}
_FUNC_ARITY = {'min': 2, 'max': 2, 'pow': 2, 'atan2': 2}

# Trigonometry takes DEGREES, matching the turtle's angle conventions --
# a grammar that writes `+(90)` and one that writes `+(cos(0)*90)` should
# agree about what 90 means.

_CONSTANTS = {'pi': math.pi, 'e': math.e, 'true': 1.0, 'false': 0.0}


def _safe_pow(a, b):
    """`a ^ b`, never raising: a negative base with a fractional
    exponent has no real value, so return 0.0 rather than a complex."""
    try:
        if a < 0.0 and b != int(b):
            return 0.0
        r = a ** b
    except (OverflowError, ZeroDivisionError, ValueError):
        return 0.0
    return float(r) if isinstance(r, (int, float)) else 0.0


def _tokenize(src, shadow=()):
    """Infix source -> [(kind, value)].  Raises ExprError on a stray
    character, which is how a malformed grammar is reported to the UI
    rather than crashing the operator.

    `shadow` names -- a production's formal parameters and the grammar's
    own `#define` constants -- ALWAYS win over the built-in constants and
    functions below.  Without that, a grammar that defines `e` (a
    perfectly ordinary name for a width exponent) silently gets Euler's
    number instead: `q^e` with q=0.53 and e=0.5 should be 0.728, but
    evaluates to 0.178, and over twelve levels of a tree that is the
    difference between a width ratio of 0.022 and 3e-12 -- every branch
    but the trunk rendering sub-pixel.
    """
    shadow = set(shadow)
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or (c == '.' and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and (src[j].isdigit() or src[j] == '.'):
                j += 1
            # exponent form 1e-3
            if j < n and src[j] in 'eE' and _is_exponent(src, j):
                k = j + 1
                if k < n and src[k] in '+-':
                    k += 1
                if k < n and src[k].isdigit():
                    while k < n and src[k].isdigit():
                        k += 1
                    j = k
            try:
                out.append((_NUM, float(src[i:j])))
            except ValueError:
                raise ExprError(f"bad number {src[i:j]!r}")
            i = j
            continue
        if c.isalpha() or c == '_':
            j = i
            while j < n and (src[j].isalnum() or src[j] == '_'):
                j += 1
            name = src[i:j]
            low = name.lower()
            if name in shadow or low in shadow:
                out.append((_VAR, name))
            elif low in _FUNCS:
                out.append((_FUN, low))
            elif low in _CONSTANTS:
                out.append((_NUM, _CONSTANTS[low]))
            else:
                out.append((_VAR, name))
            i = j
            continue
        if c == '(':
            out.append((_LPAR, '('))
            i += 1
            continue
        if c == ')':
            out.append((_RPAR, ')'))
            i += 1
            continue
        if c == ',':
            out.append((_COMMA, ','))
            i += 1
            continue
        for sym in _SYMBOLS:
            if src.startswith(sym, i):
                # ABOP prints equality as '='; C-style sources use '=='.
                # Inside an expression there is nothing to assign to, so a
                # lone '=' can only mean equality.
                out.append((_OP, '==' if sym == '=' else sym))
                i += len(sym)
                break
        else:
            raise ExprError(f"unexpected character {c!r} in {src!r}")
    return out


def _is_exponent(src, j):
    """True when src[j] ('e' or 'E') begins a scientific-notation
    exponent rather than an identifier.

    The test is structural, not a name lookup.  Checking "is this a known
    constant?" got `2e-3` wrong, because `e` IS a known constant -- so
    scientific notation never parsed at all.  An exponent marker is a
    LONE e/E followed by an optional sign and at least one digit; `2exp(x)`
    fails the length test and `2e` fails the digit test.
    """
    k = j + 1
    if k < len(src) and src[k] in '+-':
        k += 1
    # `2e-3` and `1.5e2` are exponents; `2exp(x)` is not, because the
    # character after the e is a letter rather than a digit or a sign.
    return k < len(src) and src[k].isdigit()


def _to_rpn(tokens):
    """Shunting-yard: infix tokens -> reverse-Polish program."""
    out, stack = [], []
    prev = None
    for kind, val in tokens:
        if kind in (_NUM, _VAR):
            out.append((kind, val))
        elif kind == _FUN:
            stack.append((_FUN, val))
        elif kind == _COMMA:
            while stack and stack[-1][0] != _LPAR:
                out.append(stack.pop())
            if not stack:
                raise ExprError("comma outside a function call")
        elif kind == _OP:
            # A '-' or '!' is unary when nothing value-like precedes it.
            unary = prev is None or prev[0] in (_OP, _LPAR, _COMMA)
            op = val
            if unary:
                if val == '-':
                    op = 'u-'
                elif val == '!':
                    op = 'u!'
                else:
                    raise ExprError(f"{val!r} has no left operand")
            elif val == '!':
                raise ExprError("'!' used as a binary operator")
            prec, right, _ = _OPS[op]
            while stack and stack[-1][0] == _OP:
                top = stack[-1][1]
                tprec, _tr, _ta = _OPS[top]
                if tprec > prec or (tprec == prec and not right):
                    out.append(stack.pop())
                else:
                    break
            stack.append((_OP, op))
        elif kind == _LPAR:
            stack.append((_LPAR, '('))
        elif kind == _RPAR:
            while stack and stack[-1][0] != _LPAR:
                out.append(stack.pop())
            if not stack:
                raise ExprError("unbalanced ')'")
            stack.pop()
            if stack and stack[-1][0] == _FUN:
                out.append(stack.pop())
        prev = (kind, val)
    while stack:
        top = stack.pop()
        if top[0] == _LPAR:
            raise ExprError("unbalanced '('")
        out.append(top)
    return out


def _validate(prog, src):
    """Simulate the RPN stack at COMPILE time.

    Without this, `2+` compiles happily and only fails when evaluated --
    and because constant folding swallows evaluation errors, a truncated
    expression in a grammar would reach derivation as a silent 0.0.
    Arity is static, so the check is a simple depth walk."""
    depth = 0
    for kind, val in prog:
        if kind in (_NUM, _VAR):
            depth += 1
        elif kind == _FUN:
            arity = _FUNC_ARITY.get(val, 1)
            if depth < arity:
                raise ExprError(f"{val}() is missing arguments in {src!r}")
            depth += 1 - arity
        else:
            arity = _OPS[val][2]
            if depth < arity:
                raise ExprError(
                    f"operator {val!r} is missing operands in {src!r}")
            depth += 1 - arity
    if depth != 1:
        raise ExprError(f"malformed expression {src!r}")


class Expr:
    """A compiled expression.

    `Expr(src, names)` compiles once; `ev(binding)` evaluates against a
    dict of formal-parameter values.  `names` is the production's formal
    parameter list plus any grammar constants -- an expression naming
    anything else is rejected at COMPILE time, so a typo in a grammar
    surfaces as a parse error rather than as a silent zero at derivation
    time.
    """

    __slots__ = ('src', 'prog', 'const')

    def __init__(self, src, names=(), constants=None):
        self.src = src
        constants = constants or {}
        prog = _to_rpn(_tokenize(src, set(names) | set(constants)))
        known = set(names) | set(constants)
        for kind, val in prog:
            if kind == _VAR and val not in known:
                raise ExprError(f"unknown name {val!r} in {src!r}")
        # Fold grammar constants in now; they never change per module.
        self.prog = [(_NUM, float(constants[v])) if k == _VAR and v in constants
                     and v not in names else (k, v) for k, v in prog]
        _validate(self.prog, src)
        # _fold() evaluates, and ev() consults self.const -- so it has to
        # exist (and mean "not folded") before the fold is attempted.
        self.const = None
        self.const = self._fold()

    def _fold(self):
        """Return a float when the expression has no variables at all,
        else None.  Constant conditions and constant successor arguments
        are the common case in non-parametric grammars, and folding them
        removes the evaluator from that entire hot path."""
        if any(k == _VAR for k, _ in self.prog):
            return None
        try:
            return self.ev({})
        except ExprError:
            return None

    def ev(self, binding):
        if self.const is not None:
            return self.const
        st = []
        push, pop = st.append, st.pop
        for kind, val in self.prog:
            if kind == _NUM:
                push(val)
            elif kind == _VAR:
                try:
                    push(float(binding[val]))
                except KeyError:
                    raise ExprError(f"{val!r} is unbound")
            elif kind == _FUN:
                arity = _FUNC_ARITY.get(val, 1)
                if len(st) < arity:
                    raise ExprError(f"{val}() is missing arguments")
                args = [pop() for _ in range(arity)][::-1]
                try:
                    push(float(_FUNCS[val](*args)))
                except (ValueError, OverflowError, ZeroDivisionError):
                    push(0.0)
            else:
                _p, _r, arity = _OPS[val]
                if len(st) < arity:
                    raise ExprError(f"operator {val!r} is missing operands")
                if arity == 1:
                    a = pop()
                    push(-a if val == 'u-' else float(a == 0.0))
                else:
                    b, a = pop(), pop()
                    push(_apply(val, a, b))
        if len(st) != 1:
            raise ExprError(f"malformed expression {self.src!r}")
        return st[0]

    def __repr__(self):
        return f"Expr({self.src!r})"


def _apply(op, a, b):
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        return a / b if b else 0.0
    if op == '%':
        return math.fmod(a, b) if b else 0.0
    if op == '^':
        return _safe_pow(a, b)
    if op == '<':
        return float(a < b)
    if op == '<=':
        return float(a <= b)
    if op == '>':
        return float(a > b)
    if op == '>=':
        return float(a >= b)
    if op == '==':
        return float(a == b)
    if op == '!=':
        return float(a != b)
    if op == '&&':
        return float(a != 0.0 and b != 0.0)
    if op == '||':
        return float(a != 0.0 or b != 0.0)
    raise ExprError(f"unknown operator {op!r}")


def compile_expr(src, names=(), constants=None):
    """Compile `src`, or return None for an empty/absent source.

    A missing condition is TRUE (ABOP section 1.10: "a logical statement
    specified as the empty string is assumed to have value one"), and the
    caller distinguishes that from a compiled expression by the None."""
    if src is None:
        return None
    src = src.strip()
    if not src:
        return None
    return Expr(src, names, constants)


def _selftest():
    def ev(src, **b):
        return Expr(src, tuple(b)).ev(b)

    # precedence and associativity
    assert ev('2+3*4') == 14.0
    assert ev('(2+3)*4') == 20.0
    assert ev('2^3^2') == 512.0, "^ must be right-associative"
    assert ev('-2^2') == -4.0, "unary minus binds looser than ^"
    assert ev('10-3-2') == 5.0, "- must be left-associative"

    # relational / logical produce 0.0 and 1.0
    assert ev('3 < 4') == 1.0
    assert ev('3 > 4') == 0.0
    assert ev('1 && 0') == 0.0
    assert ev('1 || 0') == 1.0
    assert ev('!0') == 1.0 and ev('!5') == 0.0

    # the ABOP '=' vs C '==' incompatibility
    assert ev('3 = 3') == 1.0, "bare '=' must read as equality"
    assert ev('3 == 3') == 1.0

    # variables, functions, constants
    assert abs(ev('x*2+y', x=4, y=1) - 9.0) < 1e-12
    assert abs(ev('sqrt(16)') - 4.0) < 1e-12
    assert abs(ev('cos(0)') - 1.0) < 1e-12
    assert abs(ev('sin(90)') - 1.0) < 1e-12, "trig takes degrees"
    assert abs(ev('max(3,7)') - 7.0) < 1e-12
    assert abs(ev('pi') - math.pi) < 1e-12

    # division and pow are total, never raising
    assert ev('1/0') == 0.0
    assert ev('(0-2)^0.5') == 0.0

    # grammar constants fold to a literal, and folding is detected
    e = Expr('R*2', (), {'R': 0.5})
    assert e.const == 1.0
    assert Expr('x+1', ('x',)).const is None

    # unknown names are a COMPILE error, not a silent zero
    for bad in ('zzz+1', '2+', '(1', '1)', '3 $ 4'):
        try:
            Expr(bad, ())
        except ExprError:
            pass
        else:
            raise AssertionError(f"{bad!r} should not compile")

    # --- grammar names must SHADOW the built-ins ---------------------
    # A grammar is entitled to `#define e 0.5` for a width exponent.  If
    # the tokeniser resolves `e` to Euler's number first, `q^e` silently
    # becomes 0.53**2.718 = 0.178 instead of 0.53**0.5 = 0.728 -- and
    # compounded over a twelve-level tree that is a width ratio of 3e-12
    # instead of 0.022, i.e. every branch but the trunk sub-pixel.
    assert abs(Expr('q^e', (), {'q': 0.53, 'e': 0.5}).ev({})
               - 0.53 ** 0.5) < 1e-12
    assert abs(Expr('pi*2', (), {'pi': 10.0}).ev({}) - 20.0) < 1e-12
    assert abs(Expr('sin+1', ('sin',)).ev({'sin': 4}) - 5.0) < 1e-12
    # ... but an UNSHADOWED built-in still works
    assert abs(Expr('e', ()).ev({}) - math.e) < 1e-12
    assert abs(Expr('pi', ()).ev({}) - math.pi) < 1e-12
    assert abs(Expr('sin(90)', ()).ev({}) - 1.0) < 1e-12
    # scientific notation is not mistaken for the constant e
    for src, want in (('2e-3', 0.002), ('1.5e2', 150.0), ('4e+2', 400.0),
                      ('2*e', 2 * math.e), ('2*exp(1)', 2 * math.e)):
        assert abs(Expr(src, ()).ev({}) - want) < 1e-9, src

    # empty condition is True (represented as None by compile_expr)
    assert compile_expr('') is None and compile_expr(None) is None
    assert compile_expr('x>1', ('x',)) is not None

    # the ABOP eq (1.7) condition set, as a smoke test
    assert ev('y <= 3', y=3) == 1.0
    assert ev('y > 3', y=4) == 1.0

    print("expr: OK -- precedence, degrees-trig, '='/'==', "
          "constant folding, compile-time name check")
