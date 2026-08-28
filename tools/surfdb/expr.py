# The exact-expression language for the surface database.
#
# Every `exact` string in data/surfaces/ is written in a small fixed
# language and must evaluate to the `value` stored beside it, so the two
# can never drift.  This is the same contract data/polyhedra/ uses --
# see tools/polydb/exact.py -- extended for surfaces in three ways:
#
#   1. Hyperbolic and exponential functions (cosh, sinh, tanh, exp, log)
#      are admitted, because the catenoid, the pseudosphere and the
#      Delaunay roulettes are not expressible without them.
#   2. Expressions may reference the free PARAMETERS a record declares
#      (a family's `h`, `k`, `a`, `c`), which a polyhedron record never
#      needed because a polyhedron has no free parameters.
#   3. Expressions may reference the coordinates x, y, z (implicit
#      surfaces) or the chart variables u, v (parametric ones), so the
#      same parser serves both the scalar measures and the defining
#      equations.
#
# Parsing is done with Python's own `ast` module restricted to a
# whitelist of node types, rather than by `eval` on the raw string: the
# database is published, and a record is data, not code.
#
# References:
# - The measure-object convention follows netlib's embedded `bc`
#   expressions and D. McCooey's published closed forms, rendered into
#   one fixed grammar rather than two ad-hoc ones.

import ast
import math

# Named constants available to every expression.
CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "phi": (1.0 + math.sqrt(5.0)) / 2.0,
    "inf": math.inf,
}

# Whitelisted unary functions.  Kept deliberately small: a grammar that
# admits arbitrary calls is not a grammar, it is an interpreter.
FUNCTIONS = {
    "sqrt": math.sqrt,
    "cbrt": lambda t: math.copysign(abs(t) ** (1.0 / 3.0), t),
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "exp": math.exp,
    "log": math.log,
}

# Binary functions, kept separate so arity is checked rather than assumed.
FUNCTIONS2 = {
    "atan2": math.atan2,
    "min": min,
    "max": max,
    "pow": math.pow,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Constant,
)


class ExprError(ValueError):
    """A record's expression is not in the language, or cannot evaluate."""


def _check(node):
    """Reject anything outside the language, at PARSE time.

    Node-type whitelisting alone is not enough, and assuming it is was a
    real bug here: `__import__('os')` is structurally just a Call on a
    Name with a Constant argument, so every node in it is whitelisted.
    What makes it illegal is that `__import__` is not one of the
    whitelisted FUNCTIONS and that a string is not a numeric literal --
    both of which have to be checked explicitly, and both of which must
    be checked when the expression is PARSED rather than when it happens
    to be evaluated, or a record can carry an expression that only fails
    on the day something reads it.
    """
    for sub in ast.walk(node):
        if not isinstance(sub, _ALLOWED_NODES):
            raise ExprError(
                "disallowed syntax %s -- the exact language is integers, "
                "+ - * / **, parentheses, the whitelisted functions and "
                "the named constants" % type(sub).__name__)
        if isinstance(sub, ast.Constant) and not isinstance(sub.value, (int, float)):
            raise ExprError(
                "only numeric literals are allowed, got %r" % (sub.value,))
        if isinstance(sub, ast.Call):
            if not isinstance(sub.func, ast.Name):
                raise ExprError("only direct calls to named functions are allowed")
            if sub.func.id not in FUNCTIONS and sub.func.id not in FUNCTIONS2:
                raise ExprError("unknown function %r" % sub.func.id)
            if sub.keywords:
                raise ExprError("keyword arguments are not part of the language")


def parse(text):
    """Parse an expression string into a checked AST. Raises ExprError."""
    if not isinstance(text, str):
        raise ExprError("expression must be a string, got %r" % type(text))
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExprError("cannot parse %r: %s" % (text, exc))
    _check(tree)
    return tree


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ExprError("only numeric literals are allowed, got %r" % (node.value,))
    if isinstance(node, ast.Name):
        if node.id in env:
            return float(env[node.id])
        raise ExprError(
            "unknown name %r -- it is neither a constant nor a declared "
            "parameter of this record" % node.id)
    if isinstance(node, ast.UnaryOp):
        val = _eval(node.operand, env)
        return -val if isinstance(node.op, ast.USub) else +val
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, env)
        right = _eval(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0.0:
                raise ExprError("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            try:
                return float(left) ** float(right)
            except (ValueError, OverflowError) as exc:
                raise ExprError("bad power: %s" % exc)
        raise ExprError("unsupported operator %s" % type(node.op).__name__)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExprError("only direct calls to named functions are allowed")
        name = node.func.id
        args = [_eval(a, env) for a in node.args]
        if name in FUNCTIONS:
            if len(args) != 1:
                raise ExprError("%s takes 1 argument, got %d" % (name, len(args)))
            try:
                return FUNCTIONS[name](args[0])
            except (ValueError, OverflowError) as exc:
                raise ExprError("%s(%g): %s" % (name, args[0], exc))
        if name in FUNCTIONS2:
            if len(args) != 2:
                raise ExprError("%s takes 2 arguments, got %d" % (name, len(args)))
            return FUNCTIONS2[name](*args)
        raise ExprError("unknown function %r" % name)
    raise ExprError("unsupported node %s" % type(node).__name__)


def evaluate(text, params=None):
    """Evaluate an expression to a float.

    `params` supplies the record's declared parameters (and, for a
    defining equation, the coordinates).  Constants are always in scope.
    """
    env = dict(CONSTANTS)
    if params:
        env.update(params)
    return _eval(parse(text), env)


def free_names(text):
    """The set of names an expression references, minus the constants.

    The validator uses this to check that every name in a definition is
    either a declared parameter or a coordinate -- which is what catches
    a formula that silently depends on something the record never states.
    """
    tree = parse(text)
    names = set()
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            names.discard(sub.func.id)
    # function names reached via ast.Name in Call position are removed above,
    # but a bare reference to a function name is still an error, so keep them
    return {n for n in names
            if n not in CONSTANTS and n not in FUNCTIONS and n not in FUNCTIONS2}


def check_measure(measure, params=None, rel_tol=1e-9, abs_tol=1e-12):
    """Check one measure object: `exact` must evaluate to `value`.

    Returns (ok, detail).  A measure with no `exact` is vacuously ok --
    the database records honest nulls rather than fabricated closed
    forms, exactly as data/polyhedra does for the snub coordinates.
    """
    if measure is None:
        return True, "null"
    exact = measure.get("exact")
    value = measure.get("value")
    if exact is None:
        return True, "no exact form"
    if value is None:
        return False, "exact form given but value is null"
    try:
        got = evaluate(exact, params)
    except ExprError as exc:
        return False, "cannot evaluate %r: %s" % (exact, exc)
    if math.isinf(got) or math.isnan(got):
        return (math.isinf(value) == math.isinf(got),
                "non-finite: %r -> %g vs %g" % (exact, got, value))
    if abs(got - value) <= max(abs_tol, rel_tol * max(abs(got), abs(value))):
        return True, "%r = %.12g" % (exact, got)
    return False, "%r evaluates to %.12g but value says %.12g" % (exact, got, value)


def _selftest():
    """Numeric self-test; raises on failure.

    Run by tests/test_selftests.py.  No __main__ guard -- see CLAUDE.md.
    """
    assert abs(evaluate("2 + 3 * 4") - 14.0) < 1e-12
    assert abs(evaluate("sqrt(2)") - math.sqrt(2)) < 1e-12
    assert abs(evaluate("phi") - (1 + math.sqrt(5)) / 2) < 1e-12
    assert abs(evaluate("-4*pi") + 4 * math.pi) < 1e-12
    assert abs(evaluate("cosh(0)") - 1.0) < 1e-12
    assert abs(evaluate("2*a + b", {"a": 1.5, "b": 2.0}) - 5.0) < 1e-12

    # the language must REFUSE things that are not in it
    for bad in ["__import__('os')", "a if b else c", "[1,2]", "lambda: 1",
                "x.y", "a and b"]:
        try:
            parse(bad)
        except ExprError:
            pass
        else:
            raise AssertionError("expression %r should have been rejected" % bad)

    # unknown names are an error, not a silent zero -- this is what
    # catches a formula depending on something the record never declares
    try:
        evaluate("q + 1")
    except ExprError:
        pass
    else:
        raise AssertionError("unknown name should raise")

    assert free_names("2*a + sqrt(b) + pi") == {"a", "b"}
    assert free_names("x^2" if False else "x*x + y*y + z*z") == {"x", "y", "z"}

    ok, _ = check_measure({"exact": "-4*pi", "value": -12.566370614359172})
    assert ok
    ok, detail = check_measure({"exact": "-4*pi", "value": -12.0})
    assert not ok and "evaluates to" in detail
    ok, _ = check_measure({"exact": None, "value": 1.0})
    assert ok, "a null exact form is honest, not an error"
    ok, _ = check_measure({"exact": "sqrt(2)", "value": None})
    assert not ok, "an exact form with no value is a drift waiting to happen"

    print("RESULT: OK  (surfdb.expr)")
