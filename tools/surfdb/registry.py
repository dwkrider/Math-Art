# Reading the generator registries WITHOUT importing them.
#
# The engine packages -- math_art/minsurf/, math_art/surfaces/ -- are
# deliberately bpy-free and import fine outside Blender.  The flat
# generator modules are not: every one of them does `import bpy` at the
# top, so `import topological_surface_generator` fails on any machine
# that is not running inside Blender.
#
# That would leave the build able to derive the minimal, TPMS and
# algebraic stages and nothing else -- roughly two thirds of the corpus
# unreachable without launching Blender for a job that needs no geometry.
#
# So the flat modules are read as SOURCE and their enum tables recovered
# with `ast`.  Two shapes cover everything in math_art/:
#
#   * a top-level table -- `PRESET_ITEMS = [...]`, `_CONOID_KINDS = [...]`,
#     `PRESETS = {...}` -- whose entries are (key, label, description)
#     tuples or key -> (label, ...) pairs.  Some values hold function
#     references, so the table cannot be `literal_eval`ed; only the
#     strings are pulled out, which is all that is wanted.
#
#   * enum items written inline in the `EnumProperty(items=[...])` call,
#     which several generators do.  These are found structurally: a tuple
#     whose first element is an ALL-CAPS string literal and whose second
#     is a human-readable string is an enum item and nothing else in
#     these files looks like that.
#
# Reading source rather than importing has a second benefit worth stating:
# the build cannot accidentally execute generator code, so a malformed
# record can never come from a module side effect.

import ast
import os
import re


def _module_source(root, dotted):
    """Source text of `dotted` under `root`, or None."""
    path = os.path.join(root, *dotted.split(".")) + ".py"
    if not os.path.exists(path):
        return None, path
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read(), path


def _strings(node):
    """Every string constant directly inside `node`, in order."""
    out = []
    for sub in ast.iter_child_nodes(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out.append(sub.value)
    return out


def _is_key(text):
    """Does `text` look like an enum key rather than a label?"""
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]*", text)) and len(text) >= 2


def top_level_table(source, var):
    """(key, label, line) triples from a top-level `var = [...]`/`{...}`.

    The LINE NUMBER matters: several modules carry more than one
    `References:` block, one per family section, and a row is cited from
    the nearest block above it (tools/surfdb/references.py).  Without the
    line, every row in an 11,000-line module would inherit the header.

    Handles both the list-of-tuples and dict-of-tuples shapes, and does
    NOT evaluate the values -- several tables hold function references.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if var not in names:
            continue
        val = node.value
        out = []
        if isinstance(val, (ast.List, ast.Tuple)):
            for item in val.elts:
                if isinstance(item, (ast.Tuple, ast.List)):
                    ss = _strings(item)
                    if len(ss) >= 2 and _is_key(ss[0]):
                        out.append((ss[0], ss[1], item.lineno))
        elif isinstance(val, ast.Dict):
            for k, v in zip(val.keys, val.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    continue
                label = None
                if isinstance(v, (ast.Tuple, ast.List)):
                    ss = _strings(v)
                    label = ss[0] if ss else None
                elif isinstance(v, ast.Constant) and isinstance(v.value, str):
                    label = v.value
                out.append((k.value, label or k.value, k.lineno))
        return out
    return []


def inline_enum_items(source):
    """(key, label) pairs for enum items written inline in the source.

    An enum item is a tuple whose first element is an ALL-CAPS string and
    whose second is a human-readable one.  Nothing else in these modules
    has that shape.
    """
    tree = ast.parse(source)
    out = []
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple):
            continue
        ss = _strings(node)
        if len(ss) >= 2 and _is_key(ss[0]) and not _is_key(ss[1]):
            if ss[0] not in seen:
                seen.add(ss[0])
                out.append((ss[0], ss[1], node.lineno))
    return out


def dict_key_lines(source, pattern):
    """key -> line number, for dict literals keyed by an ALL-CAPS string.

    Used for math_art/minsurf/zoo.py, whose rows are written both as
    entries of one big dict and as later `WE_SURFACES['KEY'] = {...}`
    assignments; both forms are matched.
    """
    out = {}
    for n, line in enumerate(source.splitlines(), 1):
        m = re.match(pattern, line)
        if m and m.group(1) not in out:
            out[m.group(1)] = n
    return out


def read_rows(root, dotted, tables=(), inline=False, only=None, exclude=()):
    """Rows for one flat generator module.

    `tables` names top-level tables to read; `inline` additionally scans
    for inline enum items.  `only` restricts to a key set (so a table
    shared with unrelated enums does not leak rows); `exclude` drops keys.

    Returns (rows, path).  An empty list with an existing path means the
    module's shape changed -- which the builder reports rather than
    silently emitting nothing.
    """
    source, path = _module_source(root, dotted)
    if source is None:
        return [], path
    rows = []
    seen = set()
    for var in tables:
        for key, label, line in top_level_table(source, var):
            if key not in seen:
                seen.add(key)
                rows.append((key, label, line))
    if inline:
        for key, label, line in inline_enum_items(source):
            if key not in seen:
                seen.add(key)
                rows.append((key, label, line))
    if only is not None:
        rows = [r for r in rows if r[0] in only]
    if exclude:
        rows = [r for r in rows if r[0] not in exclude]
    return rows, path


def _selftest():
    """Parser self-test on synthetic sources; raises on failure."""
    src = '''
import bpy
PRESET_ITEMS = [
    ('KLEIN', "Klein Bottle", "a one-sided surface"),
    ('BOY', "Boy's Surface", "an immersion of RP^2"),
]
PRESETS = {
    'PSEUDOSPHERE': ("Pseudosphere", _fn, (0.0, 3.0), True),
    'DINI': ("Dini Surface", _fn2, (0.05, 1.5), False),
}
def draw():
    prop = EnumProperty(items=[
        ('CYCLIDE_RING', "Dupin Cyclide (ring)", "inverted torus"),
        ('GABRIEL', "Gabriel's Horn", "finite volume, infinite area"),
    ])
'''
    got = [(k, l) for k, l, _n in top_level_table(src, "PRESET_ITEMS")]
    assert got == [("KLEIN", "Klein Bottle"), ("BOY", "Boy's Surface")], got

    # a dict whose values hold FUNCTION REFERENCES must still yield its
    # labels -- literal_eval would raise here, which is why it is not used
    got = [(k, l) for k, l, _n in top_level_table(src, "PRESETS")]
    assert got == [("PSEUDOSPHERE", "Pseudosphere"), ("DINI", "Dini Surface")], got

    got = [(k, l) for k, l, _n in inline_enum_items(src)]
    keys = [k for k, _ in got]
    assert "CYCLIDE_RING" in keys and "GABRIEL" in keys, got
    # the key/label test must not mistake a label for a key
    for k, lbl in got:
        assert _is_key(k) and not _is_key(lbl), (k, lbl)

    assert top_level_table(src, "NOT_THERE") == []

    assert _is_key("KLEIN") and _is_key("TET4_TRICONIC")
    assert not _is_key("Klein Bottle") and not _is_key("A")

    print("RESULT: OK  (surfdb.registry)")
