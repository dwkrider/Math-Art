# A small, bounded cache for built geometry.
#
# Blender re-runs an operator's `execute` in full whenever ANY of its
# properties changes -- there is no partial re-run.  So nudging Rim
# Thickness rebuilds the surface the rim is drawn on, even though the
# surface cannot possibly have changed, and on a large mesh that is the
# whole of the delay.
#
# The fix is not to make the rebuild conditional (the operator has no
# way to know what changed) but to make it cheap: memoise the geometry
# builders on their own arguments.  Those arguments ARE the geometry
# parameters -- a builder takes the preset, the resolution, the cell
# counts and nothing else -- so a cache keyed on them is exactly a cache
# keyed on "everything except the rim".  Nothing has to enumerate which
# properties are which, which is the part that would rot.
#
# Two things this has to get right, and both are easy to get wrong:
#
# * SIZE.  These meshes are large -- a 5x5x5 gyroid is 4.3M vertices and
#   8.6M faces, about 300 MB as float64 and int64.  An unbounded cache,
#   or even a handful of entries, would be worse than the problem.  The
#   budget below is in bytes and the eviction is oldest-first.
#
# * OWNERSHIP.  Callers here edit what they are handed -- `_empty` in
#   the TPMS module says so explicitly -- so handing out the cached
#   arrays themselves would let one caller's edit surface as another
#   caller's geometry, which is the worst kind of bug to chase.  Every
#   hit returns copies.

import numpy as np

# Roughly one 5x5x5 gyroid.  Small enough to be invisible, large enough
# that the surface you are working on stays in hand while you drag a rim
# slider back and forth.
_MAX_BYTES = 350_000_000

_cache = {}          # key -> (verts, faces)
_order = []          # keys, oldest first
_bytes = 0


def _sizeof(item):
    total = 0
    for part in item:
        if isinstance(part, np.ndarray):
            total += part.nbytes
        else:
            try:
                n = len(part)
            except TypeError:
                n = 1
            # a list of index tuples: three or four ints per face, and
            # a Python int is not 8 bytes, but this only has to be the
            # right order of magnitude to keep the budget honest
            total += n * 32
    return total


def _copy(item):
    out = []
    for part in item:
        if isinstance(part, np.ndarray):
            out.append(part.copy())
        elif isinstance(part, list):
            out.append(list(part))
        else:
            out.append(part)
    return tuple(out)


def clear():
    """Drop everything.  For the self-test, and for anyone who wants the
    memory back."""
    global _bytes
    _cache.clear()
    del _order[:]
    _bytes = 0


def stats():
    return {'entries': len(_cache), 'bytes': _bytes}


def cached(key, build):
    """Return `build()`, from the cache when the same key was asked for
    before.  `key` must be hashable; the caller owns what comes back.
    """
    global _bytes
    hit = _cache.get(key)
    if hit is not None:
        # freshen: an entry being used should not be the next evicted
        if _order and _order[-1] != key:
            try:
                _order.remove(key)
            except ValueError:
                pass
            _order.append(key)
        return _copy(hit)

    item = tuple(build())
    size = _sizeof(item)
    if size > _MAX_BYTES:
        return item                     # too big to keep; hand it over
    _cache[key] = item
    _order.append(key)
    _bytes += size
    while _bytes > _MAX_BYTES and len(_order) > 1:
        old = _order.pop(0)
        gone = _cache.pop(old, None)
        if gone is not None:
            _bytes -= _sizeof(gone)
    return _copy(item)


def memoise(fn):
    """Wrap a geometry builder so repeated calls with the same arguments
    are free.  Arguments must be hashable -- every builder here takes
    scalars, strings and small tuples, which they are."""
    def wrapper(*args, **kwargs):
        try:
            key = (fn.__module__, fn.__name__, args,
                   tuple(sorted(kwargs.items())))
            hash(key)
        except TypeError:
            return fn(*args, **kwargs)   # unhashable: just build it
        return cached(key, lambda: fn(*args, **kwargs))
    wrapper.__name__ = getattr(fn, '__name__', 'wrapped')
    wrapper.__doc__ = getattr(fn, '__doc__', None)
    wrapper.__wrapped__ = fn
    return wrapper


def _selftest():
    ok = True
    clear()

    calls = {'n': 0}

    @memoise
    def build(kind, res):
        calls['n'] += 1
        v = np.arange(res * 3, dtype=float).reshape(-1, 3) + len(kind)
        f = [(0, 1, 2)] * max(1, res // 3)
        return v, f

    a_v, a_f = build('P', 30)
    b_v, b_f = build('P', 30)
    good = calls['n'] == 1 and np.array_equal(a_v, b_v)
    ok &= good
    print("geom_cache: same arguments build once (%d call%s) %s"
          % (calls['n'], '' if calls['n'] == 1 else 's',
             'OK' if good else 'FAIL'))

    build('D', 30)
    good = calls['n'] == 2
    ok &= good
    print("geom_cache: different arguments build again (%d calls) %s"
          % (calls['n'], 'OK' if good else 'FAIL'))

    # The caller owns what it is handed.  A hit must not alias the
    # cached arrays, or one caller's edit becomes another's geometry.
    a_v[0, 0] = -999.0
    a_f.append((9, 9, 9))
    c_v, c_f = build('P', 30)
    good = (c_v[0, 0] != -999.0 and len(c_f) == len(b_f)
            and calls['n'] == 2)
    ok &= good
    print("geom_cache: a hit is a copy, not an alias %s"
          % ('OK' if good else 'FAIL'))

    # eviction keeps the budget
    global _MAX_BYTES
    keep = _MAX_BYTES
    try:
        _MAX_BYTES = 5000
        clear()
        for i in range(40):
            build('K%d' % i, 40)
        good = stats()['bytes'] <= 5000 and len(_cache) >= 1
        ok &= good
        print("geom_cache: eviction holds the budget (%d entries, %d B) %s"
              % (stats()['entries'], stats()['bytes'],
                 'OK' if good else 'FAIL'))
    finally:
        _MAX_BYTES = keep
        clear()

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("geom_cache self-test failed")
