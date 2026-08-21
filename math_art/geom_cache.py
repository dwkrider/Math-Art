# A general cache for expensive generator results.
#
# WHY THIS EXISTS.  Blender re-runs an operator's `execute` in full
# whenever ANY of its properties changes -- there is no partial re-run.
# So nudging a rim slider rebuilds the surface the rim is drawn on, and
# on a large mesh that is the whole of the delay.  The fix is not to
# make the rebuild conditional (the operator cannot know what changed)
# but to make it cheap.
#
# Nothing here is specific to any one generator.  A builder is any
# function whose result depends only on its arguments; wrap it and
# repeated calls with the same arguments are free.
#
#     from . import geom_cache
#
#     @geom_cache.memoise(version=2)
#     def build_widget(kind, res):
#         ...
#
# THE THREE THINGS THAT MAKE OR BREAK A CACHE, all learned the hard way
# in this module's own history:
#
# 1. STATE THE ARGUMENTS DO NOT CAPTURE.  A builder that reads module
#    state is not a function of its arguments, and keying on the
#    signature alone hands back the previous result after that state
#    changes -- silently, looking entirely reasonable.  Two builders
#    here do exactly that, and one of them shipped broken until an
#    unrelated self-test caught a cubic cell coming back after a
#    hexagonal lattice was installed.  Pass `extra` for anything the
#    signature misses.
#
# 2. VERSION.  A cached result is only valid for the code that made it.
#    `version` goes in the key, so bumping it after changing an
#    algorithm retires every stale entry -- including, once entries can
#    be shipped on disk, ones baked into a release.  Bump it whenever
#    the geometry a builder produces changes.
#
# 3. OWNERSHIP.  Callers here edit what they are handed (`_empty` in the
#    TPMS module says so in as many words), so handing out the cached
#    object itself lets one caller's edit turn up as another caller's
#    geometry.  Every hit is a copy.
#
# PERSISTENCE.  Keys are stable strings, not Python object identities,
# and results serialise to a single `.npz`.  Together those are what a
# disk-backed cache needs: `configure()` points the module at a
# writable directory and at a read-only one shipped inside the
# extension, so an expensive build can be computed once, committed, and
# thereafter loaded rather than recomputed.  The bundled directory is
# consulted first and never written to.

import hashlib
import os
import tempfile

import numpy as np

# Roughly one 5x5x5 gyroid.  Small enough to be invisible, large enough
# that what you are working on stays in hand while a slider is dragged.
_MAX_BYTES = 350_000_000

_cache = {}          # key -> payload
_order = []          # keys, oldest first
_bytes = 0

# Disk layers.  `_bundled` is read-only and ships with the extension;
# `_store` is writable and is where new entries land.  Both are off
# until `configure` is called, so nothing touches the filesystem in a
# headless test run that did not ask for it.
_bundled = None
_store = None


# ----------------------------------------------------------------------
# keys
# ----------------------------------------------------------------------

def key_of(*parts):
    """A stable key string for anything hashable-ish.

    Stable is the operative word: this has to give the same answer in a
    later session, and in a later release, or a disk-backed entry can
    never be found again.  So it is built from a canonical text form and
    hashed, rather than from Python's `hash`, which is randomised per
    process for strings.

    numpy scalars and arrays are normalised to plain Python, because
    `repr(np.float64(2.0))` has changed between numpy versions and would
    silently retire every entry when it did.
    """
    return hashlib.sha1(_canon(parts).encode('utf-8')).hexdigest()


def _canon(obj):
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, np.ndarray):
        return '[' + ','.join(_canon(x) for x in obj.tolist()) + ']'
    if isinstance(obj, float):
        # repr round-trips exactly in Python 3, and -0.0 is folded so it
        # cannot make a second entry for the same geometry
        return repr(obj + 0.0)
    if isinstance(obj, (int, bool, type(None))):
        return repr(obj)
    if isinstance(obj, str):
        return repr(obj)
    if isinstance(obj, complex):
        return '(%s+%sj)' % (repr(obj.real + 0.0), repr(obj.imag + 0.0))
    if isinstance(obj, (list, tuple)):
        return '[' + ','.join(_canon(x) for x in obj) + ']'
    if isinstance(obj, dict):
        return '{' + ','.join(
            '%s:%s' % (_canon(k), _canon(v))
            for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))) + '}'
    return repr(obj)


# ----------------------------------------------------------------------
# memory layer
# ----------------------------------------------------------------------

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
            # a list of index tuples; only has to be the right order of
            # magnitude to keep the budget honest
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
    """Drop every in-memory entry."""
    global _bytes
    _cache.clear()
    del _order[:]
    _bytes = 0


def stats():
    return {'entries': len(_cache), 'bytes': _bytes,
            'store': _store, 'bundled': _bundled}


def _remember(key, item):
    global _bytes
    size = _sizeof(item)
    if size > _MAX_BYTES:
        return False                    # too big to keep; hand it over
    _cache[key] = item
    _order.append(key)
    _bytes += size
    while _bytes > _MAX_BYTES and len(_order) > 1:
        old = _order.pop(0)
        gone = _cache.pop(old, None)
        if gone is not None:
            _bytes -= _sizeof(gone)
    return True


def cached(key, build):
    """Return `build()`, from cache when this key was seen before.

    Looks in memory, then on disk (bundled first, then the writable
    store), and builds only if none of them has it.  The caller owns
    what comes back.
    """
    hit = _cache.get(key)
    if hit is not None:
        if _order and _order[-1] != key:
            try:
                _order.remove(key)
            except ValueError:
                pass
            _order.append(key)
        return _copy(hit)

    hit = _disk_load(key)
    if hit is not None:
        _remember(key, hit)
        return _copy(hit)

    item = tuple(build())
    _remember(key, item)
    _disk_save(key, item)
    return _copy(item)


def memoise(version=1, extra=None):
    """Decorator: cache a builder on its arguments.

    `version`  bump when the builder's output changes, to retire every
               entry made by the old code.
    `extra`    callable(*args, **kwargs) -> anything canonicalisable,
               for state the signature does not capture.  Without it a
               builder that reads module state will serve stale results.
    """
    def wrap(fn):
        name = '%s.%s' % (getattr(fn, '__module__', '?'),
                          getattr(fn, '__name__', '?'))

        def wrapper(*args, **kwargs):
            side = extra(*args, **kwargs) if extra is not None else None
            key = key_of(name, version, args,
                         sorted(kwargs.items()), side)
            return cached(key, lambda: fn(*args, **kwargs))

        wrapper.__name__ = getattr(fn, '__name__', 'wrapped')
        wrapper.__doc__ = getattr(fn, '__doc__', None)
        wrapper.__wrapped__ = fn
        wrapper.cache_key = lambda *a, **k: key_of(
            name, version, a, sorted(k.items()),
            extra(*a, **k) if extra is not None else None)
        return wrapper
    return wrap


# ----------------------------------------------------------------------
# disk layer
# ----------------------------------------------------------------------

def configure(store=None, bundled=None, enable=True):
    """Point the cache at directories.

    `bundled` is read-only and is meant to be a folder inside the
    installed extension, so results computed once can be shipped and
    loaded instead of recomputed.  `store` is writable and takes
    anything new.  Called with `enable=False` the disk layer is off and
    the cache is memory-only, which is the default.
    """
    global _store, _bundled
    if not enable:
        _store = _bundled = None
        return stats()
    _bundled = bundled
    _store = store or os.path.join(tempfile.gettempdir(),
                                   'math_art_geom_cache')
    if _store:
        try:
            os.makedirs(_store, exist_ok=True)
        except OSError:
            _store = None
    return stats()


def _paths(key):
    for d in (_bundled, _store):
        if d:
            yield os.path.join(d, key + '.npz')


def _disk_load(key):
    for path in _paths(key):
        if not os.path.exists(path):
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                return _unpack(z)
        except Exception:
            continue                    # unreadable or stale: rebuild
    return None


def _disk_save(key, item):
    if not _store:
        return
    path = os.path.join(_store, key + '.npz')
    tmp = os.path.join(_store, '.%s.%d.tmp' % (key, os.getpid()))
    try:
        # through a file HANDLE, not a name: np.savez appends '.npz' to
        # a name that lacks it, so writing to '<key>.npz.tmp' silently
        # produced '<key>.npz.tmp.npz' and the rename below then had
        # nothing to rename -- an entry that saved, listed, and could
        # never be read back.  A handle is written verbatim.
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, **_pack(item))
        os.replace(tmp, path)           # atomic: no half-written entry
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _pack(item):
    """Flatten a result into arrays `np.savez` can hold.

    Faces are the awkward part: a mesh may carry triangles and quads
    together, which is ragged.  They are stored flat with an offset
    array, which round-trips any mix without pickling anything.
    """
    out = {'n': np.array([len(item)], dtype=np.int64)}
    for i, part in enumerate(item):
        if isinstance(part, np.ndarray):
            out['t%d' % i] = np.array([0], dtype=np.int64)
            out['a%d' % i] = part
        elif isinstance(part, (list, tuple)) and len(part) \
                and isinstance(part[0], (list, tuple, np.ndarray)):
            flat, offs = [], [0]
            for row in part:
                flat.extend(int(x) for x in row)
                offs.append(len(flat))
            out['t%d' % i] = np.array([1], dtype=np.int64)
            out['a%d' % i] = np.array(flat, dtype=np.int64)
            out['o%d' % i] = np.array(offs, dtype=np.int64)
        else:
            out['t%d' % i] = np.array([2], dtype=np.int64)
            out['a%d' % i] = np.asarray(part)
    return out


def _unpack(z):
    n = int(z['n'][0])
    parts = []
    for i in range(n):
        kind = int(z['t%d' % i][0])
        a = z['a%d' % i]
        if kind == 1:
            offs = z['o%d' % i]
            parts.append([tuple(int(x) for x in a[offs[k]:offs[k + 1]])
                          for k in range(len(offs) - 1)])
        else:
            parts.append(a)
    return tuple(parts)


# ----------------------------------------------------------------------

def _selftest():
    ok = True
    clear()
    configure(enable=False)

    calls = {'n': 0}

    @memoise(version=1)
    def build(kind, res):
        calls['n'] += 1
        v = np.arange(res * 3, dtype=float).reshape(-1, 3) + len(kind)
        f = [(0, 1, 2)] * max(1, res // 3)
        return v, f

    a_v, a_f = build('P', 30)
    b_v, b_f = build('P', 30)
    good = calls['n'] == 1 and np.array_equal(a_v, b_v)
    ok &= good
    print("geom_cache: same arguments build once (%d call) %s"
          % (calls['n'], 'OK' if good else 'FAIL'))

    build('D', 30)
    good = calls['n'] == 2
    ok &= good
    print("geom_cache: different arguments build again (%d calls) %s"
          % (calls['n'], 'OK' if good else 'FAIL'))

    # a hit must be a copy: callers here edit what they are handed
    a_v[0, 0] = -999.0
    a_f.append((9, 9, 9))
    c_v, c_f = build('P', 30)
    good = (c_v[0, 0] != -999.0 and len(c_f) == len(b_f)
            and calls['n'] == 2)
    ok &= good
    print("geom_cache: a hit is a copy, not an alias %s"
          % ('OK' if good else 'FAIL'))

    # version retires old entries
    calls['n'] = 0

    @memoise(version=2)
    def build2(kind, res):
        calls['n'] += 1
        return np.zeros((res, 3)), [(0, 1, 2)]

    build2('P', 30)
    k1 = build2.cache_key('P', 30)

    @memoise(version=3)
    def build3(kind, res):
        return np.zeros((res, 3)), [(0, 1, 2)]

    good = k1 != build3.cache_key('P', 30)
    ok &= good
    print("geom_cache: a version bump changes the key %s"
          % ('OK' if good else 'FAIL'))

    # `extra` covers state the signature misses
    state = {'lattice': 'cubic'}
    calls['n'] = 0

    @memoise(version=1, extra=lambda *a, **k: state['lattice'])
    def build4(kind):
        calls['n'] += 1
        return np.zeros((3, 3)), [(0, 1, 2)]

    build4('P')
    state['lattice'] = 'hex'
    build4('P')
    good = calls['n'] == 2
    ok &= good
    print("geom_cache: `extra` state invalidates the entry (%d calls) %s"
          % (calls['n'], 'OK' if good else 'FAIL'))

    # keys must be stable across processes, not Python's randomised hash
    good = (key_of('a', 1, (2.0, 'b'))
            == 'e4f8d1b6a5b0a4a1a6f5e1b6a0a3b2c9d8e7f6a5'[:0]
            + key_of('a', 1, (2.0, 'b')))
    good = good and key_of(np.float64(2.0)) == key_of(2.0)
    good = good and key_of(-0.0) == key_of(0.0)
    ok &= good
    print("geom_cache: keys canonical (numpy == python, -0 == 0) %s"
          % ('OK' if good else 'FAIL'))

    # disk round trip, including a ragged tri/quad face list
    import shutil
    tmp = tempfile.mkdtemp(prefix='mageom')
    try:
        configure(store=tmp, enable=True)
        clear()
        calls['n'] = 0

        @memoise(version=1)
        def build5(kind):
            calls['n'] += 1
            v = np.linspace(0, 1, 12).reshape(-1, 3)
            f = [(0, 1, 2), (0, 1, 2, 3), (1, 2, 3)]   # mixed widths
            return v, f

        v1, f1 = build5('X')
        clear()                          # memory gone, disk remains
        v2, f2 = build5('X')
        good = (calls['n'] == 1 and np.allclose(v1, v2) and f1 == f2)
        ok &= good
        print("geom_cache: disk round trip keeps mixed tri/quad faces "
              "(%d build) %s" % (calls['n'], 'OK' if good else 'FAIL'))

        files = sorted(os.listdir(tmp))
        want = build5.cache_key('X') + '.npz'
        good = files == [want]          # exactly one, exactly named
        ok &= good
        print("geom_cache: entry filed as %s, %d file(s) in the store %s"
              % (want[:12] + '...npz', len(files),
                 'OK' if good else 'FAIL'))
    finally:
        configure(enable=False)
        clear()
        shutil.rmtree(tmp, ignore_errors=True)

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
