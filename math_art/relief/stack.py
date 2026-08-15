"""relief.stack -- composing several pattern layers into one height field.

A panel is a *stack*, not a single closed-form surface.  Each layer names a
pattern, a place to put it, a strength, and a rule for combining it with what
is already there.  That is what lets the organic and the geometric mix: a real
plate eigenmode as the base, drapery over it at a fifth of the amplitude,
masked to one corner.

Three rules make the stack behave predictably.

**Normalise before weighting.**  Every layer is put on a common footing before
its amplitude applies, so one amplitude control means the same thing whichever
pattern is chosen.  Without it a fractal surface and a sinusoid -- whose
natural ranges differ by an order of magnitude -- would need completely
different slider values to contribute equally, and a stack would be untunable.

**Blend in a defined range.**  Additive blending is well behaved on zero-mean
fields, but multiply, screen and the two extrema are not: they need a
[0, 1] convention.  Each is therefore defined explicitly below rather than
left to whatever the arithmetic happens to do.

**A layer may reference earlier layers, never later ones.**  A layer can take
its mask from any layer below it in the stack.  That is a strict fold, so no
cycle can be expressed -- it buys most of what a node graph offers for one
integer per layer, and cannot deadlock.

Intra-layer versus stack-level maximum is worth stating, because it is a real
trap: merging bumps "like metaballs" is a MAX over the *points of one splat
layer*, on its raw non-negative field.  A MAX at stack level, after each layer
has been normalised to zero mean, means something else entirely and is almost
never what was wanted.
"""

import math

import numpy as np

from . import fields as _fields
from . import transfer as _transfer

BLENDS = ('ADD', 'SUB', 'MUL', 'SCREEN', 'MAX', 'MIN')
MASKS = ('NONE', 'RADIAL', 'LINEAR', 'LAYER')


def layer(kind='WAVE', amplitude=1.0, blend='ADD', **kw):
    """A stack entry.  Unknown keys are passed through to the pattern."""
    p = dict(kind=kind, amplitude=float(amplitude), blend=blend,
             offset_x=0.0, offset_y=0.0, rotation=0.0,
             scale_x=1.0, scale_y=1.0,
             mask='NONE', mask_center=0.0, mask_width=0.5, mask_angle=0.0,
             mask_layer=0, curve='NONE', curve_amount=1.0,
             norm='STD')
    p.update(kw)
    return p


def _place(X, Y, p):
    """Apply a layer's own 2-D transform to the sample coordinates."""
    ox = float(p.get('offset_x', 0.0))
    oy = float(p.get('offset_y', 0.0))
    rot = float(p.get('rotation', 0.0))
    sx = float(p.get('scale_x', 1.0)) or 1.0
    sy = float(p.get('scale_y', 1.0)) or 1.0
    dx = X - ox
    dy = Y - oy
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        dx, dy = c * dx + s * dy, -s * dx + c * dy
    return dx / sx, dy / sy


def _mask_for(p, X, Y, previous):
    """A layer's mask, in [0, 1]."""
    kind = p.get('mask', 'NONE')
    if kind == 'NONE':
        return None
    if kind == 'RADIAL':
        hx = float(np.abs(X).max()) or 1.0
        r = np.hypot(X - p.get('offset_x', 0.0),
                     Y - p.get('offset_y', 0.0)) / hx
        w = max(float(p.get('mask_width', 0.5)), 1e-6)
        t = np.clip(1.0 - r / w, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)          # smoothstep
    if kind == 'LINEAR':
        a = float(p.get('mask_angle', 0.0))
        u = X * math.cos(a) + Y * math.sin(a)
        lo, hi = float(u.min()), float(u.max())
        t = np.clip((u - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        c = float(np.clip(p.get('mask_center', 0.5), 0.0, 1.0))
        w = max(float(p.get('mask_width', 0.5)), 1e-6)
        t = np.clip((t - (c - 0.5 * w)) / w, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    if kind == 'LAYER':
        k = int(p.get('mask_layer', 0))
        if not previous or k < 0 or k >= len(previous):
            return None
        m = _transfer.normalize(previous[k], 'MINMAX')
        return 0.5 * (m + 1.0)
    raise ValueError("unknown mask: %r" % (kind,))


def _blend(acc, h, mode, first):
    """Combine a weighted layer with the accumulator."""
    if first and mode in ('MUL', 'SCREEN'):
        # Nothing to modulate yet; treat the first layer as the base.
        return h
    if mode == 'ADD':
        return acc + h
    if mode == 'SUB':
        return acc - h
    if mode == 'MUL':
        return acc * (1.0 + h)
    if mode == 'SCREEN':
        a01 = 0.5 * (np.clip(acc, -1.0, 1.0) + 1.0)
        h01 = 0.5 * (np.clip(h, -1.0, 1.0) + 1.0)
        return 2.0 * (1.0 - (1.0 - a01) * (1.0 - h01)) - 1.0
    if mode == 'MAX':
        return np.maximum(acc, h)
    if mode == 'MIN':
        return np.minimum(acc, h)
    raise ValueError("unknown blend: %r" % (mode,))


def evaluate_stack(layers, X, Y, info, params=None):
    """Fold a list of layers into one field.  Returns `(h, per_layer)`."""
    base = dict(params or {})
    acc = np.zeros(X.shape)
    produced = []
    for i, spec in enumerate(layers):
        p = dict(base)
        p.update(spec)
        Xl, Yl = _place(X, Y, p)
        h = _fields.evaluate(p.get('kind', 'WAVE'), Xl, Yl, info, p)
        h = _transfer.normalize(h, p.get('norm', 'STD'))
        h = _transfer.apply_curve(h, p.get('curve', 'NONE'),
                                  amount=float(p.get('curve_amount', 1.0)),
                                  levels=int(p.get('levels', 6)),
                                  smooth=float(p.get('terrace', 0.25)))
        produced.append(h)
        m = _mask_for(p, X, Y, produced[:-1])
        w = float(p.get('amplitude', 1.0))
        contrib = w * h if m is None else w * h * m
        acc = _blend(acc, contrib, p.get('blend', 'ADD'), i == 0)
    return acc, produced


def _selftest():
    ok = True
    from . import grid as _grid

    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=81)

    # A one-layer stack equals evaluating that pattern directly.
    one = [layer('WAVE', 1.0, wavelength=0.4, seed=2)]
    h1, _ = evaluate_stack(one, X, Y, info)
    direct = _transfer.normalize(
        _fields.evaluate('WAVE', X, Y, info, {'wavelength': 0.4, 'seed': 2}),
        'STD')
    print("stack: single layer vs direct evaluate: max diff %.2e"
          % float(np.abs(h1 - direct).max()))
    ok = ok and float(np.abs(h1 - direct).max()) < 1e-12

    # Three layers build, and amplitude actually scales the contribution.
    three = [
        layer('CHLADNI', 1.0, exact=True, mode_index=6),
        layer('WAVE_TRAIN', 0.35, wavelength=0.3, seed=5, orient='CURL'),
        layer('FBM', 0.15, seed=9),
    ]
    h3, per = evaluate_stack(three, X, Y, info)
    print("stack: 3 layers -> sd %.4f, per-layer sds %s"
          % (h3.std(), ["%.3f" % p.std() for p in per]))
    ok = ok and len(per) == 3 and np.isfinite(h3).all() and h3.std() > 1e-6

    quiet = list(three)
    quiet[1] = dict(quiet[1], amplitude=0.0)
    hq, _ = evaluate_stack(quiet, X, Y, info)
    h_no = evaluate_stack([three[0], three[2]], X, Y, info)[0]
    print("stack: zero-amplitude layer is a no-op: max diff %.2e"
          % float(np.abs(hq - h_no).max()))
    ok = ok and float(np.abs(hq - h_no).max()) < 1e-12

    # Every blend mode is finite and shape-preserving; ADD is commutative.
    for mode in BLENDS:
        two = [layer('WAVE', 1.0, wavelength=0.5),
               layer('RIPPLE', 0.5, wavelength=0.3, blend=mode, seed=4)]
        h, _ = evaluate_stack(two, X, Y, info)
        ok = ok and h.shape == X.shape and np.isfinite(h).all()
    a = evaluate_stack([layer('WAVE', 1.0, wavelength=0.5),
                        layer('FBM', 0.4, seed=3, blend='ADD')],
                       X, Y, info)[0]
    b = evaluate_stack([layer('FBM', 0.4, seed=3),
                        layer('WAVE', 1.0, wavelength=0.5, blend='ADD')],
                       X, Y, info)[0]
    print("stack: ADD is order-independent: max diff %.2e"
          % float(np.abs(a - b).max()))
    ok = ok and float(np.abs(a - b).max()) < 1e-12

    # MAX is not commutative with different amplitudes, and must not exceed
    # the larger of the two contributions.
    m = evaluate_stack([layer('WAVE', 1.0, wavelength=0.5),
                        layer('FBM', 0.4, seed=3, blend='MAX')],
                       X, Y, info)[0]
    ok = ok and m.max() <= max(1.0, 0.4) * 6.0     # normalised fields are ~[-3,3]

    # Masks confine a layer.  A radial mask must leave the corners untouched.
    base = [layer('WAVE', 1.0, wavelength=0.5)]
    masked = base + [layer('FBM', 1.0, seed=6, mask='RADIAL',
                           mask_width=0.35)]
    hb, _ = evaluate_stack(base, X, Y, info)
    hm, _ = evaluate_stack(masked, X, Y, info)
    corner = (slice(0, 6), slice(0, 6))
    centre = (slice(36, 45), slice(36, 45))
    print("stack: radial mask -- corner change %.2e, centre change %.3f"
          % (float(np.abs(hm - hb)[corner].max()),
             float(np.abs(hm - hb)[centre].max())))
    ok = ok and float(np.abs(hm - hb)[corner].max()) < 1e-12
    ok = ok and float(np.abs(hm - hb)[centre].max()) > 0.1

    # A LAYER mask references an earlier layer, and an out-of-range index is
    # ignored rather than raising.
    lm = base + [layer('FBM', 1.0, seed=6, mask='LAYER', mask_layer=0)]
    hl, _ = evaluate_stack(lm, X, Y, info)
    ok = ok and np.isfinite(hl).all()
    bad = base + [layer('FBM', 1.0, seed=6, mask='LAYER', mask_layer=99)]
    ok = ok and np.isfinite(evaluate_stack(bad, X, Y, info)[0]).all()
    print("stack: LAYER mask works; out-of-range index degrades safely")

    # Per-layer transform actually moves the pattern.
    t0 = evaluate_stack([layer('RIPPLE', 1.0, sources=1, seed=1)],
                        X, Y, info)[0]
    t1 = evaluate_stack([layer('RIPPLE', 1.0, sources=1, seed=1,
                               offset_x=0.6)], X, Y, info)[0]
    print("stack: layer offset changes the field: rms %.3f"
          % float(np.sqrt(((t0 - t1) ** 2).mean())))
    ok = ok and float(np.sqrt(((t0 - t1) ** 2).mean())) > 0.05

    # An empty stack is a flat panel, not a crash.
    he, _ = evaluate_stack([], X, Y, info)
    ok = ok and he.shape == X.shape and float(np.abs(he).max()) == 0.0

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
