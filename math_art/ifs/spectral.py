# Fractal relief by spectral synthesis: fBm and Weierstrass sums.
#
# Part of the Math Art IFS engine (`math_art/ifs/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# A rough surface is built by summing waves rather than by subdividing:
# choose directions on the sphere, give mode k an amplitude proportional
# to lacunarity^(-k H), and add them up.  H is the Hurst exponent, and
# the resulting surface has fractal dimension D = 3 - H, so the UI can
# expose whichever of the two the user prefers to think in.  Summing over
# directions on a sphere lets the same field be evaluated on a plate, a
# sphere or a torus without seams.
#
# References:
# - B. B. Mandelbrot and J. W. van Ness, "Fractional Brownian motions,
#   fractional noises and applications", SIAM Review 10, 1968,
#   pp. 422-437.
# - R. F. Voss, "Fractals in nature: from characterization to
#   simulation", in H.-O. Peitgen and D. Saupe (eds), The Science of
#   Fractal Images, Springer, 1988 -- spectral synthesis.
# - K. Weierstrass, 1872; see G. H. Hardy, "Weierstrass's
#   non-differentiable function", Transactions of the AMS 17, 1916.

import math

import numpy as np


def _fib_dirs(count):
    """`count` near-uniform directions on the unit sphere (deterministic
    Fibonacci spiral), so the Weierstrass field is close to isotropic."""
    ga = math.pi * (3.0 - math.sqrt(5.0))
    out = np.empty((count, 3))
    for i in range(count):
        z = 1.0 - 2.0 * (i + 0.5) / count
        r = math.sqrt(max(0.0, 1.0 - z * z))
        t = ga * i
        out[i] = (r * math.cos(t), r * math.sin(t), z)
    return out


def weierstrass_modes(octaves, lacunarity, dim, dirs_per_oct=4,
                      k0=math.pi):
    """Weierstrass-Mandelbrot modes: geometric frequencies with
    amplitude lacunarity^((D-3) n), several ridge directions per
    octave. Deterministic."""
    dirs = _fib_dirs(octaves * dirs_per_oct)
    K, A, P = [], [], []
    idx = 0
    for n in range(octaves):
        freq = k0 * lacunarity ** n
        amp = lacunarity ** ((dim - 3.0) * n)
        for _ in range(dirs_per_oct):
            K.append(dirs[idx] * freq)
            A.append(amp)
            P.append((idx * 0.6180339887498949) % 1.0 * 2.0 * math.pi)
            idx += 1
    return np.array(K), np.array(A), np.array(P)


def fbm_modes(count, octaves, lacunarity, hurst, seed, k0=math.pi):
    """Fractional Brownian modes by spectral synthesis: random
    directions, log-uniform frequencies over the octave band,
    amplitude |k|^-(H+1). Seeded."""
    rng = np.random.default_rng(seed)
    kmax = k0 * lacunarity ** octaves
    d = rng.normal(size=(count, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    f = k0 * (kmax / k0) ** rng.random(count)
    K = d * f[:, None]
    A = f ** (-(hurst + 1.0))
    P = rng.random(count) * 2.0 * math.pi
    return K, A, P


def eval_field(points, modes):
    """Evaluate sum_i A_i cos(K_i . P + phase_i) at each point, then
    normalise to zero mean and unit standard deviation."""
    K, A, P = modes
    ang = points @ K.T + P[None, :]
    f = (np.cos(ang) * A[None, :]).sum(axis=1)
    f -= f.mean()
    sd = f.std()
    return f / sd if sd > 1e-9 else f


def _grid_faces(nu, nv, wrap_u=False, wrap_v=False):
    faces = []
    for i in range(nu - (0 if wrap_u else 1)):
        for j in range(nv - (0 if wrap_v else 1)):
            i1 = (i + 1) % nu
            j1 = (j + 1) % nv
            faces.append([i * nv + j, i1 * nv + j,
                          i1 * nv + j1, i * nv + j1])
    return faces


def base_plate(res):
    xs = np.linspace(-1.0, 1.0, res)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=-1)
    D = np.tile((0.0, 0.0, 1.0), (P.shape[0], 1))
    return P, D, _grid_faces(res, res)


def base_sphere(res):
    nu, nv = res, res
    u = np.linspace(0.0, math.pi, nu)               # polar
    v = np.linspace(0.0, 2.0 * math.pi, nv, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing='ij')
    P = np.stack([(np.sin(U) * np.cos(Vv)).ravel(),
                  (np.sin(U) * np.sin(Vv)).ravel(),
                  np.cos(U).ravel()], axis=-1)
    return P, P.copy(), _grid_faces(nu, nv, wrap_v=True)


def base_torus(res, ring=1.0, tube=0.4):
    nu, nv = res, res
    u = np.linspace(0.0, 2.0 * math.pi, nu, endpoint=False)
    v = np.linspace(0.0, 2.0 * math.pi, nv, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing='ij')
    cu, su, cv, sv = np.cos(U), np.sin(U), np.cos(Vv), np.sin(Vv)
    P = np.stack([((ring + tube * cv) * cu).ravel(),
                  ((ring + tube * cv) * su).ravel(),
                  (tube * sv).ravel()], axis=-1)
    D = np.stack([(cv * cu).ravel(), (cv * su).ravel(), sv.ravel()],
                 axis=-1)
    return P, D, _grid_faces(nu, nv, wrap_u=True, wrap_v=True)


BASES = {'PLATE': base_plate, 'SPHERE': base_sphere,
         'TORUS': base_torus}


def _weld(V, faces, eps=1e-5):
    """Merge coincident vertices (e.g. the sphere's collapsed pole
    rows) and drop the degenerate faces that result, so the closed
    bases stay watertight."""
    keys = np.round(V / eps).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    newV = np.zeros((len(uniq), 3))
    newV[inv] = V
    out = []
    for f in faces:
        nf = [int(inv[i]) for i in f]
        dd = [nf[k] for k in range(len(nf))
              if nf[k] != nf[(k + 1) % len(nf)]]
        if len(dd) >= 3:
            out.append(dd)
    return newV, out


def build_fractal_surface(base='PLATE', method='WEIERSTRASS', res=128,
                          octaves=9, lacunarity=2.0, dim=2.3,
                          hurst=0.7, amplitude=0.3, count=240,
                          seed=1, scale=1.0):
    """Displace a base surface by a Weierstrass-Mandelbrot or fBm
    field. Returns (verts, faces) centred and fit to a 2 m cube."""
    P, D, faces = BASES[base](res)
    if method == 'WEIERSTRASS':
        modes = weierstrass_modes(octaves, lacunarity, dim)
    else:
        modes = fbm_modes(count, octaves, lacunarity, hurst, seed)
    h = eval_field(P, modes)
    V = P + (amplitude * h)[:, None] * D
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    V = (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)
    if base != 'PLATE':                 # close sphere poles / weld
        V, faces = _weld(V, faces)
    return V * scale, faces
