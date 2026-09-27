// The Hopf fibration: the geometry, in the browser.
//
// A port of the kernel and the base-point providers of
// math_art/hopf_fibration_generator.py, function for function, so the
// two can be compared directly -- tests/web/test_hopf.mjs runs the
// generator and checks this against it, point by point. Read the
// Python module's header for the mathematics; the notes here are about
// the port.
//
// THE MAP. Over a base point p on the sphere, written by its
// colatitude beta and longitude lambda, the fibre is the Clifford
// circle in S^3
//
//     z0 = cos(beta/2) e^{i(t + lambda)},  z1 = sin(beta/2) e^{i t}
//
// sampled in t and then pushed into ordinary space by stereographic
// projection from the north pole, (x1,x2,x3,x4) -> (x1,x2,x3)/(1-x4).
// Every fibre becomes a circle; any two are linked exactly once.
//
// WHAT IS LEFT OUT. The generator's RANDOM preset draws its base
// points from numpy's Generator, whose stream cannot be reproduced
// here, so the page does not offer it -- everything the page does
// offer is checked against the generator. Beads, markers, tubes and
// the Blender-side scene are not geometry and are not ported.

const TAU = 2 * Math.PI;
export const PHI = (1 + Math.sqrt(5)) / 2;

// The generator's fixed tilt of S^2: nothing lines up with the screen
// axes, so the picture never looks accidentally symmetric.
export const TILT = [0.3178, 0.2114, 0.1291];

// ------------------------------------------------------------ vectors

export function rotMatrix(ax, ay, az) {
  const cx = Math.cos(ax), sx = Math.sin(ax);
  const cy = Math.cos(ay), sy = Math.sin(ay);
  const cz = Math.cos(az), sz = Math.sin(az);
  const Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]];
  const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
  const Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]];
  return mul3(mul3(Rz, Ry), Rx);
}

function mul3(A, B) {
  const out = [];
  for (let i = 0; i < 3; i++) {
    const row = [];
    for (let j = 0; j < 3; j++) {
      row.push(A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]);
    }
    out.push(row);
  }
  return out;
}

export function apply3(M, v) {
  return [M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
          M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
          M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]];
}

export function normalize3(v) {
  const m = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / m, v[1] / m, v[2] / m];
}

// --------------------------------------- the fibre, and its projection

/** The fibre over a unit 3-vector on S^2, as `samples` points on S^3.
 *  (P, Q) winds the two phases at those rates, laying a torus curve on
 *  the same Clifford torus; 'LEFT' conjugates the second phase, which
 *  is the mirror fibration -- the other Villarceau ruling. */
export function fiberS3(base, samples, P = 1, Q = 1, chirality = 'RIGHT') {
  const [a, b, c] = base;
  const beta = Math.acos(Math.max(-1, Math.min(1, c)));
  const lam = Math.atan2(b, a);
  const ch = Math.cos(beta / 2), sh = Math.sin(beta / 2);
  const sgn = chirality === 'LEFT' ? -1 : 1;
  const out = new Float64Array(samples * 4);
  for (let i = 0; i < samples; i++) {
    const t = TAU * i / samples;          // linspace(0, 2pi, endpoint=False)
    const p0 = P * t + lam, p1 = Q * t;
    out[i * 4] = ch * Math.cos(p0);
    out[i * 4 + 1] = ch * Math.sin(p0);
    out[i * 4 + 2] = sh * Math.cos(p1);
    out[i * 4 + 3] = sgn * sh * Math.sin(p1);
  }
  return out;
}

/** Unit quaternion (cos phi, sin phi, 0, 0): one plane of a left
 *  Clifford rotation of S^3. */
export function quatLeft(phi) {
  return [Math.cos(phi), Math.sin(phi), 0, 0];
}

/** Left quaternion-multiply every S^3 point by q. A left Clifford
 *  rotation commutes with the Hopf action, so it descends to a rotation
 *  of the base sphere: the whole family flows through itself. */
export function s3Rotate(X, q) {
  const [w, i, j, k] = q;
  const out = new Float64Array(X.length);
  for (let n = 0; n < X.length; n += 4) {
    const a = X[n], b = X[n + 1], c = X[n + 2], d = X[n + 3];
    out[n] = w * a - i * b - j * c - k * d;
    out[n + 1] = w * b + i * a + j * d - k * c;
    out[n + 2] = w * c - i * d + j * a + k * b;
    out[n + 3] = w * d + i * c - j * b + k * a;
  }
  return out;
}

/** Stereographic projection from the north pole N = (0,0,0,1). */
export function stereographic(X) {
  const n = X.length / 4;
  const out = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    let denom = 1 - X[i * 4 + 3];
    if (Math.abs(denom) < 1e-9) denom = 1e-9;
    out[i * 3] = X[i * 4] / denom;
    out[i * 3 + 1] = X[i * 4 + 1] / denom;
    out[i * 3 + 2] = X[i * 4 + 2] / denom;
  }
  return out;
}

/** One fibre as a projected polyline. */
export function projectFiber(base, samples, P = 1, Q = 1) {
  return stereographic(fiberS3(base, samples, P, Q));
}

// ------------------------------------------- base points on the sphere

export function latitudes(nLat, nFiber, latMinDeg = 20, latMaxDeg = 160) {
  const pts = [];
  const betas = [];
  if (nLat === 1) {
    betas.push(0.5 * (latMinDeg + latMaxDeg) * Math.PI / 180);
  } else {
    for (let i = 0; i < nLat; i++) {
      // numpy linspace: both ends included
      const d = latMinDeg + (latMaxDeg - latMinDeg) * i / (nLat - 1);
      betas.push(d * Math.PI / 180);
    }
  }
  for (const beta of betas) {
    for (let k = 0; k < nFiber; k++) {
      const lam = TAU * k / nFiber;
      pts.push([Math.sin(beta) * Math.cos(lam),
                Math.sin(beta) * Math.sin(lam),
                Math.cos(beta)]);
    }
  }
  return pts;
}

export function flower(nFiber, ringDeg = 35) {
  return latitudes(1, nFiber, ringDeg, ringDeg);
}

export function platonic(kind) {
  let V;
  if (kind === 'TETRA') {
    V = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]];
  } else if (kind === 'OCTA') {
    V = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  } else if (kind === 'CUBE') {
    V = [];
    for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) V.push([x, y, z]);
  } else if (kind === 'ICOSA') {
    const p = PHI;
    V = [];
    for (const s1 of [-1, 1]) {
      for (const s2 of [-1, 1]) {
        V.push([0, s1, s2 * p], [s1, s2 * p, 0], [s2 * p, 0, s1]);
      }
    }
  } else if (kind === 'DODECA') {
    const p = PHI;
    V = [];
    for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) V.push([x, y, z]);
    for (const s1 of [-1, 1]) {
      for (const s2 of [-1, 1]) {
        V.push([0, s1 / p, s2 * p], [s1 / p, s2 * p, 0], [s2 * p, 0, s1 / p]);
      }
    }
  } else {
    throw new Error(kind);
  }
  return V.map(normalize3);
}

export function fibonacci(n) {
  const pts = [];
  const ga = Math.PI * (3 - Math.sqrt(5));
  for (let k = 0; k < n; k++) {
    const z = 1 - (2 * k + 1) / n;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const phi = ga * k;
    pts.push([r * Math.cos(phi), r * Math.sin(phi), z]);
  }
  return pts;
}

export function greatCircle(nFiber, tiltDeg = 55) {
  const pts = [];
  const a = tiltDeg * Math.PI / 180;
  for (let k = 0; k < nFiber; k++) {
    const u = TAU * k / nFiber;
    pts.push([Math.cos(u), Math.sin(u) * Math.cos(a), Math.sin(u) * Math.sin(a)]);
  }
  return pts;
}

export function sphericalCap(n, colatMinDeg, colatMaxDeg) {
  const zmax = Math.cos(colatMinDeg * Math.PI / 180);
  const zmin = Math.cos(colatMaxDeg * Math.PI / 180);
  const ga = Math.PI * (3 - Math.sqrt(5));
  const pts = [];
  const m = Math.max(1, n);
  for (let k = 0; k < m; k++) {
    const z = zmin + (zmax - zmin) * (k + 0.5) / m;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const ph = ga * k;
    pts.push([r * Math.cos(ph), r * Math.sin(ph), z]);
  }
  return pts;
}

export function loxodrome(n, colatMinDeg, colatMaxDeg, turns) {
  const pts = [];
  const m = Math.max(1, n);
  for (let k = 0; k < m; k++) {
    const f = (k + 0.5) / m;
    const beta = (colatMinDeg + (colatMaxDeg - colatMinDeg) * f) * Math.PI / 180;
    const lam = turns * TAU * f;
    pts.push([Math.sin(beta) * Math.cos(lam),
              Math.sin(beta) * Math.sin(lam), Math.cos(beta)]);
  }
  return pts;
}

export function curl(n, colatDeg, lobes, ampDeg) {
  const pts = [];
  const m = Math.max(1, n);
  for (let k = 0; k < m; k++) {
    const s = TAU * k / m;
    const beta = Math.min(Math.PI - 0.06, Math.max(0.06,
      colatDeg * Math.PI / 180 + ampDeg * Math.PI / 180 * Math.cos(lobes * s)));
    pts.push([Math.sin(beta) * Math.cos(s), Math.sin(beta) * Math.sin(s),
              Math.cos(beta)]);
  }
  return pts;
}

export const PROVIDERS = {
  LATITUDES: 'Nested tori (circles of latitude)',
  FLOWER: 'Flower (one small circle of base points)',
  GREATCIRCLE: 'Great-circle band',
  CAP: 'Spherical cap / band spiral',
  LOXODROME: 'Loxodrome (rhumb-line spiral)',
  CURL: 'Curl (floral rose ring)',
  FIBONACCI: 'Fibonacci sphere (quasi-uniform)',
  TETRA: 'Tetrahedron vertices',
  OCTA: 'Octahedron vertices',
  CUBE: 'Cube vertices',
  ICOSA: 'Icosahedron vertices',
  DODECA: 'Dodecahedron vertices',
};

export const RING_PRESETS = ['LATITUDES', 'FLOWER', 'GREATCIRCLE', 'CAP',
                             'LOXODROME', 'CURL'];

/** Base points for a preset, before the tilt. `extra` carries the few
 *  preset-specific knobs (turns, curl lobes and amplitude). */
export function basePoints(preset, nLat, nFiber, latMin, latMax, extra = {}) {
  switch (preset) {
    case 'LATITUDES': return latitudes(nLat, nFiber, latMin, latMax);
    case 'FLOWER': return flower(nFiber, 0.5 * (latMin + latMax));
    case 'GREATCIRCLE': return greatCircle(nFiber, 0.5 * (latMin + latMax));
    case 'CAP': return sphericalCap(nFiber, latMin, latMax);
    case 'LOXODROME': return loxodrome(nFiber, latMin, latMax, extra.turns ?? 5.0);
    case 'CURL': return curl(nFiber, 0.5 * (latMin + latMax),
                             extra.curl_lobes ?? 5, extra.curl_amp ?? 25.0);
    case 'FIBONACCI': return fibonacci(Math.max(1, nFiber));
    default: return platonic(preset);
  }
}

// ------------------------------------------------------------ assembly

/** Indices of the longest contiguous true run in a cyclic mask. */
export function longestRun(mask) {
  const n = mask.length;
  if (mask.every(Boolean)) return Array.from({ length: n }, (_, i) => i);
  let start = mask.findIndex((m) => !m);
  if (start < 0) start = 0;
  const order = [];
  for (let i = 0; i < n; i++) order.push((i + start) % n);
  let best = [0, 0];
  let i = 0;
  while (i < n) {
    if (mask[order[i]]) {
      let j = i;
      while (j < n && mask[order[j]]) j++;
      if (j - i > best[1] - best[0]) best = [i, j];
      i = j;
    } else {
      i++;
    }
  }
  return order.slice(best[0], best[1]);
}

/** numpy.percentile with the default linear interpolation. */
export function percentile(sorted, q) {
  const n = sorted.length;
  if (n === 0) return 0;
  const pos = (n - 1) * q / 100;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Every fibre of a preset, projected, centred and scaled so the
 *  95th-percentile point radius is `fitRadius`.
 *
 *  Returns {fibers, bases, closed, dropped}: `fibers[i]` is a flat
 *  Float64Array of xyz triples, `bases[i]` the tilted base point that
 *  colours it, `closed[i]` whether it is a whole loop (a near-axis
 *  fibre is kept as its in-range arc when `includeAxis`). */
export function buildFibers({
  preset = 'LATITUDES', nLat = 6, nFiber = 24, samples = 160,
  P = 1, Q = 1, latMin = 20, latMax = 160, fitRadius = 1,
  maxRadius = 12, sphereEuler = null, s3Rot = 0, chirality = 'RIGHT',
  includeAxis = false, extra = {}, points = null,
} = {}) {
  // `points` is the page's own entry, for base points the reader has
  // clicked: they are already where they will be drawn, so they skip
  // the provider and the tilt. Everything after this line -- the
  // lift, the rotation, the projection, the drop rule and the fit --
  // is the generator's, which is what keeps a hand-picked picture and
  // a preset picture the same kind of object.
  const R = rotMatrix(...(sphereEuler || TILT));
  const based = points
    ? points.map((b) => [b[0], b[1], b[2]])
    : basePoints(preset, nLat, nFiber, latMin, latMax, extra).map((b) => apply3(R, b));
  const q = s3Rot ? quatLeft(s3Rot * Math.PI / 180) : null;
  const chis = chirality === 'BOTH' ? ['RIGHT', 'LEFT'] : [chirality];

  const fibers = [], bases = [], closed = [];
  let dropped = 0;
  for (const b of based) {
    for (const chi of chis) {
      let X = fiberS3(b, samples, P, Q, chi);
      if (q) X = s3Rotate(X, q);
      const p = stereographic(X);
      let finite = true;
      for (let i = 0; i < p.length && finite; i++) finite = Number.isFinite(p[i]);
      if (!finite) { dropped++; continue; }
      const n = p.length / 3;
      const r = new Float64Array(n);
      let rmax = 0;
      for (let i = 0; i < n; i++) {
        r[i] = Math.hypot(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
        if (r[i] > rmax) rmax = r[i];
      }
      if (rmax < maxRadius) {
        fibers.push(p); bases.push(b); closed.push(true);
      } else if (includeAxis) {
        const run = longestRun(Array.from(r, (x) => x < maxRadius));
        if (run.length >= 2) {
          const arc = new Float64Array(run.length * 3);
          run.forEach((idx, i) => {
            arc[i * 3] = p[idx * 3];
            arc[i * 3 + 1] = p[idx * 3 + 1];
            arc[i * 3 + 2] = p[idx * 3 + 2];
          });
          fibers.push(arc); bases.push(b); closed.push(false);
        } else {
          dropped++;
        }
      } else {
        dropped++;
      }
    }
  }
  if (!fibers.length) return { fibers: [], bases: [], closed: [], dropped };

  // centre on the bounding box, scale by the 95th percentile radius --
  // the generator's fit, so the site and the add-on frame alike
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  let total = 0;
  for (const f of fibers) {
    total += f.length / 3;
    for (let i = 0; i < f.length; i += 3) {
      for (let a = 0; a < 3; a++) {
        if (f[i + a] < lo[a]) lo[a] = f[i + a];
        if (f[i + a] > hi[a]) hi[a] = f[i + a];
      }
    }
  }
  const centre = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
  const rad = new Float64Array(total);
  let w = 0;
  for (const f of fibers) {
    for (let i = 0; i < f.length; i += 3) {
      rad[w++] = Math.hypot(f[i] - centre[0], f[i + 1] - centre[1], f[i + 2] - centre[2]);
    }
  }
  const sorted = Array.from(rad).sort((a, b) => a - b);
  const ref = percentile(sorted, 95);
  const scale = ref > 1e-9 ? fitRadius / ref : 1;
  for (const f of fibers) {
    for (let i = 0; i < f.length; i += 3) {
      f[i] = (f[i] - centre[0]) * scale;
      f[i + 1] = (f[i + 1] - centre[1]) * scale;
      f[i + 2] = (f[i + 2] - centre[2]) * scale;
    }
  }
  return { fibers, bases, closed, dropped };
}

// ------------------------------------------------------------- colour

function hsvToRgb(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (((i % 6) + 6) % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

function rgbToHsv(r, g, b) {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  const d = mx - mn;
  let h = 0;
  if (d !== 0) {
    if (mx === r) h = ((g - b) / d) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h /= 6;
    if (h < 0) h += 1;
  }
  return [h, mx === 0 ? 0 : d / mx, mx];
}

export const MONO_DEFAULT = [0.27, 0.86, 0.80];

/** A base point's colour: hue from longitude, lightness from latitude
 *  -- the convention the generator and Niles Johnson's film share, so
 *  a dot on the base sphere and its fibre are the same colour. */
export function paletteRgb(base, style = 'RAINBOW', mono = MONO_DEFAULT) {
  const [a, b, c] = base;
  const hue = (((Math.atan2(b, a) / TAU) % 1) + 1) % 1;
  const lat = 0.5 * (c + 1);
  if (style === 'PASTEL') return hsvToRgb(hue, 0.40, 0.72 + 0.24 * lat);
  if (style === 'MONO') {
    const [h, s] = rgbToHsv(...mono);
    return hsvToRgb(h, s, 0.42 + 0.5 * lat);
  }
  return hsvToRgb(hue, 0.85, 0.35 + 0.55 * lat);
}

/** Colour by position along the fibre, which shows its direction. */
export function paramRgb(t01, style = 'RAINBOW', mono = MONO_DEFAULT) {
  if (style === 'MONO') {
    const [h, s] = rgbToHsv(...mono);
    return hsvToRgb(h, s, 0.40 + 0.5 * t01);
  }
  const sat = style === 'PASTEL' ? 0.42 : 0.85;
  return hsvToRgb(((t01 % 1) + 1) % 1, sat, 0.9);
}

// ------------------------------------------------- what the page adds

/** The Gauss linking number of two closed polylines, by the double
 *  sum over segment pairs. Any two distinct Hopf fibres link exactly
 *  once, and the page computes it rather than asserting it. */
export function linkingNumber(A, B) {
  const na = A.length / 3, nb = B.length / 3;
  let sum = 0;
  const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0], d = [0, 0, 0];
  for (let i = 0; i < na; i++) {
    const i1 = (i + 1) % na;
    a[0] = A[i * 3]; a[1] = A[i * 3 + 1]; a[2] = A[i * 3 + 2];
    b[0] = A[i1 * 3]; b[1] = A[i1 * 3 + 1]; b[2] = A[i1 * 3 + 2];
    for (let j = 0; j < nb; j++) {
      const j1 = (j + 1) % nb;
      c[0] = B[j * 3]; c[1] = B[j * 3 + 1]; c[2] = B[j * 3 + 2];
      d[0] = B[j1 * 3]; d[1] = B[j1 * 3 + 1]; d[2] = B[j1 * 3 + 2];
      sum += solidAngleTerm(a, b, c, d);
    }
  }
  return sum / (4 * Math.PI);
}

// The signed solid angle a pair of segments subtends, after the
// standard closed form for two straight segments (Klenin & Langowski).
function solidAngleTerm(p1, p2, p3, p4) {
  const r13 = sub(p3, p1), r14 = sub(p4, p1), r23 = sub(p3, p2), r24 = sub(p4, p2);
  const n1 = norm(cross(r13, r14)), n2 = norm(cross(r14, r24));
  const n3 = norm(cross(r24, r23)), n4 = norm(cross(r23, r13));
  if (!n1 || !n2 || !n3 || !n4) return 0;
  const a = unit(cross(r13, r14)), b = unit(cross(r14, r24));
  const c = unit(cross(r24, r23)), d = unit(cross(r23, r13));
  const ang = Math.asin(clamp(dot(a, b))) + Math.asin(clamp(dot(b, c)))
            + Math.asin(clamp(dot(c, d))) + Math.asin(clamp(dot(d, a)));
  const sign = Math.sign(dot(cross(sub(p4, p3), sub(p2, p1)), r13));
  return ang * sign;
}

const sub = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
const cross = (u, v) => [u[1] * v[2] - u[2] * v[1],
                         u[2] * v[0] - u[0] * v[2],
                         u[0] * v[1] - u[1] * v[0]];
const norm = (u) => Math.hypot(u[0], u[1], u[2]);
const unit = (u) => { const m = norm(u) || 1; return [u[0] / m, u[1] / m, u[2] / m]; };
const clamp = (x) => Math.max(-1, Math.min(1, x));

/** The base point under a direction the reader has clicked: just the
 *  unit vector, but named, because that is the whole interaction. */
export function basePointFromDirection(v) {
  return normalize3(v);
}
