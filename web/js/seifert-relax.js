// Relaxing and fairing a Seifert surface.
//
// A port of math_art/seifert/relax.py and the damped-Jacobi path of
// math_art/seifert/fair.py. Computation only: no DOM, no three.js, so
// tests/web/test_seifert.mjs can run it in node against the engine.
//
// WHY THERE ARE TWO STAGES, AND WHY THEY ARE DIFFERENT.
//
// The construction hands back a surface made of flat disks and swept
// ribbons. It has the right topology and looks like what it is: an
// assembly. Two separate things turn it into something a soap film
// might have made, and they pull in different ways.
//
// The RELAXATION is a physical model, borrowed from Scharein's KnotPlot
// by way of van Wijk and Cohen's section V.C: neighbouring vertices
// attract, distant ones repel, integrated explicitly with damping and a
// time step that decays so a run settles instead of ringing. The
// division of labour is what makes it produce a taut surface rather
// than a collapsed one. The LINK -- the boundary -- gets both forces,
// so it spreads out and stays embedded. The INTERIOR gets attraction
// only, so with its rim held out by the link it pulls itself tight.
// In the paper's words, the surface "follows the link, but does not
// influence it".
//
// The FAIRING is not physics but geometry: a backward Euler step of
// mean-curvature flow with the rim pinned, which drives the surface
// towards zero mean curvature -- a discrete minimal surface spanning
// the knot. It is the stage that actually makes it look like a film.
//
// The fitted mix is a LIGHT relaxation followed by heavy fairing. Long
// relaxations move away from the reference, because the particle
// model's equilibrium is not the same object as a minimal surface.
//
// References:
// - J. J. van Wijk & A. M. Cohen, "Visualization of Seifert Surfaces",
//   IEEE TVCG 12(4), 2006, section V.C -- the relaxation.
// - R. Scharein, KnotPlot -- the force model it borrows.
// - M. Desbrun, M. Meyer, P. Schroeder, A. H. Barr, "Implicit Fairing of
//   Irregular Meshes using Diffusion and Curvature Flow", SIGGRAPH 1999.
// - U. Pinkall & K. Polthier, "Computing Discrete Minimal Surfaces and
//   Their Conjugates", Experiment. Math. 2(1), 1993 -- cotangent weights.

import { Mesh, catmullClark } from './seifert-math.js';

// --------------------------------------------------------- relaxation

/** Force-model and integrator settings.
 *
 *  The paper reports alpha = 0, beta = 1 and attraction = repulsion,
 *  run for many thousands of iterations. The rest are unpublished and
 *  chosen to be stable for the meshes this builds. */
export const RELAX_DEFAULTS = {
  attraction: 1.0,
  repulsion: 0.4,
  beta: 1.0,
  alpha: 0.0,
  damping: 0.15,
  // set so the time step has halved after about a thousand steps
  stepDecay: 0.0007,
  timeStep: 0.25,
  maxMove: 0.08,          // as a multiple of the mean edge length
  // Scharein's collision guard: a move that would bring a link vertex
  // nearer than this to a non-neighbouring segment is refused. Set it
  // too low and the knot quietly passes through itself and flattens --
  // the mesh stays combinatorially correct, but the EMBEDDING untangles.
  minClearance: 1.5,
  pinBoundary: false,
  // scales the interior force only. At 1 the surface pulls itself fully
  // taut; lower values let it follow the link while keeping more of the
  // disk-and-band shape it was built with.
  surfaceAttraction: 0.6,
  // rescale about the centroid each step so the link's total length is
  // preserved. Without it the equilibrium size is wherever attraction
  // and repulsion happen to balance, and the knot inflates as it rounds.
  preserveLinkLength: true,
};

/** Undirected edges, in the engine's order: ascending (min, max). */
function edgeList(mesh) {
  const out = [];
  for (const k of mesh.edgeMap().keys()) {
    const [a, b] = k.split(',').map(Number);
    out.push([a, b]);
  }
  out.sort((p, q) => (p[0] - q[0]) || (p[1] - q[1]));
  return out;
}

/** Attraction along every edge. */
function springForces(P, edges, scale, p, into) {
  const force = into || P.map(() => [0, 0, 0]);
  for (const [a, b] of edges) {
    const dx = P[b][0] - P[a][0], dy = P[b][1] - P[a][1], dz = P[b][2] - P[a][2];
    let length = Math.hypot(dx, dy, dz);
    if (length === 0) length = 1e-12;
    const magnitude = p.attraction * Math.pow(length / scale, 1 + p.beta);
    const s = magnitude / length;
    force[a][0] += s * dx; force[a][1] += s * dy; force[a][2] += s * dz;
    force[b][0] -= s * dx; force[b][1] -= s * dy; force[b][2] -= s * dz;
  }
  return force;
}

/** Inverse-power repulsion among the given vertices, skipping
 *  neighbours. Quadratic, which is why it is run over the LINK only --
 *  a few hundred vertices, not the whole surface. */
function repulsion(P, index, excluded, scale, p, force) {
  const n = index.length;
  for (let i = 0; i < n; i++) {
    const a = P[index[i]];
    for (let j = 0; j < n; j++) {
      if (i === j || excluded.has(i < j ? `${i},${j}` : `${j},${i}`)) continue;
      const b = P[index[j]];
      const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
      const dist = Math.hypot(dx, dy, dz);
      if (dist === 0) continue;
      const magnitude = p.repulsion * Math.pow(dist / scale, -(2 + p.alpha));
      const s = magnitude / dist;
      force[index[i]][0] += s * dx;
      force[index[i]][1] += s * dy;
      force[index[i]][2] += s * dz;
    }
  }
  return force;
}

/** Which proposed moves would bring a link vertex too close to a
 *  non-neighbouring segment of its own component. */
function collisionMask(candidates, P, loop, clearance) {
  const n = loop.length;
  const blocked = new Array(n).fill(false);
  for (let k = 0; k < n; k++) {
    const c = candidates[k];
    for (let i = 0; i < n; i++) {
      // a vertex's own neighbourhood cannot block it
      const rel = ((i - k) % n + n) % n;
      if (rel === 0 || rel === 1 || rel === n - 1 || rel === n - 2) continue;
      const a = P[loop[i]], b = P[loop[(i + 1) % n]];
      const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
      let denom = abx * abx + aby * aby + abz * abz;
      if (denom === 0) denom = 1e-12;
      const rx = c[0] - a[0], ry = c[1] - a[1], rz = c[2] - a[2];
      let t = (rx * abx + ry * aby + rz * abz) / denom;
      t = t < 0 ? 0 : (t > 1 ? 1 : t);
      const dx = rx - t * abx, dy = ry - t * aby, dz = rz - t * abz;
      if (Math.hypot(dx, dy, dz) < clearance) { blocked[k] = true; break; }
    }
  }
  return blocked;
}

/** Run one relaxation cycle. There is no convergence test; the decaying
 *  time step is what makes a cycle settle. */
export function relax(mesh, params = {}, iterations = 200) {
  const p = { ...RELAX_DEFAULTS, ...params };
  const P = mesh.vertices.map((v) => v.slice());
  const edges = edgeList(mesh);
  if (!edges.length || !iterations) return mesh;

  let scale = 0;
  for (const [a, b] of edges) {
    scale += Math.hypot(P[b][0] - P[a][0], P[b][1] - P[a][1], P[b][2] - P[a][2]);
  }
  scale /= edges.length;
  const maxMove = p.maxMove * scale;
  const clearance = p.minClearance * scale;

  const loops = mesh.boundaryLoops();
  const linkSet = new Set();
  for (const loop of loops) for (const v of loop) linkSet.add(v);
  const linkIndex = [...linkSet].sort((a, b) => a - b);
  const positionInLink = new Map(linkIndex.map((v, i) => [v, i]));
  const excluded = new Set();
  const linkEdges = [];
  for (const loop of loops) {
    for (let i = 0; i < loop.length; i++) {
      const a = loop[i], b = loop[(i + 1) % loop.length];
      linkEdges.push([a, b]);
      const ia = positionInLink.get(a), ib = positionInLink.get(b);
      excluded.add(ia < ib ? `${ia},${ib}` : `${ib},${ia}`);
    }
  }
  const isLink = new Uint8Array(P.length);
  for (const v of linkIndex) isLink[v] = 1;

  let targetLinkLength = 0;
  for (const [a, b] of linkEdges) {
    targetLinkLength += Math.hypot(P[b][0] - P[a][0], P[b][1] - P[a][1],
                                   P[b][2] - P[a][2]);
  }

  const velocity = P.map(() => [0, 0, 0]);
  let dt = p.timeStep;

  for (let step = 0; step < iterations; step++) {
    // The surface follows the link but does not influence it, so the
    // two populations draw their forces from disjoint sources.
    const force = P.map(() => [0, 0, 0]);
    const interiorForce = springForces(P, edges, scale, p);
    for (let v = 0; v < P.length; v++) {
      if (isLink[v]) continue;
      for (let k = 0; k < 3; k++) force[v][k] = p.surfaceAttraction * interiorForce[v][k];
    }
    if (!p.pinBoundary && linkIndex.length >= 3) {
      const linkForce = springForces(P, linkEdges, scale, p);
      repulsion(P, linkIndex, excluded, scale, p, linkForce);
      for (const v of linkIndex) force[v] = linkForce[v];
    }

    const move = [];
    for (let v = 0; v < P.length; v++) {
      const vel = velocity[v];
      for (let k = 0; k < 3; k++) vel[k] = (1 - p.damping) * vel[k] + force[v][k] * dt;
      let m = [vel[0] * dt, vel[1] * dt, vel[2] * dt];
      const length = Math.hypot(m[0], m[1], m[2]);
      const d = length === 0 ? 1 : length;
      const capped = Math.min(length, maxMove);
      move.push([m[0] / d * capped, m[1] / d * capped, m[2] / d * capped]);
    }

    if (p.pinBoundary) {
      for (const v of linkIndex) { move[v] = [0, 0, 0]; velocity[v] = [0, 0, 0]; }
    } else if (clearance > 0) {
      for (const loop of loops) {
        const candidates = loop.map((v) => [P[v][0] + move[v][0],
                                            P[v][1] + move[v][1],
                                            P[v][2] + move[v][2]]);
        const blocked = collisionMask(candidates, P, loop, clearance);
        loop.forEach((v, i) => {
          if (blocked[i]) { move[v] = [0, 0, 0]; velocity[v] = [0, 0, 0]; }
        });
      }
    }

    for (let v = 0; v < P.length; v++) {
      for (let k = 0; k < 3; k++) P[v][k] += move[v][k];
    }

    if (p.preserveLinkLength && !p.pinBoundary && linkEdges.length) {
      let current = 0;
      for (const [a, b] of linkEdges) {
        current += Math.hypot(P[b][0] - P[a][0], P[b][1] - P[a][1],
                              P[b][2] - P[a][2]);
      }
      if (current > 1e-12) {
        const centroid = [0, 0, 0];
        for (const v of P) for (let k = 0; k < 3; k++) centroid[k] += v[k];
        for (let k = 0; k < 3; k++) centroid[k] /= P.length;
        const s = targetLinkLength / current;
        for (const v of P) {
          for (let k = 0; k < 3; k++) v[k] = centroid[k] + (v[k] - centroid[k]) * s;
        }
      }
    }

    dt *= 1 - p.stepDecay;
  }

  return new Mesh(P, mesh.faces.map((f) => f.slice()), mesh.faceGroups.slice());
}

// ------------------------------------------------------------ fairing

/** Row-normalised cotangent Laplacian, in directed-edge form.
 *
 *  Weights are clamped below so a badly shaped triangle cannot make the
 *  system indefinite; on these meshes the clamp fires on well under a
 *  percent of edges. */
export function cotangentLaplacian(mesh, clamp = 1e-3) {
  const tri = mesh.triangulated();
  const P = tri.vertices;
  const n = P.length;
  const src = [], dst = [], weight = [];
  for (const [a, b] of [[0, 1], [1, 2], [2, 0]]) {
    const c = 3 - a - b;
    for (const f of tri.faces) {
      const i = f[a], j = f[b], k = f[c];
      const u = [P[i][0] - P[k][0], P[i][1] - P[k][1], P[i][2] - P[k][2]];
      const v = [P[j][0] - P[k][0], P[j][1] - P[k][1], P[j][2] - P[k][2]];
      let cr = Math.hypot(u[1] * v[2] - u[2] * v[1],
                          u[2] * v[0] - u[0] * v[2],
                          u[0] * v[1] - u[1] * v[0]);
      if (cr < 1e-14) cr = 1e-14;
      const cot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / cr;
      const w = Math.max(0.5 * cot, clamp);
      src.push(i, j); dst.push(j, i); weight.push(w, w);
    }
  }
  const degree = new Float64Array(n);
  for (let e = 0; e < src.length; e++) degree[src[e]] += weight[e];
  for (let i = 0; i < n; i++) if (degree[i] === 0) degree[i] = 1;
  return {
    n,
    src: Int32Array.from(src),
    dst: Int32Array.from(dst),
    weight: Float64Array.from(weight),
    degree,
    /** The neighbour average (D^-1 W) x -- one sparse matvec. */
    averaged(x) {
      const acc = Array.from({ length: n }, () => [0, 0, 0]);
      for (let e = 0; e < this.src.length; e++) {
        const w = this.weight[e];
        const to = acc[this.src[e]], from = x[this.dst[e]];
        to[0] += w * from[0]; to[1] += w * from[1]; to[2] += w * from[2];
      }
      for (let i = 0; i < n; i++) {
        acc[i][0] /= this.degree[i];
        acc[i][1] /= this.degree[i];
        acc[i][2] /= this.degree[i];
      }
      return acc;
    },
  };
}

/** Solve (I + strength * L) X = rhs by damped Jacobi, rim pinned.
 *
 *  With L = I - P the update is X <- (rhs + strength * P X) /
 *  (1 + strength), and pinned rows are held at their right-hand side.
 *  The convergence factor is strength / (1 + strength) < 1, so the loop
 *  is unconditionally stable for any positive strength -- no step-size
 *  condition to get wrong, which is the point of solving it implicitly
 *  rather than stepping the flow forward. */
export function implicitSolve(L, rhs, strength, pinned, tol = 1e-6,
                              maxIterations = 4000) {
  let x = rhs.map((v) => v.slice());
  const denom = 1 + strength;
  let norm = 0;
  for (const v of rhs) norm += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const scale = Math.max(1, Math.sqrt(norm));
  for (let it = 0; it < maxIterations; it++) {
    const avg = L.averaged(x);
    const next = x.map((_, i) => [
      (rhs[i][0] + strength * avg[i][0]) / denom,
      (rhs[i][1] + strength * avg[i][1]) / denom,
      (rhs[i][2] + strength * avg[i][2]) / denom,
    ]);
    if (pinned) for (const i of pinned) next[i] = rhs[i].slice();
    let delta = 0;
    for (let i = 0; i < next.length; i++) {
      for (let k = 0; k < 3; k++) {
        const d = next[i][k] - x[i][k];
        delta += d * d;
      }
    }
    x = next;
    if (Math.sqrt(delta) <= tol * scale) break;
  }
  return x;
}

/** Flow the surface towards zero mean curvature with its rim held fixed.
 *
 *  `strength` is the implicit time step: larger means a bigger jump
 *  towards the minimal surface per solve. A handful of iterations is
 *  plenty, since the Laplacian is recomputed each time and the flow
 *  converges quickly. */
export function minimalSurface(mesh, strength = 12.0, iterations = 4,
                               fixBoundary = true) {
  let positions = mesh.vertices.map((v) => v.slice());
  const boundary = [...new Set(mesh.boundaryLoops().flat())].sort((a, b) => a - b);
  const pinned = fixBoundary && boundary.length ? boundary : null;
  for (let it = 0; it < iterations; it++) {
    const rhs = positions.map((v) => v.slice());
    const work = new Mesh(positions, mesh.faces, []);
    positions = implicitSolve(cotangentLaplacian(work), rhs, strength, pinned);
  }
  return new Mesh(positions, mesh.faces.map((f) => f.slice()),
                  mesh.faceGroups.slice());
}

/** Fair the boundary curves in place.
 *
 *  The rim of a freshly built surface is a chain of disk-rim arcs and
 *  band edges meeting at corners. A Laplacian pass along each closed
 *  loop removes the corners without moving the curve off its knot type. */
export function smoothBoundary(mesh, iterations = 20, weight = 0.5) {
  const P = mesh.vertices.map((v) => v.slice());
  const loops = mesh.boundaryLoops();
  for (let it = 0; it < iterations; it++) {
    for (const loop of loops) {
      const n = loop.length;
      const before = loop.map((v) => P[v].slice());
      for (let i = 0; i < n; i++) {
        const a = before[(i - 1 + n) % n], b = before[(i + 1) % n];
        for (let k = 0; k < 3; k++) {
          const target = 0.5 * (a[k] + b[k]);
          P[loop[i]][k] = before[i][k] + weight * (target - before[i][k]);
        }
      }
    }
  }
  return new Mesh(P, mesh.faces.map((f) => f.slice()), mesh.faceGroups.slice());
}

/** Relax, refine and fair a freshly built surface -- the engine's
 *  pipeline, in its order. The mix is deliberate: a LIGHT relaxation
 *  then heavy fairing, because the particle model's equilibrium is not
 *  the same object as a minimal surface. */
export function finish(mesh, {
  relaxSteps = 100, rimSteps = 0, levels = 2,
  fairSteps = 10, fairStrength = 2.0, relaxParams = null,
} = {}) {
  let m = mesh;
  if (relaxSteps) m = relax(m, relaxParams || {}, relaxSteps);
  if (rimSteps) m = smoothBoundary(m, rimSteps);
  if (levels) m = catmullClark(m, levels);
  if (fairSteps) m = minimalSurface(m, fairStrength, fairSteps);
  return m;
}
