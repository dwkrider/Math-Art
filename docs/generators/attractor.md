# Strange Attractor

![Strange Attractor](../images/attractor.png)

## Overview

This generator draws the trajectory of a chaotic dynamical system as a curve. It ships all 38 strange attractors of Chaotic Atmospheres' *MATHRULES* art series (plus Zhou–Chen from the same collection) as presets: the ODE systems and parameter values follow Jürgen Meier's compilation, the artist's stated source, with Lorenz / Rössler / Chua transcribed from the posters. Each system is integrated with a fixed-step RK4 (Euler where the reference render depends on numerical dissipation), the transient discarded, and the trajectory emitted as a curve — optionally arc-length resampled, with the tube radius modulated by local speed so slow regions thicken, matching the look of the original renders.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Attractor | Lorenz | Which system to integrate. The 38+ presets include: Aizawa, Anishchenko-Astakhov, Arneodo, Bouali, Burke-Shaw, Chen-Celikovsky, Chen-Lee, Chua, Coullet, Coupled Lorenz, Dadras, Dequan Li, Finance, Four-Wing, Genesio-Tesi, Hadley, Halvorsen, Liu-Chen, Lorenz, Lorenz Mod 1, Lorenz Mod 2, Lorenz-Stenflo, Lu-Chen, Newton-Leipnik, Nose-Hoover, Qi, Qi-Chen, Rayleigh-Benard, Roessler, Rucklidge, Sakarya, Shimizu-Morioka, Thomas, Three-Scroll Unified 1, Three-Scroll Unified 2, Wang-Sun, Wimol-Banlue, Yu-Wang, Zhou-Chen (each preset's tooltip lists its parameter values) |
| Steps | 0 | Integration steps (0 = preset default) |
| Time Step Scale | 1.0 | Multiplier on the preset's integration step |
| Size | 2.0 | Largest bounding-box extent |
| Spline | Poly | Poly (one point per sample) or Bezier (auto-handles, smoother at low sample counts) |
| Resample | 0 | Resample to N evenly spaced points (0 = raw integration points; even spacing disables speed taper) |
| Tube Radius | 0.05 | Curve bevel depth (0 = wire only) |
| Bevel Resolution | 4 | Bevel profile resolution |
| Speed Taper | 0.0 | Thicken the tube where the flow is slow (0 = uniform radius) |
| Profile Sides | 0 | Polygonal tube cross-section with N flat sides (0 = round; e.g. 5 gives the pentagon profile used in the reference lorentz.blend) |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/attractor__LORENZ.png" width="200"><br><sub>Lorenz</sub></td>
<td align="center"><img src="../images/variants/attractor__ROSSLER.png" width="200"><br><sub>Rossler</sub></td>
<td align="center"><img src="../images/variants/attractor__THOMAS.png" width="200"><br><sub>Thomas</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/attractor__AIZAWA.png" width="200"><br><sub>Aizawa</sub></td>
<td align="center"><img src="../images/variants/attractor__HALVORSEN.png" width="200"><br><sub>Halvorsen</sub></td>
<td align="center"><img src="../images/variants/attractor__DADRAS.png" width="200"><br><sub>Dadras</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/attractor__CHENLEE.png" width="200"><br><sub>Chen-Lee</sub></td>
<td align="center"><img src="../images/variants/attractor__FOURWING.png" width="200"><br><sub>Four-Wing</sub></td>
<td align="center"><img src="../images/variants/attractor__NOSEHOOVER.png" width="200"><br><sub>Nose-Hoover</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/attractor__ARNEODO.png" width="200"><br><sub>Arneodo</sub></td>
</tr>
</table>

## How it works

A strange attractor is the long-term set that trajectories of a chaotic autonomous ODE

$$\dot{\mathbf{x}} = f(\mathbf{x}), \qquad \mathbf{x}\in\mathbb{R}^n,$$

settle onto. Each preset supplies $f$, its parameters, an initial state $\mathbf{x}_0$, a time step $\mathrm{d}t$, a step count, and a transient length. The canonical **Lorenz** system, for instance, is

$$\dot x = a(y-x),\quad \dot y = x(b-z)-y,\quad \dot z = xy - cz,$$

with $a=10,\ b=28,\ c=8/3$.

**Integration.** The state is advanced with fixed-step **RK4**,

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \tfrac{\mathrm{d}t}{6}\big(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\big),$$

$\mathbf{k}_1 = f(\mathbf{x}_k)$, $\mathbf{k}_2 = f(\mathbf{x}_k + \tfrac{\mathrm{d}t}{2}\mathbf{k}_1)$, etc. A first block of `transient` steps is integrated and discarded so the curve lies on the attractor, not the approach to it. The integrator works in any dimension: the 4D systems (**Qi**, **Lorenz-Stenflo**) and the 6D **Coupled Lorenz** pair are integrated in full and drawn as their $(x,y,z)$ projection. Trajectories that exceed $10^6$ in any coordinate raise an escape error (the operator suggests a smaller Time Step Scale).

A few presets deliberately use **Euler** instead: *Lorenz Mod 1* with $\dot z = +z + \dots$ actually escapes to infinity under exact integration, and its reference render exists only because the original Cinema 4D plugin integrated with Euler, whose numerical dissipation keeps the trajectory bounded — so it is reproduced faithfully with Euler. Chua's diode uses the canonical inner/outer slopes ($-8/7,\,-5/7$) that give the double scroll rather than the posters' printed values (which decay to a fixed point). *Rayleigh-Bénard* keeps its transient ($r=12$ is below the chaotic regime) to show the decaying double spiral.

**Post-processing.** The point cloud is centered on its bounding-box center and scaled so the largest extent equals *Size*. **Speed taper** uses the per-point segment speed $v_i = \lVert \mathbf{x}_{i+1}-\mathbf{x}_i\rVert$, normalized to $[0,1]$; the per-point curve radius is set to $1 + 3\,\text{taper}\,(1-v)$, so slow flow gives a fat tube. Arc-length **resampling** to $N$ points evens the spacing (and therefore disables the speed taper, since even spacing carries no speed information). A round bevel, or an $N$-gon bevel object (the pentagonal-cross-section trick from the reference `lorentz.blend`), gives the tube.

## References

- Chaotic Atmospheres, *MATHRULES — Strange Attractors* art series: <https://www.chaoticatmospheres.com/mathrules-strange-attractors>
- Jürgen Meier, *3D-Meier*, Tutorial 19 (attractor ODE systems and parameter values): <http://www.3d-meier.de/tut19/Seite0.html>
- Edward N. Lorenz, *Deterministic Nonperiodic Flow*, J. Atmos. Sci. 20 (1963), 130–141.
- Otto E. Rössler, *An equation for continuous chaos*, Phys. Lett. A 57 (1976), 397–398.
- Leon O. Chua, M. Komuro, T. Matsumoto, *The double scroll family*, IEEE Trans. Circuits Syst. 33 (1986), 1072–1118.
