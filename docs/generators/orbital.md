# Atomic & Molecular Orbital

![Atomic & Molecular Orbital](../images/orbital.png)

## Overview

Hydrogenic atomic orbitals and LCAO molecular orbitals, drawn as isosurfaces of the wavefunction with the lobes coloured by the sign of $\psi$. The angular parts come from the sibling [Spherical Harmonic](spherical_harmonic.md) generator; this module supplies the radial part that turns an angular lobe picture into an actual orbital, so radial nodes are present and correct — 3s shows its inner shells, 3p its nested lobes.

**What these are and are not.** Every orbital here is a hydrogenic (or Slater-type, via the $\zeta$ exponent) function, and every molecular orbital is a fixed, symmetry-adapted combination of such functions. They are the qualitative pictures of an inorganic chemistry textbook, **not** self-consistent Hartree-Fock or DFT orbitals: the coefficients are chosen by symmetry (or by Hückel theory), not variationally optimised. They are drawn for their shape.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Mode | Atomic | One hydrogenic orbital $\psi_{n\ell m}$, or a linear combination of atomic orbitals. |
| n | 2 | Principal quantum number; the orbital has $n-\ell-1$ radial nodes. Range 1-6. |
| l | 1 | Azimuthal quantum number (0 = s, 1 = p, 2 = d, 3 = f); must be < n. Range 0-5. |
| m | 0 | Magnetic quantum number of the real orbital; clamped to $\|m\| \le \ell$. Range -5 to 5. |
| Zeta | 1.0 | Effective nuclear charge: 1 is hydrogen, larger values contract the orbital (a Slater-type basis function). Range 0.1-10. |
| Molecule | sigma 1s | Which molecular preset: the σ/σ\*, π/π\* and δ/δ\* diatomic pairs, the sp/sp²/sp³ hybrids, two water orbitals, the benzene π system, or Custom LCAO. |
| Bond Length | 1.4 | Diatomic nuclear separation in bohr (1.4 a₀ is H₂). Range 0.2-12. |
| Hückel MO | 0 | Which of the six benzene π orbitals to draw, lowest energy first. Range 0-5. |
| LCAO | `1s@0,0,-1.4 1; 1s@0,0,1.4 -1` | Custom combination: `orbital@x,y,z[:zeta] coefficient`, semicolon separated, positions in bohr. |
| Display | Single Surface | One isosurface at the enclosed probability, or a **Probability Cloud** of nested transparent contours fading outward. |
| Cloud Shells | 3 | Cloud mode: how many nested contours, each enclosing an even step of the density. Range 2-6. |
| Outer Opacity | 0.18 | Cloud mode: opacity of the outermost shell; the inner ones are progressively more solid. Range 0.02-1. |
| Enclosed Probability | 0.90 | Fraction of the electron density the isosurface encloses (in cloud mode, the fraction enclosed by the *outermost* shell). Range 0.05-0.99. |
| Level Override | 0.0 | Raw $\|\psi\|$ contour value; 0 uses the enclosed probability instead. Range 0-10. |
| Resolution | 96 | Sample grid resolution per axis; cost grows as the cube. Range 24-192. |
| Box Override | 0.0 | Sample box half-width in bohr; 0 fits it to the orbital. Range 0-200. |
| Nuclei & Bonds | On | Add a ball-and-stick skeleton for the nuclei (molecular mode). |
| Sign Colours | On | Two material slots for the positive and negative lobes of $\psi$. |
| Despeckle | 0.005 | Drop connected pieces smaller than this fraction of the mesh. Range 0-0.2. |
| Largest Lobe Only | Off | Discard all but the biggest connected piece. |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |
| Thickness | 0.0 | If > 0, add a Solidify modifier with this thickness. Range 0-1. |
| Smooth Shading | On | Shade the mesh smooth. |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/orbital__1s.png" width="200"><br><sub>1s</sub></td>
<td align="center"><img src="../images/variants/orbital__2s.png" width="200"><br><sub>2s (radial node)</sub></td>
<td align="center"><img src="../images/variants/orbital__2pz.png" width="200"><br><sub>2p_z</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/orbital__3pz.png" width="200"><br><sub>3p_z</sub></td>
<td align="center"><img src="../images/variants/orbital__3dxy.png" width="200"><br><sub>3d_xy</sub></td>
<td align="center"><img src="../images/variants/orbital__3dz2.png" width="200"><br><sub>3d_z2</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/orbital__4fz3.png" width="200"><br><sub>4f_z3</sub></td>
<td align="center"><img src="../images/variants/orbital__sigma1s.png" width="200"><br><sub>sigma 1s</sub></td>
<td align="center"><img src="../images/variants/orbital__sigmastar1s.png" width="200"><br><sub>sigma* 1s</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/orbital__pi2px.png" width="200"><br><sub>pi 2p_x</sub></td>
<td align="center"><img src="../images/variants/orbital__sp3.png" width="200"><br><sub>sp3 hybrid</sub></td>
<td align="center"><img src="../images/variants/orbital__water.png" width="200"><br><sub>H2O lone pair</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/orbital__benzene.png" width="200"><br><sub>benzene pi</sub></td>
<td align="center"><img src="../images/variants/orbital__cloud.png" width="200"><br><sub>pi 2p_x probability cloud</sub></td>
</tr>
</table>

## How it works

### The wavefunction

In atomic units ($a_0 = 1$),

$$\psi_{n\ell m}(r,\theta,\varphi) = R_{n\ell}(r)\,Y_\ell^m(\theta,\varphi), \qquad R_{n\ell}(r) = N\,e^{-\rho/2}\rho^{\ell}L_{n-\ell-1}^{2\ell+1}(\rho), \quad \rho = \frac{2\zeta r}{n},$$

with $N$ chosen so that $\int_0^\infty R_{n\ell}^2 r^2\,dr = 1$. The generalised Laguerre polynomials come from the recurrence

$$(k+1)L_{k+1}^{\alpha}(x) = (2k+1+\alpha-x)L_k^{\alpha}(x) - (k+\alpha)L_{k-1}^{\alpha}(x),$$

since Blender ships no scipy. The factor $\rho^{\ell}e^{-\rho/2}$ is evaluated as a single exponential, $\exp(\ell\log\rho - \rho/2)$: computed separately the two factors overflow and underflow into a NaN at large $\rho$. The self-test checks the normalisation and mutual orthogonality of every $R_{n\ell}$ up to $n=5$, reproduces the closed forms for $R_{10}$, $R_{20}$ and $R_{21}$ to $10^{-12}$, and counts the $n-\ell-1$ radial nodes — the assertion that catches the classic wrong Laguerre index.

### Choosing the contour

The operator exposes **enclosed probability**, not a raw level. It samples $|\psi|^2$ on a coarse grid, sorts it descending and takes the contour at which the cumulative density reaches the requested fraction. Without this, one operator could not serve both 1s and 5g, whose peak amplitudes differ by orders of magnitude.

The same pass also reports the box the contour actually needs, which is then used for the fine grid. This matters more than it sounds: an orbital's 99.9%-density radius is mostly empty space, and marching over it wastes the sample budget exactly where the nodes need it.

### The probability cloud

A single contour is a hard shell, which is exactly what an electron density is not. Cloud mode instead draws several **nested** contours — for three shells at 90%, the surfaces enclosing 30%, 60% and 90% of the density — and gives each its own transparent material, nearly solid at the core and a faint haze at the rim, so the falloff reads as a falloff.

Two details make it work. All shells are marched over the **same** sample box and put through **one** centre-and-fit at the end; fitting each to the 2 m cube on its own would scale them to identical size and stack them exactly. And a single sampling pass yields every level at once, so choosing three contours costs no more than choosing one. Sign colouring is kept, so a cloud of an antibonding orbital has $2\times$shells material slots and each lobe still shows its phase.

The transparency is set on the Principled BSDF's Alpha input, with the material's blend mode switched to blended — under whichever name the running Blender uses, since EEVEE Next renamed the switch in 4.2. Cycles honours the alpha directly.

### Meshing

The field handed to marching tetrahedra (`marching_tets` from the Minimal Surface Toolkit) is $c - |\psi|$, which is **negative inside** the lobes; the extractor winds triangles along the field gradient, so that sign convention is what puts the normals on the outside. Each face is then assigned to a material slot by the sign of $\psi$ at its centre, giving the two-colour lobe picture in a single object. In molecular mode a ball-and-stick skeleton of the nuclei is appended in a third slot, mapped through the same centre-and-fit transform so the two line up.

Where the level set grazes a sample plane tangentially — which it does at the outer edge of every orbital, the contour there being a near-sphere of nearly constant $|\psi|$ — the extractor leaves a scatter of grid-scale fragments. Despeckle drops them.

### Knowing when the grid is too coarse

A thin radial node can fall entirely between samples, fusing the lobes on either side and quietly drawing the wrong orbital. Rather than guess from a heuristic, the module **predicts the answer analytically**: it solves for the intervals of $r$ on which $|R_{n\ell}(r)|\max|Y_\ell^m| \ge c$, counts the angular lobes as $(\ell-|m|+1)\times 2|m|$ (or $\ell+1$ bands when $m=0$), and from those derives how many closed surfaces the isosurface *must* have — a region reaching the nucleus is a ball and contributes one surface, a spherical shell contributes two. If the extracted mesh comes up short, the operator says so and names the resolution to try. At the default settings 3s is such a case: it has five lobe surfaces but its innermost node gap is 0.24 a₀ against a 0.44 a₀ sample cell.

### Molecular orbitals

$\psi_{\text{MO}} = \sum_i c_i \chi_i$ with each $\chi_i$ centred on its own nucleus. The shipped presets are the textbook ones: σ and σ\* from 1s, 2s and 2p$_z$ pairs, π and π\* from 2p$_x$, δ and δ\* from 3d$_{xy}$ (a metal-metal δ bond), the sp, sp² and sp³ hybrids on one centre, two water orbitals (O at the origin, the HOH bisector along $+z$, H atoms at 104.5°, $r_{\text{OH}} = 1.81\,a_0$), and the six benzene π orbitals.

Note the sign convention on the diatomics: the two $2p_z$ lobes point *at* each other, so the **bonding** combination is the difference $2p_z(A) - 2p_z(B)$, where for s orbitals it is the sum.

For benzene the Hückel eigenvectors $c_j = e^{2\pi i jk/6}/\sqrt6$ with energies $\alpha + 2\beta\cos(2\pi k/6)$ are combined into their real cosine/sine partners; the self-test confirms they are orthonormal with adjacency eigenvalues $2, 1, 1, -1, -1, -2$. The energy is reported in the operator's status line.

Anything else can be typed into the LCAO field directly, e.g. `2pz@0,0,-1.2 1; 2pz@0,0,1.2 -1`.

## References

- E. Schrödinger, "Quantisierung als Eigenwertproblem (Erste Mitteilung)," *Annalen der Physik* 384(4), 1926, pp. 361-376.
- L. Pauling and E. B. Wilson, *Introduction to Quantum Mechanics with Applications to Chemistry*, McGraw-Hill, 1935 (the standard treatment of $R_{n\ell}$, the real orbital forms and the LCAO method).
- J. C. Slater, "Atomic Shielding Constants," *Physical Review* 36, 1930, pp. 57-64 (effective nuclear charge / Slater-type exponents).
- F. Hund, "Zur Deutung der Molekelspektren," *Zeitschrift für Physik*, 1927-1928 (series), and R. S. Mulliken, "Electronic Structures of Polyatomic Molecules and Valence," *Physical Review*, 1932 — the LCAO molecular-orbital method.
- E. Hückel, "Quantentheoretische Beiträge zum Benzolproblem," *Zeitschrift für Physik* 70, 1931 (the benzene π system).
- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Dover, 1965, chapter 22 (the generalised Laguerre recurrence used here).
