# Adopted: antiprism circle-packing ("ambo") canonicalization

Measured by `ambo_equal_time_experiment.py` (2026-08-18, branch
`antiprism-canonicalization`), the follow-up the rejected line-search
experiment (`canon_ls_RESULTS.md`) pointed to.  Same rule as there: the
verdict comes from the WALL-CLOCK curve, never the per-iteration curve.

Unlimited (adaptive-gain Hart at the bench's 400-iteration budget vs
`canonicalize_ambo` run to its own 1e-12 tolerance):

| solid | adaptive spread (time) | ambo spread (time, iters) |
|---|---|---|
| gC | 8.03e-06 (0.98s) | 4.44e-16 (0.120s, 591) |
| pC | 1.28e-05 (0.88s) | 5.55e-16 (0.157s, 943) |
| wC | 1.42e-03 (1.18s) | 5.55e-16 (0.130s, 722) |
| tI | 3.25e-05 (2.56s) | 5.55e-16 (0.532s, 565) |
| kD | 5.21e-07 (0.62s) | 1.98e-12 (0.146s, 571) |

At EQUAL WALL TIME (Hart's iteration budget scaled down to ambo's wall
time, the same budget-matching scheme canon_ls used):

| solid | Hart @ ambo budget (iters) | ambo | verdict |
|---|---|---|---|
| gC | 4.97e-03 (49) | 4.44e-16 | ambo 1.1e+13x better |
| pC | 4.04e-02 (71) | 5.55e-16 | ambo 7.3e+13x better |
| wC | 4.87e-02 (44) | 5.55e-16 | ambo 8.8e+13x better |
| tI | 2.18e-03 (83) | 5.55e-16 | ambo 3.9e+12x better |
| kD | 5.21e-07 (93) | 1.98e-12 | ambo 2.6e+05x better |

Ground-truth residuals of the ambo results (unit midsphere): worst
across the catalog max|d-1| = 1.38e-12 (kD; others <= 3.3e-16),
tangency-centroid offset <= 7.6e-17, face planarity <= 1.38e-11 (pC).

Verdict: ADOPTED.  The mechanism is why it wins: in the ambo variables
two of Hart's three competing soft nudges become exact projections
(unit-sphere tangency by renormalization, tangency centroid by exact
subtraction), and the remaining relaxed residuals vectorize over a
fixed-width (nE, 4) array with no Python face loop -- so it is
simultaneously per-iteration cheaper AND fewer-iterations, unlike the
line search, which was per-iteration stronger but wall-clock 233x
worse on pC.  Call sites use `canonicalize_best` (ambo with a checked
tangency spread and Hart fallback); `biscribe` init flipped to ambo
(8/8 classification unchanged, spreads tied ~1e-14, 1.6x faster);
Hart loop kept as `canonicalize` and pinned bitwise via
`biscribe(init="hart")` / `canonical_mode=hart` (76/76 off-path bench
metric values reproduce master bitwise).
