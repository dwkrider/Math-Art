# Rejected: parabola line search on the Hart canonicalize update

Prototype (canon_ls_experiment.py) replacing canonicalize's fixed/adaptive
gains with an Evolver-style optimizing-scale search over the combined Hart
update direction, energy = tangency residual + centring + planarity.

Unlimited-iteration comparison (400-iter cap, adaptive-gain default vs LS):

| solid | adaptive spread (time) | LS spread (time, iters) |
|---|---|---|
| gC | 8.03e-06 (0.74s) | 1.41e-08 (2.21s, 303) |
| pC | 1.28e-05 (0.77s) | 5.25e-08 (3.73s, 400) |
| wC | 1.43e-03 (1.05s) | 5.99e-05 (4.79s, 400) |
| tI | 3.25e-05 (1.10s) | 4.74e-07 (4.59s, 400) |
| kD | 5.21e-07 (0.22s) | 1.93e-09 (0.36s, 41) |

At EQUAL WALL TIME (LS iteration budget matched to the adaptive run):

| solid | adaptive | LS equal-time | verdict |
|---|---|---|---|
| gC | 8.03e-06 | 3.42e-05 | adaptive 4.3x better |
| pC | 1.28e-05 | 2.99e-03 | adaptive 233x better |
| wC | 1.43e-03 | 1.34e-02 | adaptive 9.4x better |
| tI | 3.25e-05 | 6.83e-04 | adaptive 21x better |
| kD | 5.21e-07 | 7.73e-08 | LS 6.7x better |

Verdict: REJECTED as the default.  The line search buys tighter residuals
only at 3-4.5x the wall time (each trial re-evaluates the Python face
loop); on the wall-clock curve the adaptive gain wins 4/5.  The survey's
S9a answer for canonicalize quality is the antiprism circle-packing
"ambo" method, not a line search on Hart's update -- left on the backlog.
