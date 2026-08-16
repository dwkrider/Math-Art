# Bench results: poly_ls_baseline

## canonical

time: 3.855 s

| metric | value |
|---|---|
| tangent_spread_worst | 0.009116 |
| tangent_spread_mean | 0.002064 |
| planarity_worst | 0.0002561 |
| iters_mean | 338.8 |

| solid | tangent_spread | planarity_max | iters | time_s |
|---|---|---|---|---|
| gC | 5.545e-05 | 8.439e-07 | 400 | 0.6934 |
| pC | 0.0007872 | 4.401e-06 | 400 | 0.7928 |
| wC | 0.009116 | 0.0002561 | 400 | 0.998 |
| tI | 0.0003602 | 2.281e-16 | 400 | 1.043 |
| kD | 1.028e-06 | 2.096e-16 | 94 | 0.3282 |

## biscribe

time: 13.297 s

| metric | value |
|---|---|
| n_correct | 8 |
| n_total | 8 |
| r_spread_worst_ok | 8.146e-11 |
| f_spread_worst_ok | 1.108e-10 |
| iters_mean_ok | 200.6 |

| solid | exists | converged | correct | r_spread | f_spread | iters | time_s |
|---|---|---|---|---|---|---|---|
| C | True | True | 1 | 0 | 0 | 1 | 0.02434 |
| D | True | True | 1 | 1.241e-16 | 1.21e-16 | 1 | 0.03523 |
| kC | True | True | 1 | 7.563e-11 | 1.589e-16 | 196 | 0.4386 |
| kD | True | True | 1 | 8.146e-11 | 1.89e-16 | 179 | 0.8629 |
| tO | True | True | 1 | 1.035e-16 | 1.108e-10 | 626 | 0.9903 |
| aC | False | False | 1 | 1.407e-16 | 0.07034 | 2500 | 2.315 |
| aD | False | False | 1 | 2.591e-16 | 0.04479 | 2500 | 5.241 |
| tC | False | False | 1 | 0.02759 | 0.02143 | 2500 | 3.389 |

