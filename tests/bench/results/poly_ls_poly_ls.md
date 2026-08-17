# Bench results: poly_ls_poly_ls

## canonical

time: 4.502 s

| metric | value |
|---|---|
| tangent_spread_worst | 0.001425 |
| tangent_spread_mean | 0.0002958 |
| planarity_worst | 4.311e-05 |
| iters_mean | 335 |

| solid | tangent_spread | planarity_max | iters | time_s |
|---|---|---|---|---|
| gC | 8.026e-06 | 1.537e-07 | 400 | 0.8431 |
| pC | 1.281e-05 | 3.367e-08 | 400 | 0.9464 |
| wC | 0.001425 | 4.311e-05 | 400 | 1.169 |
| tI | 3.249e-05 | 3.019e-16 | 400 | 1.273 |
| kD | 5.214e-07 | 1.797e-16 | 75 | 0.2695 |

## biscribe

time: 3.036 s

| metric | value |
|---|---|
| n_correct | 8 |
| n_total | 8 |
| r_spread_worst_ok | 4.439e-14 |
| f_spread_worst_ok | 7.463e-14 |
| iters_mean_ok | 29.4 |

| solid | exists | converged | correct | r_spread | f_spread | iters | time_s |
|---|---|---|---|---|---|---|---|
| C | True | True | 1 | 0 | 0 | 1 | 0.04799 |
| D | True | True | 1 | 4.965e-17 | 1.21e-16 | 1 | 0.04992 |
| kC | True | True | 1 | 1.839e-14 | 2.448e-16 | 18 | 0.2387 |
| kD | True | True | 1 | 4.439e-14 | 1.117e-14 | 24 | 0.6521 |
| tO | True | True | 1 | 1.327e-15 | 7.463e-14 | 103 | 0.7179 |
| aC | False | False | 1 | 1.063e-16 | 0.07034 | 1 | 0.08306 |
| aD | False | False | 1 | 9.065e-17 | 0.04479 | 1 | 0.1623 |
| tC | False | False | 1 | 1.242e-08 | 7.866e-08 | 182 | 1.084 |

