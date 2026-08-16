# Bench results: poly_ls_old_defaults

## canonical

time: 4.301 s

| metric | value |
|---|---|
| tangent_spread_worst | 0.009116 |
| tangent_spread_mean | 0.002064 |
| planarity_worst | 0.0002561 |
| iters_mean | 338.8 |

| solid | tangent_spread | planarity_max | iters | time_s |
|---|---|---|---|---|
| gC | 5.545e-05 | 8.439e-07 | 400 | 0.733 |
| pC | 0.0007872 | 4.401e-06 | 400 | 0.7853 |
| wC | 0.009116 | 0.0002561 | 400 | 1.218 |
| tI | 0.0003602 | 2.281e-16 | 400 | 1.234 |
| kD | 1.028e-06 | 2.096e-16 | 94 | 0.3308 |

## biscribe

time: 15.071 s

| metric | value |
|---|---|
| n_correct | 8 |
| n_total | 8 |
| r_spread_worst_ok | 8.146e-11 |
| f_spread_worst_ok | 1.116e-10 |
| iters_mean_ok | 200.6 |

| solid | exists | converged | correct | r_spread | f_spread | iters | time_s |
|---|---|---|---|---|---|---|---|
| C | True | True | 1 | 0 | 0 | 1 | 0.03329 |
| D | True | True | 1 | 1.024e-16 | 1.21e-16 | 1 | 0.05073 |
| kC | True | True | 1 | 7.564e-11 | 1.404e-16 | 196 | 0.4286 |
| kD | True | True | 1 | 8.146e-11 | 2.043e-16 | 179 | 0.8891 |
| tO | True | True | 1 | 1.523e-16 | 1.116e-10 | 626 | 1.078 |
| aC | False | False | 1 | 7.523e-17 | 0.07034 | 2500 | 2.747 |
| aD | False | False | 1 | 7.16e-17 | 0.04479 | 2500 | 5.985 |
| tC | False | False | 1 | 0.1057 | 0.1381 | 2500 | 3.858 |

