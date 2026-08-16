# Bench results: poly_ls_poly_ls

## canonical

time: 3.725 s

| metric | value |
|---|---|
| tangent_spread_worst | 0.001425 |
| tangent_spread_mean | 0.0002958 |
| planarity_worst | 4.311e-05 |
| iters_mean | 335 |

| solid | tangent_spread | planarity_max | iters | time_s |
|---|---|---|---|---|
| gC | 8.026e-06 | 1.537e-07 | 400 | 0.7658 |
| pC | 1.281e-05 | 3.367e-08 | 400 | 0.7069 |
| wC | 0.001425 | 4.311e-05 | 400 | 0.9815 |
| tI | 3.249e-05 | 3.019e-16 | 400 | 1.033 |
| kD | 5.214e-07 | 1.797e-16 | 75 | 0.2373 |

## biscribe

time: 2.763 s

| metric | value |
|---|---|
| n_correct | 7 |
| n_total | 8 |
| r_spread_worst_ok | 1.847e-14 |
| f_spread_worst_ok | 1.114e-13 |
| iters_mean_ok | 28.6 |

| solid | exists | converged | correct | r_spread | f_spread | iters | time_s |
|---|---|---|---|---|---|---|---|
| C | True | True | 1 | 0 | 0 | 1 | 0.036 |
| D | True | True | 1 | 1.241e-16 | 1.21e-16 | 1 | 0.0604 |
| kC | True | True | 1 | 1.847e-14 | 1.347e-16 | 18 | 0.2497 |
| kD | True | True | 1 | 5.437e-15 | 4.434e-15 | 22 | 0.5615 |
| tO | True | True | 1 | 2.324e-15 | 1.114e-13 | 101 | 0.6745 |
| aC | False | False | 1 | 4.532e-17 | 0.07034 | 1 | 0.06755 |
| aD | False | False | 1 | 2.027e-17 | 0.04479 | 1 | 0.136 |
| tC | False | True | 0 | 1.42e-08 | 8.009e-08 | 182 | 0.9777 |

