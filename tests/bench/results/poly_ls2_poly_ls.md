# Bench results: poly_ls2_poly_ls

## biscribe

time: 2.530 s

| metric | value |
|---|---|
| n_correct | 8 |
| n_total | 8 |
| r_spread_worst_ok | 1.847e-14 |
| f_spread_worst_ok | 1.114e-13 |
| iters_mean_ok | 28.6 |

| solid | exists | converged | correct | r_spread | f_spread | iters | time_s |
|---|---|---|---|---|---|---|---|
| C | True | True | 1 | 0 | 0 | 1 | 0.03633 |
| D | True | True | 1 | 1.241e-16 | 1.21e-16 | 1 | 0.05585 |
| kC | True | True | 1 | 1.847e-14 | 1.347e-16 | 18 | 0.2814 |
| kD | True | True | 1 | 5.437e-15 | 4.434e-15 | 22 | 0.53 |
| tO | True | True | 1 | 2.324e-15 | 1.114e-13 | 101 | 0.5769 |
| aC | False | False | 1 | 4.532e-17 | 0.07034 | 1 | 0.05999 |
| aD | False | False | 1 | 2.027e-17 | 0.04479 | 1 | 0.1217 |
| tC | False | False | 1 | 1.42e-08 | 8.009e-08 | 182 | 0.8678 |

