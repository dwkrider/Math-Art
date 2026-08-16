# Bench results: poly_ls2_baseline

## biscribe

time: 13.580 s

| metric | value |
|---|---|
| n_correct | 8 |
| n_total | 8 |
| r_spread_worst_ok | 8.146e-11 |
| f_spread_worst_ok | 1.108e-10 |
| iters_mean_ok | 200.6 |

| solid | exists | converged | correct | r_spread | f_spread | iters | time_s |
|---|---|---|---|---|---|---|---|
| C | True | True | 1 | 0 | 0 | 1 | 0.02954 |
| D | True | True | 1 | 1.241e-16 | 1.21e-16 | 1 | 0.04643 |
| kC | True | True | 1 | 7.563e-11 | 1.589e-16 | 196 | 0.4747 |
| kD | True | True | 1 | 8.146e-11 | 1.89e-16 | 179 | 1.015 |
| tO | True | True | 1 | 1.035e-16 | 1.108e-10 | 626 | 1.108 |
| aC | False | False | 1 | 1.407e-16 | 0.07034 | 2500 | 2.349 |
| aD | False | False | 1 | 2.591e-16 | 0.04479 | 2500 | 5.272 |
| tC | False | False | 1 | 0.02759 | 0.02143 | 2500 | 3.286 |

