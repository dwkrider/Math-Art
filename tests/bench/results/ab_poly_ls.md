# A/B: old_defaults vs poly_ls

### canonical   time 4.30s -> 4.50s (x1.05)
| metric | old_defaults | poly_ls | verdict |
|---|---|---|---|
| tangent_spread_worst | 0.009116 | 0.001425 | B wins |
| tangent_spread_mean | 0.002064 | 0.0002958 | B wins |
| planarity_worst | 0.0002561 | 4.311e-05 | B wins |
| iters_mean | 338.8 | 335 | B wins |

### biscribe   time 15.07s -> 3.04s (x0.20)
| metric | old_defaults | poly_ls | verdict |
|---|---|---|---|
| n_correct | 8 | 8 | tie |
| n_total | 8 | 8 | tie |
| r_spread_worst_ok | 8.146e-11 | 4.439e-14 | B wins |
| f_spread_worst_ok | 1.116e-10 | 7.463e-14 | B wins |
| iters_mean_ok | 200.6 | 29.4 | B wins |

TOTAL: B wins 7, A wins 0, ties 2
