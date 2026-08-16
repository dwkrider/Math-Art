# A/B: baseline vs poly_ls

### canonical   time 3.86s -> 3.72s (x0.97)
| metric | baseline | poly_ls | verdict |
|---|---|---|---|
| tangent_spread_worst | 0.009116 | 0.001425 | B wins |
| tangent_spread_mean | 0.002064 | 0.0002958 | B wins |
| planarity_worst | 0.0002561 | 4.311e-05 | B wins |
| iters_mean | 338.8 | 335 | B wins |

### biscribe   time 13.30s -> 2.76s (x0.21)
| metric | baseline | poly_ls | verdict |
|---|---|---|---|
| n_correct | 8 | 7 | A wins |
| n_total | 8 | 8 | tie |
| r_spread_worst_ok | 8.146e-11 | 1.847e-14 | B wins |
| f_spread_worst_ok | 1.108e-10 | 1.114e-13 | B wins |
| iters_mean_ok | 200.6 | 28.6 | B wins |

TOTAL: B wins 7, A wins 1, ties 1
