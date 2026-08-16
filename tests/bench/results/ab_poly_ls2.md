# A/B: baseline vs poly_ls

### biscribe   time 13.58s -> 2.53s (x0.19)
| metric | baseline | poly_ls | verdict |
|---|---|---|---|
| n_correct | 8 | 8 | tie |
| n_total | 8 | 8 | tie |
| r_spread_worst_ok | 8.146e-11 | 1.847e-14 | B wins |
| f_spread_worst_ok | 1.108e-10 | 1.114e-13 | B wins |
| iters_mean_ok | 200.6 | 28.6 | B wins |

TOTAL: B wins 3, A wins 0, ties 2
