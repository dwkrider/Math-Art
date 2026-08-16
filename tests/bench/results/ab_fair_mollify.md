# A/B: baseline vs fair_mollify

### seifert_fair   time 1.36s -> 1.06s (x0.78)
| metric | baseline | fair_mollify | verdict |
|---|---|---|---|
| area_before_fair | 49.22 | 49.22 | tie |
| area | 48.77 | 48.66 | B wins |
| area_shrink_frac | 0.009235 | 0.01145 | A wins |
| genus_preserved | 1 | 1 | tie |
| rim_max_move | 0 | 0 | tie |
| interior_mean_move | 0.01849 | 0.01023 | info |
| selfx | 0 | 0 | tie |
| time_relax_s | 0.313 | 0.2384 | info |
| time_fair_s | 1.047 | 0.8212 | info |
| H_rms | 0.04898 | 0.0503 | A wins |
| H_max | 0.3136 | 0.279 | B wins |
| q_min | 0.04218 | 0.05965 | B wins |
| q_mean | 0.6468 | 0.6475 | B wins |
| q_p05 | 0.2726 | 0.2487 | A wins |
| min_angle_deg | 6.577 | 6.416 | A wins |
| degenerate | 0 | 0 | tie |
| clamp_frac | 0.2143 | 0.2033 | B wins |
| neg_cot_frac | 0.2085 | 0.199 | B wins |

TOTAL: B wins 6, A wins 4, ties 5
