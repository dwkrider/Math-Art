# A/B: mollify vs mollify_groom

### catenoid   time 0.13s -> 0.42s (x3.29)
| metric | mollify | mollify_groom | verdict |
|---|---|---|---|
| area | 5.99 | 5.99 | A wins |
| q_min | 0.7859 | 0.8979 | B wins |
| q_mean | 0.7998 | 0.9465 | B wins |
| q_p05 | 0.786 | 0.8979 | B wins |
| min_angle_deg | 36.3 | 41.91 | B wins |
| degenerate | 0 | 0 | tie |
| clamp_frac | 0.2292 | 0 | B wins |
| neg_cot_frac | 0.1667 | 0 | B wins |
| H_rms | 0.0001845 | 0.0002909 | A wins |
| H_max | 0.0004864 | 0.0006959 | A wins |
| edge_cv | 0.2076 | 0.1181 | B wins |
| valence_rms_dev | 0 | 0 | tie |
| area_exact | 5.992 | 5.992 | tie |
| area_rel_err | -0.0003325 | -0.0002566 | A wins |
| waist_rel_err | 0.0002393 | 8.24e-05 | B wins |
| selfx | 0 | 0 | tie |
| iters | 40 | 40 | tie |

### seifert_span_q3   time 1.15s -> 1.54s (x1.34)
| metric | mollify | mollify_groom | verdict |
|---|---|---|---|
| area | 58.1 | 58.1 | B wins |
| q_min | 0.03439 | 0.02514 | A wins |
| q_mean | 0.6545 | 0.7908 | B wins |
| q_p05 | 0.1756 | 0.3502 | B wins |
| min_angle_deg | 1.61 | 1.757 | B wins |
| degenerate | 0 | 0 | tie |
| clamp_frac | 0.2042 | 0.03842 | B wins |
| neg_cot_frac | 0.1931 | 0.03021 | B wins |
| H_rms | 0.00665 | 0.0054 | B wins |
| H_max | 0.1766 | 0.1151 | B wins |
| edge_cv | 0.4478 | 0.4614 | A wins |
| valence_rms_dev | 1.892 | 1.674 | B wins |
| selfx | 0 | 0 | tie |
| rim_max_move | 0 | 0 | tie |
| iters | 8 | 8 | tie |

### seifert_span_q5   time 1.47s -> 1.30s (x0.89)
| metric | mollify | mollify_groom | verdict |
|---|---|---|---|
| area | 63.4 | 63.36 | B wins |
| q_min | 0.01453 | 0.02149 | B wins |
| q_mean | 0.6414 | 0.8204 | B wins |
| q_p05 | 0.1091 | 0.3945 | B wins |
| min_angle_deg | 1.161 | 1.807 | B wins |
| degenerate | 0 | 0 | tie |
| clamp_frac | 0.208 | 0.04068 | B wins |
| neg_cot_frac | 0.1957 | 0.03584 | B wins |
| H_rms | 0.007596 | 0.002646 | B wins |
| H_max | 0.06701 | 0.02017 | B wins |
| edge_cv | 0.4297 | 0.4392 | A wins |
| valence_rms_dev | 2.25 | 1.832 | B wins |
| selfx | 0 | 0 | tie |
| rim_max_move | 0 | 0 | tie |
| iters | 8 | 8 | tie |

TOTAL: B wins 27, A wins 7, ties 13
