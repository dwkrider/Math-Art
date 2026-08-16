# A/B: old_defaults vs mollify

### catenoid   time 0.20s -> 0.10s (x0.51)
_old_defaults effective: cotan_mode=clamp, groom_every=0, grooms_run=0_  
_mollify effective: cotan_mode=mollify, groom_every=0, grooms_run=0_  
| metric | old_defaults | mollify | verdict |
|---|---|---|---|
| area | 5.99 | 5.99 | B wins |
| q_min | 0.7882 | 0.7859 | A wins |
| q_mean | 0.8001 | 0.7998 | A wins |
| q_p05 | 0.79 | 0.786 | A wins |
| min_angle_deg | 35.99 | 36.3 | B wins |
| degenerate | 0 | 0 | tie |
| clamp_frac | 0.2495 | 0.2292 | B wins |
| neg_cot_frac | 0.1968 | 0.1667 | B wins |
| H_rms | 0.0004046 | 0.0001845 | B wins |
| H_max | 0.0008085 | 0.0004864 | B wins |
| edge_cv | 0.2101 | 0.2076 | B wins |
| valence_rms_dev | 0 | 0 | tie |
| area_exact | 5.992 | 5.992 | tie |
| area_rel_err | -0.0003278 | -0.0003325 | B wins |
| waist_rel_err | 0.0025 | 0.0002393 | B wins |
| selfx | 0 | 0 | tie |
| iters | 40 | 40 | tie |

### catenoid_fine   time 0.46s -> 0.27s (x0.59)
_old_defaults effective: cotan_mode=clamp, groom_every=0, grooms_run=0_  
_mollify effective: cotan_mode=mollify, groom_every=0, grooms_run=0_  
| metric | old_defaults | mollify | verdict |
|---|---|---|---|
| area | 5.991 | 5.991 | B wins |
| q_min | 0.7935 | 0.7888 | A wins |
| q_mean | 0.8004 | 0.7998 | A wins |
| q_p05 | 0.7945 | 0.7903 | A wins |
| min_angle_deg | 36.66 | 36.26 | A wins |
| degenerate | 0 | 0 | tie |
| clamp_frac | 0.2679 | 0.2512 | B wins |
| neg_cot_frac | 0.1933 | 0.1667 | B wins |
| H_rms | 0.0003204 | 0.0001323 | B wins |
| H_max | 0.001169 | 0.0004118 | B wins |
| edge_cv | 0.2092 | 0.2083 | B wins |
| valence_rms_dev | 0 | 0 | tie |
| area_exact | 5.992 | 5.992 | tie |
| area_rel_err | -0.0001455 | -0.0001473 | B wins |
| waist_rel_err | 0.001701 | 0.0006068 | B wins |
| selfx | 0 | 0 | tie |
| iters | 40 | 40 | tie |

### seifert_span_q3   time 3.44s -> 0.94s (x0.27)
_old_defaults effective: cotan_mode=clamp, groom_every=0, grooms_run=0_  
_mollify effective: cotan_mode=mollify, groom_every=0, grooms_run=0_  
| metric | old_defaults | mollify | verdict |
|---|---|---|---|
| area | 58.14 | 58.1 | B wins |
| q_min | 0.06674 | 0.03439 | A wins |
| q_mean | 0.6645 | 0.6545 | A wins |
| q_p05 | 0.2057 | 0.1756 | A wins |
| min_angle_deg | 2.011 | 1.61 | A wins |
| degenerate | 0 | 0 | tie |
| clamp_frac | 0.1828 | 0.2042 | A wins |
| neg_cot_frac | 0.166 | 0.1931 | A wins |
| H_rms | 0.04091 | 0.00665 | B wins |
| H_max | 0.3342 | 0.1766 | B wins |
| edge_cv | 0.4734 | 0.4478 | B wins |
| valence_rms_dev | 1.892 | 1.892 | tie |
| selfx | 0 | 0 | tie |
| rim_max_move | 0 | 0 | tie |
| iters | 8 | 8 | tie |

### seifert_span_q5   time 1.77s -> 1.42s (x0.80)
_old_defaults effective: cotan_mode=clamp, groom_every=0, grooms_run=0_  
_mollify effective: cotan_mode=mollify, groom_every=0, grooms_run=0_  
| metric | old_defaults | mollify | verdict |
|---|---|---|---|
| area | 63.44 | 63.4 | B wins |
| q_min | 0.07595 | 0.01453 | A wins |
| q_mean | 0.6531 | 0.6414 | A wins |
| q_p05 | 0.2112 | 0.1091 | A wins |
| min_angle_deg | 2.228 | 1.161 | A wins |
| degenerate | 0 | 0 | tie |
| clamp_frac | 0.1945 | 0.208 | A wins |
| neg_cot_frac | 0.1732 | 0.1957 | A wins |
| H_rms | 0.08064 | 0.007596 | B wins |
| H_max | 0.6257 | 0.06701 | B wins |
| edge_cv | 0.4667 | 0.4297 | B wins |
| valence_rms_dev | 2.25 | 2.25 | tie |
| selfx | 0 | 0 | tie |
| rim_max_move | 0 | 0 | tie |
| iters | 8 | 8 | tie |

TOTAL: B wins 25, A wins 19, ties 20
