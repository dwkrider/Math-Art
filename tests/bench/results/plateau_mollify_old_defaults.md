# Bench results: plateau_mollify_old_defaults

## catenoid

time: 0.202 s, n_verts=1088, n_tris=2048  
effective: cotan_mode=clamp, groom_every=0, grooms_run=0

| metric | value |
|---|---|
| area | 5.99 |
| q_min | 0.7882 |
| q_mean | 0.8001 |
| q_p05 | 0.79 |
| min_angle_deg | 35.99 |
| degenerate | 0 |
| clamp_frac | 0.2495 |
| neg_cot_frac | 0.1968 |
| H_rms | 0.0004046 |
| H_max | 0.0008085 |
| edge_cv | 0.2101 |
| valence_rms_dev | 0 |
| area_exact | 5.992 |
| area_rel_err | -0.0003278 |
| waist_rel_err | 0.0025 |
| selfx | 0 |
| iters | 40 |

## catenoid_fine

time: 0.462 s, n_verts=2400, n_tris=4608  
effective: cotan_mode=clamp, groom_every=0, grooms_run=0

| metric | value |
|---|---|
| area | 5.991 |
| q_min | 0.7935 |
| q_mean | 0.8004 |
| q_p05 | 0.7945 |
| min_angle_deg | 36.66 |
| degenerate | 0 |
| clamp_frac | 0.2679 |
| neg_cot_frac | 0.1933 |
| H_rms | 0.0003204 |
| H_max | 0.001169 |
| edge_cv | 0.2092 |
| valence_rms_dev | 0 |
| area_exact | 5.992 |
| area_rel_err | -0.0001455 |
| waist_rel_err | 0.001701 |
| selfx | 0 |
| iters | 40 |

## seifert_span_q3

time: 3.437 s, n_verts=5320, n_tris=10230  
effective: cotan_mode=clamp, groom_every=0, grooms_run=0

| metric | value |
|---|---|
| area | 58.14 |
| q_min | 0.06674 |
| q_mean | 0.6645 |
| q_p05 | 0.2057 |
| min_angle_deg | 2.011 |
| degenerate | 0 |
| clamp_frac | 0.1828 |
| neg_cot_frac | 0.166 |
| H_rms | 0.04091 |
| H_max | 0.3342 |
| edge_cv | 0.4734 |
| valence_rms_dev | 1.892 |
| selfx | 0 |
| rim_max_move | 0 |
| iters | 8 |

## seifert_span_q5

time: 1.769 s, n_verts=3996, n_tris=7580  
effective: cotan_mode=clamp, groom_every=0, grooms_run=0

| metric | value |
|---|---|
| area | 63.44 |
| q_min | 0.07595 |
| q_mean | 0.6531 |
| q_p05 | 0.2112 |
| min_angle_deg | 2.228 |
| degenerate | 0 |
| clamp_frac | 0.1945 |
| neg_cot_frac | 0.1732 |
| H_rms | 0.08064 |
| H_max | 0.6257 |
| edge_cv | 0.4667 |
| valence_rms_dev | 2.25 |
| selfx | 0 |
| rim_max_move | 0 |
| iters | 8 |

