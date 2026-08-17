# Bench results: plateau_groom_mollify

## catenoid

time: 0.118 s, n_verts=1088, n_tris=2048  
effective: cotan_mode=mollify, groom_every=0, grooms_run=0

| metric | value |
|---|---|
| area | 5.99 |
| q_min | 0.7859 |
| q_mean | 0.7998 |
| q_p05 | 0.786 |
| min_angle_deg | 36.3 |
| degenerate | 0 |
| clamp_frac | 0.2292 |
| neg_cot_frac | 0.1667 |
| H_rms | 0.0001845 |
| H_max | 0.0004864 |
| edge_cv | 0.2076 |
| valence_rms_dev | 0 |
| area_exact | 5.992 |
| area_rel_err | -0.0003325 |
| waist_rel_err | 0.0002393 |
| selfx | 0 |
| iters | 40 |

## catenoid_fine

time: 0.359 s, n_verts=2400, n_tris=4608  
effective: cotan_mode=mollify, groom_every=0, grooms_run=0

| metric | value |
|---|---|
| area | 5.991 |
| q_min | 0.7888 |
| q_mean | 0.7998 |
| q_p05 | 0.7903 |
| min_angle_deg | 36.26 |
| degenerate | 0 |
| clamp_frac | 0.2512 |
| neg_cot_frac | 0.1667 |
| H_rms | 0.0001323 |
| H_max | 0.0004118 |
| edge_cv | 0.2083 |
| valence_rms_dev | 0 |
| area_exact | 5.992 |
| area_rel_err | -0.0001473 |
| waist_rel_err | 0.0006068 |
| selfx | 0 |
| iters | 40 |

## seifert_span_q3

time: 1.066 s, n_verts=5320, n_tris=10230  
effective: cotan_mode=mollify, groom_every=0, grooms_run=0

| metric | value |
|---|---|
| area | 58.1 |
| q_min | 0.03439 |
| q_mean | 0.6545 |
| q_p05 | 0.1756 |
| min_angle_deg | 1.61 |
| degenerate | 0 |
| clamp_frac | 0.2042 |
| neg_cot_frac | 0.1931 |
| H_rms | 0.00665 |
| H_max | 0.1766 |
| edge_cv | 0.4478 |
| valence_rms_dev | 1.892 |
| selfx | 0 |
| rim_max_move | 0 |
| iters | 8 |

## seifert_span_q5

time: 1.692 s, n_verts=3996, n_tris=7580  
effective: cotan_mode=mollify, groom_every=0, grooms_run=0

| metric | value |
|---|---|
| area | 63.4 |
| q_min | 0.01453 |
| q_mean | 0.6414 |
| q_p05 | 0.1091 |
| min_angle_deg | 1.161 |
| degenerate | 0 |
| clamp_frac | 0.208 |
| neg_cot_frac | 0.1957 |
| H_rms | 0.007596 |
| H_max | 0.06701 |
| edge_cv | 0.4297 |
| valence_rms_dev | 2.25 |
| selfx | 0 |
| rim_max_move | 0 |
| iters | 8 |

