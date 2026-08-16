# Bench results: plateau_groom_mollify_groom

## catenoid

time: 0.446 s, n_verts=1088, n_tris=2048  
effective: cotan_mode=mollify, groom_every=4, grooms_run=9

| metric | value |
|---|---|
| area | 5.99 |
| q_min | 0.8979 |
| q_mean | 0.9465 |
| q_p05 | 0.8979 |
| min_angle_deg | 41.91 |
| degenerate | 0 |
| clamp_frac | 0 |
| neg_cot_frac | 0 |
| H_rms | 0.0002909 |
| H_max | 0.0006959 |
| edge_cv | 0.1181 |
| valence_rms_dev | 0 |
| area_exact | 5.992 |
| area_rel_err | -0.0002566 |
| waist_rel_err | 8.24e-05 |
| selfx | 0 |
| iters | 40 |

## catenoid_fine

time: 0.667 s, n_verts=2400, n_tris=4608  
effective: cotan_mode=mollify, groom_every=4, grooms_run=9

| metric | value |
|---|---|
| area | 5.991 |
| q_min | 0.7034 |
| q_mean | 0.9481 |
| q_p05 | 0.8525 |
| min_angle_deg | 35.4 |
| degenerate | 0 |
| clamp_frac | 0.003834 |
| neg_cot_frac | 0.0034 |
| H_rms | 0.0001724 |
| H_max | 0.001685 |
| edge_cv | 0.1304 |
| valence_rms_dev | 0.4267 |
| area_exact | 5.992 |
| area_rel_err | -0.0001051 |
| waist_rel_err | 0.0005197 |
| selfx | 0 |
| iters | 40 |

## seifert_span_q3

time: 1.608 s, n_verts=5320, n_tris=10230  
effective: cotan_mode=mollify, groom_every=4, grooms_run=1

| metric | value |
|---|---|
| area | 58.1 |
| q_min | 0.02514 |
| q_mean | 0.7908 |
| q_p05 | 0.3502 |
| min_angle_deg | 1.757 |
| degenerate | 0 |
| clamp_frac | 0.03842 |
| neg_cot_frac | 0.03021 |
| H_rms | 0.0054 |
| H_max | 0.1151 |
| edge_cv | 0.4614 |
| valence_rms_dev | 1.674 |
| selfx | 0 |
| rim_max_move | 0 |
| iters | 8 |

## seifert_span_q5

time: 1.587 s, n_verts=3996, n_tris=7580  
effective: cotan_mode=mollify, groom_every=4, grooms_run=1

| metric | value |
|---|---|
| area | 63.36 |
| q_min | 0.02149 |
| q_mean | 0.8204 |
| q_p05 | 0.3945 |
| min_angle_deg | 1.807 |
| degenerate | 0 |
| clamp_frac | 0.04068 |
| neg_cot_frac | 0.03584 |
| H_rms | 0.002646 |
| H_max | 0.02017 |
| edge_cv | 0.4392 |
| valence_rms_dev | 1.832 |
| selfx | 0 |
| rim_max_move | 0 |
| iters | 8 |

