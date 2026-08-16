# Bench results: post

## catenoid

time: 0.115 s, n_verts=1088, n_tris=2048  
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

time: 0.343 s, n_verts=2400, n_tris=4608  
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

time: 1.093 s, n_verts=5320, n_tris=10230  
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

time: 1.649 s, n_verts=3996, n_tris=7580  
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

## seifert_sweep

time: 85.423 s  
effective: cotan_mode=mollify, groom_every=0, grooms_run=0

| metric | value |
|---|---|
| selfx_worst | 0 |
| n_embedded | 17 |
| n_total | 17 |

| solid | selfx | min_angle_deg |
|---|---|---|
| q1_m96_r24 | 0 | 1.536 |
| q1_m140_r24 | 0 | 0.003113 |
| q1_m200_r24 | 0 | 0.9183 |
| q3_m96_r24 | 0 | 2.327 |
| q3_m140_r24 | 0 | 1.61 |
| q3_m200_r24 | 0 | 1.135 |
| q5_m96_r24 | 0 | 1.34 |
| q5_m140_r24 | 0 | 1.161 |
| q5_m200_r24 | 0 | 0.8082 |
| q7_m96_r24 | 0 | 0.7798 |
| q7_m140_r24 | 0 | 0.7798 |
| q7_m200_r24 | 0 | 0.6754 |
| q1_m48_r8 | 0 | 0.3467 |
| q3_m48_r8 | 0 | 3.125 |
| q5_m48_r8 | 0 | 1.392 |
| q7_m48_r8 | 0 | 0.813 |
| q9_m48_r8 | 0 | 0.5363 |

## seifert_fair

time: 1.128 s, n_verts=9263, n_tris=17952

| metric | value |
|---|---|
| area_before_fair | 49.22 |
| area | 48.66 |
| area_shrink_frac | 0.01145 |
| genus_preserved | 1 |
| rim_max_move | 0 |
| interior_mean_move | 0.01023 |
| selfx | 0 |
| time_relax_s | 0.2504 |
| time_fair_s | 0.8773 |
| H_rms | 0.0503 |
| H_max | 0.279 |
| q_min | 0.05965 |
| q_mean | 0.6475 |
| q_p05 | 0.2487 |
| min_angle_deg | 6.416 |
| degenerate | 0 |
| clamp_frac | 0.2033 |
| neg_cot_frac | 0.199 |

## canonical

time: 3.981 s

| metric | value |
|---|---|
| tangent_spread_worst | 0.001425 |
| tangent_spread_mean | 0.0002958 |
| planarity_worst | 4.311e-05 |
| iters_mean | 335 |

| solid | tangent_spread | planarity_max | iters | time_s |
|---|---|---|---|---|
| gC | 8.026e-06 | 1.537e-07 | 400 | 0.7949 |
| pC | 1.281e-05 | 3.367e-08 | 400 | 0.7909 |
| wC | 0.001425 | 4.311e-05 | 400 | 1.023 |
| tI | 3.249e-05 | 3.019e-16 | 400 | 1.136 |
| kD | 5.214e-07 | 1.797e-16 | 75 | 0.2359 |

## biscribe

time: 2.580 s

| metric | value |
|---|---|
| n_correct | 8 |
| n_total | 8 |
| r_spread_worst_ok | 4.439e-14 |
| f_spread_worst_ok | 7.463e-14 |
| iters_mean_ok | 29.4 |

| solid | exists | converged | correct | r_spread | f_spread | iters | time_s |
|---|---|---|---|---|---|---|---|
| C | True | True | 1 | 0 | 0 | 1 | 0.04668 |
| D | True | True | 1 | 4.965e-17 | 1.21e-16 | 1 | 0.05479 |
| kC | True | True | 1 | 1.839e-14 | 2.448e-16 | 18 | 0.1943 |
| kD | True | True | 1 | 4.439e-14 | 1.117e-14 | 24 | 0.5666 |
| tO | True | True | 1 | 1.327e-15 | 7.463e-14 | 103 | 0.636 |
| aC | False | False | 1 | 1.063e-16 | 0.07034 | 1 | 0.06313 |
| aD | False | False | 1 | 9.065e-17 | 0.04479 | 1 | 0.1417 |
| tC | False | False | 1 | 1.242e-08 | 7.866e-08 | 182 | 0.8772 |

## knot_trefoil

time: 0.093 s, n_verts=120

| metric | value |
|---|---|
| min_far_gap | 0.3494 |
| length | 18.51 |
| ropelength_proxy | 52.98 |
| turning_rms_deg | 6.088 |

## knot_hopf

time: 0.068 s, n_verts=200

| metric | value |
|---|---|
| lk_before | -1.001 |
| lk_after | -1.001 |
| lk_preserved | 1 |
| inter_component_gap | 0.8845 |
| length | 12.4 |

## crochet

time: 0.151 s, n_verts=430, n_tris=767

| metric | value |
|---|---|
| edge_err_rms | 0.06841 |
| edge_err_max | 0.1224 |
| selfx | 0 |
| median_K | -2.166 |
| z_extent | 0.3109 |

## planarize

time: 0.154 s, n_verts=28

| metric | value |
|---|---|
| planar_dev | 1.79e-16 |
| aspect | 0.6856 |
| crossings | 0 |
| map | Heptagonal Dodecahedron B |

