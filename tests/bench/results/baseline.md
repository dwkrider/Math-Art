# Bench results: baseline

## catenoid

time: 0.200 s, n_verts=1088, n_tris=2048

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

time: 0.449 s, n_verts=2400, n_tris=4608

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

time: 3.222 s, n_verts=5320, n_tris=10230

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

time: 1.773 s, n_verts=3996, n_tris=7580

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

## seifert_fair

time: 1.312 s, n_verts=9263, n_tris=17952

| metric | value |
|---|---|
| area_before_fair | 49.22 |
| area | 48.77 |
| area_shrink_frac | 0.009235 |
| genus_preserved | 1 |
| rim_max_move | 0 |
| interior_mean_move | 0.01849 |
| selfx | 0 |
| time_relax_s | 0.2478 |
| time_fair_s | 1.065 |
| H_rms | 0.04898 |
| H_max | 0.3136 |
| q_min | 0.04218 |
| q_mean | 0.6468 |
| q_p05 | 0.2726 |
| min_angle_deg | 6.577 |
| degenerate | 0 |
| clamp_frac | 0.2143 |
| neg_cot_frac | 0.2085 |

## canonical

time: 3.692 s

| metric | value |
|---|---|
| tangent_spread_worst | 0.009116 |
| tangent_spread_mean | 0.002064 |
| planarity_worst | 0.0002561 |
| iters_mean | 338.8 |

| solid | tangent_spread | planarity_max | iters | time_s |
|---|---|---|---|---|
| gC | 5.545e-05 | 8.439e-07 | 400 | 0.7201 |
| pC | 0.0007872 | 4.401e-06 | 400 | 0.7115 |
| wC | 0.009116 | 0.0002561 | 400 | 0.9585 |
| tI | 0.0003602 | 2.281e-16 | 400 | 1.03 |
| kD | 1.028e-06 | 2.096e-16 | 94 | 0.2717 |

## biscribe

time: 13.112 s

| metric | value |
|---|---|
| n_correct | 8 |
| n_total | 8 |
| r_spread_worst_ok | 8.146e-11 |
| f_spread_worst_ok | 1.108e-10 |
| iters_mean_ok | 200.6 |

| solid | exists | converged | correct | r_spread | f_spread | iters | time_s |
|---|---|---|---|---|---|---|---|
| C | True | True | 1 | 0 | 0 | 1 | 0.02476 |
| D | True | True | 1 | 1.241e-16 | 1.21e-16 | 1 | 0.04026 |
| kC | True | True | 1 | 7.563e-11 | 1.589e-16 | 196 | 0.4479 |
| kD | True | True | 1 | 8.146e-11 | 1.89e-16 | 179 | 0.8627 |
| tO | True | True | 1 | 1.035e-16 | 1.108e-10 | 626 | 0.9799 |
| aC | False | False | 1 | 1.407e-16 | 0.07034 | 2500 | 2.164 |
| aD | False | False | 1 | 2.591e-16 | 0.04479 | 2500 | 5.249 |
| tC | False | False | 1 | 0.02759 | 0.02143 | 2500 | 3.343 |

## knot_trefoil

time: 0.091 s, n_verts=120

| metric | value |
|---|---|
| min_far_gap | 0.3494 |
| length | 18.51 |
| ropelength_proxy | 52.98 |
| turning_rms_deg | 6.088 |

## knot_hopf

time: 0.067 s, n_verts=200

| metric | value |
|---|---|
| lk_before | -1.001 |
| lk_after | -1.001 |
| lk_preserved | 1 |
| inter_component_gap | 0.8845 |
| length | 12.4 |

## crochet

time: 0.153 s, n_verts=430, n_tris=767

| metric | value |
|---|---|
| edge_err_rms | 0.06841 |
| edge_err_max | 0.1224 |
| selfx | 0 |
| median_K | -2.166 |
| z_extent | 0.3109 |

## planarize

time: 0.166 s, n_verts=28

| metric | value |
|---|---|
| planar_dev | 1.79e-16 |
| aspect | 0.6856 |
| crossings | 0 |
| map | Heptagonal Dodecahedron B |

