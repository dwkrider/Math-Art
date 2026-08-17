# Bench results: bubble_groom_bubble_groom

## bubble_single

time: 1.923 s, n_verts=642, n_tris=1280  
effective: iters=300, iters_run=300, groom_every=4, grooms_run=74, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 12.58 |
| area_rel_excess | 0.0009778 |
| area_rel_excess_ref | 0.0009864 |
| radius_cv | 0.0003151 |
| radius_max_rel_dev | 0.001733 |
| pressure | 2.002 |
| pressure_rel_err | 0.0009767 |
| vol_drift_max | 8.573e-13 |
| vol_drift_pre_max | 0.01687 |
| area_max_rise | -1.66e-08 |
| groom_max_rise | -1.741e-08 |
| iters | 300 |

## bubble_double

time: 24.427 s, n_verts=2019, n_tris=4080  
effective: iters=1200, iters_run=1200, groom_every=4, grooms_run=299, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 21.22 |
| area_analytic | 21.21 |
| area_rel_excess | 0.0005585 |
| angle_rms_fit | 0.03374 |
| angle_min_fit | 120 |
| angle_max_fit | 120.1 |
| angle_rms_raw | 3.239 |
| angle_rms_raw_seed | 4.784 |
| fit_rms_worst | 0.0004481 |
| r1_fit | 1.002 |
| r2_fit | 1.002 |
| r3_fit | None |
| curv_resid | 5.713e-08 |
| p1 | 2.001 |
| p2 | 2.001 |
| dp_err | 8.134e-06 |
| pressure_rel_err_worst | 0.0004971 |
| vol_drift_max | 9.389e-13 |
| vol_drift_pre_max | 5.393e-05 |
| area_max_rise | -6.676e-09 |
| groom_max_rise | -8.094e-09 |
| iters | 1200 |

## bubble_double_unequal

time: 41.628 s, n_verts=2307, n_tris=4656  
effective: iters=1800, iters_run=1800, groom_every=4, grooms_run=449, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 22.67 |
| area_analytic | 22.66 |
| area_rel_excess | 0.0005385 |
| angle_rms_fit | 0.06333 |
| angle_min_fit | 119.9 |
| angle_max_fit | 120.1 |
| angle_rms_raw | 2.639 |
| angle_rms_raw_seed | 4.573 |
| fit_rms_worst | 0.0006374 |
| r1_fit | 0.8017 |
| r2_fit | 1.202 |
| r3_fit | 2.404 |
| curv_resid | 0.0007792 |
| p1 | 2.501 |
| p2 | 1.668 |
| dp_err | 0.0003574 |
| pressure_rel_err_worst | 0.0005331 |
| vol_drift_max | 6.376e-13 |
| vol_drift_pre_max | 0.000105 |
| area_max_rise | -7.277e-09 |
| groom_max_rise | -7.239e-09 |
| iters | 1800 |

