# Bench results: bubble_cg_baseline

## bubble_single

time: 1.201 s, n_verts=642, n_tris=1280  
effective: iters=300, iters_run=300, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 12.58 |
| area_rel_excess | 0.000957 |
| area_rel_excess_ref | 0.0009864 |
| radius_cv | 0.0002993 |
| radius_max_rel_dev | 0.001498 |
| pressure | 2.002 |
| pressure_rel_err | 0.0009569 |
| vol_drift_max | 9.79e-13 |
| vol_drift_pre_max | 0.01687 |
| area_max_rise | -3.517e-09 |
| groom_max_rise | 0 |
| iters | 300 |

## bubble_double

time: 21.162 s, n_verts=2019, n_tris=4080  
effective: iters=1200, iters_run=1200, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 21.22 |
| area_analytic | 21.21 |
| area_rel_excess | 0.0007271 |
| angle_rms_fit | 0.2534 |
| angle_min_fit | 119.8 |
| angle_max_fit | 120.4 |
| angle_rms_raw | 4.418 |
| angle_rms_raw_seed | 4.784 |
| fit_rms_worst | 0.0005121 |
| r1_fit | 1.003 |
| r2_fit | 1.003 |
| r3_fit | None |
| curv_resid | 2.306e-06 |
| p1 | 2.001 |
| p2 | 2.001 |
| dp_err | 5.781e-05 |
| pressure_rel_err_worst | 0.0006778 |
| vol_drift_max | 8.77e-13 |
| vol_drift_pre_max | 5.393e-05 |
| area_max_rise | -6.52e-09 |
| groom_max_rise | 0 |
| iters | 1200 |

## bubble_double_unequal

time: 32.594 s, n_verts=2307, n_tris=4656  
effective: iters=1800, iters_run=1800, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 22.68 |
| area_analytic | 22.66 |
| area_rel_excess | 0.0007181 |
| angle_rms_fit | 0.0974 |
| angle_min_fit | 119.9 |
| angle_max_fit | 120.1 |
| angle_rms_raw | 4.174 |
| angle_rms_raw_seed | 4.573 |
| fit_rms_worst | 0.0002866 |
| r1_fit | 0.8013 |
| r2_fit | 1.202 |
| r3_fit | 2.396 |
| curv_resid | 0.0014 |
| p1 | 2.502 |
| p2 | 1.668 |
| dp_err | 0.001097 |
| pressure_rel_err_worst | 0.0008104 |
| vol_drift_max | 9.468e-13 |
| vol_drift_pre_max | 0.000105 |
| area_max_rise | -5.335e-11 |
| groom_max_rise | 0 |
| iters | 1800 |

