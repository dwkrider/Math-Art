# Bench results: bubble_cg_bubble_nocg

## bubble_single

time: 1.246 s, n_verts=642, n_tris=1280  
effective: iters=300, iters_run=300, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 12.58 |
| area_rel_excess | 0.001177 |
| area_rel_excess_ref | 0.0009864 |
| radius_cv | 0.00208 |
| radius_max_rel_dev | 0.004313 |
| pressure | 2.002 |
| pressure_rel_err | 0.001151 |
| vol_drift_max | 9.171e-13 |
| vol_drift_pre_max | 0.01687 |
| area_max_rise | -1.603e-07 |
| groom_max_rise | 0 |
| iters | 300 |

## bubble_double

time: 17.635 s, n_verts=2019, n_tris=4080  
effective: iters=1200, iters_run=1200, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 21.26 |
| area_analytic | 21.21 |
| area_rel_excess | 0.002661 |
| angle_rms_fit | 8.732 |
| angle_min_fit | 106.6 |
| angle_max_fit | 126.7 |
| angle_rms_raw | 4.788 |
| angle_rms_raw_seed | 4.784 |
| fit_rms_worst | 0.01723 |
| r1_fit | 0.9855 |
| r2_fit | 0.9855 |
| r3_fit | None |
| curv_resid | 3.222e-11 |
| p1 | 1.99 |
| p2 | 1.99 |
| dp_err | 3.367e-11 |
| pressure_rel_err_worst | 0.005079 |
| vol_drift_max | 6.22e-13 |
| vol_drift_pre_max | 5.869e-06 |
| area_max_rise | -6.29e-07 |
| groom_max_rise | 0 |
| iters | 1200 |

## bubble_double_unequal

time: 28.464 s, n_verts=2307, n_tris=4656  
effective: iters=1800, iters_run=1800, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 22.72 |
| area_analytic | 22.66 |
| area_rel_excess | 0.002606 |
| angle_rms_fit | 7.703 |
| angle_min_fit | 108.4 |
| angle_max_fit | 126.7 |
| angle_rms_raw | 4.69 |
| angle_rms_raw_seed | 4.573 |
| fit_rms_worst | 0.03097 |
| r1_fit | 0.7776 |
| r2_fit | 1.201 |
| r3_fit | 2.535 |
| curv_resid | 0.05892 |
| p1 | 2.498 |
| p2 | 1.657 |
| dp_err | 0.007387 |
| pressure_rel_err_worst | 0.005633 |
| vol_drift_max | 8.002e-13 |
| vol_drift_pre_max | 2.47e-05 |
| area_max_rise | -4.698e-07 |
| groom_max_rise | 0 |
| iters | 1800 |

