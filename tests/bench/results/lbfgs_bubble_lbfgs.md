# Bench results: lbfgs_bubble_lbfgs

## bubble_single

time: 0.729 s, n_verts=642, n_tris=1280  
effective: iters=300, iters_run=41, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs

| metric | value |
|---|---|
| area | 12.58 |
| area_rel_excess | 0.0009569 |
| area_rel_excess_ref | 0.0009864 |
| radius_cv | 0.0002891 |
| radius_max_rel_dev | 0.001433 |
| pressure | 2.002 |
| pressure_rel_err | 0.0009569 |
| vol_drift_max | 8.407e-13 |
| vol_drift_pre_max | 0.02403 |
| area_max_rise | -2.698e-12 |
| groom_max_rise | 0 |
| iters | 41 |

## bubble_double

time: 7.037 s, n_verts=2019, n_tris=4080  
effective: iters=1200, iters_run=90, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs

| metric | value |
|---|---|
| area | 21.22 |
| area_analytic | 21.21 |
| area_rel_excess | 0.0005647 |
| angle_rms_fit | 0.05501 |
| angle_min_fit | 120 |
| angle_max_fit | 120.1 |
| angle_rms_raw | 3.027 |
| angle_rms_raw_seed | 4.784 |
| fit_rms_worst | 0.0003696 |
| r1_fit | 1.002 |
| r2_fit | 1.002 |
| r3_fit | 3.762e+05 |
| curv_resid | 2.513e-06 |
| p1 | 2.001 |
| p2 | 2.001 |
| dp_err | 6.504e-07 |
| pressure_rel_err_worst | 0.0005649 |
| vol_drift_max | 7.726e-13 |
| vol_drift_pre_max | 0.001531 |
| area_max_rise | -5.435e-12 |
| groom_max_rise | 0 |
| iters | 90 |

## bubble_double_unequal

time: 19.726 s, n_verts=2307, n_tris=4656  
effective: iters=1800, iters_run=228, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs

| metric | value |
|---|---|
| area | 22.67 |
| area_analytic | 22.66 |
| area_rel_excess | 0.0004921 |
| angle_rms_fit | 0.0401 |
| angle_min_fit | 120 |
| angle_max_fit | 120.1 |
| angle_rms_raw | 2.695 |
| angle_rms_raw_seed | 4.573 |
| fit_rms_worst | 0.0004018 |
| r1_fit | 0.8017 |
| r2_fit | 1.202 |
| r3_fit | 2.399 |
| curv_resid | 0.001686 |
| p1 | 2.502 |
| p2 | 1.667 |
| dp_err | 0.000852 |
| pressure_rel_err_worst | 0.0006369 |
| vol_drift_max | 9.471e-13 |
| vol_drift_pre_max | 0.00276 |
| area_max_rise | -1.877e-11 |
| groom_max_rise | 0 |
| iters | 228 |

## bubble_triple

time: 156.504 s, n_verts=3489, n_tris=7071  
effective: iters=1600, iters_run=1600, groom_every=4, grooms_run=399, mobility=star, cotan_mode=mollify, optimizer=lbfgs

| metric | value |
|---|---|
| area | 27.94 |
| angle_rms_fit | 0.04886 |
| angle_min_fit | 119.9 |
| angle_max_fit | 120.1 |
| angle_rms_raw | 1.913 |
| tetra_rms_fit | 0.0736 |
| tetra_min_fit | 109.4 |
| tetra_max_fit | 109.5 |
| tetra_rms_raw | 3.675 |
| n_tetra_angles | 12 |
| fit_rms_worst | 0.0003112 |
| p1 | 2.001 |
| p2 | 2.001 |
| p3 | 2.001 |
| yl_worst | 0.001382 |
| area_analytic | 27.93 |
| area_rel_excess | 0.000376 |
| pressure_rel_err_worst | 0.0004133 |
| vol_drift_max | 7.346e-13 |
| vol_drift_pre_max | 0.000263 |
| area_max_rise | -5.721e-09 |
| groom_max_rise | -1.999e-11 |
| iters | 1600 |

## bubble_double_fine_hard

time: 19.469 s, n_verts=8355, n_tris=16800  
effective: iters=2400, iters_run=122, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs

| metric | value |
|---|---|
| area | 21.21 |
| area_analytic | 21.21 |
| area_rel_excess | 0.0001372 |
| angle_rms_fit | 0.01184 |
| angle_min_fit | 120 |
| angle_max_fit | 120 |
| angle_rms_raw | 1.488 |
| angle_rms_raw_seed | 2.311 |
| fit_rms_worst | 9.653e-05 |
| r1_fit | 1 |
| r2_fit | 1 |
| r3_fit | 3.877e+04 |
| curv_resid | 2.71e-05 |
| p1 | 2 |
| p2 | 2 |
| dp_err | 1.069e-06 |
| pressure_rel_err_worst | 0.000136 |
| vol_drift_max | 8.914e-13 |
| vol_drift_pre_max | 0.001873 |
| area_max_rise | -3.005e-11 |
| groom_max_rise | 0 |
| iters | 122 |

