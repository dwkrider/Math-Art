# Bench results: cmc

## cmc_bridge_cyl

time: 1.906 s, n_verts=816, n_tris=1536  
effective: iters=400, iters_run=400, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 6.279 |
| pressure | 1.002 |
| H_mean | 0.5 |
| H_cv | 0.001081 |
| p_minus_2H | 0.002101 |
| radius_cv | 8.19e-06 |
| pressure_rel_err | 0.002017 |
| H_row_cv | 0.0002866 |
| F_cv | 3.599e-08 |
| vol_drift_max | 9.886e-13 |
| vol_drift_pre_max | 0.0001532 |
| area_max_rise | -1.956e-10 |
| groom_max_rise | 0 |
| iters | 400 |

## cmc_bridge_fat

time: 4.547 s, n_verts=816, n_tris=1536  
effective: iters=1200, iters_run=959, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 7.526 |
| pressure | 2.075 |
| H_mean | 1.035 |
| H_cv | 0.002641 |
| p_minus_2H | 0.005518 |
| H_row_cv | 0.0004726 |
| F_cv | 0.0008248 |
| vol_drift_max | 9.961e-13 |
| vol_drift_pre_max | 9.505e-05 |
| area_max_rise | -3.958e-13 |
| groom_max_rise | 0 |
| iters | 959 |

## cmc_bridge_thin

time: 1.092 s, n_verts=816, n_tris=1536  
effective: iters=600, iters_run=197, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 6.01 |
| pressure | -0.2425 |
| H_mean | -0.1209 |
| H_cv | 0.001107 |
| p_minus_2H | 0.0006892 |
| H_row_cv | 0.000293 |
| F_cv | 0.0004499 |
| vol_drift_max | 7.01e-13 |
| vol_drift_pre_max | 0.001894 |
| area_max_rise | -7.657e-11 |
| groom_max_rise | 0 |
| iters | 197 |

## cmc_bridge_fat_fine

time: 24.379 s, n_verts=3168, n_tris=6144  
effective: iters=2400, iters_run=1161, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 7.53 |
| pressure | 2.071 |
| H_mean | 1.035 |
| H_cv | 0.002423 |
| p_minus_2H | 0.001329 |
| H_row_cv | 0.0003347 |
| F_cv | 0.000397 |
| vol_drift_max | 8.44e-13 |
| vol_drift_pre_max | 0.0001013 |
| area_max_rise | -2.453e-14 |
| groom_max_rise | 0 |
| iters | 1161 |

## cmc_catenoid

time: 1.012 s, n_verts=816, n_tris=1536  
effective: iters=400, iters_run=199, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 5.987 |
| pressure | 0.01113 |
| p_minus_2H | 2.651e-05 |
| F_cv | 0.0002118 |
| vol_drift_max | 6.709e-13 |
| vol_drift_pre_max | 0.0008691 |
| area_max_rise | -3.032e-11 |
| groom_max_rise | 0 |
| iters | 199 |
| area_exact | 5.992 |
| area_rel_err | 0.0007417 |
| pressure_abs | 0.01113 |
| H_abs_mean | 0.005553 |

## cmc_catenoid_fine

time: 11.831 s, n_verts=3168, n_tris=6144  
effective: iters=1200, iters_run=561, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| area | 5.991 |
| pressure | 0.002742 |
| p_minus_2H | 6.59e-06 |
| F_cv | 5.78e-05 |
| vol_drift_max | 7.708e-13 |
| vol_drift_pre_max | 0.0004961 |
| area_max_rise | -8.688e-11 |
| groom_max_rise | 0 |
| iters | 561 |
| area_exact | 5.992 |
| area_rel_err | 0.0001867 |
| pressure_abs | 0.002742 |
| H_abs_mean | 0.001374 |

## cmc_drop45

time: 9.796 s, n_verts=577, n_tris=1104  
effective: iters=1500, iters_run=1500, groom_every=4, grooms_run=374, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| angle_achieved_deg | 44.97 |
| angle_err_deg | 0.03375 |
| contact_r | 1.452 |
| contact_r_rel_err | 0.001754 |
| area | 7.737 |
| area_rel_err | 0.0005818 |
| E_rel_err | 0.0004813 |
| pressure | 0.976 |
| pressure_rel_err | 0.0002597 |
| H_mean | 0.4877 |
| H_cv | 0.001339 |
| p_minus_2H | 0.0006163 |
| fit_rms | 0.0001674 |
| wall_resid_max | 0 |
| vt_normal_max | 0 |
| min_interior_z | 0.07801 |
| E_max_rise | -5.17e-10 |
| vol_drift_max | 9.981e-13 |
| iters | 1500 |

## cmc_drop135

time: 24.558 s, n_verts=865, n_tris=1680  
effective: iters=3000, iters_run=3000, groom_every=4, grooms_run=749, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| angle_achieved_deg | 134.8 |
| angle_err_deg | 0.1835 |
| contact_r | 0.5752 |
| contact_r_rel_err | 0.004624 |
| area | 7.032 |
| area_rel_err | 8.758e-05 |
| E_rel_err | 0.0006789 |
| pressure | 2.472 |
| pressure_rel_err | 0.0006874 |
| H_mean | 1.232 |
| H_cv | 0.00166 |
| p_minus_2H | 0.006984 |
| fit_rms | 0.0004004 |
| wall_resid_max | 0 |
| vt_normal_max | 0 |
| min_interior_z | 0.07393 |
| E_max_rise | -2.531e-09 |
| vol_drift_max | 9.824e-13 |
| iters | 3000 |

## cmc_drop45_fine

time: 65.724 s, n_verts=2305, n_tris=4512  
effective: iters=3000, iters_run=2965, groom_every=4, grooms_run=741, mobility=star, cotan_mode=mollify

| metric | value |
|---|---|
| angle_achieved_deg | 45 |
| angle_err_deg | 0.003887 |
| contact_r | 1.45 |
| contact_r_rel_err | 0.0003637 |
| area | 7.732 |
| area_rel_err | 5.595e-05 |
| E_rel_err | 0.000121 |
| pressure | 0.9759 |
| pressure_rel_err | 0.0001647 |
| H_mean | 0.4879 |
| H_cv | 0.0007112 |
| p_minus_2H | 5.051e-05 |
| fit_rms | 3.411e-05 |
| wall_resid_max | 0 |
| vt_normal_max | 0 |
| min_interior_z | 0.04458 |
| E_max_rise | -8.429e-11 |
| vol_drift_max | 9.991e-13 |
| iters | 2965 |

