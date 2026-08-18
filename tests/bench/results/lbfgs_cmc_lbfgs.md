# Bench results: lbfgs_cmc_lbfgs

## cmc_bridge_cyl

time: 5.660 s, n_verts=816, n_tris=1536  
effective: iters=400, iters_run=400, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs

| metric | value |
|---|---|
| area | 6.278 |
| pressure | 0.9985 |
| H_mean | 0.5109 |
| H_cv | 3.259 |
| p_minus_2H | 0.02338 |
| radius_cv | 0.000469 |
| pressure_rel_err | 0.001513 |
| H_row_cv | 0.5053 |
| F_cv | 2.219e-05 |
| vol_drift_max | 8.799e-13 |
| vol_drift_pre_max | 0.001042 |
| area_max_rise | -3.031e-11 |
| groom_max_rise | 0 |
| iters | 400 |

## cmc_bridge_fat

time: 0.435 s, n_verts=816, n_tris=1536  
effective: iters=1200, iters_run=73, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs

| metric | value |
|---|---|
| area | 7.526 |
| pressure | 2.074 |
| H_mean | 1.034 |
| H_cv | 0.0001832 |
| p_minus_2H | 0.00566 |
| H_row_cv | 9.701e-05 |
| F_cv | 0.001327 |
| vol_drift_max | 9.54e-13 |
| vol_drift_pre_max | 0.000457 |
| area_max_rise | -1.61e-11 |
| groom_max_rise | 0 |
| iters | 73 |

## cmc_drop45

time: 15.619 s, n_verts=577, n_tris=1104  
effective: iters=1500, iters_run=1500, groom_every=4, grooms_run=374, mobility=star, cotan_mode=mollify, optimizer=lbfgs

| metric | value |
|---|---|
| angle_achieved_deg | 44.99 |
| angle_err_deg | 0.00807 |
| contact_r | 1.452 |
| contact_r_rel_err | 0.001719 |
| area | 7.736 |
| area_rel_err | 0.0005636 |
| E_rel_err | 0.0005415 |
| pressure | 0.9762 |
| pressure_rel_err | 0.0005344 |
| H_mean | 0.4878 |
| H_cv | 0.000565 |
| p_minus_2H | 0.0006168 |
| fit_rms | 0.0002967 |
| wall_resid_max | 6.829e-13 |
| vt_normal_max | 0 |
| min_interior_z | 0.08306 |
| E_max_rise | -5.692e-11 |
| vol_drift_max | 5.723e-13 |
| iters | 1500 |

## cmc_drop135

time: 43.522 s, n_verts=865, n_tris=1680  
effective: iters=3000, iters_run=3000, groom_every=4, grooms_run=749, mobility=star, cotan_mode=mollify, optimizer=lbfgs

| metric | value |
|---|---|
| angle_achieved_deg | 134.8 |
| angle_err_deg | 0.1691 |
| contact_r | 0.5751 |
| contact_r_rel_err | 0.004441 |
| area | 7.033 |
| area_rel_err | 0.0001189 |
| E_rel_err | 0.000673 |
| pressure | 2.472 |
| pressure_rel_err | 0.0006717 |
| H_mean | 1.232 |
| H_cv | 0.0005356 |
| p_minus_2H | 0.006916 |
| fit_rms | 0.000376 |
| wall_resid_max | 0 |
| vt_normal_max | 0 |
| min_interior_z | 0.07406 |
| E_max_rise | -7.504e-12 |
| vol_drift_max | 9.44e-13 |
| iters | 3000 |

## film_sphere_eq

time: 0.013 s, n_verts=240, n_tris=384  
effective: iters=800, iters_run=7, groom_every=4, grooms_run=1, mobility=star, cotan_mode=mollify, optimizer=lbfgs

| metric | value |
|---|---|
| area | 6.015 |
| dev90_fit_rms | 1.568e-05 |
| dev90_fit_max | 3.158e-05 |
| dev90_raw_rms | 1.563e-05 |
| wall_resid_max | 9.782e-13 |
| vt_normal_max | 3.296e-17 |
| area_max_rise | -1.26e-13 |
| n_pressures | 0 |
| iters | 7 |
| area_exact | 6.032 |
| area_rel_err | 0.002853 |
| boundary_abs_z_max | 1.877e-09 |

## film_cyl_disk

time: 7.965 s, n_verts=577, n_tris=1104  
effective: iters=400, iters_run=400, groom_every=4, grooms_run=99, mobility=star, cotan_mode=mollify, optimizer=lbfgs

| metric | value |
|---|---|
| area | 0.0001321 |
| dev90_fit_rms | 2.582 |
| dev90_fit_max | 14.33 |
| dev90_raw_rms | 16.89 |
| wall_resid_max | 9.994e-13 |
| vt_normal_max | 5.684e-14 |
| area_max_rise | -2.089e-07 |
| n_pressures | 0 |
| iters | 400 |
| area_exact | 3.133 |
| area_rel_err | 1 |
| boundary_z_spread | 1.711e-05 |

