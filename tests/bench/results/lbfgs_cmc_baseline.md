# Bench results: lbfgs_cmc_baseline

## cmc_bridge_cyl

time: 1.951 s, n_verts=816, n_tris=1536  
effective: iters=400, iters_run=400, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=cg

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

time: 3.951 s, n_verts=816, n_tris=1536  
effective: iters=1200, iters_run=959, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=cg

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

## cmc_drop45

time: 10.058 s, n_verts=577, n_tris=1104  
effective: iters=1500, iters_run=1500, groom_every=4, grooms_run=374, mobility=star, cotan_mode=mollify, optimizer=cg

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

time: 25.828 s, n_verts=865, n_tris=1680  
effective: iters=3000, iters_run=3000, groom_every=4, grooms_run=749, mobility=star, cotan_mode=mollify, optimizer=cg

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

## film_sphere_eq

time: 0.025 s, n_verts=240, n_tris=384  
effective: iters=800, iters_run=16, groom_every=4, grooms_run=3, mobility=star, cotan_mode=mollify, optimizer=cg

| metric | value |
|---|---|
| area | 6.015 |
| dev90_fit_rms | 5.032e-05 |
| dev90_fit_max | 6.1e-05 |
| dev90_raw_rms | 3.677e-05 |
| wall_resid_max | 9.871e-13 |
| vt_normal_max | 1.778e-14 |
| area_max_rise | -5.514e-13 |
| n_pressures | 0 |
| iters | 16 |
| area_exact | 6.032 |
| area_rel_err | 0.002853 |
| boundary_abs_z_max | 6.996e-07 |

## film_cyl_disk

time: 1.505 s, n_verts=577, n_tris=1104  
effective: iters=400, iters_run=400, groom_every=4, grooms_run=99, mobility=star, cotan_mode=mollify, optimizer=cg

| metric | value |
|---|---|
| area | 3.133 |
| dev90_fit_rms | 0.01422 |
| dev90_fit_max | 0.02951 |
| dev90_raw_rms | 0.08149 |
| wall_resid_max | 9.981e-13 |
| vt_normal_max | 8.816e-14 |
| area_max_rise | -9.647e-07 |
| n_pressures | 0 |
| iters | 400 |
| area_exact | 3.133 |
| area_rel_err | 0.0002766 |
| boundary_z_spread | 0.04308 |

