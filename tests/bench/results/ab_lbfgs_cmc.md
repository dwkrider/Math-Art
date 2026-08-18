# A/B: baseline vs lbfgs

### cmc_bridge_cyl   time 1.95s -> 5.66s (x2.90)
_baseline effective: iters=400, iters_run=400, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=cg_  
_lbfgs effective: iters=400, iters_run=400, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs_  
| metric | baseline | lbfgs | verdict |
|---|---|---|---|
| area | 6.279 | 6.278 | B wins |
| pressure | 1.002 | 0.9985 | info |
| H_mean | 0.5 | 0.5109 | info |
| H_cv | 0.001081 | 3.259 | A wins |
| p_minus_2H | 0.002101 | 0.02338 | A wins |
| radius_cv | 8.19e-06 | 0.000469 | A wins |
| pressure_rel_err | 0.002017 | 0.001513 | B wins |
| H_row_cv | 0.0002866 | 0.5053 | A wins |
| F_cv | 3.599e-08 | 2.219e-05 | A wins |
| vol_drift_max | 9.886e-13 | 8.799e-13 | tie |
| vol_drift_pre_max | 0.0001532 | 0.001042 | A wins |
| area_max_rise | -1.956e-10 | -3.031e-11 | A wins |
| groom_max_rise | 0 | 0 | tie |
| iters | 400 | 400 | tie |

### cmc_bridge_fat   time 3.95s -> 0.43s (x0.11)
_baseline effective: iters=1200, iters_run=959, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=cg_  
_lbfgs effective: iters=1200, iters_run=73, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs_  
| metric | baseline | lbfgs | verdict |
|---|---|---|---|
| area | 7.526 | 7.526 | A wins |
| pressure | 2.075 | 2.074 | info |
| H_mean | 1.035 | 1.034 | info |
| H_cv | 0.002641 | 0.0001832 | B wins |
| p_minus_2H | 0.005518 | 0.00566 | A wins |
| H_row_cv | 0.0004726 | 9.701e-05 | B wins |
| F_cv | 0.0008248 | 0.001327 | A wins |
| vol_drift_max | 9.961e-13 | 9.54e-13 | tie |
| vol_drift_pre_max | 9.505e-05 | 0.000457 | A wins |
| area_max_rise | -3.958e-13 | -1.61e-11 | B wins |
| groom_max_rise | 0 | 0 | tie |
| iters | 959 | 73 | B wins |

### cmc_drop45   time 10.06s -> 15.62s (x1.55)
_baseline effective: iters=1500, iters_run=1500, groom_every=4, grooms_run=374, mobility=star, cotan_mode=mollify, optimizer=cg_  
_lbfgs effective: iters=1500, iters_run=1500, groom_every=4, grooms_run=374, mobility=star, cotan_mode=mollify, optimizer=lbfgs_  
| metric | baseline | lbfgs | verdict |
|---|---|---|---|
| angle_achieved_deg | 44.97 | 44.99 | info |
| angle_err_deg | 0.03375 | 0.00807 | B wins |
| contact_r | 1.452 | 1.452 | info |
| contact_r_rel_err | 0.001754 | 0.001719 | B wins |
| area | 7.737 | 7.736 | B wins |
| area_rel_err | 0.0005818 | 0.0005636 | B wins |
| E_rel_err | 0.0004813 | 0.0005415 | A wins |
| pressure | 0.976 | 0.9762 | info |
| pressure_rel_err | 0.0002597 | 0.0005344 | A wins |
| H_mean | 0.4877 | 0.4878 | info |
| H_cv | 0.001339 | 0.000565 | B wins |
| p_minus_2H | 0.0006163 | 0.0006168 | A wins |
| fit_rms | 0.0001674 | 0.0002967 | A wins |
| wall_resid_max | 0 | 6.829e-13 | tie |
| vt_normal_max | 0 | 0 | tie |
| min_interior_z | 0.07801 | 0.08306 | info |
| E_max_rise | -5.17e-10 | -5.692e-11 | A wins |
| vol_drift_max | 9.981e-13 | 5.723e-13 | tie |
| iters | 1500 | 1500 | tie |

### cmc_drop135   time 25.83s -> 43.52s (x1.69)
_baseline effective: iters=3000, iters_run=3000, groom_every=4, grooms_run=749, mobility=star, cotan_mode=mollify, optimizer=cg_  
_lbfgs effective: iters=3000, iters_run=3000, groom_every=4, grooms_run=749, mobility=star, cotan_mode=mollify, optimizer=lbfgs_  
| metric | baseline | lbfgs | verdict |
|---|---|---|---|
| angle_achieved_deg | 134.8 | 134.8 | info |
| angle_err_deg | 0.1835 | 0.1691 | B wins |
| contact_r | 0.5752 | 0.5751 | info |
| contact_r_rel_err | 0.004624 | 0.004441 | B wins |
| area | 7.032 | 7.033 | A wins |
| area_rel_err | 8.758e-05 | 0.0001189 | A wins |
| E_rel_err | 0.0006789 | 0.000673 | B wins |
| pressure | 2.472 | 2.472 | info |
| pressure_rel_err | 0.0006874 | 0.0006717 | B wins |
| H_mean | 1.232 | 1.232 | info |
| H_cv | 0.00166 | 0.0005356 | B wins |
| p_minus_2H | 0.006984 | 0.006916 | B wins |
| fit_rms | 0.0004004 | 0.000376 | B wins |
| wall_resid_max | 0 | 0 | tie |
| vt_normal_max | 0 | 0 | tie |
| min_interior_z | 0.07393 | 0.07406 | info |
| E_max_rise | -2.531e-09 | -7.504e-12 | A wins |
| vol_drift_max | 9.824e-13 | 9.44e-13 | tie |
| iters | 3000 | 3000 | tie |

### film_sphere_eq   time 0.02s -> 0.01s (x0.55)
_baseline effective: iters=800, iters_run=16, groom_every=4, grooms_run=3, mobility=star, cotan_mode=mollify, optimizer=cg_  
_lbfgs effective: iters=800, iters_run=7, groom_every=4, grooms_run=1, mobility=star, cotan_mode=mollify, optimizer=lbfgs_  
| metric | baseline | lbfgs | verdict |
|---|---|---|---|
| area | 6.015 | 6.015 | tie |
| dev90_fit_rms | 5.032e-05 | 1.568e-05 | info |
| dev90_fit_max | 6.1e-05 | 3.158e-05 | info |
| dev90_raw_rms | 3.677e-05 | 1.563e-05 | info |
| wall_resid_max | 9.871e-13 | 9.782e-13 | tie |
| vt_normal_max | 1.778e-14 | 3.296e-17 | tie |
| area_max_rise | -5.514e-13 | -1.26e-13 | tie |
| n_pressures | 0 | 0 | tie |
| iters | 16 | 7 | B wins |
| area_exact | 6.032 | 6.032 | tie |
| area_rel_err | 0.002853 | 0.002853 | tie |
| boundary_abs_z_max | 6.996e-07 | 1.877e-09 | info |

### film_cyl_disk   time 1.50s -> 7.97s (x5.29)
_baseline effective: iters=400, iters_run=400, groom_every=4, grooms_run=99, mobility=star, cotan_mode=mollify, optimizer=cg_  
_lbfgs effective: iters=400, iters_run=400, groom_every=4, grooms_run=99, mobility=star, cotan_mode=mollify, optimizer=lbfgs_  
| metric | baseline | lbfgs | verdict |
|---|---|---|---|
| area | 3.133 | 0.0001321 | B wins |
| dev90_fit_rms | 0.01422 | 2.582 | info |
| dev90_fit_max | 0.02951 | 14.33 | info |
| dev90_raw_rms | 0.08149 | 16.89 | info |
| wall_resid_max | 9.981e-13 | 9.994e-13 | tie |
| vt_normal_max | 8.816e-14 | 5.684e-14 | tie |
| area_max_rise | -9.647e-07 | -2.089e-07 | A wins |
| n_pressures | 0 | 0 | tie |
| iters | 400 | 400 | tie |
| area_exact | 3.133 | 3.133 | tie |
| area_rel_err | 0.0002766 | 1 | A wins |
| boundary_z_spread | 0.04308 | 1.711e-05 | info |

TOTAL: B wins 20, A wins 21, ties 25
