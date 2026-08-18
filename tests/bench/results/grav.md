# Bench results: grav

## cmc_drop_grav_tiny

time: 9.596 s, n_verts=577, n_tris=1104  
effective: iters=1500, iters_run=1500, groom_every=4, grooms_run=374, mobility=star, cotan_mode=mollify, bond=0.001

| metric | value |
|---|---|
| invariant_mean | 0.9754 |
| invariant_std | 0.001384 |
| invariant_band_std | 0.0004027 |
| invariant_vs_p | 0.0006168 |
| pressure | 0.9761 |
| apex_z | 0.6027 |
| local_angle_mean | 44.95 |
| local_angle_err | 0.05011 |
| contact_r | 1.452 |
| area | 7.737 |
| wall_resid_max | 0 |
| E_max_rise | -8.489e-10 |
| vol_drift_max | 9.919e-13 |
| iters | 1500 |
| angle_achieved_deg | 44.96 |
| angle_err_deg | 0.03531 |
| contact_r_rel_err | 0.001763 |
| pressure_rel_err | 0.0003464 |

## cmc_drop_grav

time: 10.200 s, n_verts=577, n_tris=1104  
effective: iters=1500, iters_run=1500, groom_every=4, grooms_run=374, mobility=star, cotan_mode=mollify, bond=2.0

| metric | value |
|---|---|
| invariant_mean | 1.653 |
| invariant_std | 0.002837 |
| invariant_band_std | 0.0007182 |
| invariant_vs_p | 0.000355 |
| pressure | 1.653 |
| apex_z | 0.6624 |
| local_angle_mean | 60.09 |
| local_angle_err | 0.0947 |
| contact_r | 1.331 |
| area | 7.061 |
| wall_resid_max | 0 |
| E_max_rise | -2.625e-09 |
| vol_drift_max | 9.815e-13 |
| iters | 1500 |

## cmc_drop_grav_fine

time: 57.816 s, n_verts=2305, n_tris=4512  
effective: iters=3000, iters_run=2609, groom_every=4, grooms_run=652, mobility=star, cotan_mode=mollify, bond=2.0

| metric | value |
|---|---|
| invariant_mean | 1.652 |
| invariant_std | 0.0008215 |
| invariant_band_std | 0.0002953 |
| invariant_vs_p | 0.0002269 |
| pressure | 1.652 |
| apex_z | 0.6615 |
| local_angle_mean | 60.03 |
| local_angle_err | 0.02527 |
| contact_r | 1.329 |
| area | 7.058 |
| wall_resid_max | 0 |
| E_max_rise | -7.736e-11 |
| vol_drift_max | 9.998e-13 |
| iters | 2609 |

## cmc_puddle_sweep

time: 86.084 s

| metric | value |
|---|---|
| ratio_bo4 | 1.002 |
| ratio_bo10 | 1.068 |
| ratio_bo25 | 1.048 |
| ratio_bo50 | 1.036 |
| approach_monotone | 1 |
| worst_final_ratio_err | 0.03586 |

| solid | h_inf | apex | ratio | excess_sqrtBo | local_angle_mean | lc_over_h | drift | iters |
|---|---|---|---|---|---|---|---|---|
| bo4 | 0.7274 | 0.7292 | 1.002 | 0.004993 | 117.7 | 2.966 | 4.236e-13 | 1500 |
| bo10 | 0.4601 | 0.4912 | 1.068 | 0.2141 | 116.9 | 2.103 | 9.675e-13 | 878 |
| bo25 | 0.291 | 0.3049 | 1.048 | 0.2401 | 116.5 | 1.64 | 9.771e-13 | 1404 |
| bo50 | 0.2057 | 0.2131 | 1.036 | 0.2536 | 114 | 1.321 | 9.872e-13 | 1394 |

