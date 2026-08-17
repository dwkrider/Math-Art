# A/B: baseline vs bubble_groom

### bubble_single   time 1.62s -> 1.92s (x1.19)
_baseline effective: iters=300, iters_run=300, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify_  
_bubble_groom effective: iters=300, iters_run=300, groom_every=4, grooms_run=74, mobility=star, cotan_mode=mollify_  
| metric | baseline | bubble_groom | verdict |
|---|---|---|---|
| area | 12.58 | 12.58 | A wins |
| area_rel_excess | 0.000957 | 0.0009778 | A wins |
| area_rel_excess_ref | 0.0009864 | 0.0009864 | tie |
| radius_cv | 0.0002993 | 0.0003151 | A wins |
| radius_max_rel_dev | 0.001498 | 0.001733 | A wins |
| pressure | 2.002 | 2.002 | info |
| pressure_rel_err | 0.0009569 | 0.0009767 | A wins |
| vol_drift_max | 9.79e-13 | 8.573e-13 | tie |
| vol_drift_pre_max | 0.01687 | 0.01687 | tie |
| area_max_rise | -3.517e-09 | -1.66e-08 | B wins |
| groom_max_rise | 0 | -1.741e-08 | B wins |
| iters | 300 | 300 | tie |

### bubble_double   time 25.29s -> 24.43s (x0.97)
_baseline effective: iters=1200, iters_run=1200, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify_  
_bubble_groom effective: iters=1200, iters_run=1200, groom_every=4, grooms_run=299, mobility=star, cotan_mode=mollify_  
| metric | baseline | bubble_groom | verdict |
|---|---|---|---|
| area | 21.22 | 21.22 | B wins |
| area_analytic | 21.21 | 21.21 | tie |
| area_rel_excess | 0.0007271 | 0.0005585 | B wins |
| angle_rms_fit | 0.2534 | 0.03374 | B wins |
| angle_min_fit | 119.8 | 120 | info |
| angle_max_fit | 120.4 | 120.1 | info |
| angle_rms_raw | 4.418 | 3.239 | B wins |
| angle_rms_raw_seed | 4.784 | 4.784 | tie |
| fit_rms_worst | 0.0005121 | 0.0004481 | B wins |
| r1_fit | 1.003 | 1.002 | info |
| r2_fit | 1.003 | 1.002 | info |
| r3_fit | None | None | info |
| curv_resid | 2.306e-06 | 5.713e-08 | B wins |
| p1 | 2.001 | 2.001 | info |
| p2 | 2.001 | 2.001 | info |
| dp_err | 5.781e-05 | 8.134e-06 | B wins |
| pressure_rel_err_worst | 0.0006778 | 0.0004971 | B wins |
| vol_drift_max | 8.77e-13 | 9.389e-13 | tie |
| vol_drift_pre_max | 5.393e-05 | 5.393e-05 | tie |
| area_max_rise | -6.52e-09 | -6.676e-09 | B wins |
| groom_max_rise | 0 | -8.094e-09 | B wins |
| iters | 1200 | 1200 | tie |

### bubble_double_unequal   time 40.04s -> 41.63s (x1.04)
_baseline effective: iters=1800, iters_run=1800, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify_  
_bubble_groom effective: iters=1800, iters_run=1800, groom_every=4, grooms_run=449, mobility=star, cotan_mode=mollify_  
| metric | baseline | bubble_groom | verdict |
|---|---|---|---|
| area | 22.68 | 22.67 | B wins |
| area_analytic | 22.66 | 22.66 | tie |
| area_rel_excess | 0.0007181 | 0.0005385 | B wins |
| angle_rms_fit | 0.0974 | 0.06333 | B wins |
| angle_min_fit | 119.9 | 119.9 | info |
| angle_max_fit | 120.1 | 120.1 | info |
| angle_rms_raw | 4.174 | 2.639 | B wins |
| angle_rms_raw_seed | 4.573 | 4.573 | tie |
| fit_rms_worst | 0.0002866 | 0.0006374 | A wins |
| r1_fit | 0.8013 | 0.8017 | info |
| r2_fit | 1.202 | 1.202 | info |
| r3_fit | 2.396 | 2.404 | info |
| curv_resid | 0.0014 | 0.0007792 | B wins |
| p1 | 2.502 | 2.501 | info |
| p2 | 1.668 | 1.668 | info |
| dp_err | 0.001097 | 0.0003574 | B wins |
| pressure_rel_err_worst | 0.0008104 | 0.0005331 | B wins |
| vol_drift_max | 9.468e-13 | 6.376e-13 | tie |
| vol_drift_pre_max | 0.000105 | 0.000105 | tie |
| area_max_rise | -5.335e-11 | -7.277e-09 | B wins |
| groom_max_rise | 0 | -7.239e-09 | B wins |
| iters | 1800 | 1800 | tie |

TOTAL: B wins 21, A wins 6, ties 14
