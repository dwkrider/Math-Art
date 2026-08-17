# A/B: bubble_nocg vs baseline

### bubble_single   time 1.25s -> 1.20s (x0.96)
_bubble_nocg effective: iters=300, iters_run=300, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify_  
_baseline effective: iters=300, iters_run=300, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify_  
| metric | bubble_nocg | baseline | verdict |
|---|---|---|---|
| area | 12.58 | 12.58 | B wins |
| area_rel_excess | 0.001177 | 0.000957 | B wins |
| area_rel_excess_ref | 0.0009864 | 0.0009864 | tie |
| radius_cv | 0.00208 | 0.0002993 | B wins |
| radius_max_rel_dev | 0.004313 | 0.001498 | B wins |
| pressure | 2.002 | 2.002 | info |
| pressure_rel_err | 0.001151 | 0.0009569 | B wins |
| vol_drift_max | 9.171e-13 | 9.79e-13 | tie |
| vol_drift_pre_max | 0.01687 | 0.01687 | tie |
| area_max_rise | -1.603e-07 | -3.517e-09 | A wins |
| groom_max_rise | 0 | 0 | tie |
| iters | 300 | 300 | tie |

### bubble_double   time 17.64s -> 21.16s (x1.20)
_bubble_nocg effective: iters=1200, iters_run=1200, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify_  
_baseline effective: iters=1200, iters_run=1200, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify_  
| metric | bubble_nocg | baseline | verdict |
|---|---|---|---|
| area | 21.26 | 21.22 | B wins |
| area_analytic | 21.21 | 21.21 | tie |
| area_rel_excess | 0.002661 | 0.0007271 | B wins |
| angle_rms_fit | 8.732 | 0.2534 | B wins |
| angle_min_fit | 106.6 | 119.8 | info |
| angle_max_fit | 126.7 | 120.4 | info |
| angle_rms_raw | 4.788 | 4.418 | B wins |
| angle_rms_raw_seed | 4.784 | 4.784 | tie |
| fit_rms_worst | 0.01723 | 0.0005121 | B wins |
| r1_fit | 0.9855 | 1.003 | info |
| r2_fit | 0.9855 | 1.003 | info |
| r3_fit | None | None | info |
| curv_resid | 3.222e-11 | 2.306e-06 | A wins |
| p1 | 1.99 | 2.001 | info |
| p2 | 1.99 | 2.001 | info |
| dp_err | 3.367e-11 | 5.781e-05 | A wins |
| pressure_rel_err_worst | 0.005079 | 0.0006778 | B wins |
| vol_drift_max | 6.22e-13 | 8.77e-13 | tie |
| vol_drift_pre_max | 5.869e-06 | 5.393e-05 | A wins |
| area_max_rise | -6.29e-07 | -6.52e-09 | A wins |
| groom_max_rise | 0 | 0 | tie |
| iters | 1200 | 1200 | tie |

### bubble_double_unequal   time 28.46s -> 32.59s (x1.15)
_bubble_nocg effective: iters=1800, iters_run=1800, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify_  
_baseline effective: iters=1800, iters_run=1800, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify_  
| metric | bubble_nocg | baseline | verdict |
|---|---|---|---|
| area | 22.72 | 22.68 | B wins |
| area_analytic | 22.66 | 22.66 | tie |
| area_rel_excess | 0.002606 | 0.0007181 | B wins |
| angle_rms_fit | 7.703 | 0.0974 | B wins |
| angle_min_fit | 108.4 | 119.9 | info |
| angle_max_fit | 126.7 | 120.1 | info |
| angle_rms_raw | 4.69 | 4.174 | B wins |
| angle_rms_raw_seed | 4.573 | 4.573 | tie |
| fit_rms_worst | 0.03097 | 0.0002866 | B wins |
| r1_fit | 0.7776 | 0.8013 | info |
| r2_fit | 1.201 | 1.202 | info |
| r3_fit | 2.535 | 2.396 | info |
| curv_resid | 0.05892 | 0.0014 | B wins |
| p1 | 2.498 | 2.502 | info |
| p2 | 1.657 | 1.668 | info |
| dp_err | 0.007387 | 0.001097 | B wins |
| pressure_rel_err_worst | 0.005633 | 0.0008104 | B wins |
| vol_drift_max | 8.002e-13 | 9.468e-13 | tie |
| vol_drift_pre_max | 2.47e-05 | 0.000105 | A wins |
| area_max_rise | -4.698e-07 | -5.335e-11 | A wins |
| groom_max_rise | 0 | 0 | tie |
| iters | 1800 | 1800 | tie |

TOTAL: B wins 19, A wins 7, ties 15
