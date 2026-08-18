# A/B: baseline vs lbfgs

### bubble_single   time 1.22s -> 0.73s (x0.60)
_baseline effective: iters=300, iters_run=300, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=cg_  
_lbfgs effective: iters=300, iters_run=41, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs_  
| metric | baseline | lbfgs | verdict |
|---|---|---|---|
| area | 12.58 | 12.58 | tie |
| area_rel_excess | 0.000957 | 0.0009569 | B wins |
| area_rel_excess_ref | 0.0009864 | 0.0009864 | tie |
| radius_cv | 0.0002993 | 0.0002891 | B wins |
| radius_max_rel_dev | 0.001498 | 0.001433 | B wins |
| pressure | 2.002 | 2.002 | tie |
| pressure_rel_err | 0.0009569 | 0.0009569 | B wins |
| vol_drift_max | 9.79e-13 | 8.407e-13 | tie |
| vol_drift_pre_max | 0.01687 | 0.02403 | A wins |
| area_max_rise | -3.517e-09 | -2.698e-12 | A wins |
| groom_max_rise | 0 | 0 | tie |
| iters | 300 | 41 | B wins |

### bubble_double   time 18.02s -> 7.04s (x0.39)
_baseline effective: iters=1200, iters_run=1200, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=cg_  
_lbfgs effective: iters=1200, iters_run=90, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs_  
| metric | baseline | lbfgs | verdict |
|---|---|---|---|
| area | 21.22 | 21.22 | B wins |
| area_analytic | 21.21 | 21.21 | tie |
| area_rel_excess | 0.0007271 | 0.0005647 | B wins |
| angle_rms_fit | 0.2534 | 0.05501 | B wins |
| angle_min_fit | 119.8 | 120 | info |
| angle_max_fit | 120.4 | 120.1 | info |
| angle_rms_raw | 4.418 | 3.027 | B wins |
| angle_rms_raw_seed | 4.784 | 4.784 | tie |
| fit_rms_worst | 0.0005121 | 0.0003696 | B wins |
| r1_fit | 1.003 | 1.002 | info |
| r2_fit | 1.003 | 1.002 | info |
| r3_fit | None | 3.762e+05 | info |
| curv_resid | 2.306e-06 | 2.513e-06 | A wins |
| p1 | 2.001 | 2.001 | info |
| p2 | 2.001 | 2.001 | info |
| dp_err | 5.781e-05 | 6.504e-07 | B wins |
| pressure_rel_err_worst | 0.0006778 | 0.0005649 | B wins |
| vol_drift_max | 8.77e-13 | 7.726e-13 | tie |
| vol_drift_pre_max | 5.393e-05 | 0.001531 | A wins |
| area_max_rise | -6.52e-09 | -5.435e-12 | A wins |
| groom_max_rise | 0 | 0 | tie |
| iters | 1200 | 90 | B wins |

### bubble_double_unequal   time 29.09s -> 19.73s (x0.68)
_baseline effective: iters=1800, iters_run=1800, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=cg_  
_lbfgs effective: iters=1800, iters_run=228, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs_  
| metric | baseline | lbfgs | verdict |
|---|---|---|---|
| area | 22.68 | 22.67 | B wins |
| area_analytic | 22.66 | 22.66 | tie |
| area_rel_excess | 0.0007181 | 0.0004921 | B wins |
| angle_rms_fit | 0.0974 | 0.0401 | B wins |
| angle_min_fit | 119.9 | 120 | info |
| angle_max_fit | 120.1 | 120.1 | info |
| angle_rms_raw | 4.174 | 2.695 | B wins |
| angle_rms_raw_seed | 4.573 | 4.573 | tie |
| fit_rms_worst | 0.0002866 | 0.0004018 | A wins |
| r1_fit | 0.8013 | 0.8017 | info |
| r2_fit | 1.202 | 1.202 | info |
| r3_fit | 2.396 | 2.399 | info |
| curv_resid | 0.0014 | 0.001686 | A wins |
| p1 | 2.502 | 2.502 | info |
| p2 | 1.668 | 1.667 | info |
| dp_err | 0.001097 | 0.000852 | B wins |
| pressure_rel_err_worst | 0.0008104 | 0.0006369 | B wins |
| vol_drift_max | 9.468e-13 | 9.471e-13 | tie |
| vol_drift_pre_max | 0.000105 | 0.00276 | A wins |
| area_max_rise | -5.335e-11 | -1.877e-11 | A wins |
| groom_max_rise | 0 | 0 | tie |
| iters | 1800 | 228 | B wins |

### bubble_triple   time 52.45s -> 156.50s (x2.98)
_baseline effective: iters=1600, iters_run=1600, groom_every=4, grooms_run=399, mobility=star, cotan_mode=mollify, optimizer=cg_  
_lbfgs effective: iters=1600, iters_run=1600, groom_every=4, grooms_run=399, mobility=star, cotan_mode=mollify, optimizer=lbfgs_  
| metric | baseline | lbfgs | verdict |
|---|---|---|---|
| area | 27.94 | 27.94 | B wins |
| angle_rms_fit | 0.05664 | 0.04886 | B wins |
| angle_min_fit | 119.9 | 119.9 | info |
| angle_max_fit | 120.2 | 120.1 | info |
| angle_rms_raw | 2.236 | 1.913 | B wins |
| tetra_rms_fit | 0.08796 | 0.0736 | info |
| tetra_min_fit | 109.4 | 109.4 | info |
| tetra_max_fit | 109.6 | 109.5 | info |
| tetra_rms_raw | 3.944 | 3.675 | info |
| n_tetra_angles | 12 | 12 | tie |
| fit_rms_worst | 0.0003149 | 0.0003112 | B wins |
| p1 | 2.001 | 2.001 | info |
| p2 | 2.001 | 2.001 | info |
| p3 | 2.001 | 2.001 | info |
| yl_worst | 0.0001723 | 0.001382 | info |
| area_analytic | 27.93 | 27.93 | tie |
| area_rel_excess | 0.0003813 | 0.000376 | B wins |
| pressure_rel_err_worst | 0.000374 | 0.0004133 | A wins |
| vol_drift_max | 9.508e-13 | 7.346e-13 | tie |
| vol_drift_pre_max | 9.623e-07 | 0.000263 | A wins |
| area_max_rise | -2.403e-09 | -5.721e-09 | B wins |
| groom_max_rise | -2.4e-09 | -1.999e-11 | A wins |
| iters | 1600 | 1600 | tie |

### bubble_double_fine_hard   time 190.64s -> 19.47s (x0.10)
_baseline effective: iters=2400, iters_run=2400, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=cg_  
_lbfgs effective: iters=2400, iters_run=122, groom_every=0, grooms_run=0, mobility=star, cotan_mode=mollify, optimizer=lbfgs_  
| metric | baseline | lbfgs | verdict |
|---|---|---|---|
| area | 21.22 | 21.21 | B wins |
| area_analytic | 21.21 | 21.21 | tie |
| area_rel_excess | 0.0005381 | 0.0001372 | B wins |
| angle_rms_fit | 3.986 | 0.01184 | B wins |
| angle_min_fit | 114.2 | 120 | info |
| angle_max_fit | 122.9 | 120 | info |
| angle_rms_raw | 2.33 | 1.488 | B wins |
| angle_rms_raw_seed | 2.311 | 2.311 | tie |
| fit_rms_worst | 0.004934 | 9.653e-05 | B wins |
| r1_fit | 0.9916 | 1 | info |
| r2_fit | 0.9916 | 1 | info |
| r3_fit | None | 3.877e+04 | info |
| curv_resid | 1.389e-07 | 2.71e-05 | A wins |
| p1 | 1.993 | 2 | info |
| p2 | 1.993 | 2 | info |
| dp_err | 3.179e-05 | 1.069e-06 | B wins |
| pressure_rel_err_worst | 0.003271 | 0.000136 | B wins |
| vol_drift_max | 9.273e-13 | 8.914e-13 | tie |
| vol_drift_pre_max | 1.369e-06 | 0.001873 | A wins |
| area_max_rise | -1.443e-07 | -3.005e-11 | A wins |
| groom_max_rise | 0 | 0 | tie |
| iters | 2400 | 122 | B wins |

TOTAL: B wins 34, A wins 15, ties 21
