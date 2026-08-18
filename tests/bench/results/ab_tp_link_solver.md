# A/B: baseline vs tp_link_lagged

### tp_hopf   time 6.40s -> 6.33s (x0.99)
_baseline effective: iters=150, solver=dense, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=32, converged=True, factorizations=33_  
_tp_link_lagged effective: iters=150, solver=lagged, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=150, converged=False, factorizations=16_  
| metric | baseline | tp_link_lagged | verdict |
|---|---|---|---|
| lk_max_abs_before | 1 | 1 | tie |
| lk_max_abs_after | 1 | 1 | tie |
| lk_preserved | 1 | 1 | tie |
| lk_dev_max | 0.001029 | 0.001029 | tie |
| alex_link_before | 9 | 9 | tie |
| alex_link_after | 9 | 9 | tie |
| alex_link_preserved | 1 | 1 | tie |
| percomp_alex_preserved | 1 | 1 | tie |
| inter_gap_min | 0.3269 | 0.3269 | tie |
| inter_gap_final | 0.7349 | 0.7349 | tie |
| viol_max | 3.439e-07 | 9.954e-07 | A wins |
| rise_max | 0 | 0.03861 | A wins |
| E_final | 37.65 | 37.65 | tie |
| iters | 32 | 150 | A wins |
| plane_angle_err_deg | 0 | 1.421e-14 | tie |
| planarity_max | 1.921e-08 | 1.63e-07 | A wins |
| radius_cv_max | 0.0119 | 0.0119 | B wins |
| d_over_R | 1.229 | 1.229 | tie |
| d_star_family | 1.253 | 1.253 | tie |
| d_vs_family_err | 0.02374 | 0.02374 | A wins |
| E_over_family | 0.9989 | 0.9989 | tie |

### tp_whitehead   time 34.52s -> 13.83s (x0.40)
_baseline effective: iters=150, solver=dense, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=150, converged=False, factorizations=150_  
_tp_link_lagged effective: iters=150, solver=lagged, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=150, converged=False, factorizations=32_  
| metric | baseline | tp_link_lagged | verdict |
|---|---|---|---|
| lk_max_abs_before | 0 | 0 | tie |
| lk_max_abs_after | 0 | 0 | tie |
| lk_preserved | 1 | 1 | tie |
| lk_dev_max | 0.002204 | 0.001365 | B wins |
| alex_link_before | 729 | 729 | tie |
| alex_link_after | 729 | 729 | tie |
| alex_link_preserved | 1 | 1 | tie |
| percomp_alex_preserved | 1 | 1 | tie |
| inter_gap_min | 0.2501 | 0.2501 | tie |
| inter_gap_final | 0.6541 | 0.7304 | B wins |
| viol_max | 9.942e-07 | 9.953e-07 | A wins |
| rise_max | 0.0007759 | 0.157 | A wins |
| E_final | 92.39 | 85.71 | B wins |
| iters | 150 | 150 | tie |
| ropelength_gm | 91.19 | 81.67 | B wins |
| thickness_gm | 0.3272 | 0.3653 | B wins |

### tp_borromean   time 45.33s -> 13.42s (x0.30)
_baseline effective: iters=150, solver=dense, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=150, converged=False, factorizations=150_  
_tp_link_lagged effective: iters=150, solver=lagged, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=150, converged=False, factorizations=23_  
| metric | baseline | tp_link_lagged | verdict |
|---|---|---|---|
| lk_max_abs_before | 0 | 0 | tie |
| lk_max_abs_after | 0 | 0 | tie |
| lk_preserved | 1 | 1 | tie |
| lk_dev_max | 9.277e-05 | 9.277e-05 | tie |
| alex_link_before | 6561 | 6561 | tie |
| alex_link_after | 6561 | 6561 | tie |
| alex_link_preserved | 1 | 1 | tie |
| percomp_alex_preserved | 1 | 1 | tie |
| inter_gap_min | 0.3007 | 0.3007 | tie |
| inter_gap_final | 0.9461 | 0.9462 | B wins |
| viol_max | 9.694e-07 | 9.994e-07 | A wins |
| rise_max | 0.008509 | 0.2742 | A wins |
| E_final | 116.9 | 116.9 | tie |
| iters | 150 | 150 | tie |
| ropelength_gm | 64.61 | 64.61 | B wins |
| thickness_gm | 0.4731 | 0.4731 | B wins |

### tp_chain3   time 5.48s -> 5.47s (x1.00)
_baseline effective: iters=150, solver=dense, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=18, converged=True, factorizations=19_  
_tp_link_lagged effective: iters=150, solver=lagged, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=106, converged=True, factorizations=16_  
| metric | baseline | tp_link_lagged | verdict |
|---|---|---|---|
| lk_max_abs_before | 1 | 1 | tie |
| lk_max_abs_after | 1 | 1 | tie |
| lk_preserved | 1 | 1 | tie |
| lk_dev_max | 0.001651 | 0.001651 | tie |
| alex_link_before | 81 | 81 | tie |
| alex_link_after | 81 | 81 | tie |
| alex_link_preserved | 1 | 1 | tie |
| percomp_alex_preserved | 1 | 1 | tie |
| inter_gap_min | 0.4745 | 0.4745 | tie |
| inter_gap_final | 0.6252 | 0.6252 | tie |
| viol_max | 2.812e-07 | 9.455e-07 | A wins |
| rise_max | 0 | 0.005266 | A wins |
| E_final | 75.48 | 75.48 | tie |
| iters | 18 | 106 | A wins |
| lk_adjacent_ok | 1 | 1 | tie |
| lk_ends_zero | 1 | 1 | tie |

TOTAL: B wins 9, A wins 12, ties 48
