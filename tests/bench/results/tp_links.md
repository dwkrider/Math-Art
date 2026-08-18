# Bench results: tp_links

## tp_hopf

time: 6.412 s, n_verts=160  
effective: iters=150, solver=dense, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=32, converged=True, factorizations=33

| metric | value |
|---|---|
| lk_max_abs_before | 1 |
| lk_max_abs_after | 1 |
| lk_preserved | 1 |
| lk_dev_max | 0.001029 |
| alex_link_before | 9 |
| alex_link_after | 9 |
| alex_link_preserved | 1 |
| percomp_alex_preserved | 1 |
| inter_gap_min | 0.3269 |
| inter_gap_final | 0.7349 |
| viol_max | 3.439e-07 |
| rise_max | 0 |
| E_final | 37.65 |
| iters | 32 |
| plane_angle_err_deg | 0 |
| planarity_max | 1.921e-08 |
| radius_cv_max | 0.0119 |
| d_over_R | 1.229 |
| d_star_family | 1.253 |
| d_vs_family_err | 0.02374 |
| E_over_family | 0.9989 |

## tp_whitehead

time: 34.128 s, n_verts=200  
effective: iters=150, solver=dense, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=150, converged=False, factorizations=150

| metric | value |
|---|---|
| lk_max_abs_before | 0 |
| lk_max_abs_after | 0 |
| lk_preserved | 1 |
| lk_dev_max | 0.002204 |
| alex_link_before | 729 |
| alex_link_after | 729 |
| alex_link_preserved | 1 |
| percomp_alex_preserved | 1 |
| inter_gap_min | 0.2501 |
| inter_gap_final | 0.6541 |
| viol_max | 9.942e-07 |
| rise_max | 0.0007759 |
| E_final | 92.39 |
| iters | 150 |
| ropelength_gm | 91.19 |
| thickness_gm | 0.3272 |

## tp_borromean

time: 44.786 s, n_verts=252  
effective: iters=150, solver=dense, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=150, converged=False, factorizations=150

| metric | value |
|---|---|
| lk_max_abs_before | 0 |
| lk_max_abs_after | 0 |
| lk_preserved | 1 |
| lk_dev_max | 9.277e-05 |
| alex_link_before | 6561 |
| alex_link_after | 6561 |
| alex_link_preserved | 1 |
| percomp_alex_preserved | 1 |
| inter_gap_min | 0.3007 |
| inter_gap_final | 0.9461 |
| viol_max | 9.694e-07 |
| rise_max | 0.008509 |
| E_final | 116.9 |
| iters | 150 |
| ropelength_gm | 64.61 |
| thickness_gm | 0.4731 |

## tp_chain3

time: 3.640 s, n_verts=192  
effective: iters=150, solver=dense, length_mode=edge, precondition=True, refresh_every=10, refresh_drift=1.0, iters_run=18, converged=True, factorizations=19

| metric | value |
|---|---|
| lk_max_abs_before | 1 |
| lk_max_abs_after | 1 |
| lk_preserved | 1 |
| lk_dev_max | 0.001651 |
| alex_link_before | 81 |
| alex_link_after | 81 |
| alex_link_preserved | 1 |
| percomp_alex_preserved | 1 |
| inter_gap_min | 0.4745 |
| inter_gap_final | 0.6252 |
| viol_max | 2.812e-07 |
| rise_max | 0 |
| E_final | 75.48 |
| iters | 18 |
| lk_adjacent_ok | 1 |
| lk_ends_zero | 1 |

## tp_scale

time: 35.103 s, n_verts=192  
effective: ns=[200, 400, 800], iters_agreement=80

| metric | value |
|---|---|
| t_iter_dense_200 | 0.2117 |
| t_iter_lagged_200 | 0.06462 |
| factorizations_lagged_200 | 2 |
| t_iter_dense_400 | 0.4334 |
| t_iter_lagged_400 | 0.1922 |
| factorizations_lagged_400 | 3 |
| t_iter_dense_800 | 1.234 |
| t_iter_lagged_800 | 0.7118 |
| factorizations_lagged_800 | 4 |
| exp_dense | 1.272 |
| exp_lagged | 1.731 |
| speedup_800 | 1.734 |
| E_dense_192 | 47.18 |
| E_lagged_192 | 47.18 |
| E_rel_diff_192 | 2.408e-06 |
| alex_preserved | 1 |
| speedup_192_wall | 3.65 |

