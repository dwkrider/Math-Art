# Bench results: tp_l2_tp_l2

## tp_circle

time: 0.165 s, n_verts=128  
effective: iters=5, precondition=False, length_mode=edge, alpha=3.0, beta=6.0, iters_run=0, converged=True

| metric | value |
|---|---|
| E_rel_err_n32 | 0.03436 |
| E_rel_err_n256 | 0.003956 |
| conv_order | 1.018 |
| radius_cv | 7.008e-17 |
| viol_max | 0 |
| rise_max | 0 |

## tp_unknot

time: 29.419 s, n_verts=160  
effective: iters=150, precondition=False, length_mode=edge, alpha=3.0, beta=6.0, iters_run=150, converged=False

| metric | value |
|---|---|
| alex_before | 1 |
| alex_after | 1 |
| alex_preserved | 1 |
| radius_cv | 0.3205 |
| E_final | 120.4 |
| E_over_circle | 158.4 |
| min_far_gap | 0.7639 |
| viol_max | 9.832e-07 |
| rise_max | 0 |
| iters | 150 |

## tp_trefoil

time: 138.060 s, n_verts=192  
effective: iters=300, precondition=False, length_mode=edge, alpha=3.0, beta=6.0, iters_run=300, converged=False

| metric | value |
|---|---|
| alex_before | 91 |
| alex_after | 91 |
| alex_preserved | 1 |
| ropelength_gm | 45.22 |
| thickness_gm | 0.43 |
| E_final | 55.87 |
| min_far_gap | 0.6885 |
| viol_max | 9.96e-07 |
| rise_max | 0 |
| iters | 300 |

