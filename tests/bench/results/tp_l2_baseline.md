# Bench results: tp_l2_baseline

## tp_circle

time: 0.217 s, n_verts=128  
effective: iters=5, precondition=True, length_mode=edge, alpha=3.0, beta=6.0, iters_run=0, converged=True

| metric | value |
|---|---|
| E_rel_err_n32 | 0.03436 |
| E_rel_err_n256 | 0.003956 |
| conv_order | 1.018 |
| radius_cv | 7.008e-17 |
| viol_max | 0 |
| rise_max | 0 |

## tp_unknot

time: 26.860 s, n_verts=160  
effective: iters=150, precondition=True, length_mode=edge, alpha=3.0, beta=6.0, iters_run=150, converged=False

| metric | value |
|---|---|
| alex_before | 1 |
| alex_after | 1 |
| alex_preserved | 1 |
| radius_cv | 0.001604 |
| E_final | 0.7597 |
| E_over_circle | 1 |
| min_far_gap | 1.723 |
| viol_max | 9.669e-07 |
| rise_max | 0 |
| iters | 150 |

## tp_trefoil

time: 17.145 s, n_verts=192  
effective: iters=300, precondition=True, length_mode=edge, alpha=3.0, beta=6.0, iters_run=57, converged=True

| metric | value |
|---|---|
| alex_before | 91 |
| alex_after | 91 |
| alex_preserved | 1 |
| ropelength_gm | 39.67 |
| thickness_gm | 0.4901 |
| E_final | 47.18 |
| min_far_gap | 0.6894 |
| viol_max | 8.339e-07 |
| rise_max | 0 |
| iters | 57 |

