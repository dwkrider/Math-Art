# A/B: baseline vs tp_l2

### tp_circle   time 0.22s -> 0.17s (x0.76)
_baseline effective: iters=5, precondition=True, length_mode=edge, alpha=3.0, beta=6.0, iters_run=0, converged=True_  
_tp_l2 effective: iters=5, precondition=False, length_mode=edge, alpha=3.0, beta=6.0, iters_run=0, converged=True_  
| metric | baseline | tp_l2 | verdict |
|---|---|---|---|
| E_rel_err_n32 | 0.03436 | 0.03436 | tie |
| E_rel_err_n256 | 0.003956 | 0.003956 | tie |
| conv_order | 1.018 | 1.018 | tie |
| radius_cv | 7.008e-17 | 7.008e-17 | tie |
| viol_max | 0 | 0 | tie |
| rise_max | 0 | 0 | tie |

### tp_unknot   time 26.86s -> 29.42s (x1.10)
_baseline effective: iters=150, precondition=True, length_mode=edge, alpha=3.0, beta=6.0, iters_run=150, converged=False_  
_tp_l2 effective: iters=150, precondition=False, length_mode=edge, alpha=3.0, beta=6.0, iters_run=150, converged=False_  
| metric | baseline | tp_l2 | verdict |
|---|---|---|---|
| alex_before | 1 | 1 | tie |
| alex_after | 1 | 1 | tie |
| alex_preserved | 1 | 1 | tie |
| radius_cv | 0.001604 | 0.3205 | A wins |
| E_final | 0.7597 | 120.4 | A wins |
| E_over_circle | 1 | 158.4 | A wins |
| min_far_gap | 1.723 | 0.7639 | A wins |
| viol_max | 9.669e-07 | 9.832e-07 | A wins |
| rise_max | 0 | 0 | tie |
| iters | 150 | 150 | tie |

### tp_trefoil   time 17.15s -> 138.06s (x8.05)
_baseline effective: iters=300, precondition=True, length_mode=edge, alpha=3.0, beta=6.0, iters_run=57, converged=True_  
_tp_l2 effective: iters=300, precondition=False, length_mode=edge, alpha=3.0, beta=6.0, iters_run=300, converged=False_  
| metric | baseline | tp_l2 | verdict |
|---|---|---|---|
| alex_before | 91 | 91 | tie |
| alex_after | 91 | 91 | tie |
| alex_preserved | 1 | 1 | tie |
| ropelength_gm | 39.67 | 45.22 | A wins |
| thickness_gm | 0.4901 | 0.43 | A wins |
| E_final | 47.18 | 55.87 | A wins |
| min_far_gap | 0.6894 | 0.6885 | A wins |
| viol_max | 8.339e-07 | 9.96e-07 | A wins |
| rise_max | 0 | 0 | tie |
| iters | 57 | 300 | A wins |

TOTAL: B wins 0, A wins 11, ties 15
