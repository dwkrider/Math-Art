# Committed benchmark results — provenance

Every file here is reproducible from the committed code and the committed
configs (`tests/bench/configs/`).  Quality metrics are deterministic;
`time_s`/`t` fields are wall-clock noise, quoted only as ratios.  Run all
commands from the repo/worktree root.

| artifact(s) | command |
|---|---|
| `baseline.*` | `python tests/bench/run.py --config tests/bench/configs/old_defaults.json --tag baseline` |
| `post.*` | `python tests/bench/run.py --tag post` |
| `ab_plateau_mollify.md`, `plateau_mollify_*` | `python tests/bench/run.py catenoid catenoid_fine seifert_span_q3 seifert_span_q5 --ab tests/bench/configs/old_defaults.json tests/bench/configs/mollify.json --tag plateau_mollify` |
| `ab_plateau_groom.md`, `plateau_groom_*` | `python tests/bench/run.py catenoid catenoid_fine seifert_span_q3 seifert_span_q5 --ab tests/bench/configs/mollify.json tests/bench/configs/mollify_groom.json --tag plateau_groom` |
| `ab_poly_ls.md`, `poly_ls_*` | `python tests/bench/run.py canonical biscribe --ab tests/bench/configs/old_defaults.json tests/bench/configs/poly_ls.json --tag poly_ls` |
| `ab_fair_mollify.md`, `fair_mollify_*` | `python tests/bench/run.py seifert_fair --ab tests/bench/configs/old_defaults.json tests/bench/configs/fair_mollify.json --tag fair_mollify` |
| `sweep_groom.*` | `python tests/bench/run.py seifert_sweep --config tests/bench/configs/mollify_groom.json --tag sweep_groom` |
| `bubble.*` | `python tests/bench/run.py bubble_single bubble_single_fine bubble_double bubble_double_unequal bubble_double_fine --tag bubble` |
| `ab_bubble_cg.md`, `bubble_cg_*` | `python tests/bench/run.py bubble_single bubble_double bubble_double_unequal --ab tests/bench/configs/bubble_nocg.json tests/bench/configs/baseline.json --tag bubble_cg` |
| `ab_bubble_groom.md`, `bubble_groom_*` | `python tests/bench/run.py bubble_single bubble_double bubble_double_unequal --ab tests/bench/configs/baseline.json tests/bench/configs/bubble_groom.json --tag bubble_groom` |
| `cmc.*` | `python tests/bench/run.py cmc_bridge_cyl cmc_bridge_fat cmc_bridge_thin cmc_bridge_fat_fine cmc_catenoid cmc_catenoid_fine cmc_drop45 cmc_drop135 cmc_drop45_fine --tag cmc` |
| `grav.*` | `python tests/bench/run.py cmc_drop_grav_tiny cmc_drop_grav cmc_drop_grav_fine cmc_puddle_sweep --tag grav` |
| `films.*` | `python tests/bench/run.py film_sphere_eq film_sphere_eq_fine film_sphere_off film_sphere_off_fine film_cyl_disk film_cyl_disk_fine film_cyl_tilt film_cyl_tilt_fine --tag films` |
| `triple.*` | `python tests/bench/run.py bubble_triple bubble_triple_fine bubble_triple_unequal --tag triple` |

Config semantics: `{}` / `baseline.json` = **current defaults** (they moved
when measured winners shipped); `old_defaults.json` = the pre-branch
(master @ 3873571) behaviour, kept A/B-able forever.  A/B columns are
labelled by config-file basename, and each side's `effective:` line in the
tables records the flags that actually ran (`grooms_run` > 0 proves the
groom cycle fired) — added after a stale artifact was found reporting a
groomed config whose grooming had been silently disabled by a tracer bug.

Known reproduction tolerance: in `baseline.*`, the biscribe residual
spreads (`r/f_spread_worst_ok`, scale ~1e-10) can differ from a fresh
old-defaults run in their 4th significant digit: `biscribe`'s *internal*
`canonicalize` init uses the current adaptive default (there is no knob to
re-create the pre-branch init), which shifts the fixed-step solve's start
point minutely.  Both values sit five orders of magnitude below the 1e-5
convergence criterion; every decision-bearing metric reproduces exactly.
`baseline.*` as committed IS the fresh old-defaults run against current
code, so re-runs match it bitwise.

History note: artifacts predating the final harness (including one whose
"mollify_groom" column had grooming silently off, and a "sweep_baseline"
generated after the default flip that actually measured mollify) were
removed and regenerated on 2026-08-16; no verdict changed.  See
`research/plans/solver-unification-bench-plan.md` §"Reconciliation".
