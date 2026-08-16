# Solver benchmark runner.
#
#   python tests/bench/run.py                          # all cases, baseline
#   python tests/bench/run.py catenoid biscribe        # a subset
#   python tests/bench/run.py --tag baseline           # tag the results files
#   python tests/bench/run.py --config cfg.json        # opt-in solver flags
#   python tests/bench/run.py --ab a.json b.json case  # A/B delta table
#
# Results land in tests/bench/results/<tag>.json (machine-readable) and
# <tag>.md (readable table).  Quality metrics are deterministic; wall
# times are environment noise and recorded for ratio judgements only.
import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bench import cases as C
else:
    from . import cases as C

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_cases(names, config):
    out = {}
    for name in names:
        fn = C.CASES[name]
        print(f"[bench] {name} ...", flush=True)
        res = fn(dict(config))
        res["config"] = config
        out[name] = res
        mets = res["metrics"]
        brief = ", ".join(f"{k}={_fmt(v)}" for k, v in list(mets.items())[:6])
        print(f"[bench] {name}: {res['time_s']:.2f}s  {brief}", flush=True)
    return out


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_results(results, tag):
    os.makedirs(RESULTS, exist_ok=True)
    jpath = os.path.join(RESULTS, f"{tag}.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)
    mpath = os.path.join(RESULTS, f"{tag}.md")
    with open(mpath, "w") as f:
        f.write(f"# Bench results: {tag}\n\n")
        for name, res in results.items():
            f.write(f"## {name}\n\n")
            f.write(f"time: {res['time_s']:.3f} s")
            for k in ("n_verts", "n_tris"):
                if k in res:
                    f.write(f", {k}={res[k]}")
            f.write("\n\n| metric | value |\n|---|---|\n")
            for k, v in res["metrics"].items():
                f.write(f"| {k} | {_fmt(v)} |\n")
            if res.get("per_solid"):
                f.write("\n| solid | " + " | ".join(
                    next(iter(res["per_solid"].values())).keys()) + " |\n")
                f.write("|---" * (1 + len(next(iter(
                    res["per_solid"].values())))) + "|\n")
                for s, p in res["per_solid"].items():
                    f.write(f"| {s} | " + " | ".join(
                        _fmt(v) for v in p.values()) + " |\n")
            f.write("\n")
    print(f"[bench] wrote {jpath} and {mpath}")


def ab_table(res_a, res_b, label_a, label_b):
    """Print win/loss/tie per metric.  'Better' directions are encoded
    here; metrics not listed are informational only."""
    lower_better = {
        "area", "area_rel_err", "waist_rel_err", "H_rms", "H_max",
        "edge_cv", "valence_rms_dev", "clamp_frac", "neg_cot_frac",
        "selfx", "degenerate", "rim_max_move", "tangent_spread_worst",
        "tangent_spread_mean", "planarity_worst", "iters_mean",
        "iters_mean_ok", "r_spread_worst_ok", "f_spread_worst_ok",
        "ropelength_proxy", "turning_rms_deg", "edge_err_rms",
        "edge_err_max", "planar_dev", "crossings", "area_shrink_frac",
        "iters",
    }
    higher_better = {
        "q_min", "q_mean", "q_p05", "min_angle_deg", "min_far_gap",
        "n_correct", "genus_preserved", "lk_preserved", "aspect",
        "inter_component_gap", "z_extent",
    }
    lines = []
    wins = losses = ties = 0
    for case in res_a:
        if case not in res_b:
            continue
        ma, mb = res_a[case]["metrics"], res_b[case]["metrics"]
        ta, tb = res_a[case]["time_s"], res_b[case]["time_s"]
        lines.append(f"\n### {case}   time {ta:.2f}s -> {tb:.2f}s "
                     f"(x{tb / max(ta, 1e-9):.2f})")
        lines.append(f"| metric | {label_a} | {label_b} | verdict |")
        lines.append("|---|---|---|---|")
        for k in ma:
            if k not in mb:
                continue
            va, vb = ma[k], mb[k]
            verdict = "info"
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                if abs(float(va) - float(vb)) <= 1e-12 + 1e-6 * abs(float(va)):
                    verdict = "tie"
                elif k in lower_better:
                    verdict = "B wins" if vb < va else "A wins"
                elif k in higher_better:
                    verdict = "B wins" if vb > va else "A wins"
            if verdict == "B wins":
                wins += 1
            elif verdict == "A wins":
                losses += 1
            elif verdict == "tie":
                ties += 1
            lines.append(f"| {k} | {_fmt(va)} | {_fmt(vb)} | {verdict} |")
    lines.append(f"\nTOTAL: B wins {wins}, A wins {losses}, ties {ties}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=[])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--config", default=None,
                    help="JSON file of solver flags")
    ap.add_argument("--ab", nargs=2, metavar=("CFG_A", "CFG_B"),
                    help="run twice with two config files, print deltas")
    args = ap.parse_args()

    names = args.cases or list(C.CASES.keys())
    for n in names:
        if n not in C.CASES:
            ap.error(f"unknown case {n!r}; have {sorted(C.CASES)}")

    if args.ab:
        cfg_a = json.load(open(args.ab[0]))
        cfg_b = json.load(open(args.ab[1]))
        la = os.path.splitext(os.path.basename(args.ab[0]))[0]
        lb = os.path.splitext(os.path.basename(args.ab[1]))[0]
        res_a = run_cases(names, cfg_a)
        res_b = run_cases(names, cfg_b)
        table = ab_table(res_a, res_b, la, lb)
        print(table)
        os.makedirs(RESULTS, exist_ok=True)
        path = os.path.join(RESULTS, f"ab_{args.tag}.md")
        with open(path, "w") as f:
            f.write(f"# A/B: {la} vs {lb}\n{table}\n")
        write_results(res_a, f"{args.tag}_{la}")
        write_results(res_b, f"{args.tag}_{lb}")
        print(f"[bench] wrote {path}")
        return

    config = json.load(open(args.config)) if args.config else {}
    results = run_cases(names, config)
    write_results(results, args.tag)


if __name__ == "__main__":
    main()
