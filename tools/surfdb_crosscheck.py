# Cross-check every record against the local Ferreol mirror.
#
#     python tools/surfdb_crosscheck.py          # run and write results
#     python tools/surfdb_crosscheck.py -v       # list the disagreements
#
# SEPARATE FROM THE BUILD, deliberately and for the same reason
# data/polyhedra keeps its `crosscheck` stage separate: it depends only on
# what is in the source mirror, so more sources can be cached and the
# verification re-run without rebuilding any geometry.
#
# Disagreements are RECORDED, not reconciled. polydb reports two records
# that disagree with McCooey rather than forcing them to agree; the same
# standard applies here.

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from surfdb import crosscheck, sources  # noqa: E402

DB = os.path.join(ROOT, "data", "surfaces", "surfaces")


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    if not sources.mirror_present():
        print("mirror not mounted at %s -- nothing to check against"
              % sources.SURFACES)
        return 0

    records, paths = {}, {}
    for dp, _d, fs in os.walk(DB):
        for f in fs:
            if not f.endswith(".json"):
                continue
            p = os.path.join(dp, f)
            with open(p, encoding="utf-8") as fh:
                rec = json.load(fh)
            records[rec["slug"]] = rec
            paths[rec["slug"]] = p

    checked, agree, disagree, skipped = crosscheck.run(records, verbose=verbose)

    for slug, rec in records.items():
        if (rec.get("provenance") or {}).get("cross_checked"):
            with open(paths[slug], "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2, ensure_ascii=False)
                fh.write("\n")

    print("cross-checked %d records against the Ferreol mirror" % checked)
    print("  %d agreements, %d disagreements" % (agree, disagree))
    print("  %d records skipped (no resolvable mathcurve id, or the page "
          "states nothing comparable)" % skipped)
    if disagree and not verbose:
        print("  re-run with -v to list the disagreements")
    return 0


sys.exit(main())
