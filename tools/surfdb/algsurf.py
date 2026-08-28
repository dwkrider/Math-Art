"""PLACEHOLDER -- filled in by the algsurf pass.

Contract identical to ferreol.py / vmm.py:

  records() -> {slug: spec}    new records, spec shaped as in tail.py
  ids()     -> {slug: {"algsurf": "chNNN_stem", ...}}
"""


def records():
    return {}


def ids():
    return {}


def _selftest():
    print("RESULT: OK  (surfdb.algsurf, %d records, %d id sets)"
          % (len(records()), len(ids())))
