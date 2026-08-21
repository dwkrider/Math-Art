#!/usr/bin/env bash
# Populate the polyhedron-database source cache.
#
# Published tables are used as a VERIFICATION ORACLE and as a source for exact
# scalar closed forms, which are mathematical facts. The database's stored
# vertex tables remain this repository's own derivation, so no third party's
# compilation is republished. See data/polyhedra/README.md.
#
# Deliberately polite: one request at a time, a pause between them, and
# everything cached so a rebuild never re-hits the servers. Already-downloaded
# files are skipped.
#
#   bash tools/polydb_fetch.sh mccooey Cube Dodecahedron ...
#   bash tools/polydb_fetch.sh netlib 0 1 2 ...
#   bash tools/polydb_fetch.sh index
#
# Requires the Git Bash curl (it handles the servers' content encoding; the
# Windows system curl does not).

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$ROOT/.polydb_cache"
PAUSE="${POLYDB_PAUSE:-0.4}"

get() {   # get <url> <destination>
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then return 0; fi
  mkdir -p "$(dirname "$dest")"
  if curl -sL --compressed -m 30 -o "$dest.part" "$url"; then
    if [ -s "$dest.part" ] && ! head -c 200 "$dest.part" | grep -qi "<html"; then
      mv "$dest.part" "$dest"
      sleep "$PAUSE"
      return 0
    fi
  fi
  rm -f "$dest.part"
  return 1
}

mode="${1:-}"; shift || true
ok=0; fail=0

case "$mode" in
  mccooey)
    for stem in "$@"; do
      if get "https://dmccooey.com/polyhedra/$stem.txt" "$CACHE/mccooey/$stem.txt"
      then ok=$((ok+1)); else fail=$((fail+1)); echo "  MISS $stem"; fi
    done
    ;;
  netlib)
    for n in "$@"; do
      if get "https://www.netlib.org/polyhedra/$n" "$CACHE/netlib/$n.txt"
      then ok=$((ok+1)); else fail=$((fail+1)); echo "  MISS netlib/$n"; fi
    done
    ;;
  index)
    get "https://dmccooey.com/polyhedra/" "$CACHE/mccooey_index.html" \
      && ok=1 || fail=1
    ;;
  *)
    echo "usage: $0 {mccooey <stem>... | netlib <num>... | index}" >&2
    exit 2
    ;;
esac

echo "cached: $ok ok, $fail missing  ->  $CACHE"
