"""Pack the surface tiles into one sprite sheet for the clustered view.

    python tools/build_thumb_atlas.py

Writes web/thumbs/surfaces-atlas.png and web/thumbs/surfaces-atlas.json.

WHY.  The clustered view draws every surface at once, so it needs all 466
tiles before it can finish drawing. Fetched individually that is 466
requests and ~34 MB, and the field fills in visibly over several seconds.
The same tiles at the size the view actually draws them fit in a single
image of a couple of megabytes: one request, one decode, and the view is
complete the moment it arrives.

The grid keeps using the individual PNGs. It shows a few rows at a time
and lazy-loads them, so it wants the full-resolution tile and does not
want to pay for 465 it will never show.

The cell size is the atlas's only real parameter. It has to cover the
largest a tile is ever drawn in the clustered view -- 0.85*sqrt(area/n),
which for a small filtered set hits the 132 px clamp -- so 128 is the
honest choice, and a filtered handful is the one case where the view
falls back to the full-resolution PNGs anyway.

Transparency is preserved: the tiles are rendered against a hidden
backdrop precisely so they do not clip each other in the layout, and an
atlas that flattened them onto black would undo that.
"""
import json
import math
import os
import sys

try:
    from PIL import Image
except ImportError:                                          # pragma: no cover
    sys.stderr.write("Pillow is required: pip install pillow\n")
    raise SystemExit(1)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILES = os.path.join(PROJ, "web", "thumbs", "surfaces")
DB = os.path.join(PROJ, "data", "surfaces")
OUT_PNG = os.path.join(PROJ, "web", "thumbs", "surfaces-atlas.png")
OUT_JSON = os.path.join(PROJ, "web", "thumbs", "surfaces-atlas.json")

CELL = 128


def main():
    with open(os.path.join(DB, "index.json"), encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]

    slugs = []
    for e in entries:
        if os.path.exists(os.path.join(TILES, e["slug"] + ".png")):
            slugs.append(e["slug"])
    if not slugs:
        sys.stderr.write("no tiles in %s\n" % TILES)
        return 1

    # Square-ish sheet: fewer very long rows keeps the PNG's own filtering
    # working on something closer to 2-D locality.
    cols = int(math.ceil(math.sqrt(len(slugs))))
    rows = int(math.ceil(len(slugs) / cols))
    sheet = Image.new("RGBA", (cols * CELL, rows * CELL), (0, 0, 0, 0))

    index = {}
    for i, slug in enumerate(slugs):
        im = Image.open(os.path.join(TILES, slug + ".png")).convert("RGBA")
        if im.size != (CELL, CELL):
            im = im.resize((CELL, CELL), Image.LANCZOS)
        cx, cy = (i % cols) * CELL, (i // cols) * CELL
        sheet.paste(im, (cx, cy))
        index[slug] = [cx, cy]
        im.close()

    sheet.save(OUT_PNG, optimize=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"cell": CELL, "cols": cols, "rows": rows,
                   "tiles": index}, fh, separators=(",", ":"))

    png = os.path.getsize(OUT_PNG)
    loose = sum(os.path.getsize(os.path.join(TILES, s + ".png"))
                for s in slugs)
    print("atlas: %d tiles, %dx%d cells of %dpx -> %s"
          % (len(slugs), cols, rows, CELL,
             os.path.relpath(OUT_PNG, PROJ)))
    print("  %.1f MB in one request, against %.1f MB in %d"
          % (png / 1e6, loose / 1e6, len(slugs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
