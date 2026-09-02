"""Find variant galleries whose renders are identical to one another.

A gallery exists to show what a generator's selector actually changes.
If several of its thumbnails are the same picture, the selector is not
reaching the geometry -- usually because another property (a `preset`,
a `mode`) overrides it -- and the grid silently claims a difference
that is not there.

Compare DECODED PIXELS, not file bytes.  Blender's PNGs of the very
same frame differ in their non-pixel chunks, so byte hashing reports
five identical polystix renders as five distinct files; every one of
them is pixel-for-pixel the same image.  Pixel equality is still exact
rather than perceptual, which is what we want: the rig is
deterministic, so two genuinely different shapes never collide.

Run:  python tools/check_variant_dupes.py
      python tools/check_variant_dupes.py --quiet    problems only

Exits non-zero if any gallery has a duplicate group.
"""
import collections
import hashlib
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = os.path.join(HERE, "docs", "images", "variants")


def pixel_digest(path):
    """Hash of the decoded RGB pixels, ignoring PNG metadata."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(a.tobytes()).hexdigest()


def main(argv):
    quiet = "--quiet" in argv
    by_slug = collections.defaultdict(dict)
    for name in sorted(os.listdir(VARIANTS)):
        if not name.endswith(".png") or "__" not in name:
            continue
        slug, _, vid = name[:-4].partition("__")
        by_slug[slug][vid] = pixel_digest(os.path.join(VARIANTS, name))

    bad = []
    for slug in sorted(by_slug):
        groups = collections.defaultdict(list)
        for vid, d in by_slug[slug].items():
            groups[d].append(vid)
        dupes = {d: sorted(v) for d, v in groups.items() if len(v) > 1}
        if dupes:
            bad.append((slug, len(by_slug[slug]), len(groups), dupes))

    for slug, n_files, distinct, dupes in bad:
        flag = "   <-- ALL IDENTICAL" if distinct == 1 else ""
        print("%-26s %3d renders, %3d distinct%s"
              % (slug, n_files, distinct, flag))
        if not quiet:
            for vids in sorted(dupes.values()):
                print("      identical: %s" % ", ".join(vids))

    involved = sum(len(v) for _, _, _, d in bad for v in d.values())
    print("\n%d gallery(ies) with duplicate renders, %d image(s) involved, "
          "out of %d galleries" % (len(bad), involved, len(by_slug)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
