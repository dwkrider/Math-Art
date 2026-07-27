"""Insert a '## Variants' gallery into each generator doc page that has
rendered option variants.

Reads docs/images/variants/_manifest.json (slug -> [[variant_id, label], ...]),
and for each slug inserts a thumbnail grid into docs/generators/<slug>.md,
placed after the '## Options' section and before '## How it works'
(falling back to before '## References', else appended). Idempotent: any
existing '## Variants' block is replaced.

Run:  python docs/insert_variants.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "images", "variants", "_manifest.json")
GEN = os.path.join(HERE, "generators")
PER_ROW = 3
THUMB = 200


def _grid(slug, variants):
    """One <table> grid for a flat list of [vid, label, ...] entries."""
    lines = ["<table>"]
    for i in range(0, len(variants), PER_ROW):
        lines.append("<tr>")
        for v in variants[i:i + PER_ROW]:
            vid, label = v[0], v[1]
            img = f"../images/variants/{slug}__{vid}.png"
            lines.append(
                f'<td align="center"><img src="{img}" width="{THUMB}">'
                f'<br><sub>{label}</sub></td>')
        lines.append("</tr>")
    lines.append("</table>")
    return lines


def build_section(slug, variants):
    lines = ["## Variants", "",
             "Renders of each selectable option:", ""]
    # grouped when entries carry a third field (the group label)
    grouped = any(len(v) > 2 for v in variants)
    if grouped:
        order, byg = [], {}
        for v in variants:
            g = v[2] if len(v) > 2 else "Other"
            if g not in byg:
                byg[g] = []
                order.append(g)
            byg[g].append(v)
        for g in order:
            lines.append(f"### {g}")
            lines.append("")
            lines.extend(_grid(slug, byg[g]))
            lines.append("")
    else:
        lines.extend(_grid(slug, variants))
        lines.append("")
    return "\n".join(lines)


def strip_existing(text):
    # Remove a prior '## Variants' block, INCLUDING the blank lines
    # right before it, up to the next '## ' heading (or end of file),
    # collapsing to a single '\n'. Consuming the leading blank lines
    # (\n+) is what keeps this idempotent -- otherwise each re-run
    # left one extra blank line before the heading.
    return re.sub(r"\n+## Variants\b.*?(?=\n## |\Z)", "\n", text,
                  flags=re.DOTALL)


def insert(text, section):
    for anchor in ("\n## How it works", "\n## References"):
        idx = text.find(anchor)
        if idx != -1:
            return text[:idx] + "\n" + section + "\n" + text[idx + 1:]
    return text.rstrip() + "\n\n" + section + "\n"


def main():
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    changed = 0
    for slug, variants in manifest.items():
        path = os.path.join(GEN, slug + ".md")
        if not os.path.exists(path):
            print("SKIP no doc page:", slug)
            continue
        # only keep variants whose image actually exists
        variants = [v for v in variants if os.path.exists(
            os.path.join(HERE, "images", "variants",
                         f"{slug}__{v[0]}.png"))]
        if not variants:
            print("SKIP no images:", slug)
            continue
        text = open(path, encoding="utf-8").read()
        text = strip_existing(text)
        text = insert(text, build_section(slug, variants))
        open(path, "w", encoding="utf-8").write(text)
        print(f"OK {slug} ({len(variants)} variants)")
        changed += 1
    print(f"DONE {changed} pages updated")


if __name__ == "__main__":
    main()
