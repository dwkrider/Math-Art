"""Display names for the surface registries.

One home for the rule that turns a registry label into the name a
person reads.  The Add-menu shows these, and `tools/surfdb_build.py`
writes the same string into each database record's `name`, so the
catalogue and the UI cannot drift apart -- they did: the database
carried raw dict keys (`DISPHENOID_FAMILY_A_GENUS_31 (exact)`) for
forty-six surfaces that the menu had been showing correctly as
`Disphenoid 31` all along, because the builder synthesised a name from
the key instead of reading the label sitting next to it.

The rule itself: a registry label may carry parenthetical notes, and
some of those say how a surface was BUILT rather than what it is --
`(Evolver cell)`, `(nodal approximation)`, `(exact)` -- or repeat a
lattice word the record already stores as structured data.  Those are
provenance, not name, so they come out.  Mathematical qualifiers
stay: `(genus 4)`, `(square-torus member)`, `(a=0.1 b=0.3 member)`
distinguish one surface from another and belong in the name.
"""

import re

#: Parenthetical parts that describe the CONSTRUCTION or the lattice
#: rather than the surface.  Matched case-insensitively against each
#: comma-separated part, whole -- so "exact" goes and "exact
#: fundamental piece" (which says which PART of the surface this is)
#: stays.
PROVENANCE = ('evolver cell', 'nodal approximation', 'nodal',
              'relaxed', 'exact', 'cubic', 'hexagonal', 'tetragonal',
              'trigonal', 'rhombohedral', 'orthorhombic')


def clean_label(label):
    """Drop provenance and lattice words from a registry label."""
    def _fix(m):
        lead, inner = m.group(1), m.group(2)
        keep = [p.strip() for p in inner.split(',')
                if p.strip() and p.strip().lower() not in PROVENANCE]
        if not keep:
            return ''
        # Put back the spacing the name had.  Without this every
        # parenthesis gains a space in front of it and names that OWN
        # their brackets come apart: Fischer-Koch C(S) turns into
        # "Fischer-Koch C (S)", and C(I2-Y**) into "C (I2-Y**)".
        return '%s(%s)' % (lead, ', '.join(keep))

    out = re.sub(r'(\s*)\(([^()]*)\)', _fix, label)
    return re.sub(r'\s{2,}', ' ', out).strip()


def _selftest():
    # provenance goes
    assert clean_label("Disphenoid 31 (Evolver cell)") == "Disphenoid 31"
    assert clean_label("Schwarz P (nodal approximation)") == "Schwarz P"
    assert clean_label("Schwarz CLP (exact)") == "Schwarz CLP"
    # mathematics stays
    assert (clean_label("Wei Triply Periodic Surface "
                        "(genus 4, a=0.1 b=0.3 member)")
            == "Wei Triply Periodic Surface (genus 4, a=0.1 b=0.3 member)")
    assert (clean_label("CLP with Handle (exact fundamental piece, genus 4)")
            == "CLP with Handle (exact fundamental piece, genus 4)")
    # mixed: the provenance part goes, the rest survives with its comma
    assert (clean_label("Schoen F-RD (exact, genus 6)")
            == "Schoen F-RD (genus 6)")
    # names that OWN their brackets keep their spacing -- the regression
    # that put a space in front of every parenthesis
    assert clean_label("Fischer-Koch C(S)") == "Fischer-Koch C(S)"
    assert clean_label("C(I2-Y**) Rod Packing") == "C(I2-Y**) Rod Packing"
    assert clean_label("Schoen C21(P)") == "Schoen C21(P)"
    # a label that is nothing but provenance loses the bracket entirely
    assert clean_label("Batwing (relaxed)") == "Batwing"
    # no name may come back empty or still carry a raw registry key
    for lab in ("Disphenoid 31 (Evolver cell)", "Schwarz CLP (exact)"):
        out = clean_label(lab)
        assert out and not re.search(r'[A-Z][A-Z0-9]*_[A-Z0-9]', out), out
    print("labels: provenance stripped, mathematics and owned "
          "brackets kept OK")
