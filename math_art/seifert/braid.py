"""Braid words and the disk/band combinatorics of a Seifert surface.

Braid words use *letter notation*: an uppercase letter is a right-hand
crossing, the matching lowercase letter a left-hand one, and the letter's
position in the alphabet says which pair of adjacent strands is involved --
``A`` twists strands 1 and 2, ``B`` strands 2 and 3, and so on.  A letter may be
followed by an integer repeat count, so ``A3`` is three successive right-hand
crossings of the first two strands.

Applying Seifert's algorithm to the closure of an n-strand braid gives exactly
n Seifert circles, one per strand row, and one band per letter.  That makes the
whole combinatorial layer a few lines of arithmetic:

    chi   = n_disks - n_bands
    chi   = 2 - 2 * genus - n_components
    genus = (2 - n_components - n_disks + n_bands) / 2

Reference: J. J. van Wijk and A. M. Cohen, "Visualization of Seifert Surfaces",
IEEE TVCG 12(4), 2006, sections II.C and II.E.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

try:                                  # inside the math_art package
    from ..knots.braid import count_cycles as _count_cycles
    from ..knots.braid import strand_permutation as _strand_permutation
except ImportError:                   # flat import (headless test runner)
    from knots.braid import count_cycles as _count_cycles
    from knots.braid import strand_permutation as _strand_permutation

__all__ = ["Crossing", "Braid", "torus_knot"]

_TOKEN = re.compile(r"([A-Za-z])(\d*)")


@dataclass(frozen=True)
class Crossing:
    """A single crossing of the braid.

    Seifert's algorithm gives one band per crossing, each with one half-twist,
    so crossings are stored expanded: ``A3`` becomes three of these.

    Attributes:
        row: 0-based index of the upper strand; the crossing joins Seifert
            circles ``row`` and ``row + 1``.
        sign: ``+1`` for a right-hand crossing (uppercase letter), ``-1`` for a
            left-hand one (lowercase).
    """

    row: int
    sign: int

    @property
    def half_turns(self) -> int:
        """Half-twists of the band this crossing produces."""
        return self.sign

    def __str__(self) -> str:
        letter = chr(ord("A") + self.row)
        return letter if self.sign > 0 else letter.lower()


class Braid:
    """A braid word, parsed from letter notation.

    >>> Braid("AAA")                    # trefoil
    Braid('AAA', strands=2)
    >>> Braid("AbAb").genus             # figure-eight
    1
    >>> Braid("AbAb").euler_characteristic
    -1
    """

    def __init__(self, word: str, strands: int | None = None) -> None:
        self.word = word.strip()
        #: crossings, expanded -- one entry per crossing, not per letter
        self.crossings = tuple(self._parse(self.word))
        if not self.crossings:
            raise ValueError(f"no crossings parsed from {word!r}")
        implied = max(c.row for c in self.crossings) + 2
        if strands is not None and strands < implied:
            raise ValueError(f"{word!r} needs at least {implied} strands")
        self.strands = strands if strands is not None else implied

    # -- parsing ----------------------------------------------------------
    @staticmethod
    def _parse(word: str) -> Iterator[Crossing]:
        pos = 0
        for match in _TOKEN.finditer(word):
            if match.start() != pos:
                raise ValueError(f"cannot parse {word!r} at index {pos}")
            pos = match.end()
            letter, digits = match.groups()
            repeats = int(digits) if digits else 1
            if repeats < 1:
                raise ValueError(f"repeat count must be >= 1 in {match.group()!r}")
            row = ord(letter.upper()) - ord("A")
            sign = 1 if letter.isupper() else -1
            for _ in range(repeats):
                yield Crossing(row, sign)
        if pos != len(word):
            raise ValueError(f"trailing junk in {word!r} at index {pos}")

    # -- combinatorics ----------------------------------------------------
    @property
    def n_disks(self) -> int:
        """One Seifert circle per strand row."""
        return self.strands

    @property
    def n_bands(self) -> int:
        """One half-twisted band per crossing."""
        return len(self.crossings)

    n_crossings = n_bands

    def signed_word(self) -> list[int]:
        """This braid as a signed-integer word: `+i` is s_i, `-i` its
        inverse, indices 1-based.

        The canonical interchange form between this package and
        :mod:`math_art.knots`.  Note the two engines' LETTER notations are
        opposites -- here an uppercase letter is the right-hand (positive)
        crossing, after van Wijk and Cohen, while ``knots`` follows
        Gittings, where lowercase is positive.  So ``'AAA'`` is a
        different braid to each of them, and this integer form is what
        they exchange instead of strings.
        """
        return [(c.row + 1) * c.sign for c in self.crossings]

    @classmethod
    def from_signed_word(cls, word, strands: int | None = None) -> "Braid":
        """Build a braid from a signed-integer word (see `signed_word`)."""
        letters = []
        for g in word:
            letter = chr(ord("A") + abs(g) - 1)
            letters.append(letter if g > 0 else letter.lower())
        return cls("".join(letters), strands)

    def permutation(self) -> list[int]:
        """Strand permutation of the closure.

        Each crossing transposes its two strands; the link components are the
        cycles of the product.  Shared with :mod:`math_art.knots.braid`,
        which needs the same permutation from its own word form -- the
        computation ignores crossing signs, so the engines' opposite
        letter conventions do not matter here.
        """
        return _strand_permutation(self.signed_word(), self.strands)

    @property
    def n_components(self) -> int:
        """Number of link components: cycles of the strand permutation."""
        return _count_cycles(self.permutation())

    @property
    def euler_characteristic(self) -> int:
        """chi = #disks - #bands, the disk/band count of the surface."""
        return self.n_disks - self.n_bands

    @property
    def genus(self) -> int:
        """Genus of *this* surface -- not necessarily the genus of the knot."""
        twice = 2 - self.n_components - self.euler_characteristic
        if twice % 2:
            raise ValueError(
                f"inconsistent invariants for {self.word!r}: "
                f"chi={self.euler_characteristic}, m={self.n_components}"
            )
        return twice // 2

    def band_rows(self) -> list[int]:
        return [c.row for c in self.crossings]

    def __repr__(self) -> str:
        return f"Braid({self.word!r}, strands={self.strands})"

    def summary(self) -> str:
        return (
            f"{self.word}: {self.n_disks} disks, {self.n_bands} bands, "
            f"{self.n_components} component"
            f"{'s' if self.n_components != 1 else ''}, "
            f"chi={self.euler_characteristic}, genus={self.genus}"
        )


def torus_knot(p: int, q: int) -> "Braid":
    """Braid word for the (p, q) torus knot or link: (s1 s2 ... s_{p-1}) ** q.

    >>> torus_knot(2, 3).word          # trefoil
    'AAA'
    >>> torus_knot(3, 4).genus
    3
    """
    if p < 2 or q < 1:
        raise ValueError("need p >= 2 and q >= 1")
    return Braid("".join(chr(ord("A") + i) for i in range(p - 1)) * q)


def _selftest():
    ok = True

    # Letter parsing, including the repeat-count grammar this notation has
    # and the Gittings-style one does not.
    good = (Braid("A3").signed_word() == [1, 1, 1]
            and Braid("AAA").signed_word() == [1, 1, 1]
            and Braid("Ab").signed_word() == [1, -2]
            and Braid("AbAb").strands == 3)
    ok &= good
    print(f"seifert.braid: letter notation + repeat counts "
          f"{'OK' if good else 'FAIL'}")

    # THE CONVENTION TRAP.  This package follows van Wijk and Cohen, where
    # an UPPERCASE letter is the positive crossing; math_art.knots follows
    # Gittings, where LOWERCASE is positive.  The same string therefore
    # denotes mirror-image braids, and the two engines must exchange
    # signed-integer words rather than strings.  Assert that explicitly, so
    # nobody later "unifies" the parsers and silently mirrors output.
    try:
        from ..knots.braid import parse_letters
    except ImportError:
        from knots.braid import parse_letters
    bad = []
    for w in ("AAA", "AbAb", "AABacBc", "aBaBa"):
        mine, theirs = Braid(w).signed_word(), parse_letters(w)
        if mine != [-g for g in theirs]:
            bad.append(w)
    good = not bad
    ok &= good
    print(f"seifert.braid: sign convention is opposite to knots' on every "
          f"word {'OK' if good else 'FAIL ' + ','.join(bad)}")

    # The signed-integer bridge round-trips, preserving strand count.
    bad = []
    for w in ("AAA", "AbAb", "AABacBc", "A3", "aBaBa"):
        b = Braid(w)
        rt = Braid.from_signed_word(b.signed_word(), b.strands)
        if rt.signed_word() != b.signed_word() or rt.strands != b.strands:
            bad.append(w)
    good = not bad
    ok &= good
    print(f"seifert.braid: signed-word round-trip "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # Component count is sign-independent, so despite the opposite letter
    # conventions BOTH engines must agree on it for the same string.
    try:
        from ..knots.braid import closure_components
    except ImportError:
        from knots.braid import closure_components
    bad = []
    for w in ("AAA", "AbAb", "AABacBc", "aBaBa", "ABABAB"):
        if Braid(w).n_components != closure_components(parse_letters(w)):
            bad.append(w)
    good = not bad
    ok &= good
    print(f"seifert.braid: component count agrees with knots "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # Seifert's algorithm on a braid closure: n disks, one band per
    # crossing, chi = disks - bands, and genus from chi.  The trefoil and
    # figure-eight both have genus 1; the (2,5) torus knot has genus 2.
    cases = [("AAA", 2, 3, -1, 1), ("AbAb", 3, 4, -1, 1),
             ("A5", 2, 5, -3, 2)]
    bad = []
    for w, disks, bands, chi, genus in cases:
        b = Braid(w)
        if (b.n_disks, b.n_bands, b.euler_characteristic) != (disks, bands,
                                                              chi):
            bad.append(f"{w}:{b.n_disks},{b.n_bands},"
                       f"{b.euler_characteristic}")
        elif b.genus != genus:
            bad.append(f"{w}:genus {b.genus}!={genus}")
    good = not bad
    ok &= good
    print(f"seifert.braid: disk/band/chi/genus on {len(cases)} braids "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # torus_knot(p, q) must build a braid whose closure is a single
    # component when gcd(p, q) = 1, and gcd components otherwise.
    from math import gcd
    bad = []
    for p, q in ((2, 3), (2, 5), (3, 4), (2, 4), (3, 6)):
        got = torus_knot(p, q).n_components
        if got != gcd(p, q):
            bad.append(f"({p},{q}):{got}!={gcd(p, q)}")
    good = not bad
    ok &= good
    print(f"seifert.braid: torus_knot component counts "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("seifert.braid self-test failed")
