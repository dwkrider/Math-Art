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

    def permutation(self) -> list[int]:
        """Strand permutation of the closure.

        Each crossing transposes its two strands; the link components are the
        cycles of the product.
        """
        perm = list(range(self.strands))
        for c in self.crossings:
            k = c.row
            perm[k], perm[k + 1] = perm[k + 1], perm[k]
        return perm

    @property
    def n_components(self) -> int:
        """Number of link components: cycles of the strand permutation."""
        perm = self.permutation()
        seen, cycles = set(), 0
        for start in range(self.strands):
            if start in seen:
                continue
            cycles += 1
            i = start
            while i not in seen:
                seen.add(i)
                i = perm[i]
        return cycles

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
