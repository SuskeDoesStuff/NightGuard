"""Seeded random-number plumbing. PROJECT.md 1.2.

All randomness flows through an injected :class:`numpy.random.Generator`. Nothing here or anywhere
else in the package calls ``random.*`` or ``np.random.*`` at module level; this is tested.

The injected generator is split into one independent substream per consumer, in a **fixed order that
already reserves slots for WARDEN, SPRINTER and blackout** even though v0.1 simulates none of them.
That is the point of the design: adding an entity in v0.2 must not shift the draws seen by the
entities that already exist, or every seeded v0.1 fixture silently changes. It also keeps PROJECT.md
8.3's "with all other entities disabled" a clean ablation rather than a reshuffle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.random import Generator, SeedSequence

STREAM_NAMES: Final[tuple[str, ...]] = (
    "reset",
    "warden",
    "drifter",
    "prowler",
    "sprinter",
    "blackout",
)


@dataclass(frozen=True)
class SimRng:
    """One independent generator per consumer.

    Attributes:
        reset: Draws made once at reset, such as night 4's rolled WARDEN level.
        warden: Reserved for v0.2.
        drifter: DRIFTER's opportunity rolls and pool choices.
        prowler: PROWLER's opportunity rolls and chain choices.
        sprinter: Reserved for v0.2.
        blackout: Reserved for the three-phase sequence in v0.3.
    """

    reset: Generator
    warden: Generator
    drifter: Generator
    prowler: Generator
    sprinter: Generator
    blackout: Generator

    @classmethod
    def from_generator(cls, rng: Generator) -> SimRng:
        """Split an injected generator into the fixed set of substreams."""
        children = _spawn(rng, len(STREAM_NAMES))
        return cls(**dict(zip(STREAM_NAMES, children, strict=True)))

    @classmethod
    def from_seed(cls, seed: int) -> SimRng:
        """Build substreams directly from an integer seed."""
        return cls.from_generator(np.random.default_rng(seed))


def _spawn(rng: Generator, count: int) -> list[Generator]:
    """Derive ``count`` independent generators from ``rng``.

    Prefers ``SeedSequence.spawn``, which leaves the parent untouched. Generators built from a raw
    bit generator may not carry a seed sequence, so fall back to drawing child seeds from the parent
    itself; that consumes parent state but stays fully deterministic.
    """
    seed_seq = getattr(rng.bit_generator, "seed_seq", None)
    if isinstance(seed_seq, SeedSequence):
        return [np.random.default_rng(child) for child in seed_seq.spawn(count)]
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=count)
    return [np.random.default_rng(int(seed)) for seed in seeds]
