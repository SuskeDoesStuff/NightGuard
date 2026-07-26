"""All randomness must flow through an injected Generator. PROJECT.md 1.2, and CLAUDE.md.

The invariant is about *global* randomness, not about the numpy namespace: constructing a seeded
``np.random.default_rng`` is exactly the intended plumbing, while ``np.random.random()`` and friends
draw from a hidden global state and would silently break determinism.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

# Constructors and types, as opposed to draws from the global generator.
ALLOWED_NUMPY_RANDOM_ATTRS = frozenset(
    {"default_rng", "Generator", "SeedSequence", "BitGenerator", "PCG64"}
)

NUMPY_RANDOM = re.compile(r"\bnp\.random\.(\w+)|\bnumpy\.random\.(\w+)")
STDLIB_RANDOM = re.compile(r"^\s*(?:import random\b|from random import\b)", re.MULTILINE)


def source_files() -> list[Path]:
    return sorted(SOURCE_ROOT.rglob("*.py"))


def test_there_is_source_to_scan() -> None:
    assert source_files(), f"no Python sources found under {SOURCE_ROOT}"


def test_no_draws_from_the_global_numpy_generator() -> None:
    offences: list[str] = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        for match in NUMPY_RANDOM.finditer(text):
            attribute = match.group(1) or match.group(2)
            if attribute not in ALLOWED_NUMPY_RANDOM_ATTRS:
                line = text.count("\n", 0, match.start()) + 1
                offences.append(f"{path.relative_to(SOURCE_ROOT)}:{line}: np.random.{attribute}")
    assert not offences, "global numpy randomness: " + "; ".join(offences)


def test_no_stdlib_random() -> None:
    offences = [
        str(path.relative_to(SOURCE_ROOT))
        for path in source_files()
        if STDLIB_RANDOM.search(path.read_text(encoding="utf-8"))
    ]
    assert not offences, "stdlib random imported in: " + ", ".join(offences)


def test_core_does_not_import_gymnasium_or_torch() -> None:
    """PROJECT.md 1: core must not import gymnasium, torch, or any rendering library."""
    forbidden = re.compile(r"^\s*(?:import|from)\s+(gymnasium|gym|torch)\b", re.MULTILINE)
    offences = [
        str(path.relative_to(SOURCE_ROOT))
        for path in (SOURCE_ROOT / "nightguard" / "core").rglob("*.py")
        if forbidden.search(path.read_text(encoding="utf-8"))
    ]
    assert not offences, "forbidden import in core/: " + ", ".join(offences)
