# NightGuard

Full specification: read PROJECT.md before implementing anything.
Section 3 of PROJECT.md is normative for all simulation behaviour.

## Invariants, never violate
- No franchise names, character names, or game assets anywhere. See PROJECT.md §0.2.
- core/ must not import gymnasium or torch.
- All randomness goes through an injected np.random.Generator. No np.random.* calls.
- No constants in logic; they go in config dataclasses.

## Commands
- Test:   pytest
- Lint:   ruff check && ruff format --check
- Types:  mypy --strict src/nightguard/core src/nightguard/env
- Validate: python scripts/validate.py

## Current milestone
v0.1 (see PROJECT.md §7). Do not start v0.2 until v0.1 exit criteria pass.