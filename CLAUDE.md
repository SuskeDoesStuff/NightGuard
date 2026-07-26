# NightGuard

Full specification: read PROJECT.md before implementing anything.
Section 3 of PROJECT.md is normative for all simulation behaviour — it outranks the exit criteria
in §7 where they conflict.

## Invariants, never violate
- No franchise names, character names, or game assets anywhere. See PROJECT.md §0.2.
- core/ must not import gymnasium or torch.
- All randomness goes through an injected `np.random.Generator`, split into the fixed per-consumer
  substreams in `core/rng.py`. Never draw from the global generator (`np.random.random`, `.randint`,
  `.choice`, `.seed`) and never `import random`. Constructing `np.random.default_rng(seed)` is the
  intended plumbing, not a violation. Enforced by `tests/test_no_global_rng.py`.
- No constants in logic; they go in config dataclasses with defaults, overridable from YAML.
- Never reorder node IDs (PROJECT.md §2.2) — they index the trace format and the observation.
- Never reorder the entity resolution order `[WARDEN, DRIFTER, PROWLER, SPRINTER]` (§3.13), and never
  reorder or insert into `rng.STREAM_NAMES` — either would invalidate every seeded fixture.

## Commands
The venv is at `.venv/` and is gitignored; prefix with `.venv/bin/` or activate it first.
- Test:   `pytest`
- Lint:   `ruff check && ruff format --check`
- Types:  `mypy --strict src/nightguard/core src/nightguard/env`
- Validate: `python scripts/validate.py`  (exit criteria for the current milestone)

All four must pass before any milestone is called done. See PROJECT.md §11.

## Progression

| Version | Scope | Status |
|---|---|---|
| v0.1 | Clock, power, office controls, DRIFTER + PROWLER | **Done** — tag `v0.1`, commit `19362e1` |
| v0.2 | WARDEN + SPRINTER, trace format | **Current** |
| v0.3 | Three-phase blackout, full validation suite (§8) | Not started |
| v1.0 | Gymnasium wrapper, wrappers, vectorised runner | Not started |
| v1.1 | RecurrentPPO baseline | Not started |
| v1.2 | Replay viewer | Not started |
| v2.0 | Randomised configs, benchmark table | Not started |
| v2.1 | Packaging and release | Not started |

Do not begin a version until the previous version's exit criteria pass. v0.3 is the critical-path
milestone: everything after it depends on the simulator being correct.

## Settled decisions — do not re-litigate

PROJECT.md §10 is the register; `CHANGELOG.md` carries the reasoning. Three were settled in v0.1:

- **`active = clamp(count, 1, 4)`.** One active control drains at exactly the idle rate. §7's exit
  criterion 3 was corrected from "one door" to "two doors" to match.
- **Office entry requires `monitor_up`.** A closed door turns the entity away regardless; an open
  door with the monitor down is a stalemate where the entity camps at the corner.
- **Office invasion shipped in v0.1**, not v0.2. v0.2 is WARDEN and SPRINTER only.

## Traps

- **§8.2's night-1 assertion will come under pressure in v0.2.** SPRINTER charges only while the
  monitor is *down*, so `do_nothing` is maximally exposed to it. Expect a conflict between §3.7 and
  §8.2's "`do_nothing` survives night 1 ≥ 0.8" and resolve it deliberately.
- **Determinism tests are vacuous under an all-NOOP script.** With the monitor never raised, the door
  entities end camped at their corners and every seed produces the same terminal state. §8.5's
  byte-identical trace check must use scripts that raise the monitor.
- **The §7 idle-power table is rounded to 3 dp** and cannot be met at 1e-6 as printed. Assert against
  the closed form; nights 1, 2, 5 and 6 are off by 8e-5 to 3e-4.
- **Entity opportunity intervals are not multiples of the sim tick.** They are scheduled in exact
  integer units and recomputed from the firing count. Never accumulate them in floats, and never
  round them to the tick grid — that collapses DRIFTER and PROWLER onto one period.
