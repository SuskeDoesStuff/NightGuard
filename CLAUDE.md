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

## Evidence discipline

- Never derive an expected value by running the implementation. Derive from PROJECT.md, then
  compare. If they disagree, investigate; do not adopt the measured value.
- Never weaken a tolerance, threshold, or assertion to make a test pass. If you believe an exit
  criterion is wrong, stop and say so.
- Every statistical assertion needs a non-vacuity check (PROJECT.md §8.0).

## Settled decisions — do not re-litigate

PROJECT.md §10 is the register; `CHANGELOG.md` carries the reasoning.

- **`active = clamp(count, 1, 4)`.** The first active control is free — one door shut drains at
  exactly the idle rate. §7's exit criterion 3 needs *two* doors.
- **Office entry requires `monitor_up`, for all three path entities.** A closed door turns the
  entity away regardless; an open door with the monitor down is a stalemate where the entity camps
  at the corner. SPRINTER is therefore the only entity that punishes keeping the monitor down.
- **WARDEN's `E_CORNER` "attack" is a move into `OFFICE`, not a kill.** The kill is the 25%/s roll
  afterwards, while the monitor is down. Read as a kill, `OFFICE` is unreachable.
- **§8.2's night-1 threshold is an agreement test**, not `≥ 0.8`: measured `do_nothing` survival
  must match the analytic 0.2397 within binomial error. Nights 2–6 are near zero by design.

## Traps

- **Timing resolution is 1/300 s = lcm(60, 100),** expressed as `timing.time_units_per_second: 300`
  so the value stays an exact integer. The countdown table divides by 60; the opportunity intervals
  are two-decimal. 1/300 is the coarsest grid on which both are exact — at 1/1000 six of WARDEN's
  ten countdowns are non-terminating. Do not change it without re-running `test_config.py`'s
  exactness test. It is a resolution, **not** a frame rate.
- **Never accumulate timers in floats, and never round an interval to the tick grid.** Firing ticks
  are recomputed from the firing count. Rounding collapses DRIFTER and PROWLER onto one period.
- **WARDEN's countdown is not paused by monitor raises.** Only new opportunity *rolls* are
  suppressed by the monitor. This is v0.2 exit criterion 3 and the single most likely place for a
  plausible-looking bug to survive testing.
- **Determinism tests are vacuous under an all-NOOP script.** With the monitor never raised, no
  entity can enter the office and every seed produces an identical surviving episode. §8.5's
  byte-identical trace check must assert distinct outcomes across seeds first (§8.0).
- **The §7 idle-power table is rounded to 3 dp** and is display only. The closed form
  `(0.1 + 0.1/D) × 535` is normative; nights 1, 2, 5 and 6 differ by 8e-5 to 3e-4.
