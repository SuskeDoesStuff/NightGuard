# NightGuard

A partially-observable reinforcement learning environment modelling an asymmetric
surveillance-and-resource-management game: a stationary agent must survive a fixed-length night
against four autonomous adversaries, using a camera system that is simultaneously its only source of
information and one of its costliest actions.

`PROJECT.md` is the normative specification. Read it before changing anything.

## Status

**v0.1 — Skeleton.** Pure simulation core only: clock, power model, the five office controls, and the
two door entities (`DRIFTER`, `PROWLER`). Blackout is a placeholder that terminates immediately. No
Gymnasium wrapper, no trace format, no observations, no reward. The deliverable is a `NightSim` you
drive with a list of actions.

## Install

```
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Use

```python
from nightguard.core import Action, NightSim, load_night_config

config = load_night_config(1)
sim = NightSim.from_seed(config, seed=918442)

# run() takes an action script and pads with NOOP until the night ends.
result = sim.run([Action.SELECT_CAM_5, Action.MONITOR_DOWN, Action.TOGGLE_DOOR_LEFT])

print(result.cause, result.final_power_pct, sim.clock.decision_steps)
```

## Commands

| Task | Command |
|---|---|
| Test | `pytest` |
| Lint | `ruff check && ruff format --check` |
| Types | `mypy --strict src/nightguard/core src/nightguard/env` |
| Validate | `python scripts/validate.py` |

`validate.py` checks the v0.1 exit criteria from `PROJECT.md` §7.

## Layout

```
src/nightguard/core/   Pure simulation. No gymnasium, no torch, no global randomness.
src/nightguard/env/    Gymnasium wrapper. Empty until v1.0.
configs/               Topology, night presets, and run presets. All numeric constants live here.
scripts/               Exit-criteria validation and episode runners.
```
