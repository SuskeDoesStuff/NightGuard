# NightGuard

A partially-observable reinforcement learning environment modelling an asymmetric
surveillance-and-resource-management game: a stationary agent must survive a fixed-length night
against four autonomous adversaries, using a camera system that is simultaneously its only source of
information and one of its costliest actions.

`PROJECT.md` is the normative specification. Read it before changing anything.

## Status

**v0.2 — Full roster.** All four entities (`WARDEN`, `DRIFTER`, `PROWLER`, `SPRINTER`), the JSONL
trace format, the audio channel, and the scripted reference policies. Blackout is still a
placeholder that terminates immediately; the three-phase sequence lands in v0.3. No Gymnasium
wrapper, no observations, no reward — `env/actions.py` is the only `env/` code.

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

`validate.py` checks the v0.1 and v0.2 exit criteria from `PROJECT.md` §7, printing each
measured value alongside its target.

## Layout

```
src/nightguard/core/       Pure simulation. No gymnasium, no torch, no global randomness.
src/nightguard/trace/      JSONL serialisation of ground truth. Depends on core only.
src/nightguard/policies/   Scripted reference policies: do_nothing, rhythm, monitor_down.
src/nightguard/env/        Gymnasium layer. Only the action space until v1.0.
configs/                   Topology and night presets. All numeric constants live here.
scripts/                   Exit-criteria validation.
```
