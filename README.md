# NightGuard

A partially-observable reinforcement learning environment modelling an asymmetric
surveillance-and-resource-management game: a stationary agent must survive a fixed-length night
against four autonomous adversaries, using a camera system that is simultaneously its only source of
information and one of its costliest actions.

`PROJECT.md` is the normative specification. Read it before changing anything.

## Status

**v1.0 — Environment.** A Gymnasium interface over the validated simulator: a 100-dim observation,
§6.3's reward with `sparse_mode`, and the `Oracle`, `PreviousAction`, `AudioMask` and `FrameStack`
wrappers. Ground truth reaches the caller through `info`, never through `obs`. No training yet —
that is v1.1.

## Install

```
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Use

```python
from nightguard.env import NightGuardEnv

env = NightGuardEnv(night=4)
obs, info = env.reset(seed=918442)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Or drive the simulator directly, without the RL layer:

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
| Test | `pytest` (fast) · `pytest -m slow` (statistical) |
| Lint | `ruff check && ruff format --check` |
| Types | `mypy --strict src/nightguard` |
| Validate | `python scripts/validate.py` |

`validate.py` checks every v0.1, v0.2 and v0.3 exit criterion from `PROJECT.md` §7 and the whole
of §8, printing each measured value alongside its target.

## Layout

```
src/nightguard/core/       Pure simulation. No gymnasium, no torch, no global randomness.
src/nightguard/trace/      JSONL serialisation of ground truth. Depends on core only.
src/nightguard/policies/   Scripted reference policies: do_nothing, rhythm, monitor_down.
src/nightguard/env/        Gymnasium layer: env, observation, reward, wrappers.
src/nightguard/viewer/     Static HTML/JS trace viewer. No server, no build step.
src/nightguard/derivations.py  Analytic predictions for the section 8 suite.
configs/                   Topology and night presets. All numeric constants live here.
scripts/                   Exit-criteria validation and trace export.
```
