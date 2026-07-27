# NightGuard

A partially-observable reinforcement learning environment modelling an asymmetric
surveillance-and-resource-management game: a stationary agent must survive a fixed-length night
against four autonomous adversaries, using a camera system that is simultaneously its only source of
information and one of its costliest actions.

`PROJECT.md` is the normative specification. Read it before changing anything.

## Status

**v1.1 — Baseline policy.** `RecurrentPPO` with an LSTM over the v1.0 environment, a three-stage
curriculum set by measured headroom, and the measurements that make partial observability a number.

**The harness works; the policy does not yet.** Four 2M-step arms on night 5 all converge on the
`monitor_down` degenerate strategy — 824.9 mean steps against the scripted policy's 820.4, dying to
a blackout in 88% of episodes. Five of §7's eight v1.1 exit criteria are met and three are reported
as misses. The one clean pass came by transfer: **0.140 survival on night 4**, a night the policy
never trained on, against `do_nothing`'s 0.000. `rhythm` still holds the field at 0.980 on night 5.

Escaping the plateau needs a sequence — close both doors, raise the monitor to freeze SPRINTER, drop
it, reopen — and every prefix of it is worse than not starting, so §6.3's reward gives the optimiser
no gradient toward it. `CHANGELOG.md` has the full account, including a `sparse_mode` result that
contradicted its own pre-registered expectation.

## Install

```
python -m venv .venv
.venv/bin/pip install -e ".[dev]"          # simulator, environment, tests
.venv/bin/pip install -e ".[dev,train]"    # adds torch, stable-baselines3, sb3-contrib
```

## Use

```python
import gymnasium as gym
import nightguard  # noqa: F401 - registers the env IDs

env = gym.make("NightGuard-v0", night=4)
obs, info = env.reset(seed=918442)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`NightGuard-CustomMax-v0` is the same environment with every adversary at the maximum level.
`NightGuardEnv` can also be constructed directly, with `night=`, `preset=` or a prebuilt `config=`.

## Train

```
python scripts/train.py                                  # the three-stage curriculum
python scripts/train.py --oracle --tag oracle            # the matched Oracle arm
python scripts/evaluate.py --night 6 --model runs/.../policy.zip
python scripts/profile_training.py --steps 20000         # env-versus-network split
```

Each run writes `runs/<timestamp>-<algorithm>[-<tag>]/` with a manifest carrying the git SHA, config
hash, seed and device, one directory per stage holding the learning curve and the saved policy, and
a rolling `summary.json`. Hyperparameters live in `configs/train/baseline.yaml`, not in code.

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

`validate.py` checks every exit criterion from `PROJECT.md` §7 and the whole of §8, printing each
measured value alongside its target. The v1.1 criteria that cost hours of training are checked
against the committed `runs/summary.json`; a missing summary fails, so the milestone cannot be
called done on a repository where the runs were never made.

## Layout

```
src/nightguard/core/       Pure simulation. No gymnasium, no torch, no global randomness.
src/nightguard/trace/      JSONL serialisation of ground truth. Depends on core only.
src/nightguard/policies/   Scripted reference policies: do_nothing, rhythm, monitor_down.
src/nightguard/env/        Gymnasium layer: env, registration, observation, reward, wrappers.
src/nightguard/train/      RecurrentPPO, curriculum, evaluation, run manifests. May import torch.
src/nightguard/viewer/     Static HTML/JS trace viewer. No server, no build step.
src/nightguard/derivations.py  Analytic predictions for the section 8 suite.
configs/                   Topology, night presets and training hyperparameters.
runs/                      Training artefacts. Gitignored except summary.json.
scripts/                   Validation, trace export, training, evaluation, profiling.
```
