#!/usr/bin/env python3
"""Where the training wall clock actually goes. PROJECT.md 6.5, 7 (v1.0 criterion 4, v1.1).

§5 of ``docs/V1.0-DISCREPANCY-RESOLUTION.md`` left ``env/vector.py`` unbuilt by decision, on the
argument that the environment is a single-digit percentage of a training run. That is a claim, and
this script is what turns it into a measurement: it times ``learn()`` end to end, then times the
same number of environment steps on their own, and reports the split.

Build the vectorised runner only if the environment genuinely dominates -- and if it is built, the
object-versus-vectorised equivalence test is a hard gate, because a vectorised simulator that
diverges from the validated one silently invalidates the whole of §8.

::

    python scripts/profile_training.py --steps 20000 --device cpu
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from nightguard.train.config import load_train_config
from nightguard.train.recurrent_ppo import build_model, make_vec_env, stage_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--stage", default="stage1-night5")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--device", default=None, help="auto, cpu or cuda")
    parser.add_argument("--n-envs", type=int, default=None)
    return parser.parse_args(argv)


def time_environment_only(venv, steps: int, seed: int) -> float:  # type: ignore[no-untyped-def]
    """Seconds to take ``steps`` environment steps with random actions and no network."""
    rng = np.random.default_rng(seed)
    venv.reset()
    n = venv.num_envs
    taken = 0
    start = time.perf_counter()
    while taken < steps:
        venv.step(rng.integers(0, venv.action_space.n, size=n))
        taken += n
    return time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    """Profile and print JSON."""
    args = parse_args(argv)
    train = load_train_config(args.config)
    if args.device is not None:
        train = replace(train, algo=replace(train.algo, device=args.device))
    if args.n_envs is not None:
        train = replace(train, algo=replace(train.algo, n_envs=args.n_envs))

    night = stage_config(train.stage(args.stage), train.sparse)
    venv = make_vec_env(night, train)
    model = build_model(venv, train)

    start = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    total_s = time.perf_counter() - start
    steps = int(model.num_timesteps)

    env_s = time_environment_only(venv, steps, train.seed)
    venv.close()

    report = {
        "config": args.config,
        "stage": args.stage,
        "device": str(model.device),
        "n_envs": train.algo.n_envs,
        "steps": steps,
        "total_s": round(total_s, 2),
        "env_only_s": round(env_s, 2),
        "env_fraction": round(env_s / total_s, 4),
        "train_steps_per_s": round(steps / total_s, 1),
        "env_steps_per_s": round(steps / env_s, 1),
        "machine": f"{platform.machine()} {platform.system()}, Python {platform.python_version()}",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
