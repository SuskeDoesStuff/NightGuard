#!/usr/bin/env python3
"""Run the v1.1 curriculum. PROJECT.md 7 (v1.1).

::

    python scripts/train.py                          # the baseline
    python scripts/train.py --oracle --tag oracle    # the matched Oracle arm
    python scripts/train.py --sparse --stages stage1-night5
    python scripts/train.py --algorithm ppo --tag nomem

Writes everything to ``runs/<timestamp>-<algorithm>[-<tag>]/``: a manifest carrying the git SHA,
config hash, seed and device, one directory per stage with the learning curve as CSV and the saved
policy, and a rolling ``summary.json``. Exit criterion 7 is that every number reported anywhere is
reachable from one of those manifests.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nightguard.train.config import TrainConfig, load_train_config
from nightguard.train.curriculum import run_curriculum


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line. Every flag maps onto a field of :class:`TrainConfig`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline", help="name under configs/train/, or a path")
    parser.add_argument("--tag", default="", help="suffix for the run directory")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--algorithm", choices=("recurrent_ppo", "ppo"), default=None)
    parser.add_argument("--oracle", action="store_true", help="wrap the env in Oracle")
    parser.add_argument("--sparse", action="store_true", help="PROJECT.md 6.3 sparse_mode")
    parser.add_argument("--device", default=None, help="auto, cpu or cuda")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="run only these stages, by name, in the order given",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="override every stage's step budget; for smoke runs",
    )
    return parser.parse_args(argv)


def apply_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Fold the command-line overrides into the loaded config, so the hash reflects them."""
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.algorithm is not None:
        config = replace(config, algorithm=args.algorithm)
    if args.oracle:
        config = replace(config, oracle=True)
    if args.sparse:
        config = replace(config, sparse=True)
    if args.device is not None:
        config = replace(config, algo=replace(config.algo, device=args.device))
    if args.n_envs is not None:
        config = replace(config, algo=replace(config.algo, n_envs=args.n_envs))
    if args.stages:
        chosen = tuple(config.stage(name) for name in args.stages)
        config = replace(config, stages=chosen)
    if args.total_steps is not None:
        config = replace(
            config,
            stages=tuple(replace(s, total_steps=args.total_steps) for s in config.stages),
        )
    return config


def main(argv: list[str] | None = None) -> int:
    """Load, override, run, and print the summary."""
    args = parse_args(argv)
    config = apply_overrides(load_train_config(args.config), args)
    summary = run_curriculum(config, tag=args.tag)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
