#!/usr/bin/env python3
"""Score policies on held-out seeds. PROJECT.md 7 (v1.1), criteria 3, 4 and 6.

::

    python scripts/evaluate.py --night 6                        # the reference policies
    python scripts/evaluate.py --night 6 --model runs/.../policy.zip
    python scripts/evaluate.py --preset custom_max --episodes 500

Learned and scripted policies go through the same evaluation path over the same seeds, because the
milestone's substantive claim -- that a learned policy beats `rhythm` on night 6 -- is a comparison,
and a comparison run through two harnesses compares the harnesses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nightguard.core.config import NightConfig, load_night_config, load_preset
from nightguard.core.topology import load_topology
from nightguard.policies import DoNothing, MonitorDown, Rhythm
from nightguard.train.evaluate import EvalResult, eval_seeds, evaluate_scripted

REFERENCES = {"do_nothing": DoNothing, "rhythm": Rhythm, "monitor_down": MonitorDown}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--night", type=int, default=6)
    parser.add_argument("--preset", default=None, help="a configs/nights/ stem, overriding --night")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed-offset", type=int, default=1_000_000)
    parser.add_argument("--model", default=None, help="path to a saved SB3 policy")
    parser.add_argument("--oracle", action="store_true", help="the model expects the Oracle block")
    parser.add_argument("--algorithm", choices=("recurrent_ppo", "ppo"), default="recurrent_ppo")
    parser.add_argument("--stochastic", action="store_true", help="sample rather than argmax")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> NightConfig:
    """Pick the night config the run is scored on."""
    return load_preset(args.preset) if args.preset else load_night_config(args.night)


def score_model(args: argparse.Namespace, config: NightConfig, seeds: list[int]) -> EvalResult:
    """Load a saved policy and score it. Imports torch only on this path."""
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    from nightguard.train.evaluate import evaluate_model
    from nightguard.train.recurrent_ppo import make_env

    topology = load_topology(config.topology_path())
    env = make_env(config, args.oracle, topology)
    loader = PPO if args.algorithm == "ppo" else RecurrentPPO
    model = loader.load(args.model, device="auto")
    return evaluate_model(env, model, seeds, deterministic=not args.stochastic)


def main(argv: list[str] | None = None) -> int:
    """Score whatever was asked for and print JSON."""
    args = parse_args(argv)
    config = resolve_config(args)
    topology = load_topology(config.topology_path())
    seeds = eval_seeds(args.episodes, args.seed_offset)

    results: dict[str, object] = {
        "config": args.preset or f"night{args.night}",
        "episodes": args.episodes,
        "seed_offset": args.seed_offset,
    }
    for name, factory in REFERENCES.items():
        results[name] = evaluate_scripted(config, factory, seeds, topology).as_dict()
    if args.model:
        results["model"] = score_model(args, config, seeds).as_dict()
        results["model_path"] = args.model

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
