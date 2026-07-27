#!/usr/bin/env python3
"""Collect the training runs into ``runs/summary.json``. PROJECT.md 7 (v1.1).

``validate.py`` reads the v1.1 criteria from that file. They cannot be recomputed inside a
validation pass -- each one costs hours of training -- so the artefact is what the milestone is
checked against, and it is generated here rather than written by hand.

::

    python scripts/summarise.py --base runs/<id> --oracle runs/<id> --sparse runs/<id>

Every policy and every reference is scored on the **same held-out seeds** by the same evaluation
function, because criteria 3 and 4 are comparisons and a comparison run through two harnesses
compares the harnesses.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nightguard.core.config import NightConfig, load_night_config, load_preset
from nightguard.core.topology import load_topology
from nightguard.policies import DoNothing, Rhythm
from nightguard.train.evaluate import EvalResult, eval_seeds, evaluate_scripted

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"

# The configs criteria 3 and 4 name, plus stage 3's.
EVAL_CONFIGS: dict[str, dict[str, Any]] = {
    "night4": {"night": 4},
    "night5": {"night": 5},
    "night6": {"night": 6},
    "custom_max": {"preset": "custom_max"},
}

# `rhythm`'s measured camera duty cycle on night 6, PROJECT.md 10.
RHYTHM_DUTY_CYCLE = 0.0793


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line. Each flag names one run directory under ``runs/``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="the baseline curriculum run directory")
    parser.add_argument("--oracle", default=None, help="the Oracle arm of the matched pair")
    parser.add_argument("--matched-base", default=None, help="the no-Oracle arm of that pair")
    parser.add_argument("--sparse", default=None, help="the sparse_mode ablation")
    parser.add_argument("--no-memory", default=None, help="the feed-forward PPO ablation")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed-offset", type=int, default=1_000_000)
    parser.add_argument("--out", default=str(RUNS_DIR / "summary.json"))
    return parser.parse_args(argv)


def load_run(directory: str | None) -> dict[str, Any] | None:
    """Read one run's manifest and rolling summary."""
    if directory is None:
        return None
    path = Path(directory)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with (path / "summary.json").open(encoding="utf-8") as handle:
        summary: dict[str, Any] = json.load(handle)
    with (path / "manifest.json").open(encoding="utf-8") as handle:
        summary["manifest"] = json.load(handle)
    summary["directory"] = str(path.relative_to(REPO_ROOT))
    return summary


def resolve(name: str) -> NightConfig:
    """The night config behind an evaluation key."""
    spec = EVAL_CONFIGS[name]
    return load_preset(spec["preset"]) if "preset" in spec else load_night_config(spec["night"])


def final_policy_path(run: dict[str, Any]) -> str:
    """The policy the curriculum finished on."""
    return str(run["stages"][-1]["model_path"])


def score_policy(
    model_path: str, config: NightConfig, seeds: list[int], oracle: bool
) -> EvalResult:
    """Score a saved policy. Imports torch only on this path."""
    from sb3_contrib import RecurrentPPO

    from nightguard.train.evaluate import evaluate_model
    from nightguard.train.recurrent_ppo import make_env

    topology = load_topology(config.topology_path())
    env = make_env(config, oracle, topology)
    model = RecurrentPPO.load(model_path, device="cpu")
    return evaluate_model(env, model, seeds)


def margin_sigma(policy: EvalResult, reference: float) -> float:
    """How far the policy's survival sits above a reference, in its own standard errors."""
    error = policy.standard_error()
    if error == 0.0:
        return float("inf") if policy.survival != reference else 0.0
    return (policy.survival - reference) / error


def curve_criterion(run: dict[str, Any]) -> dict[str, Any]:
    """Criterion 2: did stage 1's curve rise?

    Judged on ``mean_steps`` as well as survival. On a night where a random policy never reaches
    dawn, survival is a flat zero for a long time and the only thing that moves early is how long
    the agent stays alive -- which is precisely what 6.3's dense term pays for.
    """
    stage = run["stages"][0]
    points = stage["curve"]
    if not points:
        return {"stage": stage["stage"], "rises": False, "points": 0}
    first, best_steps = points[0], max(p["mean_steps"] for p in points)
    best_survival = max(p["survival"] for p in points)
    return {
        "stage": stage["stage"],
        "points": len(points),
        "first_mean_steps": round(first["mean_steps"], 2),
        "best_mean_steps": round(best_steps, 2),
        "first_survival": round(first["survival"], 4),
        "best_survival": round(best_survival, 4),
        "final_survival": round(stage["final"]["survival"], 4),
        "rises": best_steps > first["mean_steps"] or best_survival > first["survival"],
    }


def duty_cycle_criterion(run: dict[str, Any]) -> dict[str, Any]:
    """Criterion 6: the trend, reported either way."""
    points = [point for stage in run["stages"] for point in stage["curve"]]
    if not points:
        return {"logged": False, "points": 0}
    first, last = points[0]["camera_duty_cycle"], points[-1]["camera_duty_cycle"]
    lowest = min(point["camera_duty_cycle"] for point in points)
    if last < first * 0.9:
        trend = "falling"
    elif last > first * 1.1:
        trend = "rising"
    else:
        trend = "flat"
    return {
        "logged": True,
        "points": len(points),
        "first": round(first, 4),
        "last": round(last, 4),
        "lowest": round(lowest, 4),
        "reference": RHYTHM_DUTY_CYCLE,
        "trend": trend,
        "below_reference": last < RHYTHM_DUTY_CYCLE,
    }


def main(argv: list[str] | None = None) -> int:
    """Score everything on one seed set and write the summary."""
    args = parse_args(argv)
    seeds = eval_seeds(args.episodes, args.seed_offset)

    runs = {
        name: load_run(directory)
        for name, directory in (
            ("base", args.base),
            ("oracle", args.oracle),
            ("matched_base", args.matched_base),
            ("sparse", args.sparse),
            ("no_memory", args.no_memory),
        )
    }
    present = {name: run for name, run in runs.items() if run is not None}

    references: dict[str, dict[str, Any]] = {}
    policy_scores: dict[str, dict[str, Any]] = {}
    base_policy = final_policy_path(present["base"])
    for name in EVAL_CONFIGS:
        config = resolve(name)
        topology = load_topology(config.topology_path())
        references[name] = {
            "do_nothing": evaluate_scripted(config, DoNothing, seeds, topology).as_dict(),
            "rhythm": evaluate_scripted(config, Rhythm, seeds, topology).as_dict(),
        }
        print(f"scored references on {name}", file=sys.stderr)
        policy_scores[name] = score_policy(base_policy, config, seeds, oracle=False).as_dict()
        print(f"scored the trained policy on {name}", file=sys.stderr)

    beats_do_nothing: dict[str, Any] = {}
    for name in ("night4", "night5", "night6"):
        policy = policy_scores[name]
        baseline = references[name]["do_nothing"]["survival"]
        error = policy["standard_error"]
        sigma = float("inf") if error == 0 else (policy["survival"] - baseline) / error
        beats_do_nothing[name] = {
            "policy": round(policy["survival"], 4),
            "do_nothing": round(baseline, 4),
            "episodes": args.episodes,
            "margin_sigma": None if sigma == float("inf") else round(sigma, 2),
            "passed": policy["survival"] > baseline + 2.0 * error,
        }

    rhythm6 = references["night6"]["rhythm"]["survival"]
    policy6 = policy_scores["night6"]
    error6 = policy6["standard_error"]
    sigma6 = float("inf") if error6 == 0 else (policy6["survival"] - rhythm6) / error6

    oracle_gap: dict[str, Any] = {"config": None, "base": None, "oracle": None, "gap": None}
    if present.get("oracle") and present.get("matched_base"):
        arm = present["oracle"]["stages"][-1]["final"]["survival"]
        base_arm = present["matched_base"]["stages"][-1]["final"]["survival"]
        oracle_gap = {
            "config": present["oracle"]["stages"][-1]["stage"],
            "base": round(base_arm, 4),
            "oracle": round(arm, 4),
            "gap": round(arm - base_arm, 4),
            "note": "a lower bound: PROJECT.md 6.4",
        }

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "episodes": args.episodes,
        "seed_offset": args.seed_offset,
        "runs": {
            name: {
                "directory": run["directory"],
                "run_id": run["run_id"],
                "git_sha": run["git_sha"],
                "config_hash": run["config_hash"],
                "seed": run["seed"],
                "wall_clock_s": run["wall_clock_s"],
                "device": run.get("device"),
                "stages": [
                    {
                        "stage": stage["stage"],
                        "timesteps": stage["timesteps"],
                        "survival": stage["final"]["survival"],
                        "reference": stage["reference"],
                        "beats_reference": stage["beats_reference"],
                        "graduated": stage["graduated"],
                        "camera_duty_cycle": stage["final"]["camera_duty_cycle"],
                        "causes": stage["final"]["causes"],
                    }
                    for stage in run["stages"]
                ],
            }
            for name, run in present.items()
        },
        "references": references,
        "trained_policy": {"path": base_policy, "scores": policy_scores},
        "criteria": {
            "learning_curve": curve_criterion(present["base"]),
            "beats_do_nothing": beats_do_nothing,
            "beats_rhythm_night6": {
                "policy": round(policy6["survival"], 4),
                "rhythm": round(rhythm6, 4),
                "episodes": args.episodes,
                "margin_sigma": None if sigma6 == float("inf") else round(sigma6, 2),
                "passed": policy6["survival"] > rhythm6 + 2.0 * error6,
            },
            "oracle_gap": oracle_gap,
            "camera_duty_cycle": duty_cycle_criterion(present["base"]),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["criteria"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
