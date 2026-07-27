#!/usr/bin/env python3
"""Write a JSONL trace for the viewer. PROJECT.md 5 and 9.

Examples::

    python scripts/export_trace.py --night 4 --seed 11 --policy rhythm
    python scripts/export_trace.py --night 1 --force-blackout-at 500 --no-entities
    python scripts/export_trace.py --night 6 --stride 5 --out trace.jsonl

Drop the result next to `src/nightguard/viewer/index.html` as `trace.jsonl`, or load it with the
viewer's file picker.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nightguard.core import Action, NightConfig, NightSim, load_night_config
from nightguard.policies import DoNothing, MonitorDown, RandomPolicy, Rhythm, observe
from nightguard.trace import TraceWriter

POLICIES = {
    "do_nothing": DoNothing,
    "random": RandomPolicy,
    "rhythm": Rhythm,
    "monitor_down": MonitorDown,
}


def disable_entities(config: NightConfig) -> NightConfig:
    """Turn the whole roster off, so a run reaches a forced blackout without being killed first."""
    entities = config.entities
    return replace(
        config,
        entities=replace(
            entities,
            warden=replace(entities.warden, enabled=False),
            drifter=replace(entities.drifter, enabled=False),
            prowler=replace(entities.prowler, enabled=False),
            sprinter=replace(entities.sprinter, enabled=False),
        ),
    )


def main() -> int:
    """Parse arguments, run one episode, and write its trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--night", type=int, default=4, choices=range(1, 7))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="rhythm")
    parser.add_argument("--stride", type=int, default=1, help="1 record per N sim ticks")
    parser.add_argument("--out", type=Path, default=Path("trace.jsonl"))
    parser.add_argument(
        "--force-blackout-at",
        type=float,
        default=None,
        metavar="SECONDS",
        help="force blackout onset at this time, as PROJECT.md 8.7 does",
    )
    parser.add_argument("--no-entities", action="store_true", help="disable the whole roster")
    args = parser.parse_args()

    config = load_night_config(args.night)
    if args.no_entities:
        config = disable_entities(config)

    sim = NightSim.from_seed(config, seed=args.seed)
    policy = POLICIES[args.policy]()
    policy.reset()
    onset_tick = (
        None if args.force_blackout_at is None else sim.clock.to_ticks(args.force_blackout_at)
    )

    with TraceWriter(
        args.out, sim, night=args.night, seed=args.seed, stride=args.stride, policy=args.policy
    ):
        # Force by zeroing power one tick early and letting 3.13 step 3 detect the crossing, so
        # onset happens in-band: the event stamp and phase_started_tick agree, and the trace shows
        # a blackout with an empty battery rather than a quarter of one.
        emit = sim.on_tick

        def hook(state, action):
            if onset_tick is not None and state.tick == onset_tick - 1:
                state.power_pct = 0.0
            if emit is not None:
                emit(state, action)

        sim.on_tick = hook
        step = 0
        while not sim.state.terminated:
            action = Action.NOOP if sim.state.blackout else policy(observe(sim, sim.state, step))
            sim.step(action)
            step += 1
        sim.on_tick = emit

    cause = sim.state.cause.value if sim.state.cause else "NONE"
    print(f"{args.out}: night {args.night}, seed {args.seed}, {args.policy} -> {cause}")
    print(f"  {sim.state.tick} ticks, stride {args.stride}, {args.out.stat().st_size // 1024} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
