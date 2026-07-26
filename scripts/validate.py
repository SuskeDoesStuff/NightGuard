#!/usr/bin/env python3
"""Check the v0.1 exit criteria from PROJECT.md 7.

Run with ``python scripts/validate.py``. Exits non-zero if any criterion fails.

This is the v0.1 slice of the validation suite that PROJECT.md 8 grows into; the fidelity
assertions there (reference-policy survival rates, WARDEN latency, SPRINTER attack frequency,
blackout survivability) need entities and a blackout sequence that v0.1 does not have.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nightguard.core import (
    Action,
    NightSim,
    TerminationCause,
    load_night_config,
)
from nightguard.core.power import idle_drain_per_second

TOLERANCE = 1e-6
SEED = 918442

# PROJECT.md 3.3: night-1 end-of-night levels, ordered [WARDEN, DRIFTER, PROWLER, SPRINTER].
NIGHT_1_END_LEVELS = (0, 3, 2, 2)
EXPECTED_ESCALATIONS = 3
EXPECTED_BLACKOUT_TICK = 4705
EXPECTED_BLACKOUT_HOUR = 5


@dataclass
class Check:
    """One assertion and its outcome."""

    criterion: str
    detail: str
    passed: bool


def check_determinism() -> list[Check]:
    """Criterion 1: a seed plus an action script is fully deterministic."""
    config = load_night_config(4)
    rng = np.random.default_rng(0)
    script = [Action(int(v)) for v in rng.integers(0, len(Action), size=600)]

    checks = []
    for seed in (1, 2, 3):
        first = NightSim.from_seed(config, seed=seed).run(script)
        second = NightSim.from_seed(config, seed=seed).run(script)
        checks.append(
            Check("1 determinism", f"night 4, seed {seed}: identical episode", first == second)
        )

    injected = NightSim(config, np.random.default_rng(7)).run(script)
    from_seed = NightSim.from_seed(config, seed=7).run(script)
    checks.append(
        Check("1 determinism", "injected Generator matches from_seed", injected == from_seed)
    )
    return checks


def check_idle_power() -> list[Check]:
    """Criterion 2: idle power at 6AM matches the exit table to 1e-6."""
    checks = []
    for night in range(1, 7):
        config = load_night_config(night)
        result = NightSim.from_seed(config, seed=SEED).run()
        drained = config.power.start_pct - result.final_power_pct
        expected = idle_drain_per_second(config.power) * 535.0
        ok = result.cause is TerminationCause.SURVIVED and abs(drained - expected) <= TOLERANCE
        checks.append(
            Check(
                "2 idle power",
                f"night {night}: drained {drained:.7f} pp, expected {expected:.7f}",
                ok,
            )
        )
    return checks


def check_blackout_timing() -> list[Check]:
    """Criterion 3: two doors held closed all night on night 1 black out inside the 5AM block.

    Corrected from "one door": with ``active`` clamped below at 1, a single control drains at
    exactly the idle rate. See CHANGELOG.
    """
    sim = NightSim.from_seed(load_night_config(1), seed=SEED)
    sim.state.office.door_left = True
    sim.state.office.door_right = True
    result = sim.run()
    hour = sim.clock.hour_at(result.time_s)

    return [
        Check(
            "3 blackout",
            f"night 1, two doors: {result.cause.value} at t={result.time_s:.1f}s "
            f"(tick {result.ticks}), hour {hour}",
            result.cause is TerminationCause.KILLED_BLACKOUT
            and result.ticks == EXPECTED_BLACKOUT_TICK
            and hour == EXPECTED_BLACKOUT_HOUR,
        )
    ]


def check_escalation() -> list[Check]:
    """Criterion 4: boundaries fire escalation exactly once each; night 1 ends at 0/3/2/2."""
    sim = NightSim.from_seed(load_night_config(1), seed=SEED)
    result = sim.run()
    ticks = [tick for tick, name in result.events if name.startswith("escalation")]
    expected_ticks = [sim.clock.hour_boundary_ticks[hour] for hour in (2, 3, 4)]

    return [
        Check(
            "4 escalation",
            f"night 1 end-of-night levels {result.ai_levels}, expected {NIGHT_1_END_LEVELS}",
            result.ai_levels == NIGHT_1_END_LEVELS,
        ),
        Check(
            "4 escalation",
            f"{result.escalations_applied} events at ticks {ticks}",
            result.escalations_applied == EXPECTED_ESCALATIONS and ticks == expected_ticks,
        ),
    ]


def main() -> int:
    """Run every check and print a table."""
    checks: list[Check] = []
    checks += check_determinism()
    checks += check_idle_power()
    checks += check_blackout_timing()
    checks += check_escalation()

    width = max(len(check.criterion) for check in checks)
    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"[{mark}] {check.criterion:<{width}}  {check.detail}")

    failed = [check for check in checks if not check.passed]
    print()
    if failed:
        print(f"{len(failed)} of {len(checks)} v0.1 exit criteria checks FAILED")
        return 1
    print(f"all {len(checks)} v0.1 exit criteria checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
