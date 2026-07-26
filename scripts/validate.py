#!/usr/bin/env python3
"""Check the v0.1 and v0.2 exit criteria from PROJECT.md 7.

Run with ``python scripts/validate.py``. Exits non-zero if any criterion fails.

Every threshold prints its measured value alongside the target, per PROJECT.md 8.0: a check that
passes by a factor of ten should be visible, not silently green.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nightguard.core import (
    Action,
    EntityId,
    NightConfig,
    NightSim,
    Node,
    TerminationCause,
    load_night_config,
    load_topology,
)
from nightguard.core.power import idle_drain_per_second
from nightguard.policies import DoNothing, MonitorDown, Rhythm, run_policy
from nightguard.trace import write_episode

TOLERANCE = 1e-6
SEED = 918442
NIGHT_LENGTH_S = 535.0

NIGHT_1_END_LEVELS = (0, 3, 2, 2)
EXPECTED_ESCALATIONS = 3
EXPECTED_ESCALATION_TICKS = [1790, 2680, 3570]
EXPECTED_BLACKOUT_TICK = 4705
EXPECTED_BLACKOUT_HOUR = 5

# PROJECT.md 8.2, derived in CHANGELOG from 3.1, 3.3 and 3.7 before SPRINTER was implemented.
DERIVED_NIGHT_1_SURVIVAL = 0.2397
SURVIVAL_EPISODES = 10_000
SIGMA_TOLERANCE = 4.0

TRACE_SEEDS = 40
MIN_DISTINCT_TRACES = 10
PROBE_EPISODES = 300

TOPOLOGY = load_topology(NightConfig().topology_path())


@dataclass
class Check:
    """One assertion and its outcome."""

    criterion: str
    detail: str
    passed: bool


def _no_entities(config: NightConfig) -> NightConfig:
    """Disable every entity: for checks that measure the power or AI subsystems, not survival."""
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


def _only(config: NightConfig, entity: EntityId, levels: list[int]) -> NightConfig:
    """Enable exactly one entity, at fixed levels with no escalation."""
    entities = config.entities
    config = replace(config, ai=replace(config.ai, levels=tuple(levels), escalation=()))
    return replace(
        config,
        entities=replace(
            config.entities,
            warden=replace(entities.warden, enabled=entity is EntityId.WARDEN),
            drifter=replace(entities.drifter, enabled=entity is EntityId.DRIFTER),
            prowler=replace(entities.prowler, enabled=entity is EntityId.PROWLER),
            sprinter=replace(entities.sprinter, enabled=entity is EntityId.SPRINTER),
        ),
    )


def _sim(config: NightConfig, seed: int) -> NightSim:
    return NightSim.from_seed(config, seed=seed, topology=TOPOLOGY)


def _random_script(seed: int, length: int = 1200) -> list[Action]:
    rng = np.random.default_rng(seed)
    return [Action(int(value)) for value in rng.integers(0, len(Action), size=length)]


# --- v0.1 --------------------------------------------------------------------------------------


def check_determinism() -> list[Check]:
    """v0.1 criterion 1: a seed plus an action script is fully deterministic."""
    config = load_night_config(4)
    script = _random_script(0)
    checks = []
    for seed in (1, 2, 3):
        first = _sim(config, seed).run(script)
        second = _sim(config, seed).run(script)
        checks.append(
            Check("v0.1-1 determinism", f"night 4, seed {seed}: identical episode", first == second)
        )
    injected = NightSim(config, np.random.default_rng(7), topology=TOPOLOGY).run(script)
    checks.append(
        Check(
            "v0.1-1 determinism",
            "injected Generator matches from_seed",
            injected == _sim(config, 7).run(script),
        )
    )
    return checks


def check_idle_power() -> list[Check]:
    """v0.1 criterion 2: idle power at 6AM matches the closed form to 1e-6."""
    checks = []
    for night in range(1, 7):
        config = load_night_config(night)
        result = _sim(_no_entities(config), SEED).run()
        drained = config.power.start_pct - result.final_power_pct
        expected = idle_drain_per_second(config.power) * NIGHT_LENGTH_S
        ok = result.cause is TerminationCause.SURVIVED and abs(drained - expected) <= TOLERANCE
        checks.append(
            Check(
                "v0.1-2 idle power",
                f"night {night}: drained {drained:.7f} pp, closed form {expected:.7f}, "
                f"err {abs(drained - expected):.2e}",
                ok,
            )
        )
    return checks


def check_blackout_timing() -> list[Check]:
    """v0.1 criterion 3: two doors held closed all night on night 1 black out inside 5AM."""
    sim = _sim(_no_entities(load_night_config(1)), SEED)
    sim.state.office.door_left = True
    sim.state.office.door_right = True
    result = sim.run()
    hour = sim.clock.hour_at(result.time_s)
    return [
        Check(
            "v0.1-3 blackout",
            f"night 1, two doors: {result.cause.value} at t={result.time_s:.1f}s "
            f"(tick {result.ticks}, target {EXPECTED_BLACKOUT_TICK}), hour {hour}",
            result.cause is TerminationCause.KILLED_BLACKOUT
            and result.ticks == EXPECTED_BLACKOUT_TICK
            and hour == EXPECTED_BLACKOUT_HOUR,
        )
    ]


def check_escalation() -> list[Check]:
    """v0.1 criterion 4: three escalation events at the 2AM, 3AM and 4AM boundaries."""
    sim = _sim(_no_entities(load_night_config(1)), SEED)
    result = sim.run()
    ticks = [tick for tick, name in result.events if name.startswith("escalation")]
    return [
        Check(
            "v0.1-4 escalation",
            f"night 1 end levels {result.ai_levels}, expected {NIGHT_1_END_LEVELS}",
            result.ai_levels == NIGHT_1_END_LEVELS,
        ),
        Check(
            "v0.1-4 escalation",
            f"{result.escalations_applied} events at ticks {ticks} "
            f"(expected {EXPECTED_ESCALATIONS} at {EXPECTED_ESCALATION_TICKS})",
            result.escalations_applied == EXPECTED_ESCALATIONS
            and ticks == EXPECTED_ESCALATION_TICKS,
        ),
    ]


# --- v0.2 --------------------------------------------------------------------------------------


def check_all_entities_kill() -> list[Check]:
    """v0.2 criterion 1: all four entities can produce a kill, by seeded scenario."""
    base = load_night_config(6)
    seen: dict[EntityId, TerminationCause | None] = {}

    sim = _sim(_only(base, EntityId.WARDEN, [20, 0, 0, 0]), 0)
    sim.state.warden.node = Node.OFFICE
    sim.state.warden.path_index = len(sim.warden.path) - 1
    while not sim.state.terminated and sim.state.tick < 2000:
        sim.step(Action.NOOP)
    seen[EntityId.WARDEN] = sim.state.cause

    for entity, levels, name in (
        (EntityId.DRIFTER, [0, 20, 0, 0], "drifter"),
        (EntityId.PROWLER, [0, 0, 20, 0], "prowler"),
    ):
        sim = _sim(_only(base, entity, levels), 0)
        getattr(sim.state, name).node = getattr(sim, name).corner
        while not getattr(sim.state, name).in_office and sim.state.tick < 600:
            sim.step(Action.SELECT_CAM_0)
        sim.step(Action.MONITOR_DOWN)
        seen[entity] = sim.state.cause

    sim = _sim(_only(base, EntityId.SPRINTER, [0, 0, 0, 20]), 0)
    sim.state.sprinter.stage = sim.config.entities.sprinter.stages_to_arm
    sim.state.sprinter.armed_at_tick = sim.state.tick
    while not sim.state.terminated and sim.state.tick < 2000:
        sim.step(Action.NOOP)
    seen[EntityId.SPRINTER] = sim.state.cause

    return [
        Check(
            "v0.2-1 kills",
            f"{entity.name}: {None if cause is None else cause.value}",
            cause is not None and cause.name == f"KILLED_{entity.name}",
        )
        for entity, cause in seen.items()
    ]


def check_trace_determinism() -> list[Check]:
    """v0.2 criterion 2: byte-identical traces, with the 8.0 non-vacuity check first."""
    config = load_night_config(4)
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        digests = set()
        for seed in range(TRACE_SEEDS):
            path = write_episode(
                root / f"a{seed}.jsonl",
                _sim(config, seed),
                night=4,
                actions=_random_script(seed),
                seed=seed,
            )
            digests.add(hashlib.sha256(path.read_bytes()).hexdigest())
        checks.append(
            Check(
                "v0.2-2 trace",
                f"non-vacuity: {len(digests)} distinct traces over {TRACE_SEEDS} seeds "
                f"(need >= {MIN_DISTINCT_TRACES})",
                len(digests) >= MIN_DISTINCT_TRACES,
            )
        )
        identical = True
        for seed in (1, 2, 3):
            script = _random_script(seed)
            first = write_episode(
                root / f"x{seed}.jsonl", _sim(config, seed), night=4, actions=script, seed=seed
            )
            second = write_episode(
                root / f"y{seed}.jsonl", _sim(config, seed), night=4, actions=script, seed=seed
            )
            identical &= first.read_bytes() == second.read_bytes()
        checks.append(
            Check("v0.2-2 trace", "same seed and script give byte-identical files", identical)
        )
    return checks


def check_warden_countdown() -> list[Check]:
    """v0.2 criterion 3: the countdown is not paused by monitor raises."""
    sim = _sim(_only(load_night_config(6), EntityId.WARDEN, [1, 0, 0, 0]), 0)
    sim.state.drifter.node = Node.COMMONS
    sim.state.prowler.node = Node.COMMONS
    sim.state.warden.countdown_units = sim.clock.to_units(15.0)
    start = sim.state.warden.node
    while not sim.state.terminated and sim.state.warden.node == start and sim.state.tick < 400:
        sim.step(Action.SELECT_CAM_0)
    moved = sim.state.warden.node != start
    return [
        Check(
            "v0.2-3 countdown",
            f"moved to {sim.state.warden.node.name} at tick {sim.state.tick} "
            f"with the monitor held up (expected ~150 ticks)",
            moved and sim.state.office.monitor_up,
        )
    ]


def check_sprinter_freeze() -> list[Check]:
    """v0.2 criterion 4: no advance with the monitor up, none during the immunity window."""
    config = _only(load_night_config(6), EntityId.SPRINTER, [0, 0, 0, 20])

    frozen = _sim(config, 0)
    for _ in range(300):
        if frozen.state.terminated:
            break
        frozen.step(Action.SELECT_CAM_7)

    immune = _sim(config, 0)
    immune.state.sprinter.immune_until_tick = immune.clock.total_ticks
    for _ in range(300):
        if immune.state.terminated:
            break
        immune.step(Action.NOOP)

    free = _sim(config, 0)
    while not free.state.terminated and free.state.sprinter.stage == 0 and free.state.tick < 400:
        free.step(Action.NOOP)

    return [
        Check(
            "v0.2-4 sprinter",
            f"monitor up: stage {frozen.state.sprinter.stage} after "
            f"{frozen.state.sprinter.fire_count} opportunities",
            frozen.state.sprinter.stage == 0 and frozen.state.sprinter.fire_count > 0,
        ),
        Check(
            "v0.2-4 sprinter",
            f"immunity window: stage {immune.state.sprinter.stage}",
            immune.state.sprinter.stage == 0,
        ),
        Check(
            "v0.2-4 sprinter",
            f"non-vacuity: unfrozen stage reaches {free.state.sprinter.stage}",
            free.state.sprinter.stage > 0,
        ),
    ]


def check_stage_lock() -> list[Check]:
    """v0.2 criterion 5: WARDEN cannot leave STAGE while a door entity is there."""
    sim = _sim(_only(load_night_config(6), EntityId.WARDEN, [20, 0, 0, 0]), 0)
    for _ in range(300):
        if sim.state.terminated:
            break
        sim.step(Action.NOOP)
    locked = sim.state.warden.node is Node.STAGE

    free = _sim(_only(load_night_config(6), EntityId.WARDEN, [20, 0, 0, 0]), 0)
    free.state.drifter.node = Node.COMMONS
    free.state.prowler.node = Node.COMMONS
    while (
        not free.state.terminated and free.state.warden.node is Node.STAGE and free.state.tick < 400
    ):
        free.step(Action.NOOP)

    return [
        Check(
            "v0.2-5 stage lock",
            f"held for {sim.state.tick} ticks with both door entities on STAGE",
            locked,
        ),
        Check(
            "v0.2-5 stage lock",
            f"non-vacuity: with STAGE clear, WARDEN reached {free.state.warden.node.name}",
            free.state.warden.node is not Node.STAGE,
        ),
    ]


def _survival(night: int, factory, episodes: int) -> float:
    config = load_night_config(night)
    survived = 0
    for seed in range(episodes):
        sim = _sim(config, seed)
        run_policy(sim, factory())
        survived += sim.state.cause is TerminationCause.SURVIVED
    return survived / episodes


def check_night_one_survival() -> list[Check]:
    """v0.2 criterion 6: measured night-1 do_nothing survival matches the derivation."""
    measured = _survival(1, DoNothing, SURVIVAL_EPISODES)
    sigma = (DERIVED_NIGHT_1_SURVIVAL * (1 - DERIVED_NIGHT_1_SURVIVAL) / SURVIVAL_EPISODES) ** 0.5
    deviation = abs(measured - DERIVED_NIGHT_1_SURVIVAL) / sigma
    return [
        Check(
            "v0.2-6 derivation",
            f"do_nothing night 1: measured {measured:.4f}, derived "
            f"{DERIVED_NIGHT_1_SURVIVAL:.4f}, {deviation:.2f} sigma at n={SURVIVAL_EPISODES}",
            deviation <= SIGMA_TOLERANCE,
        )
    ]


def check_monitor_down_probe() -> list[Check]:
    """v0.2 criterion 7: report the monitor-down probe. A finding, not a pass/fail gate."""
    checks = []
    for night in (5, 6):
        rhythm = _survival(night, Rhythm, PROBE_EPISODES)
        probe = _survival(night, MonitorDown, PROBE_EPISODES)
        verdict = "probe wins - investigate" if probe > rhythm else "rhythm holds"
        checks.append(
            Check(
                "v0.2-7 probe",
                f"night {night}: rhythm {rhythm:.3f} vs monitor_down {probe:.3f} ({verdict})",
                True,  # reporting obligation only
            )
        )
    return checks


def main() -> int:
    """Run every check and print a table."""
    checks: list[Check] = []
    for step in (
        check_determinism,
        check_idle_power,
        check_blackout_timing,
        check_escalation,
        check_all_entities_kill,
        check_trace_determinism,
        check_warden_countdown,
        check_sprinter_freeze,
        check_stage_lock,
        check_night_one_survival,
        check_monitor_down_probe,
    ):
        checks += step()

    width = max(len(check.criterion) for check in checks)
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.criterion:<{width}}  {check.detail}")

    failed = [check for check in checks if not check.passed]
    print()
    if failed:
        print(f"{len(failed)} of {len(checks)} exit-criteria checks FAILED")
        return 1
    print(f"all {len(checks)} exit-criteria checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
