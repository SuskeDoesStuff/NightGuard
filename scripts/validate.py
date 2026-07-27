#!/usr/bin/env python3
"""Check the v0.1 and v0.2 exit criteria from PROJECT.md 7.

Run with ``python scripts/validate.py``. Exits non-zero if any criterion fails.

Every threshold prints its measured value alongside the target, per PROJECT.md 8.0: a check that
passes by a factor of ten should be visible, not silently green.
"""

from __future__ import annotations

import hashlib
import platform
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nightguard import derivations
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
from nightguard.core.blackout import apply_onset
from nightguard.core.power import idle_drain_per_second
from nightguard.core.state import action_for_camera
from nightguard.env import NightGuardEnv
from nightguard.env import obs as obs_mod
from nightguard.env.actions import ACTION_COUNT
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
    onset = [tick for tick, name in result.events if name == "blackout"]
    hour = sim.clock.hour_at(sim.clock.time_s(onset[0])) if onset else -1
    return [
        Check(
            "v0.1-3 blackout",
            f"night 1, two doors: onset at tick {onset[0] if onset else None} "
            f"(target {EXPECTED_BLACKOUT_TICK}), hour {hour}, ended {result.cause.value}",
            onset == [EXPECTED_BLACKOUT_TICK] and hour == EXPECTED_BLACKOUT_HOUR,
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


# --- v0.3: the section 8 validation suite -------------------------------------------------------

BLACKOUT_ONSET_TICK = 5000  # PROJECT.md 8.7: power forced to 0 at t = 500 s
BLACKOUT_EPISODES = 10_000
LATENCY_EPISODES = 600
LATENCY_BAND = 0.08
LATENCY_MEAN_BAND = 0.04
NIGHT_SCALE = 4
NEVER_ARMS = 10**6
UNFROZEN_BAND = 0.02
HARD_ZERO_PERIODS = (0.5, 1.0)
CURVE_PERIODS = (1.5, 2.0, 4.0, 6.0, 10.0, 20.0, None)


def check_do_nothing_survival() -> list[Check]:
    """8.2: every night's do_nothing survival against its derivation."""
    checks = []
    rates = []
    for night in range(1, 7):
        config = load_night_config(night)
        derived = derivations.do_nothing_survival(config)
        episodes = SURVIVAL_EPISODES if night == 1 else 2000
        measured = _survival(night, DoNothing, episodes)
        rates.append(measured)
        sigma = max((derived * (1 - derived) / episodes) ** 0.5, 1.0 / episodes)
        deviation = abs(measured - derived) / sigma
        checks.append(
            Check(
                "v0.3-8.2 survival",
                f"night {night}: measured {measured:.4f}, derived {derived:.4f}, "
                f"{deviation:.2f} sigma at n={episodes}",
                deviation <= SIGMA_TOLERANCE,
            )
        )
    checks.append(
        Check(
            "v0.3-8.2 survival",
            f"non-increasing across nights: {[round(r, 4) for r in rates]}",
            all(a >= b for a, b in pairwise(rates)),
        )
    )
    for night in (3, 4, 5, 6):
        rhythm = _survival(night, Rhythm, 300)
        nothing = _survival(night, DoNothing, 300)
        checks.append(
            Check(
                "v0.3-8.2 rhythm",
                f"night {night}: rhythm {rhythm:.3f} beats do_nothing {nothing:.3f}",
                rhythm > nothing,
            )
        )
    return checks


def _latency_config(level: int) -> NightConfig:
    config = _only(load_night_config(6), EntityId.WARDEN, [level, 0, 0, 0])
    return replace(
        config,
        timing=replace(
            config.timing,
            hour_durations_s=tuple(d * NIGHT_SCALE for d in config.timing.hour_durations_s),
        ),
    )


def check_warden_latency() -> list[Check]:
    """8.3: STAGE to E_CORNER latency against the derived curve."""
    residuals = []
    checks = []
    means = []
    for level in range(1, 11):
        config = _latency_config(level)
        predicted = derivations.warden_latency_s(config, level)
        total = reached = 0
        for seed in range(LATENCY_EPISODES):
            sim = _sim(config, seed)
            sim.state.drifter.node = Node.COMMONS
            sim.state.prowler.node = Node.COMMONS
            while not sim.state.terminated and sim.state.warden.node is not Node.E_CORNER:
                sim.step(Action.NOOP)
            if sim.state.warden.node is Node.E_CORNER:
                total += sim.clock.time_s(sim.state.tick)
                reached += 1
        measured = total / reached
        means.append(measured)
        residuals.append((measured - predicted) / predicted)
        checks.append(
            Check(
                "v0.3-8.3 latency",
                f"AI {level:>2}: measured {measured:7.2f}s, derived {predicted:7.2f}s, "
                f"{residuals[-1]:+.1%}, reached {reached}/{LATENCY_EPISODES}",
                abs(residuals[-1]) <= LATENCY_BAND and reached >= LATENCY_EPISODES * 0.95,
            )
        )
    average = sum(abs(r) for r in residuals) / len(residuals)
    checks.append(
        Check(
            "v0.3-8.3 latency",
            f"mean absolute residual {average:.2%}",
            average <= LATENCY_MEAN_BAND,
        )
    )
    checks.append(
        Check(
            "v0.3-8.3 latency",
            "monotonically decreasing in AI level",
            all(a > b for a, b in pairwise(means)),
        )
    )
    return checks


def _peek_action(period_s: float | None, step: int) -> Action:
    if period_s is None:
        return Action.NOOP
    steps = round(period_s / 0.5)
    return action_for_camera(Node.COVE) if step % steps == 0 else Action.MONITOR_DOWN


def check_sprinter_curve() -> list[Check]:
    """8.4: the hard zero, and agreement with the unfrozen-fraction curve."""
    checks = []
    base = load_night_config(6)
    bound = derivations.sprinter_hard_zero_bound_s(base)

    for period in HARD_ZERO_PERIODS:
        config = _only(base, EntityId.SPRINTER, [0, 0, 0, 20])
        attacks = 0
        for seed in range(40):
            sim = _sim(config, seed)
            sim.state.office.door_left = True
            step = 0
            while not sim.state.terminated:
                sim.step(_peek_action(period, step))
                step += 1
            attacks += sim.state.sprinter.bang_count
        checks.append(
            Check(
                "v0.3-8.4 hard zero",
                f"k={period}s (bound {bound}s): {attacks} attacks over 40 seeds",
                attacks == 0,
            )
        )

    for period in HARD_ZERO_PERIODS + CURVE_PERIODS:
        config = _only(base, EntityId.SPRINTER, [0, 0, 0, 20])
        config = replace(
            config,
            entities=replace(
                config.entities,
                sprinter=replace(config.entities.sprinter, stages_to_arm=NEVER_ARMS),
            ),
        )
        predicted = derivations.sprinter_unfrozen_fraction(config, period)
        unfrozen = total = 0
        for seed in range(30):
            sim = _sim(config, seed)
            counters = [0, 0]

            def hook(state, _action, s=sim, c=counters):
                if not state.blackout:
                    c[1] += 1
                    c[0] += 0 if s.sprinter.is_frozen(state) else 1

            sim.on_tick = hook
            step = 0
            while not sim.state.terminated:
                sim.step(_peek_action(period, step))
                step += 1
            unfrozen += counters[0]
            total += counters[1]
        measured = unfrozen / total
        checks.append(
            Check(
                "v0.3-8.4 curve",
                f"k={'inf' if period is None else period}: measured {measured:.4f}, "
                f"derived {predicted:.4f}",
                abs(measured - predicted) <= UNFROZEN_BAND,
            )
        )
    return checks


def check_blackout() -> list[Check]:
    """8.7: blackout survivability against the derivation."""
    config = _no_entities(load_night_config(1))
    derived = derivations.blackout_survival(config, 35.0)
    causes: dict[str, int] = {}
    for seed in range(BLACKOUT_EPISODES):
        sim = _sim(config, seed)
        while sim.state.tick < BLACKOUT_ONSET_TICK:
            sim.step(Action.NOOP)
        apply_onset(sim.state)
        sim.run()
        cause = sim.state.cause.value if sim.state.cause else "NONE"
        causes[cause] = causes.get(cause, 0) + 1
    measured = causes.get("SURVIVED", 0) / BLACKOUT_EPISODES
    sigma = (derived * (1 - derived) / BLACKOUT_EPISODES) ** 0.5
    deviation = abs(measured - derived) / sigma
    return [
        Check(
            "v0.3-8.7 blackout",
            f"non-vacuity: both outcomes occur - {causes}",
            causes.get("SURVIVED", 0) > 0 and causes.get("KILLED_BLACKOUT", 0) > 0,
        ),
        Check(
            "v0.3-8.7 blackout",
            f"measured {measured:.4f}, derived {derived:.4f}, {deviation:.2f} sigma "
            f"at n={BLACKOUT_EPISODES}",
            deviation <= SIGMA_TOLERANCE,
        ),
    ]


# --- v1.0: the Gymnasium environment ------------------------------------------------------------

RANDOM_EPISODES_PER_NIGHT = 10_000 // 6
ROLLOUT_STEPS = 100_000
ROLLOUT_BUDGET_S = 60.0


def _rollout(env: NightGuardEnv, seed: int) -> list[Any]:
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = [observation]
    while True:
        observation, _, terminated, truncated, _ = env.step(int(rng.integers(ACTION_COUNT)))
        out.append(observation)
        if terminated or truncated:
            return out


def check_env_api() -> list[Check]:
    """v1.0 criterion 1: check_env passes with no warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(NightGuardEnv(night=4), skip_render_check=True)
    return [
        Check(
            "v1.0-1 check_env",
            f"{len(caught)} warnings"
            + (f": {[str(w.message)[:60] for w in caught]}" if caught else ""),
            not caught,
        )
    ]


def check_random_episodes() -> list[Check]:
    """v1.0 criteria 2 and 6: episodes complete, and observations stay inside the Box."""
    checks = []
    total = 0
    out_of_bounds = 0
    negative_power = 0.0
    for night in range(1, 7):
        env = NightGuardEnv(night=night, topology=TOPOLOGY)
        causes: dict[str, int] = {}
        for seed in range(RANDOM_EPISODES_PER_NIGHT):
            for observation in _rollout(env, seed):
                if not env.observation_space.contains(observation):
                    out_of_bounds += 1
            negative_power = min(negative_power, env.sim.state.power_pct)
            cause = env.sim.state.cause.value if env.sim.state.cause else "NONE"
            causes[cause] = causes.get(cause, 0) + 1
            total += 1
        checks.append(
            Check(
                "v1.0-2 episodes", f"night {night}: {sum(causes.values())} episodes, {causes}", True
            )
        )
    checks.append(
        Check("v1.0-2 episodes", f"{total} episodes completed without exception", total > 0)
    )
    checks.append(
        Check(
            "v1.0-6 bounds",
            f"{out_of_bounds} observations outside the declared Box",
            out_of_bounds == 0,
        )
    )
    checks.append(
        Check(
            "v1.0-6 power",
            f"lowest power_pct across the run: {negative_power:.6f}",
            negative_power >= 0.0,
        )
    )
    return checks


def check_no_position_leak() -> list[Check]:
    """v1.0 criterion 3: the most important test in the milestone."""
    checks = []
    for entity in EntityId:
        env = NightGuardEnv(night=6, topology=TOPOLOGY)
        env.reset(seed=0)
        for _ in range(20):
            env.step(Action.NOOP)
        office = env.sim.state.office
        office.monitor_up = False
        office.light_left = office.light_right = False

        def encode(bound: NightGuardEnv = env) -> Any:
            return obs_mod.encode(bound.sim, bound.sim.state, bound._belief, Action.NOOP)

        baseline = encode()
        leaked = False
        values = range(4) if entity is EntityId.SPRINTER else [int(n) for n in Node]
        for value in values:
            if entity is EntityId.SPRINTER:
                env.sim.state.sprinter.stage = value
            elif entity is EntityId.WARDEN:
                env.sim.state.warden.node = Node(value)
            else:
                env.sim.state.entity(entity).node = Node(value)
            leaked |= not bool((encode() == baseline).all())
        checks.append(
            Check(
                "v1.0-3 no leak",
                f"{entity.name}: hidden position does not reach the observation",
                not leaked,
            )
        )

    # Non-vacuity: the observation must be capable of changing at all.
    env = NightGuardEnv(night=6, topology=TOPOLOGY)
    env.reset(seed=3)
    env.sim.state.office.monitor_up = True
    env.sim.state.office.selected_camera = env.sim.drifter.corner
    env.sim.state.drifter.node = env.sim.drifter.corner
    env._belief.update(env.sim, env.sim.state)
    visible = obs_mod.encode(env.sim, env.sim.state, env._belief, Action.NOOP)
    env.sim.state.drifter.node = Node.W_BACKSTAGE
    env._belief.update(env.sim, env.sim.state)
    hidden = obs_mod.encode(env.sim, env.sim.state, env._belief, Action.NOOP)
    checks.append(
        Check(
            "v1.0-3 no leak",
            "non-vacuity: a visible entity does change the observation",
            not bool((visible == hidden).all()),
        )
    )
    return checks


def check_reset_determinism() -> list[Check]:
    """v1.0 criterion 5."""
    env = NightGuardEnv(night=5, topology=TOPOLOGY)
    script = [int(v) for v in np.random.default_rng(0).integers(0, ACTION_COUNT, size=400)]

    def run() -> list[tuple[float, float]]:
        observation, _ = env.reset(seed=99)
        out = [(float(observation.sum()), 0.0)]
        for action in script:
            observation, reward, terminated, truncated, _ = env.step(action)
            out.append((float(observation.sum()), reward))
            if terminated or truncated:
                break
        return out

    return [
        Check(
            "v1.0-5 determinism",
            "reset(seed=99) twice gives identical trajectories",
            run() == run(),
        )
    ]


def check_throughput() -> list[Check]:
    """v1.0 criterion 4, restated by its purpose. Records the figure and the hardware."""
    env = NightGuardEnv(night=4, topology=TOPOLOGY)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    start = time.perf_counter()
    done = 0
    while done < ROLLOUT_STEPS:
        _, _, terminated, truncated, _ = env.step(int(rng.integers(ACTION_COUNT)))
        done += 1
        if terminated or truncated:
            env.reset(seed=done)
    elapsed = time.perf_counter() - start
    machine = f"{platform.machine()} {platform.system()}, Python {platform.python_version()}"
    return [
        Check(
            "v1.0-4 throughput",
            f"{ROLLOUT_STEPS:,} steps with observation encoding in {elapsed:.1f}s "
            f"({ROLLOUT_STEPS / elapsed:,.0f} steps/s) on {machine}",
            elapsed < ROLLOUT_BUDGET_S,
        )
    ]


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
        check_do_nothing_survival,
        check_warden_latency,
        check_sprinter_curve,
        check_blackout,
        check_env_api,
        check_random_episodes,
        check_no_position_leak,
        check_reset_determinism,
        check_throughput,
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
