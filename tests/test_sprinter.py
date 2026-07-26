"""SPRINTER. PROJECT.md 3.7, and v0.2 exit criterion 4."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise

import pytest

from nightguard import derivations
from nightguard.core import (
    Action,
    EntityId,
    NightConfig,
    NightSim,
    Node,
    TerminationCause,
    Topology,
    action_for_camera,
    load_night_config,
)
from tests.conftest import step_until, with_only

SPRINTER_ONLY = [EntityId.SPRINTER]


def sprinter_sim(make_sim: Callable[..., NightSim], level: int = 20, seed: int = 0) -> NightSim:
    """A SPRINTER-only night with no escalation."""
    return make_sim(levels=[0, 0, 0, level], escalation=(), only=SPRINTER_ONLY, seed=seed)


def arm(sim: NightSim) -> None:
    """Put SPRINTER in the armed state without waiting for the rolls."""
    sim.state.sprinter.stage = sim.config.entities.sprinter.stages_to_arm
    sim.state.sprinter.armed_at_tick = sim.state.tick


class TestFreeze:
    """v0.2 exit criterion 4."""

    def test_cannot_advance_while_the_monitor_is_up(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        """The freeze is universal: any camera, not specifically COVE."""
        sim = sprinter_sim(make_sim)
        for _ in range(300):
            if sim.state.terminated:
                break
            sim.step(Action.SELECT_CAM_7)
        assert sim.state.sprinter.stage == 0
        assert sim.state.sprinter.fire_count > 0, "opportunities must still fire and be spent"

    def test_any_camera_freezes_not_just_the_home_node(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        for camera in (Action.SELECT_CAM_0, Action.SELECT_CAM_2, Action.SELECT_CAM_9):
            sim = sprinter_sim(make_sim)
            for _ in range(120):
                sim.step(camera)
            assert sim.state.sprinter.stage == 0

    def test_immunity_is_sampled_on_the_monitor_down_edge(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        sim = sprinter_sim(make_sim)
        sim.step(Action.SELECT_CAM_0)
        assert sim.state.sprinter.immune_until_tick == 0

        lowered_at = sim.state.tick
        sim.step(Action.MONITOR_DOWN)
        window = sim.state.sprinter.immune_until_tick - lowered_at
        low = sim.clock.to_ticks(sim.config.entities.sprinter.immunity_range_s[0])
        high = sim.clock.to_ticks(sim.config.entities.sprinter.immunity_range_s[1])
        assert low <= window <= high

    def test_immunity_windows_vary(self, make_sim: Callable[..., NightSim]) -> None:
        """A constant window would make the mechanic predictable and the test above vacuous."""
        windows = set()
        for seed in range(30):
            sim = sprinter_sim(make_sim, seed=seed)
            sim.step(Action.SELECT_CAM_0)
            lowered_at = sim.state.tick
            sim.step(Action.MONITOR_DOWN)
            windows.add(sim.state.sprinter.immune_until_tick - lowered_at)
        assert len(windows) > 5

    def test_cannot_advance_during_the_immunity_window(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        sim = sprinter_sim(make_sim)
        sim.state.sprinter.immune_until_tick = sim.clock.total_ticks
        for _ in range(300):
            if sim.state.terminated:
                break
            sim.step(Action.NOOP)
        assert sim.state.sprinter.stage == 0

    def test_advances_freely_with_the_monitor_down(self, make_sim: Callable[..., NightSim]) -> None:
        """The converse, so the two tests above cannot both pass on a permanently frozen entity."""
        sim = sprinter_sim(make_sim)
        assert step_until(sim, lambda s: s.state.sprinter.stage > 0, max_steps=40)


class TestCharging:
    def test_arms_at_the_configured_stage(self, make_sim: Callable[..., NightSim]) -> None:
        sim = sprinter_sim(make_sim)
        assert step_until(sim, lambda s: s.state.sprinter.armed, max_steps=60)
        assert sim.state.sprinter.stage == sim.config.entities.sprinter.stages_to_arm

    def test_level_zero_never_charges(self, make_sim: Callable[..., NightSim]) -> None:
        sim = sprinter_sim(make_sim, level=0)
        sim.run()
        assert sim.state.sprinter.stage == 0


class TestAttack:
    def test_raising_the_monitor_triggers_an_armed_attack(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        sim = sprinter_sim(make_sim)
        arm(sim)
        assert sim.state.sprinter.attack_at_tick is None
        sim.step(Action.SELECT_CAM_0)
        assert sim.state.sprinter.attack_at_tick is not None

    def test_forced_attack_after_the_timeout(self, make_sim: Callable[..., NightSim]) -> None:
        """The other half of "whichever comes first": 25 s with the monitor never raised."""
        sim = sprinter_sim(make_sim)
        arm(sim)
        armed_at = sim.state.tick
        assert step_until(sim, lambda s: s.state.sprinter.attacking, max_steps=80)
        elapsed = sim.state.sprinter.attack_at_tick
        assert elapsed is not None
        assert elapsed - armed_at >= sim.clock.to_ticks(25.0)

    def test_open_door_kills(self, make_sim: Callable[..., NightSim]) -> None:
        sim = sprinter_sim(make_sim)
        arm(sim)
        step_until(sim, lambda s: s.state.terminated, max_steps=80)
        assert sim.state.cause is TerminationCause.KILLED_SPRINTER

    def test_the_grace_period_lets_the_door_save_the_agent(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        """0.5 s of reaction time, i.e. exactly one decision step."""
        sim = sprinter_sim(make_sim)
        arm(sim)
        step_until(sim, lambda s: s.state.sprinter.attacking, max_steps=80)
        assert sim.state.audio.running, "the running cue must be audible during the grace period"

        sim.step(Action.TOGGLE_DOOR_LEFT)
        assert not sim.state.terminated
        assert sim.state.sprinter.bang_count == 1

    def test_a_closed_door_banks_a_bang_not_a_death(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        sim = sprinter_sim(make_sim)
        sim.state.office.door_left = True
        arm(sim)
        step_until(sim, lambda s: s.state.sprinter.bang_count > 0, max_steps=80)
        assert not sim.state.terminated
        assert sim.state.audio.bang or sim.state.sprinter.bang_count == 1


class TestBangs:
    def test_bang_costs_escalate_by_five(self, make_sim: Callable[..., NightSim]) -> None:
        sim = sprinter_sim(make_sim)
        costs = [sim.sprinter.bang_cost_pct(n) for n in range(4)]
        assert costs == [1.0, 6.0, 11.0, 16.0]

    def test_a_bang_drains_power(self, make_sim: Callable[..., NightSim]) -> None:
        """The bang is 1.0 pp on top of the drain the night was spending anyway.

        One door closed is `active == 1` by the clamp floor, so the background rate here is the
        idle rate; the wait for the forced attack accounts for the rest of the drop.
        """
        from nightguard.core.power import drain_per_tick

        sim = sprinter_sim(make_sim)
        sim.state.office.door_left = True
        arm(sim)
        before = sim.state.power_pct
        start_tick = sim.state.tick

        step_until(sim, lambda s: s.state.sprinter.bang_count > 0, max_steps=80)

        elapsed = sim.state.tick - start_tick
        background = drain_per_tick(1, sim.config.power, sim.clock.sim_tick_s) * elapsed
        expected = sim.sprinter.bang_cost_pct(0) + background
        assert before - sim.state.power_pct == pytest.approx(expected, abs=1e-9)

    def test_stage_resets_into_the_configured_choices(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        seen = set()
        for seed in range(40):
            sim = sprinter_sim(make_sim, seed=seed)
            sim.state.office.door_left = True
            arm(sim)
            step_until(sim, lambda s: s.state.sprinter.bang_count > 0, max_steps=80)
            seen.add(sim.state.sprinter.stage)
            assert not sim.state.sprinter.armed
        assert seen == set(sim.config.entities.sprinter.reset_stage_choices)


def test_attack_frequency_increases_with_camera_gap(make_sim: Callable[..., NightSim]) -> None:
    """PROJECT.md 8.4, in miniature.

    Raising the monitor for one step every k seconds should let SPRINTER attack more often as k
    grows, and be roughly flat for k below the lower bound of the immunity window.
    """
    measured: dict[float, float] = {}
    for gap_s in (2.0, 6.0, 20.0):
        gap_steps = int(gap_s / 0.5)
        total = 0
        episodes = 12
        for seed in range(episodes):
            sim = make_sim(levels=[0, 0, 0, 10], escalation=(), only=SPRINTER_ONLY, seed=seed)
            sim.state.office.door_left = True  # bang instead of dying, so the night runs on
            step = 0
            while not sim.state.terminated:
                sim.step(Action.SELECT_CAM_2 if step % gap_steps == 0 else Action.MONITOR_DOWN)
                step += 1
            total += sim.state.sprinter.bang_count
        measured[gap_s] = total / episodes

    assert measured[2.0] <= measured[6.0] <= measured[20.0], measured
    assert measured[20.0] > measured[2.0], measured


# --- PROJECT.md 8.4: attack frequency versus camera duty ---------------------------------------

# The unfrozen-fraction curve lives in nightguard.derivations, derived in CHANGELOG from 3.7's
# freeze and immunity rules. Two constraints on the family that the original 8.4 wording missed:
# `k` must be a whole number of 0.5 s decision steps, so 0.75 and 1.25 are unreachable; and the
# hard-zero bound is 1.4 s rather than the continuous 1.33 s, because immunity is sampled in grid
# units and 249 units ceilings to 9 ticks.
HARD_ZERO_PERIODS = (0.5, 1.0)
CURVE_PERIODS = (1.5, 2.0, 4.0, 6.0, 10.0, 20.0, None)
UNFROZEN_BAND = 0.02  # absolute; the residual is tick quantisation, which shortens the window
NEVER_ARMS = 10**6  # isolates charging: with no arming there are no armed-but-idle ticks


def peek_config(period_s: float | None, level: int = 20, stages: int | None = None) -> NightConfig:
    """SPRINTER alone at a fixed level, optionally unable to arm."""
    config = load_night_config(6)
    config = replace(config, ai=replace(config.ai, levels=(0, 0, 0, level), escalation=()))
    config = with_only(config, [EntityId.SPRINTER])
    if stages is not None:
        config = replace(
            config,
            entities=replace(
                config.entities, sprinter=replace(config.entities.sprinter, stages_to_arm=stages)
            ),
        )
    return config


def peek_action(period_s: float | None, step: int, config: NightConfig) -> Action:
    """One step of a policy that raises the monitor for one step every ``period_s`` seconds."""
    if period_s is None:
        return Action.NOOP
    steps = round(period_s / config.timing.decision_step_s)
    return action_for_camera(Node.COVE) if step % steps == 0 else Action.MONITOR_DOWN


def measure_unfrozen(period_s: float | None, seeds: int, topology: Topology) -> float:
    """Fraction of non-blackout ticks on which SPRINTER is unfrozen.

    Sampled per *tick* via the trace hook, not per decision step: a step is five ticks, and at
    small `k` the unfrozen window is a fraction of a step, so per-step sampling misreads it badly.
    """
    config = peek_config(period_s, stages=NEVER_ARMS)
    unfrozen = total = 0
    for seed in range(seeds):
        sim = NightSim.from_seed(config, seed=seed, topology=topology)
        counters = [0, 0]

        def hook(state, _action, s=sim, c=counters):
            if not state.blackout:
                c[1] += 1
                c[0] += 0 if s.sprinter.is_frozen(state) else 1

        sim.on_tick = hook
        step = 0
        while not sim.state.terminated:
            sim.step(peek_action(period_s, step, config))
            step += 1
        unfrozen += counters[0]
        total += counters[1]
    return unfrozen / total


def measure_attacks(period_s: float | None, seeds: int, topology: Topology) -> float:
    """Mean SPRINTER attacks per night, with the left door shut so a night survives its bangs."""
    config = peek_config(period_s)
    attacks = 0
    for seed in range(seeds):
        sim = NightSim.from_seed(config, seed=seed, topology=topology)
        sim.state.office.door_left = True
        step = 0
        while not sim.state.terminated:
            sim.step(peek_action(period_s, step, config))
            step += 1
        attacks += sim.state.sprinter.bang_count
    return attacks / seeds


@pytest.mark.parametrize("period_s", HARD_ZERO_PERIODS)
def test_no_attacks_below_the_hard_zero_bound(period_s: float, topology: Topology) -> None:
    """PROJECT.md 8.4's sharpest assertion: deterministic, not statistical.

    Below the bound the unfrozen window cannot open at all, so a single attack on any seed means
    either the freeze or the immunity window is wrong.
    """
    config = peek_config(period_s)
    assert period_s <= derivations.sprinter_hard_zero_bound_s(config)
    assert measure_attacks(period_s, 40, topology) == 0.0


def test_the_hard_zero_bound_is_the_quantised_one() -> None:
    """1.4 s, not the continuous 1.33 s, because immunity ceilings to whole ticks."""
    config = load_night_config(6)
    assert derivations.sprinter_hard_zero_bound_s(config) == pytest.approx(1.4)


def test_every_period_in_the_family_is_reachable() -> None:
    """`k` must be a whole number of decision steps; 0.75 and 1.25 are not expressible."""
    step = load_night_config(6).timing.decision_step_s
    for period in HARD_ZERO_PERIODS + tuple(p for p in CURVE_PERIODS if p is not None):
        assert abs(period / step - round(period / step)) < 1e-9, period


@pytest.mark.slow
def test_unfrozen_fraction_agrees_with_the_derivation(topology: Topology) -> None:
    """PROJECT.md 8.4's replacement for the untestable flatness assertion."""
    report = []
    for period_s in HARD_ZERO_PERIODS + CURVE_PERIODS:
        config = peek_config(period_s)
        predicted = derivations.sprinter_unfrozen_fraction(config, period_s)
        measured = measure_unfrozen(period_s, 30, topology)
        report.append(f"k={period_s}: {measured:.4f} vs {predicted:.4f}")
        assert abs(measured - predicted) <= UNFROZEN_BAND, "; ".join(report)


@pytest.mark.slow
def test_attacks_increase_with_the_camera_gap(topology: Topology) -> None:
    """The original 8.4 direction, kept: longer gaps mean more attacks."""
    measured = [measure_attacks(period, 25, topology) for period in (2.0, 6.0, 20.0, None)]
    assert all(a <= b for a, b in pairwise(measured)), measured
    assert measured[-1] > measured[0], measured
