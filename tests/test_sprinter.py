"""SPRINTER. PROJECT.md 3.7, and v0.2 exit criterion 4."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from nightguard.core import Action, EntityId, NightSim, TerminationCause
from tests.conftest import step_until

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
