"""WARDEN. PROJECT.md 3.4, and v0.2 exit criteria 3 and 5."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from nightguard.core import Action, EntityId, NightSim, Node, TerminationCause
from tests.conftest import clear_stage, step_until

WARDEN_ONLY = [EntityId.WARDEN]
PATH = (
    Node.STAGE,
    Node.COMMONS,
    Node.E_RESTROOMS,
    Node.E_KITCHEN,
    Node.E_HALL,
    Node.E_CORNER,
    Node.OFFICE,
)

# PROJECT.md 3.4's countdown table, in seconds.
COUNTDOWN_TABLE = {
    1: 15.0,
    2: 40.0 / 3,
    3: 35.0 / 3,
    4: 10.0,
    5: 25.0 / 3,
    6: 20.0 / 3,
    7: 5.0,
    8: 10.0 / 3,
    9: 5.0 / 3,
    10: 0.0,
    15: 0.0,
    20: 0.0,
}


def warden_sim(make_sim: Callable[..., NightSim], level: int = 20, seed: int = 0) -> NightSim:
    """A WARDEN-only night with the stage lock cleared and no escalation."""
    sim = make_sim(levels=[level, 0, 0, 0], escalation=(), only=WARDEN_ONLY, seed=seed)
    clear_stage(sim)
    return sim


class TestCountdown:
    """The countdown is the heart of 3.4 and of exit criterion 3."""

    @pytest.mark.parametrize(("level", "seconds"), sorted(COUNTDOWN_TABLE.items()))
    def test_table_matches_the_spec(
        self, make_sim: Callable[..., NightSim], level: int, seconds: float
    ) -> None:
        sim = warden_sim(make_sim, level=level)
        units = sim.warden.countdown_units[level]
        assert units == pytest.approx(seconds * sim.clock.time_units_per_second)

    def test_countdown_is_held_on_the_exact_grid(self, make_sim: Callable[..., NightSim]) -> None:
        """Six of the ten values are not whole ticks; none may be rounded."""
        sim = warden_sim(make_sim)
        assert sim.warden.countdown_units[2] == 4000  # 13.333... s
        assert sim.warden.countdown_units[9] == 500  # 1.666... s
        assert sim.warden.countdown_units[1] == 4500  # 15.0 s

    def test_raising_the_monitor_does_not_pause_an_in_flight_countdown(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        """v0.2 exit criterion 3. Only new *rolls* are suppressed, never a running countdown."""
        sim = warden_sim(make_sim, level=1)
        start_node = sim.state.warden.node
        sim.state.warden.countdown_units = sim.clock.to_units(15.0)

        # Hold the monitor up for the entire countdown.
        moved = step_until(
            sim,
            lambda s: s.state.warden.node != start_node,
            max_steps=40,
            action=Action.SELECT_CAM_0,
        )

        assert moved, "the countdown did not expire while the monitor was up"
        assert sim.state.warden.node is Node.COMMONS
        assert sim.state.office.monitor_up

    def test_the_move_lands_on_the_tick_the_countdown_expires(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        sim = warden_sim(make_sim, level=1)
        units = sim.clock.to_units(15.0)
        sim.state.warden.countdown_units = units
        expected_ticks = sim.clock.units_to_ticks(units)

        start_tick = sim.state.tick
        step_until(sim, lambda s: s.state.warden.node is not Node.STAGE, max_steps=40)
        assert sim.state.tick - start_tick == pytest.approx(expected_ticks, abs=5)

    def test_a_raised_monitor_suppresses_new_rolls(self, make_sim: Callable[..., NightSim]) -> None:
        """Level 20 always succeeds, so any movement here would be a suppression failure."""
        sim = warden_sim(make_sim, level=20)
        for _ in range(120):
            sim.step(Action.SELECT_CAM_5)
        assert sim.state.warden.node is Node.STAGE
        assert sim.state.warden.fire_count > 0, "opportunities must still fire and be spent"


class TestStageLock:
    def test_warden_cannot_leave_stage_while_a_door_entity_is_there(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        """v0.2 exit criterion 5."""
        sim = make_sim(levels=[20, 0, 0, 0], escalation=(), only=WARDEN_ONLY)
        assert sim.state.drifter.node is Node.STAGE
        for _ in range(200):
            sim.step(Action.NOOP)
        assert sim.state.warden.node is Node.STAGE

    def test_either_door_entity_alone_holds_the_lock(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        for lingering in ("drifter", "prowler"):
            sim = make_sim(levels=[20, 0, 0, 0], escalation=(), only=WARDEN_ONLY)
            other = "prowler" if lingering == "drifter" else "drifter"
            getattr(sim.state, other).node = Node.COMMONS
            for _ in range(100):
                sim.step(Action.NOOP)
            assert sim.state.warden.node is Node.STAGE, f"{lingering} did not hold the lock"

    def test_warden_moves_once_the_stage_clears(self, make_sim: Callable[..., NightSim]) -> None:
        sim = warden_sim(make_sim, level=20)
        assert step_until(sim, lambda s: s.state.warden.node is not Node.STAGE, max_steps=40)
        assert sim.state.warden.node is Node.COMMONS


class TestPath:
    def test_warden_walks_the_path_in_order_and_never_reverses(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        sim = warden_sim(make_sim, level=20)
        seen = [sim.state.warden.node]
        for _ in range(400):
            if sim.state.terminated:
                break
            sim.step(Action.NOOP)
            if sim.state.warden.node != seen[-1]:
                seen.append(sim.state.warden.node)
        assert seen == list(PATH[: len(seen)])
        assert Node.E_HALL in seen, "E_HALL is on the path; PROJECT.md 10 locks this"

    def test_warden_stops_at_the_corner_with_the_monitor_down(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        """Monitor down and door open is a stalemate: WARDEN camps at E_CORNER."""
        sim = warden_sim(make_sim, level=20)
        assert step_until(sim, lambda s: s.state.warden.node is Node.E_CORNER, max_steps=200)
        for _ in range(200):
            sim.step(Action.NOOP)
        assert sim.state.warden.node is Node.E_CORNER
        assert not sim.state.terminated


class TestCorner:
    """The rules invert at E_CORNER. PROJECT.md 3.4."""

    def _at_corner(self, make_sim: Callable[..., NightSim], level: int = 20) -> NightSim:
        sim = warden_sim(make_sim, level=level)
        sim.state.warden.node = Node.E_CORNER
        sim.state.warden.path_index = PATH.index(Node.E_CORNER)
        return sim

    def test_closed_right_door_retreats_to_e_hall_not_commons(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        sim = self._at_corner(make_sim)
        sim.state.office.door_right = True
        assert step_until(sim, lambda s: s.state.warden.node is not Node.E_CORNER, max_steps=40)
        assert sim.state.warden.node is Node.E_HALL

    def test_open_door_with_the_monitor_up_enters_the_office(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        """PROJECT.md 10: this is a move, not a kill."""
        sim = self._at_corner(make_sim)
        assert step_until(
            sim, lambda s: s.state.warden.in_office, max_steps=40, action=Action.SELECT_CAM_0
        )
        assert sim.state.warden.node is Node.OFFICE
        assert not sim.state.terminated, "entering the office is a move; the kill comes later"
        assert not sim.state.office.jam_right, "WARDEN has no door-jam mechanic"

    def test_looking_at_the_corner_camera_suppresses(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        """The inversion: at E_CORNER a raised monitor no longer protects unless aimed here."""
        sim = self._at_corner(make_sim)
        for _ in range(60):
            sim.step(Action.SELECT_CAM_10)
        assert sim.state.warden.node is Node.E_CORNER
        assert not sim.state.warden.in_office

    def test_a_closed_door_still_retreats_while_looking_elsewhere(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        sim = self._at_corner(make_sim)
        sim.state.office.door_right = True
        assert step_until(
            sim,
            lambda s: s.state.warden.node is Node.E_HALL,
            max_steps=40,
            action=Action.SELECT_CAM_0,
        )


class TestOfficeKill:
    def _in_office(self, make_sim: Callable[..., NightSim], seed: int = 0) -> NightSim:
        sim = warden_sim(make_sim, level=20, seed=seed)
        sim.state.warden.node = Node.OFFICE
        sim.state.warden.path_index = PATH.index(Node.OFFICE)
        return sim

    def test_kills_while_the_monitor_is_down(self, make_sim: Callable[..., NightSim]) -> None:
        """25% per second: death within a few seconds on essentially every seed."""
        deaths = 0
        for seed in range(20):
            sim = self._in_office(make_sim, seed=seed)
            step_until(sim, lambda s: s.state.terminated, max_steps=200)
            if sim.state.cause is TerminationCause.KILLED_WARDEN:
                deaths += 1
        assert deaths == 20

    def test_holding_the_monitor_up_is_a_legal_survival_strategy(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        """PROJECT.md 3.4: no forced monitor-down, and no timeout. Do not add one."""
        sim = self._in_office(make_sim)
        for _ in range(400):
            if sim.state.terminated:
                break
            sim.step(Action.SELECT_CAM_3)
        assert sim.state.cause is not TerminationCause.KILLED_WARDEN
        assert sim.state.warden.in_office

    def test_only_monitor_down_time_accumulates(self, make_sim: Callable[..., NightSim]) -> None:
        sim = self._in_office(make_sim)
        for _ in range(50):
            sim.step(Action.SELECT_CAM_3)
        assert sim.state.warden.office_kill_units == 0
