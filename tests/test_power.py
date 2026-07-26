"""The power drain model. PROJECT.md 3.10, and v0.1 exit criteria 2 and 3.

Power is never compared for equality; every assertion carries a tolerance (PROJECT.md 1.3).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from nightguard.core import (
    Action,
    NightSim,
    OfficeState,
    PowerConfig,
    TerminationCause,
    load_night_config,
)
from nightguard.core.power import active_units, drain_per_second, idle_drain_per_second
from tests.conftest import NIGHT_DIVISORS

TOLERANCE = 1e-6

# PROJECT.md 7's exit table, rounded to three decimals as printed there. The assertions below run
# against the closed form instead, because the printed table cannot be met to 1e-6: night 1's true
# value is 59.0729167, not 59.073.
PRINTED_TABLE = {1: 59.073, 2: 62.417, 3: 64.200, 4: 66.875, 5: 71.333, 6: 71.333}


def expected_idle_drain(night: int) -> float:
    """Closed-form percentage points drained by an idle office over a full night."""
    config = PowerConfig(night_divisor=NIGHT_DIVISORS[night])
    return idle_drain_per_second(config) * 535.0


@pytest.mark.parametrize("night", sorted(NIGHT_DIVISORS))
def test_idle_power_at_dawn_matches_the_exit_table(
    night: int, make_sim: Callable[..., NightSim]
) -> None:
    """v0.1 exit criterion 2, to 1e-6.

    Every entity is disabled: this measures the power model, and from v0.2 SPRINTER kills a
    do_nothing policy on most seeds, which would otherwise end the night before dawn.
    """
    config = load_night_config(night)
    result = make_sim(night=night, seed=4242, only=()).run()

    assert result.cause is TerminationCause.SURVIVED
    drained = config.power.start_pct - result.final_power_pct
    assert drained == pytest.approx(expected_idle_drain(night), abs=TOLERANCE)
    assert result.final_power_pct == pytest.approx(
        config.power.start_pct - expected_idle_drain(night), abs=TOLERANCE
    )


@pytest.mark.parametrize("night", sorted(NIGHT_DIVISORS))
def test_closed_form_agrees_with_the_printed_table(night: int) -> None:
    """The printed table is the same numbers, rounded to three decimals."""
    assert expected_idle_drain(night) == pytest.approx(PRINTED_TABLE[night], abs=5e-4)


def test_two_doors_all_night_blacks_out_inside_the_5am_block(
    make_sim: Callable[..., NightSim],
) -> None:
    """v0.1 exit criterion 3.

    Corrected from "one door" to "two doors": with ``active`` clamped below at 1, a single control
    active drains at exactly the idle rate, so one door alone never exhausts the budget. Two doors
    give 0.2104167 pp/s and blackout at t = 470.495 s, the figure the criterion already quoted.

    The doors are set closed at t = 0 rather than toggled, because only one action fits in a decision
    step and "held closed for the entire night" means from the start.
    """
    sim = make_sim(night=1, seed=7, only=())
    sim.state.office.door_left = True
    sim.state.office.door_right = True

    result = sim.run()

    assert result.cause is TerminationCause.KILLED_BLACKOUT
    assert result.ticks == 4705
    assert result.time_s == pytest.approx(470.5, abs=TOLERANCE)
    assert sim.clock.hour_at(result.time_s) == 5


def test_toggling_both_doors_costs_two_steps_of_ramp_up(make_sim: Callable[..., NightSim]) -> None:
    """Driving the same scenario from an action script is three ticks slower, and that is real."""
    sim = make_sim(night=1, seed=7, only=())
    result = sim.run([Action.TOGGLE_DOOR_LEFT, Action.TOGGLE_DOOR_RIGHT])

    assert result.cause is TerminationCause.KILLED_BLACKOUT
    assert result.ticks == 4708


class TestActiveUnits:
    """The clamp is load-bearing in both directions. PROJECT.md 3.10."""

    config = PowerConfig()

    def test_idle_office_still_drains(self) -> None:
        assert active_units(OfficeState(), self.config) == 1

    def test_one_control_costs_the_same_as_none(self) -> None:
        """The resolved source conflict: the floor of 1 makes a single control free."""
        one_door = OfficeState(door_left=True)
        assert active_units(one_door, self.config) == active_units(OfficeState(), self.config)
        assert drain_per_second(active_units(one_door, self.config), self.config) == pytest.approx(
            idle_drain_per_second(self.config)
        )

    def test_counts_each_control(self) -> None:
        office = OfficeState(door_left=True, door_right=True)
        assert active_units(office, self.config) == 2
        office.monitor_up = True
        assert active_units(office, self.config) == 3

    def test_ceiling_makes_the_monitor_free_at_four(self) -> None:
        """Both doors plus both lights costs the same as that plus the monitor."""
        four = OfficeState(door_left=True, door_right=True, light_left=True, light_right=True)
        five = OfficeState(
            door_left=True, door_right=True, light_left=True, light_right=True, monitor_up=True
        )
        assert active_units(four, self.config) == active_units(five, self.config) == 4


@pytest.mark.parametrize("night", sorted(NIGHT_DIVISORS))
def test_doing_nothing_is_always_survivable_on_power_alone(night: int) -> None:
    """PROJECT.md 3.10: the environment is about *when* to spend, not *whether*."""
    config = PowerConfig(night_divisor=NIGHT_DIVISORS[night])
    assert expected_idle_drain(night) < config.start_pct


@pytest.mark.parametrize("night", sorted(NIGHT_DIVISORS))
def test_holding_both_doors_all_night_is_impossible(night: int) -> None:
    """PROJECT.md 3.10, stated as a consequence worth understanding before implementing."""
    config = PowerConfig(night_divisor=NIGHT_DIVISORS[night])
    both_doors = drain_per_second(2, config)
    assert config.start_pct / both_doors < 535.0
