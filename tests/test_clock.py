"""Clock arithmetic. PROJECT.md 3.1."""

from __future__ import annotations

import pytest

from nightguard.core import Clock, NightConfig
from nightguard.core.entities import OpportunityTimer

# PROJECT.md 3.1 states this table outright and asks that it be tested directly.
HOUR_BOUNDARIES_S = (0.0, 90.0, 179.0, 268.0, 357.0, 446.0, 535.0)


def test_hour_boundaries_match_the_spec_table(clock: Clock) -> None:
    assert clock.hour_boundaries() == pytest.approx(HOUR_BOUNDARIES_S)


def test_night_length(clock: Clock) -> None:
    assert clock.total_ticks == 5350
    assert clock.total_seconds == pytest.approx(535.0)
    assert clock.decision_steps == 1070
    assert clock.ticks_per_decision_step == 5
    assert clock.hours == 6


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        (0.0, 0),
        (89.9, 0),
        (90.0, 1),  # half-open: a boundary belongs to the hour it opens
        (178.9, 1),
        (179.0, 2),
        (268.0, 3),
        (357.0, 4),
        (446.0, 5),
        (534.9, 5),
        (535.0, 6),  # dawn is not a real hour; it marks the night as over
    ],
)
def test_hour_at_is_half_open(clock: Clock, t: float, expected: int) -> None:
    assert clock.hour_at(t) == expected


def test_hour_at_tick_agrees_with_hour_at(clock: Clock) -> None:
    for tick in range(0, clock.total_ticks + 1, 7):
        assert clock.hour_at_tick(tick) == clock.hour_at(clock.time_s(tick))


def test_hour_at_rejects_negative_time(clock: Clock) -> None:
    with pytest.raises(ValueError):
        clock.hour_at(-0.1)


def test_interior_boundaries_are_the_escalation_points(clock: Clock) -> None:
    assert list(clock.hour_boundary_ticks[1:-1]) == [900, 1790, 2680, 3570, 4460]


def test_seconds_convert_to_ticks_by_rounding_up(clock: Clock) -> None:
    assert clock.to_ticks(0.1) == 1
    assert clock.to_ticks(4.97) == 50
    assert clock.to_ticks(3.02) == 31
    assert clock.to_ticks(30.0) == 300


def test_units_convert_by_integer_multiplication(clock: Clock) -> None:
    """Multiplying by an integer count avoids the reciprocal of 1/300, which has no decimal form."""
    assert clock.time_units_per_second == 300
    assert clock.units_per_tick == 30
    assert clock.to_units(1.0) == 300
    assert clock.to_units(4.97) == 1491
    assert clock.units_to_ticks(1491) == 50
    assert clock.units_to_ticks(30) == 1


def test_unrepresentable_durations_are_rejected() -> None:
    config = NightConfig()
    timing = type(config.timing)(hour_durations_s=(90.001,))
    with pytest.raises(ValueError):
        Clock.from_config(timing)


class TestOpportunityTimer:
    """The schedule must be exact, and must not synchronise the entities. PROJECT.md 3.3."""

    def test_first_firing_covers_the_interval(self, clock: Clock) -> None:
        assert OpportunityTimer.from_interval(4.97, clock).fire_tick(0) == 50
        assert OpportunityTimer.from_interval(3.02, clock).fire_tick(0) == 31

    def test_schedule_does_not_drift(self, clock: Clock) -> None:
        """Naive float accumulation puts the 5th WARDEN firing at tick 152 instead of 151."""
        timer = OpportunityTimer.from_interval(3.02, clock)
        assert timer.fire_tick(4) == 151  # 5 * 3.02 s = 15.10 s exactly
        assert timer.fire_tick(49) == 1510  # 50 * 3.02 s = 151.0 s exactly

    def test_intervals_desynchronise(self, clock: Clock) -> None:
        """4.97 s and 4.98 s share a tick early on, then separate. Neither may be rounded to 5.0."""
        drifter = OpportunityTimer.from_interval(4.97, clock)
        prowler = OpportunityTimer.from_interval(4.98, clock)
        assert drifter.fire_tick(0) == prowler.fire_tick(0) == 50
        assert drifter.fire_tick(9) == 497
        assert prowler.fire_tick(9) == 498
