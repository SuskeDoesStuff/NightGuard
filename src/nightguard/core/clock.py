"""Tick and hour arithmetic. PROJECT.md 3.1.

Three clocks are kept explicit, because conflating any two of them is the most common source of
unexplainable training dynamics: the sim tick (0.1 s), the per-entity movement opportunity timers
(3.3), and the agent decision step (0.5 s).

All internal time is an integer tick index. Seconds are derived from ticks on demand and never
accumulated, so no float drift can build up over the 5350 ticks of a night.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from .config import TimingConfig


@dataclass(frozen=True)
class Clock:
    """Derived timing quantities for one night.

    Attributes:
        sim_tick_s: Length of one tick, in seconds.
        decision_step_s: Length of one decision step, in seconds.
        time_units_per_second: Integer units per second of the exact scheduling grid.
        conversion_tolerance: Epsilon for validating those conversions.
        units_per_tick: Grid units in one tick.
        ticks_per_decision_step: Sim ticks advanced by one ``env.step()``.
        hour_boundary_ticks: Tick index of each hour boundary, including 0 and dawn.
    """

    sim_tick_s: float
    decision_step_s: float
    time_units_per_second: int
    conversion_tolerance: float
    units_per_tick: int
    ticks_per_decision_step: int
    hour_boundary_ticks: tuple[int, ...]

    @classmethod
    def from_config(cls, timing: TimingConfig) -> Clock:
        """Build a clock from a :class:`TimingConfig`."""
        per_second = timing.time_units_per_second
        tolerance = timing.conversion_tolerance
        units_per_tick = _to_units(timing.sim_tick_s, per_second, tolerance, "sim_tick_s")
        step_units = _to_units(timing.decision_step_s, per_second, tolerance, "decision_step_s")
        if step_units % units_per_tick != 0:
            raise ValueError("decision_step_s must be a whole number of sim ticks")

        boundaries = [0]
        for index, duration in enumerate(timing.hour_durations_s):
            duration_units = _to_units(
                duration, per_second, tolerance, f"hour_durations_s[{index}]"
            )
            if duration_units % units_per_tick != 0:
                raise ValueError(f"hour_durations_s[{index}] must be a whole number of sim ticks")
            boundaries.append(boundaries[-1] + duration_units // units_per_tick)

        return cls(
            sim_tick_s=timing.sim_tick_s,
            decision_step_s=timing.decision_step_s,
            time_units_per_second=per_second,
            conversion_tolerance=tolerance,
            units_per_tick=units_per_tick,
            ticks_per_decision_step=step_units // units_per_tick,
            hour_boundary_ticks=tuple(boundaries),
        )

    def to_units(self, seconds: float, name: str = "value") -> int:
        """Convert ``seconds`` to exact integer grid units."""
        return _to_units(seconds, self.time_units_per_second, self.conversion_tolerance, name)

    def units_to_ticks(self, units: int) -> int:
        """Convert grid units to whole sim ticks, rounding up."""
        return -(-units // self.units_per_tick)

    def to_ticks(self, seconds: float, name: str = "value") -> int:
        """Convert ``seconds`` to a whole number of sim ticks, rounding up.

        Rounding up rather than to nearest matters for entity opportunity intervals, which are not
        multiples of the tick: an interval of 4.97 s first fires on the tick covering t = 4.97 s.
        """
        units = self.to_units(seconds, name)
        return -(-units // self.units_per_tick)

    @property
    def total_ticks(self) -> int:
        """Ticks in a full night. 5350 by default."""
        return self.hour_boundary_ticks[-1]

    @property
    def total_seconds(self) -> float:
        """Length of a full night, in seconds. 535.0 by default."""
        return self.time_s(self.total_ticks)

    @property
    def decision_steps(self) -> int:
        """Decision steps in a full night. 1070 by default."""
        return self.total_ticks // self.ticks_per_decision_step

    @property
    def hours(self) -> int:
        """Number of in-game hours in a night. 6 by default."""
        return len(self.hour_boundary_ticks) - 1

    def time_s(self, tick: int) -> float:
        """Elapsed seconds at ``tick``."""
        return tick * self.sim_tick_s

    def hour_boundaries(self) -> tuple[float, ...]:
        """Hour boundaries in seconds: ``[0, 90, 179, 268, 357, 446, 535]`` by default."""
        return tuple(self.time_s(tick) for tick in self.hour_boundary_ticks)

    def hour_at(self, t: float) -> int:
        """The in-game hour containing time ``t``, in seconds.

        Intervals are half-open, so t = 90.0 is hour 1, not hour 0. Times at or beyond dawn return
        the hour count (6 by default), which is not a real hour but marks the night as over.
        """
        if t < 0.0:
            raise ValueError(f"time must be non-negative, found {t}")
        return bisect_right(self.hour_boundaries(), t) - 1

    def hour_at_tick(self, tick: int) -> int:
        """The in-game hour containing ``tick``. Half-open, as :meth:`hour_at`."""
        if tick < 0:
            raise ValueError(f"tick must be non-negative, found {tick}")
        return bisect_right(self.hour_boundary_ticks, tick) - 1


def _to_units(seconds: float, per_second: int, tolerance: float, name: str) -> int:
    """Convert seconds to integer grid units, rejecting values that are not representable.

    Multiplying by an integer count of units per second, rather than dividing by a fractional
    resolution, keeps the conversion exact: 1/300 s has no finite decimal form, so the reciprocal
    would carry representation error into every timing constant in the config.
    """
    scaled = seconds * per_second
    units = round(scaled)
    if abs(scaled - units) > tolerance * max(1.0, abs(scaled)):
        raise ValueError(f"{name}={seconds} is not a whole number of 1/{per_second}s grid units")
    return units
