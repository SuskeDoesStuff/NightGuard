"""The power drain model. PROJECT.md 3.10.

Drain is a function of how many office controls are active, plus a per-night constant::

    active           = clamp(count of true controls, active_min, active_max)
    night_constant   = night_constant_numerator / night_divisor
    drain_per_second = active / divisor + night_constant     # percentage points per second
    drain_per_tick   = drain_per_second * sim_tick_s

Two consequences of the clamp are load-bearing and must not be "fixed":

* The floor of 1 means an entirely idle office still drains, and that exactly one control active
  costs the same as none. Sources disagree on the second point; see CHANGELOG for the resolution.
* The ceiling of 4 means camera use is free once you are already spending on four things.

Never compare drained power for equality; use a tolerance (PROJECT.md 1.3).
"""

from __future__ import annotations

from .config import PowerConfig
from .state import OfficeState


def active_units(office: OfficeState, config: PowerConfig) -> int:
    """Count the active office controls, clamped to the configured range.

    The five controls are both doors, both lights and the monitor.
    """
    count = sum(
        (
            office.door_left,
            office.door_right,
            office.light_left,
            office.light_right,
            office.monitor_up,
        )
    )
    return min(max(count, config.active_min), config.active_max)


def night_constant(config: PowerConfig) -> float:
    """The per-night baseline drain, in percentage points per second."""
    return config.night_constant_numerator / config.night_divisor


def drain_per_second(active: int, config: PowerConfig) -> float:
    """Drain in percentage points per second at the given active-control count."""
    return active / config.divisor + night_constant(config)


def drain_per_tick(active: int, config: PowerConfig, sim_tick_s: float) -> float:
    """Drain in percentage points for one sim tick."""
    return drain_per_second(active, config) * sim_tick_s


def idle_drain_per_second(config: PowerConfig) -> float:
    """Drain with nothing active, in percentage points per second."""
    return drain_per_second(config.active_min, config)
