"""Configuration schema and the exact-timing-grid invariant. PROJECT.md 4.

The exactness test is the durable half of the 1/300 s change. The value itself is a one-off fix;
this assertion catches the next constant that does not fit the grid, which would otherwise surface
as an inexplicable off-by-one-tick fidelity failure much later.
"""

from __future__ import annotations

import pytest

from nightguard.core import ConfigError, NightConfig, config_from_mapping, load_night_config
from nightguard.core.clock import Clock

NIGHTS = range(1, 7)


def timing_constants(config: NightConfig) -> dict[str, float]:
    """Every second-valued quantity in a config that must land on the scheduling grid."""
    entities = config.entities
    values: dict[str, float] = {
        "timing.sim_tick_s": config.timing.sim_tick_s,
        "timing.decision_step_s": config.timing.decision_step_s,
        "entities.warden.interval_s": entities.warden.interval_s,
        "entities.drifter.interval_s": entities.drifter.interval_s,
        "entities.prowler.interval_s": entities.prowler.interval_s,
        "entities.sprinter.interval_s": entities.sprinter.interval_s,
        "entities.sprinter.forced_attack_after_s": entities.sprinter.forced_attack_after_s,
        "entities.sprinter.grace_period_s": entities.sprinter.grace_period_s,
        "entities.sprinter.immunity_range_s[0]": entities.sprinter.immunity_range_s[0],
        "entities.sprinter.immunity_range_s[1]": entities.sprinter.immunity_range_s[1],
        "office.invasion_kill_timeout_s": config.office.invasion_kill_timeout_s,
    }
    for index, duration in enumerate(config.timing.hour_durations_s):
        values[f"timing.hour_durations_s[{index}]"] = duration
    for name in ("approach", "song", "kill"):
        phase = getattr(config.blackout, name)
        values[f"blackout.{name}.interval_s"] = phase.interval_s
        if phase.max_s is not None:
            values[f"blackout.{name}.max_s"] = phase.max_s
    # WARDEN's countdown table, PROJECT.md 3.4. Six of these are non-terminating at 1 ms.
    warden = entities.warden
    for level in range(config.ai.level_min, config.ai.level_max + 1):
        seconds = (
            max(0.0, (warden.countdown_numerator - warden.countdown_per_level * level))
            / warden.countdown_divisor
        )
        values[f"warden.countdown[ai={level}]"] = seconds
    return values


@pytest.mark.parametrize("night", NIGHTS)
def test_every_timing_constant_lands_on_the_grid(night: int) -> None:
    """PROJECT.md 4: no timing constant may need rounding to be scheduled."""
    config = load_night_config(night)
    clock = Clock.from_config(config.timing)
    for name, seconds in timing_constants(config).items():
        clock.to_units(seconds, name)  # raises if not an exact integer count of units


def test_the_grid_is_the_documented_lcm() -> None:
    from math import lcm

    timing = NightConfig().timing
    assert timing.time_units_per_second == lcm(60, 100) == 300


def test_documented_unit_counts() -> None:
    """The specific counts recorded in PROJECT.md 4's comment."""
    config = NightConfig()
    clock = Clock.from_config(config.timing)
    assert clock.units_per_tick == 30
    assert clock.ticks_per_decision_step == 5
    assert clock.to_units(config.entities.warden.interval_s) == 906
    assert clock.to_units(config.entities.drifter.interval_s) == 1491
    assert clock.to_units(config.entities.prowler.interval_s) == 1494
    assert clock.to_units(config.entities.sprinter.interval_s) == 1503
    assert clock.to_units(config.entities.sprinter.immunity_range_s[0]) == 249
    assert clock.to_units(config.entities.sprinter.immunity_range_s[1]) == 5001


@pytest.mark.parametrize("per_second", [100, 1000])
def test_millisecond_grids_cannot_represent_the_countdown(per_second: int) -> None:
    """Why the grid is not 1/1000 s: six of WARDEN's countdowns are non-terminating there."""
    from dataclasses import replace

    config = NightConfig()
    timing = replace(config.timing, time_units_per_second=per_second)
    clock = Clock.from_config(timing)
    warden = config.entities.warden
    failures = []
    for level in range(1, 10):
        seconds = (warden.countdown_numerator - warden.countdown_per_level * level) / (
            warden.countdown_divisor
        )
        try:
            clock.to_units(seconds, f"countdown[{level}]")
        except ValueError:
            failures.append(level)
    assert failures == [2, 3, 5, 6, 8, 9]


def test_blackout_enabled_is_rejected() -> None:
    """The flag was removed in v0.2; a config that still sets it must fail loudly."""
    with pytest.raises(ConfigError, match="blackout.enabled"):
        config_from_mapping({"blackout": {"enabled": False}})


def test_night_files_override_only_what_they_change() -> None:
    defaults = NightConfig()
    night6 = load_night_config(6)
    assert night6.power.night_divisor == 3.0
    assert night6.ai.levels == (4, 10, 12, 16)
    assert night6.timing == defaults.timing
    assert night6.entities == defaults.entities
