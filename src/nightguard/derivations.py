"""Analytic predictions for the §8 validation suite. PROJECT.md 8.0.

Every statistical assertion in §8 compares a measurement against a prediction derived from the
spec. These are those predictions, computed from configuration rather than hard-coded, so an
assertion can never quietly become "whatever the implementation happened to produce".

**Nothing here may consult the simulator.** Each function derives from PROJECT.md's stated rules and
the config values alone; the derivations are written out in full in `CHANGELOG.md` under
"Derivations (written before measuring)". If a measurement disagrees with one of these, one of the
two is wrong — do not adjust either to match.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import comb, floor

from .core.clock import Clock
from .core.config import NightConfig

# --- §8.2: do_nothing survival -----------------------------------------------------------------


def sprinter_opportunity_windows(config: NightConfig) -> tuple[int, int, int]:
    """Opportunity counts in each of SPRINTER's three level windows on a night. PROJECT.md 3.3, 3.7.

    A `do_nothing` policy never raises the monitor, so no camera freeze and no immunity window ever
    applies and every opportunity is live. Levels step at the 3AM and 4AM boundaries. The third
    success must land early enough for the forced attack to resolve before dawn, so the counting
    stops one forced-attack interval short of the end.

    Returns:
        Opportunities at the starting level, at +1, and at +2 before the kill deadline.
    """
    clock = Clock.from_config(config.timing)
    sprinter = config.entities.sprinter
    interval_units = clock.to_units(sprinter.interval_s)
    boundaries = clock.hour_boundary_ticks
    third, fourth = boundaries[3], boundaries[4]
    deadline = clock.total_ticks - clock.to_ticks(sprinter.forced_attack_after_s)

    counts = [0, 0, 0]
    fire = 1
    while True:
        tick = -(-(fire * interval_units) // clock.units_per_tick)
        if tick > clock.total_ticks:
            break
        if tick < third:
            counts[0] += 1
        elif tick < fourth:
            counts[1] += 1
        elif tick <= deadline:
            counts[2] += 1
        fire += 1
    return counts[0], counts[1], counts[2]


def do_nothing_survival(config: NightConfig) -> float:
    """Analytic P(survive) for a `do_nothing` policy. PROJECT.md 3.3, 3.7, 8.2.

    SPRINTER is the only kill path: under the §3.4 and §3.5 monitor gates no other entity can enter
    the office against a policy that never raises the monitor. Survival is therefore the probability
    of fewer than `stages_to_arm` successes before the kill deadline.
    """
    sprinter = config.entities.sprinter
    start = int(config.ai.levels[3]) if isinstance(config.ai.levels[3], int) else 0
    deltas = _sprinter_escalation_steps(config)
    windows = sprinter_opportunity_windows(config)
    levels = [min(config.ai.level_max, start + d) for d in deltas]

    threshold = sprinter.stages_to_arm
    distribution = {0: 1.0}
    for level, trials in zip(levels, windows, strict=True):
        p = level / config.ai.level_max
        updated: dict[int, float] = {}
        for successes, weight in distribution.items():
            for extra in range(min(trials, threshold - 1 - successes) + 1):
                mass = comb(trials, extra) * p**extra * (1 - p) ** (trials - extra)
                updated[successes + extra] = updated.get(successes + extra, 0.0) + weight * mass
        distribution = updated
    return sum(weight for successes, weight in distribution.items() if successes < threshold)


def _sprinter_escalation_steps(config: NightConfig) -> tuple[int, int, int]:
    """Cumulative SPRINTER level increase in each of the three windows."""
    gained = 0
    steps = [0]
    for hour in (3, 4):
        for event in config.ai.escalation:
            if event.hour == hour and "SPRINTER" in event.entities:
                gained += event.delta
        steps.append(gained)
    return steps[0], steps[1], steps[2]


# --- §8.3: WARDEN latency ----------------------------------------------------------------------


def warden_countdown_ticks(config: NightConfig, level: int) -> int:
    """WARDEN's move countdown at an AI level, in whole ticks. PROJECT.md 3.4.

    Ceiling division: countdowns are exact on the 1/300 s grid but six of the ten values are not
    whole ticks, so AI 2's 13.333 s countdown expires at 13.4 s.
    """
    clock = Clock.from_config(config.timing)
    warden = config.entities.warden
    seconds = max(0.0, warden.countdown_numerator - warden.countdown_per_level * level)
    return clock.units_to_ticks(clock.to_units(seconds / warden.countdown_divisor))


def warden_latency_s(config: NightConfig, level: int) -> float:
    """Predicted STAGE→E_CORNER latency at an AI level, in seconds. PROJECT.md 3.4, 8.3.

    Per hop: ``C + Δ(C) + (level_max/level − 1) × interval``, where `C` is the quantised countdown
    and ``Δ(C) = interval × (⌊C/interval⌋ + 1) − C`` is the wait from the countdown expiring to the
    next opportunity firing.

    Δ is a **credit, not a penalty**: the countdown consumes part of an inter-firing interval, so
    the next opportunity arrives sooner than a full interval after the move. At the top level there
    is no countdown and ``Δ(0) = interval`` recovers the uncorrected value.
    """
    if level <= 0:
        raise ValueError("level 0 never succeeds, so the walk never completes")
    clock = Clock.from_config(config.timing)
    warden = config.entities.warden
    interval = warden.interval_s
    countdown = warden_countdown_ticks(config, level) * clock.sim_tick_s
    phase = interval * (floor(countdown / interval) + 1) - countdown
    expected_tries = config.ai.level_max / level
    hops = len(warden.path) - 2  # STAGE..E_CORNER, excluding OFFICE
    return hops * (countdown + phase + (expected_tries - 1) * interval)


# --- §8.4: SPRINTER freeze ---------------------------------------------------------------------


def sprinter_unfrozen_fraction(config: NightConfig, peek_period_s: float | None) -> float:
    """Fraction of the night SPRINTER can act, for a policy peeking one step every `k` seconds.

    PROJECT.md 3.7. SPRINTER is frozen while the monitor is up and for a uniform immunity window
    sampled on each monitor-down edge, so the unfrozen time per cycle is ``max(0, k − peek − U)``.
    ``None`` means never peeking.
    """
    if peek_period_s is None:
        return 1.0
    low, high = config.entities.sprinter.immunity_range_s
    down = peek_period_s - config.timing.decision_step_s
    if down <= low:
        expected = 0.0
    elif down >= high:
        expected = down - (low + high) / 2
    else:
        expected = (down - low) ** 2 / (2 * (high - low))
    return expected / peek_period_s


def sprinter_hard_zero_bound_s(config: NightConfig) -> float:
    """Largest peek period at which SPRINTER provably never acts. PROJECT.md 8.4.

    The continuous bound is ``peek + immunity_min``, but the immunity window is sampled in grid
    units and rounded up to whole ticks, so the realised minimum is longer and the zero region is
    correspondingly wider — 1.4 s rather than 1.33 s at the shipped configuration.
    """
    clock = Clock.from_config(config.timing)
    low = config.entities.sprinter.immunity_range_s[0]
    min_ticks = clock.units_to_ticks(clock.to_units(low))
    return config.timing.decision_step_s + min_ticks * clock.sim_tick_s


# --- §8.7: blackout ----------------------------------------------------------------------------


def blackout_phase_mass(config: NightConfig, phase: str) -> dict[int, Fraction]:
    """Completion-time mass for one capped blackout phase, in seconds. PROJECT.md 3.11.

    Rolls land one interval in, and the cap replaces the roll at the cap rather than following one.
    """
    settings = getattr(config.blackout, phase)
    if settings.max_s is None:
        raise ValueError(f"{phase} has no cap and never completes on its own")
    p = Fraction(settings.prob).limit_denominator(10**6)
    q = 1 - p
    interval, cap = settings.interval_s, settings.max_s
    rolls = int(cap / interval) - 1
    mass = {int(interval * (index + 1)): q**index * p for index in range(rolls)}
    mass[int(cap)] = q**rolls
    return mass


def blackout_survival(config: NightConfig, budget_s: float) -> float:
    """Analytic P(surviving `budget_s` of blackout). PROJECT.md 3.11, 8.7.

    Convolves the two capped phases, then applies the uncapped kill phase. A kill roll landing
    exactly on the budget boundary counts (the inclusive convention of §3.11); the strict reading
    moves the 35 s value from 0.6148 to 0.6367.
    """
    approach = blackout_phase_mass(config, "approach")
    song = blackout_phase_mass(config, "song")
    kill = config.blackout.kill
    q = 1 - Fraction(kill.prob).limit_denominator(10**6)
    interval = int(kill.interval_s)
    budget = int(budget_s)

    total = Fraction(0)
    for (first, weight_a), (second, weight_b) in product(approach.items(), song.items()):
        remaining = max(0, budget - first - second)
        rolls = 0 if remaining < interval else (remaining - interval) // interval + 1
        total += weight_a * weight_b * q**rolls
    return float(total)
