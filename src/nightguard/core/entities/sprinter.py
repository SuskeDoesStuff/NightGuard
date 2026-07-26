"""SPRINTER: stage counter, no position, rushes the left door. PROJECT.md 3.7.

This is the entity that punishes camera *usage* rather than camera *inattention*, and it is what
makes the observation cost asymmetric. Every other entity can only enter the office while the
monitor is up; SPRINTER charges only while it is down. That asymmetry is the environment.

Two rules are commonly misremembered and are implemented as specified rather than as believed:

* The camera freeze is universal. *Any* raised camera freezes SPRINTER, not specifically `COVE`.
* Lowering the monitor does not immediately release it: a uniform immunity window is sampled on
  each monitor-down transition, and opportunities keep failing until it expires.
"""

from __future__ import annotations

from numpy.random import Generator

from ..clock import Clock
from ..config import SprinterConfig
from ..state import DoorSide, EntityId, SimState, door_side_from_name
from .base import OpportunityTimer


class Sprinter:
    """The positionless west-door entity."""

    def __init__(self, config: SprinterConfig, clock: Clock) -> None:
        self.entity_id = EntityId.SPRINTER
        self.config = config
        self.clock = clock
        self.door: DoorSide = door_side_from_name(config.door)
        self.timer = OpportunityTimer.from_interval(config.interval_s, clock)
        # Sampled in grid units rather than seconds so the window always lands on the exact grid.
        # At 1/300 s the discretisation is 15x finer than the sim tick, so it is not observable.
        self.immunity_min_units = clock.to_units(
            config.immunity_range_s[0], "sprinter.immunity_range_s[0]"
        )
        self.immunity_max_units = clock.to_units(
            config.immunity_range_s[1], "sprinter.immunity_range_s[1]"
        )
        self.forced_attack_ticks = clock.to_ticks(
            config.forced_attack_after_s, "sprinter.forced_attack_after_s"
        )
        self.grace_ticks = clock.to_ticks(config.grace_period_s, "sprinter.grace_period_s")

    # --- freeze --------------------------------------------------------------------------------

    def is_frozen(self, state: SimState) -> bool:
        """Whether SPRINTER cannot advance this tick. PROJECT.md 3.7.

        True while the monitor is up — for *any* camera — and during the immunity window that
        follows the monitor coming down.
        """
        return state.office.monitor_up or state.tick < state.sprinter.immune_until_tick

    def on_monitor_lowered(self, state: SimState, rng: Generator) -> None:
        """Sample the post-camera immunity window. Called once per monitor-down transition."""
        units = int(rng.integers(self.immunity_min_units, self.immunity_max_units, endpoint=True))
        state.sprinter.immune_until_tick = state.tick + self.clock.units_to_ticks(units)

    def on_monitor_raised(self, state: SimState) -> None:
        """An armed SPRINTER attacks the instant the monitor goes up. PROJECT.md 3.7."""
        sprinter = state.sprinter
        if sprinter.armed and sprinter.attack_at_tick is None:
            self._fire(state)

    # --- charging ------------------------------------------------------------------------------

    def resolve(self, state: SimState, rng: Generator) -> None:
        """Advance the stage counter on a successful, unfrozen opportunity."""
        sprinter = state.sprinter
        if sprinter.armed:
            return
        sprinter.stage += 1
        if sprinter.stage >= self.config.stages_to_arm:
            sprinter.stage = self.config.stages_to_arm
            sprinter.armed_at_tick = state.tick
            state.record("sprinter_armed")

    # --- attack --------------------------------------------------------------------------------

    def _fire(self, state: SimState) -> None:
        """Start the attack and its grace period."""
        sprinter = state.sprinter
        sprinter.attack_at_tick = state.tick
        sprinter.resolve_at_tick = state.tick + self.grace_ticks
        state.record("sprinter_attack")

    def tick_attack(self, state: SimState, rng: Generator) -> bool:
        """Advance the attack timers and resolve an expiring grace period.

        Returns:
            Whether the agent was killed this tick.
        """
        sprinter = state.sprinter
        if (
            sprinter.armed_at_tick is not None
            and sprinter.attack_at_tick is None
            and state.tick - sprinter.armed_at_tick >= self.forced_attack_ticks
        ):
            self._fire(state)

        if sprinter.resolve_at_tick is None:
            return False

        state.audio.running = True
        if state.tick < sprinter.resolve_at_tick:
            return False
        return self._resolve(state, rng)

    def _resolve(self, state: SimState, rng: Generator) -> bool:
        """Resolve the attack against the door. PROJECT.md 3.7."""
        sprinter = state.sprinter
        if not state.office.door(self.door):
            return True

        cost = self.config.bang_base_pct + self.config.bang_increment_pct * sprinter.bang_count
        sprinter.bang_count += 1
        state.power_pct -= cost
        state.audio.bang = True
        state.record("bang")

        choices = self.config.reset_stage_choices
        sprinter.stage = int(choices[int(rng.integers(len(choices)))])
        sprinter.armed_at_tick = None
        sprinter.attack_at_tick = None
        sprinter.resolve_at_tick = None
        return False

    def bang_cost_pct(self, bang_index: int) -> float:
        """Power cost of the ``bang_index``-th bang: 1.0, 6.0, 11.0, ... percentage points."""
        return self.config.bang_base_pct + self.config.bang_increment_pct * bang_index
