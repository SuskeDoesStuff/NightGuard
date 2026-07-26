"""WARDEN: fixed path, camera-suppressed, countdown before each move. PROJECT.md 3.4.

The most intricate entity and the main reason the environment has depth. Four mechanics interact:

* **Camera suppression.** While the monitor is up, every opportunity fails — regardless of which
  camera is selected. The rules invert at `E_CORNER`, where the monitor no longer suppresses but
  looking *at* `E_CORNER` does.
* **The countdown.** A successful opportunity does not move WARDEN; it starts a countdown of
  ``(1000 - 100 × ai_level) / 60`` seconds. Raising the monitor does **not** pause it. Only new
  opportunity *rolls* are suppressed by the monitor, never an in-flight countdown.
* **The stage lock.** WARDEN cannot leave `STAGE` while `DRIFTER` or `PROWLER` is still there, so
  even a maximum-aggression configuration is inert early in a night.
* **The office.** Entry is a move gated on `monitor_up`, not a kill; once inside, WARDEN kills at
  25% per second of monitor-down time, and never forces the monitor down.

The countdown is held in grid units, not ticks: six of the ten countdown values are not whole
ticks, and rounding them would reintroduce exactly the drift the grid exists to prevent.
"""

from __future__ import annotations

from numpy.random import Generator

from ..clock import Clock
from ..config import AIConfig, WardenConfig
from ..state import DoorSide, EntityId, SimState, door_side_from_name
from ..topology import Node
from .base import OpportunityTimer


class Warden:
    """The east-path entity."""

    def __init__(self, config: WardenConfig, clock: Clock, ai_config: AIConfig) -> None:
        self.entity_id = EntityId.WARDEN
        self.config = config
        self.ai_config = ai_config
        self.path = tuple(Node[name] for name in config.path)
        if len(self.path) < 3:
            raise ValueError("warden.path needs at least start, corner and office")
        self.start = self.path[0]
        self.office = self.path[-1]
        self.corner = self.path[-2]
        self.corner_index = len(self.path) - 2
        self.door: DoorSide = door_side_from_name(config.door)
        self.timer = OpportunityTimer.from_interval(config.interval_s, clock)
        self.units_per_tick = clock.units_per_tick
        self.office_kill_interval_units = clock.to_units(
            config.office_kill_interval_s, "warden.office_kill_interval_s"
        )
        # Countdown per AI level, precomputed on the exact grid: (1000 - 100*ai)/60 s.
        self.countdown_units = tuple(
            clock.to_units(self._countdown_seconds(level), f"warden.countdown[ai={level}]")
            for level in range(ai_config.level_min, ai_config.level_max + 1)
        )

    def _countdown_seconds(self, level: int) -> float:
        """The countdown for an AI level, in seconds. Floored at 0 for levels at or above 10."""
        remaining = self.config.countdown_numerator - self.config.countdown_per_level * level
        return max(0.0, remaining) / self.config.countdown_divisor

    @property
    def start_node(self) -> Node:
        """The node WARDEN begins the night on."""
        return self.start

    # --- suppression ---------------------------------------------------------------------------

    def is_suppressed(self, state: SimState) -> bool:
        """Whether the monitor suppresses this opportunity. PROJECT.md 3.4.

        Everywhere but `E_CORNER`, any raised monitor suppresses. At `E_CORNER` the rule inverts:
        a raised monitor no longer suppresses, but looking specifically at `E_CORNER` does.
        """
        office = state.office
        if state.warden.node == self.corner:
            return office.monitor_up and office.selected_camera == self.corner
        return office.monitor_up

    def is_stage_locked(self, state: SimState) -> bool:
        """Whether the stage lock blocks WARDEN. PROJECT.md 3.4.

        A real mechanic, not a bug: WARDEN is inert until the door entities have escalated enough
        to leave `STAGE`.
        """
        if not self.config.stage_lock or state.warden.node != self.start:
            return False
        return state.drifter.node == self.start or state.prowler.node == self.start

    # --- resolution ----------------------------------------------------------------------------

    def resolve(self, state: SimState, rng: Generator) -> None:
        """Resolve one successful, unsuppressed movement opportunity."""
        warden = state.warden
        if warden.in_office:
            return

        if warden.node == self.corner:
            self._resolve_corner(state)
            return

        # Anywhere else on the path, success starts a countdown rather than moving.
        units = self.countdown_units[state.ai_levels[EntityId.WARDEN]]
        if units <= 0:
            self.advance(state)  # AI level 10 and above move instantly
            return
        warden.countdown_units = units
        state.record("warden_countdown_start")

    def _resolve_corner(self, state: SimState) -> None:
        """Resolve at `E_CORNER`, where entry is a move gated on the monitor. PROJECT.md 3.4, 10.

        Entry requires the monitor to be up, exactly as 3.5's door entities do. Read as an
        immediate kill instead, `OFFICE` would be unreachable and the 25%/s mechanic dead code.
        A closed door retreats to the previous path node — `E_HALL`, not `COMMONS`.
        """
        warden = state.warden
        office = state.office
        if office.door(self.door):
            warden.path_index = self.corner_index - 1
            warden.node = self.path[warden.path_index]
            state.record("warden_retreat")
        elif office.monitor_up and office.selected_camera != self.corner:
            warden.path_index = self.corner_index + 1
            warden.node = self.office
            state.record("invasion_warden")
        # Monitor down with the door open: WARDEN camps at the corner, opportunity spent.

    def advance(self, state: SimState) -> None:
        """Take the pending move: one step along the path, never reversing."""
        warden = state.warden
        if warden.path_index >= self.corner_index:
            return
        warden.path_index += 1
        warden.node = self.path[warden.path_index]

    def tick_countdown(self, state: SimState) -> bool:
        """Decrement an in-flight countdown and execute the move if it expires.

        PROJECT.md 3.13 step 5. Deliberately independent of monitor state: the move lands the
        instant the countdown expires even if the monitor is up at that moment.
        """
        warden = state.warden
        if warden.countdown_units is None:
            return False
        warden.countdown_units -= self.units_per_tick
        if warden.countdown_units > 0:
            return False
        warden.countdown_units = None
        self.advance(state)
        return True

    def office_kill_roll(self, state: SimState, rng: Generator) -> bool:
        """Roll the office kill for this tick. PROJECT.md 3.4.

        25% per second, resolved once per whole second of *monitor-down* time. Time spent with the
        monitor up does not accumulate, which is why holding the monitor up indefinitely is a
        legal if power-expensive survival strategy. There is deliberately no timeout.
        """
        warden = state.warden
        if not warden.in_office or state.office.monitor_up:
            return False
        warden.office_kill_units += self.units_per_tick
        while warden.office_kill_units >= self.office_kill_interval_units:
            warden.office_kill_units -= self.office_kill_interval_units
            if float(rng.random()) < self.config.office_kill_prob_per_s:
                return True
        return False
