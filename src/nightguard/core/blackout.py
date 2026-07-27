"""Blackout: the three-phase power-out sequence. PROJECT.md 3.11.

Power reaching 0 is **not terminal**. It is a high-hazard absorbing state the agent can sometimes
survive, and an optimal policy near 5AM should occasionally accept it. The sequence can take 40+
seconds, so reaching 6AM after a blackout is genuinely reachable — and must stay so.

Three roll conventions are settled in §3.11 and §10, each because the alternative moves §8.7's
survival rate materially:

* **The first roll of a phase happens one interval in**, not at t=0, so a phase always takes at
  least one full interval and the 20 s cap is a real ceiling. Rolling at t=0 moves §8.7 from
  0.6148 to 0.4389.
* **The 20 s guarantee replaces the roll at 20 s.** Approach and Song roll at 5, 10 and 15 s; if
  all three fail the phase advances at 20 s with certainty. Completion mass per phase is exactly
  ``{5: 0.2, 10: 0.16, 15: 0.128, 20: 0.512}``.
* **A kill roll landing on the survival boundary counts.** The strict reading moves §8.7 from
  0.6148 to 0.6367, which is 4.5σ at n=10,000 — a correct implementation would fail on convention
  alone and the failure would look like a bug.

Every timer is on the 1/300 s grid via :meth:`Clock.to_units`, consistent with the rest of the
simulation. There is no ``enabled`` flag: blackout is unconditional (removed in v0.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from numpy.random import Generator

from .clock import Clock
from .config import BlackoutConfig, BlackoutPhaseConfig
from .state import SimState, TerminationCause


class BlackoutPhase(IntEnum):
    """Which phase of the sequence is active. PROJECT.md 3.11.

    ``DONE`` is unreachable in practice: the kill phase has no cap and only ends by killing.
    """

    APPROACH = 0
    SONG = 1
    KILL = 2


@dataclass
class BlackoutState:
    """Progress through the sequence.

    Attributes:
        phase: The active phase.
        phase_started_tick: Tick the current phase began, i.e. the onset tick for `APPROACH`.
        rolls_taken: Rolls resolved so far in the current phase, used to schedule the next one.
    """

    phase: BlackoutPhase = BlackoutPhase.APPROACH
    phase_started_tick: int = 0
    rolls_taken: int = 0


def apply_onset(state: SimState) -> None:
    """Enter blackout: all doors open, all lights off, monitor forced down. PROJECT.md 3.11.

    Door jams are irrelevant here — the doors open regardless. All entities except WARDEN are
    removed from consideration, including any already in the office and including an armed
    SPRINTER; the simulator enforces that by skipping them once ``state.blackout`` is set.

    **The onset tick is the tick on which this runs.** Both the ``blackout`` event stamp and
    ``phase_started_tick`` use it, so they agree by construction. Power is zeroed here rather than
    left at whatever the crossing produced: the state this function names is "power exhausted", and
    a trace depicting a blackout with a quarter of the battery left is internally contradictory.
    """
    state.blackout = True
    state.power_pct = 0.0
    state.blackout_state = BlackoutState(phase_started_tick=state.tick)
    state.office.door_left = False
    state.office.door_right = False
    state.office.clear_lights()
    state.office.monitor_up = False
    state.record("blackout")


class BlackoutSequence:
    """Drives the three-phase sequence, one tick at a time."""

    def __init__(self, config: BlackoutConfig, clock: Clock) -> None:
        self.config = config
        self.clock = clock
        self._phases: tuple[BlackoutPhaseConfig, ...] = (config.approach, config.song, config.kill)
        self._interval_ticks = tuple(
            clock.to_ticks(phase.interval_s, f"blackout.{name}.interval_s")
            for name, phase in zip(("approach", "song", "kill"), self._phases, strict=True)
        )
        self._cap_ticks = tuple(
            None if phase.max_s is None else clock.to_ticks(phase.max_s, "blackout.max_s")
            for phase in self._phases
        )

    def phase_config(self, phase: BlackoutPhase) -> BlackoutPhaseConfig:
        """The configuration for ``phase``."""
        return self._phases[int(phase)]

    def interval_ticks(self, phase: BlackoutPhase) -> int:
        """Ticks between rolls in ``phase``."""
        return self._interval_ticks[int(phase)]

    def cap_ticks(self, phase: BlackoutPhase) -> int | None:
        """Ticks after which ``phase`` advances with certainty, or ``None`` for no cap."""
        return self._cap_ticks[int(phase)]

    def resolve(self, state: SimState, rng: Generator) -> TerminationCause | None:
        """Advance the sequence by one tick. PROJECT.md 3.13 step 4.

        Returns:
            :attr:`TerminationCause.KILLED_BLACKOUT` if WARDEN killed this tick, else ``None``.
        """
        blackout = state.blackout_state
        if blackout is None:  # pragma: no cover - onset always installs one
            return None

        phase = blackout.phase
        elapsed = state.tick - blackout.phase_started_tick
        interval = self.interval_ticks(phase)
        cap = self.cap_ticks(phase)

        # The guarantee replaces the roll at the cap rather than following one.
        if cap is not None and elapsed >= cap:
            self._advance(state, blackout)
            return None

        # Rolls land one interval in, then every interval after.
        if elapsed == 0 or elapsed % interval != 0:
            return None
        if elapsed // interval <= blackout.rolls_taken:
            return None

        blackout.rolls_taken += 1
        if float(rng.random()) >= self.phase_config(phase).prob:
            return None

        if phase is BlackoutPhase.KILL:
            state.record("death_blackout")
            return TerminationCause.KILLED_BLACKOUT
        self._advance(state, blackout)
        return None

    def _advance(self, state: SimState, blackout: BlackoutState) -> None:
        """Move to the next phase, restarting its timer from this tick."""
        blackout.phase = BlackoutPhase(int(blackout.phase) + 1)
        blackout.phase_started_tick = state.tick
        blackout.rolls_taken = 0
        state.record(f"blackout_{blackout.phase.name.lower()}")
