"""Blackout. PROJECT.md 3.11.

**v0.1 placeholder.** Power reaching 0 terminates immediately with ``KILLED_BLACKOUT``.

The real mechanic is not terminal: it is a high-hazard absorbing state that the agent can sometimes
survive, and an optimal policy near 5AM should occasionally accept it. The three-phase sequence
(approach, song, kill) can take 40+ seconds, so reaching 6AM after a blackout must be reachable in
simulation. That lands in v0.3, where PROJECT.md 8.7 asserts the survival rate is strictly between 0
and 1 — if it is 0, the sequence is resolving too fast and the mechanic has been flattened.
"""

from __future__ import annotations

from .state import SimState, TerminationCause


def apply_onset(state: SimState) -> None:
    """Enter blackout: all doors open, all lights off, monitor forced down.

    Door jams are irrelevant here — the doors open regardless.
    """
    state.blackout = True
    state.office.door_left = False
    state.office.door_right = False
    state.office.clear_lights()
    state.office.monitor_up = False
    state.record("blackout")


def resolve(state: SimState) -> TerminationCause:
    """Advance the blackout state machine and check for a kill.

    v0.1 placeholder: immediately fatal. The three-phase sequence replaces this in v0.3.
    """
    state.record("death_blackout")
    return TerminationCause.KILLED_BLACKOUT
