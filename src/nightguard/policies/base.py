"""Scripted policies and the restricted view they are allowed to see.

PROJECT.md 8.2 ships two reference policies permanently as a regression suite. They must not cheat:
the design rule in 6.2 is "give the agent exactly what a human player perceives, and nothing more",
so policies here consume a :class:`Percept` rather than :class:`SimState`.

``env/`` will encode the same information as a Box observation in v1.0. This is deliberately a
smaller, hand-written subset — enough to write a reference policy against, and no ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core.sim import NightSim
from ..core.state import Action, SimState
from ..core.topology import Node


@dataclass(frozen=True)
class Percept:
    """What a human player can perceive at a decision step. PROJECT.md 3.9 and 6.2.

    Attributes:
        step: Decision steps elapsed.
        power_pct: The power readout.
        time_s: Elapsed time, which the in-game clock shows.
        door_left: Whether the left door is closed.
        door_right: Whether the right door is closed.
        monitor_up: Whether the monitor is raised.
        footstep: Audio: a door entity completed a move near the office.
        kitchen: Audio: an entity occupies the blind node.
        running: Audio: SPRINTER's attack has fired and is in its grace period.
        bang: Audio: SPRINTER banged on a closed door.
        left_occupied: Door proximity, revealed **only** on a step where the left light was
            flashed. ``None`` otherwise — absence of information, not absence of a threat.
        right_occupied: As above, for the right light.
    """

    step: int
    power_pct: float
    time_s: float
    door_left: bool
    door_right: bool
    monitor_up: bool
    footstep: bool
    kitchen: bool
    running: bool
    bang: bool
    left_occupied: bool | None
    right_occupied: bool | None


class Policy(Protocol):
    """A scripted policy. Called once per decision step."""

    def __call__(self, percept: Percept) -> Action:
        """Choose an action."""

    def reset(self) -> None:
        """Clear any per-episode state."""


def observe(sim: NightSim, state: SimState, step: int) -> Percept:
    """Build the restricted view. Reveals corner occupancy only through a flashed light."""
    office = state.office
    left = state.drifter.node == sim.drifter.corner if office.light_left else None
    right = (
        state.prowler.node == sim.prowler.corner or state.warden.node == sim.warden.corner
        if office.light_right
        else None
    )
    audio = state.step_audio
    return Percept(
        step=step,
        power_pct=max(0.0, state.power_pct),
        time_s=sim.clock.time_s(state.tick),
        door_left=office.door_left,
        door_right=office.door_right,
        monitor_up=office.monitor_up,
        footstep=audio.footstep,
        kitchen=audio.kitchen,
        running=audio.running,
        bang=audio.bang,
        left_occupied=left,
        right_occupied=right,
    )


def run_policy(sim: NightSim, policy: Policy) -> None:
    """Drive ``sim`` to termination with ``policy``."""
    policy.reset()
    step = 0
    while not sim.state.terminated:
        action = policy(observe(sim, sim.state, step))
        sim.step(action)
        step += 1


OFFICE_CAMERA = Node.STAGE
