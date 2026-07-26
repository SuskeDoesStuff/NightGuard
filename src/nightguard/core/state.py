"""Simulation state, actions and termination causes. PROJECT.md 3.2, 3.12, 3.13.

Configuration objects are frozen; simulation state is not. Nothing here is module-level mutable.

:class:`Action` gives the 17 actions of PROJECT.md 3.2 semantic names. Their integer values are the
normative indices from that table, and core owns them because ``trace/`` must serialise them
(PROJECT.md 5's ``"action": 5``) while 1's dependency rule permits only ``trace -> core``. The
``Discrete(17)`` space and the encode/decode boundary live in ``env/actions.py`` (1.3): core knows
nothing about spaces, observations or reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

from .topology import Node


class Action(IntEnum):
    """The agent's 17 actions. PROJECT.md 3.2.

    Actions 6-16 fold "raise monitor" and "switch camera" into one, removing a two-step sequence the
    agent would otherwise waste capacity learning.
    """

    NOOP = 0
    TOGGLE_DOOR_LEFT = 1
    TOGGLE_DOOR_RIGHT = 2
    FLASH_LIGHT_LEFT = 3
    FLASH_LIGHT_RIGHT = 4
    MONITOR_DOWN = 5
    SELECT_CAM_0 = 6
    SELECT_CAM_1 = 7
    SELECT_CAM_2 = 8
    SELECT_CAM_3 = 9
    SELECT_CAM_4 = 10
    SELECT_CAM_5 = 11
    SELECT_CAM_6 = 12
    SELECT_CAM_7 = 13
    SELECT_CAM_8 = 14
    SELECT_CAM_9 = 15
    SELECT_CAM_10 = 16


def is_camera_action(action: Action) -> bool:
    """Whether ``action`` selects a camera and raises the monitor."""
    return action >= Action.SELECT_CAM_0


def camera_for_action(action: Action) -> Node:
    """The node selected by a camera action."""
    if not is_camera_action(action):
        raise ValueError(f"{action!r} is not a camera action")
    return Node(int(action) - int(Action.SELECT_CAM_0))


def action_for_camera(node: Node) -> Action:
    """The camera action that selects ``node``."""
    return Action(int(Action.SELECT_CAM_0) + int(node))


class EntityId(IntEnum):
    """The four adversaries, in the fixed resolution order of PROJECT.md 3.13."""

    WARDEN = 0
    DRIFTER = 1
    PROWLER = 2
    SPRINTER = 3


class DoorSide(IntEnum):
    """The two office doors."""

    LEFT = 0
    RIGHT = 1


def door_side_from_name(name: str) -> DoorSide:
    """Resolve a config ``door:`` value to a :class:`DoorSide`."""
    try:
        return DoorSide[name.upper()]
    except KeyError as error:
        raise ValueError(f"unknown door side {name!r}") from error


class TerminationCause(Enum):
    """Why an episode ended. PROJECT.md 3.12."""

    SURVIVED = "SURVIVED"
    KILLED_WARDEN = "KILLED_WARDEN"
    KILLED_DRIFTER = "KILLED_DRIFTER"
    KILLED_PROWLER = "KILLED_PROWLER"
    KILLED_SPRINTER = "KILLED_SPRINTER"
    KILLED_BLACKOUT = "KILLED_BLACKOUT"


def killed_by(entity: EntityId) -> TerminationCause:
    """The termination cause for a kill by ``entity``."""
    return TerminationCause[f"KILLED_{entity.name}"]


@dataclass
class OfficeState:
    """The five office controls plus camera selection and door jams. PROJECT.md 3.2.

    Lights are momentary: a light action sets the light for exactly one decision step, after which
    it turns itself off. ``selected_camera`` is meaningful only while ``monitor_up``.
    """

    door_left: bool = False
    door_right: bool = False
    light_left: bool = False
    light_right: bool = False
    monitor_up: bool = False
    selected_camera: Node = Node.STAGE
    jam_left: bool = False
    jam_right: bool = False

    def door(self, side: DoorSide) -> bool:
        """Whether the door on ``side`` is closed."""
        return self.door_left if side is DoorSide.LEFT else self.door_right

    def set_door(self, side: DoorSide, closed: bool) -> None:
        """Set the door on ``side``."""
        if side is DoorSide.LEFT:
            self.door_left = closed
        else:
            self.door_right = closed

    def jammed(self, side: DoorSide) -> bool:
        """Whether the door on ``side`` is permanently jammed."""
        return self.jam_left if side is DoorSide.LEFT else self.jam_right

    def set_jammed(self, side: DoorSide) -> None:
        """Jam the door on ``side`` for the rest of the episode."""
        if side is DoorSide.LEFT:
            self.jam_left = True
        else:
            self.jam_right = True

    def clear_lights(self) -> None:
        """Expire both momentary lights."""
        self.light_left = False
        self.light_right = False


@dataclass
class DoorEntityState:
    """Position and timer state for DRIFTER or PROWLER. PROJECT.md 3.5, 3.6.

    Attributes:
        node: Current node, or :attr:`Node.OFFICE` once it has invaded.
        fire_count: Number of movement opportunities that have fired so far. The schedule is
            recomputed from this count, never accumulated, so it cannot drift.
        invaded_at_tick: Tick at which it entered the office, or ``None``.
    """

    node: Node
    fire_count: int = 0
    invaded_at_tick: int | None = None

    @property
    def in_office(self) -> bool:
        """Whether this entity is inside the office awaiting kill resolution."""
        return self.invaded_at_tick is not None


@dataclass
class WardenState:
    """WARDEN's position, countdown and office timer. PROJECT.md 3.4.

    Attributes:
        node: Current node on the fixed path.
        path_index: Index into the configured path; kept in step with ``node``.
        fire_count: Movement opportunities fired so far.
        countdown_units: Grid units remaining on an in-flight move countdown, or ``None``.
            Held in grid units rather than ticks because six of the ten countdown values are
            not whole ticks; raising the monitor does **not** pause it.
        office_kill_units: Accumulated monitor-down time inside the office, in grid units.
            The 25%/s roll fires once per whole second of it.
    """

    node: Node
    path_index: int = 0
    fire_count: int = 0
    countdown_units: int | None = None
    office_kill_units: int = 0

    @property
    def in_office(self) -> bool:
        """Whether WARDEN has reached the office."""
        return self.node is Node.OFFICE

    @property
    def counting_down(self) -> bool:
        """Whether a move countdown is in flight."""
        return self.countdown_units is not None


@dataclass
class SprinterState:
    """SPRINTER's stage counter and timers. PROJECT.md 3.7.

    SPRINTER has no position at all. ``immune_until_tick`` is exclusive: the immunity window
    sampled when the monitor comes down blocks opportunities on ticks before it.
    """

    stage: int = 0
    fire_count: int = 0
    immune_until_tick: int = 0
    bang_count: int = 0
    armed_at_tick: int | None = None
    attack_at_tick: int | None = None
    resolve_at_tick: int | None = None

    @property
    def armed(self) -> bool:
        """Whether the stage counter has reached the arming threshold."""
        return self.armed_at_tick is not None

    @property
    def attacking(self) -> bool:
        """Whether an attack has fired and is inside its grace period."""
        return self.attack_at_tick is not None


@dataclass
class AudioState:
    """The four binary audio signals of PROJECT.md 3.9.

    ``kitchen`` is a *state* signal — the compensation for `E_KITCHEN` having no video feed — and
    is recomputed every tick. The other three are *event* signals, set by the event that raises
    them and cleared at the start of the next tick.
    """

    footstep: bool = False
    kitchen: bool = False
    running: bool = False
    bang: bool = False

    def clear_events(self) -> None:
        """Clear the event signals, leaving the state signal to be recomputed."""
        self.footstep = False
        self.running = False
        self.bang = False

    def clear(self) -> None:
        """Clear every signal."""
        self.clear_events()
        self.kitchen = False

    def absorb(self, other: AudioState) -> None:
        """OR another tick's signals into this one.

        PROJECT.md 3.9 emits audio "for the decision step in which the triggering event occurred",
        but a tick is a fifth of a decision step. The per-tick flags feed the trace; this
        accumulates them so a single-tick cue such as `footstep` or `bang` is still audible to the
        agent at the end of the step in which it fired.
        """
        self.footstep |= other.footstep
        self.kitchen |= other.kitchen
        self.running |= other.running
        self.bang |= other.bang


@dataclass
class SimState:
    """Complete ground-truth state of one night.

    ``ai_levels`` is indexed by :class:`EntityId`, so ``ai_levels[EntityId.DRIFTER]`` is DRIFTER's
    current level including any within-night escalation applied so far.
    """

    tick: int
    power_pct: float
    office: OfficeState
    ai_levels: list[int]
    warden: WardenState
    drifter: DoorEntityState
    prowler: DoorEntityState
    sprinter: SprinterState
    audio: AudioState = field(default_factory=AudioState)
    step_audio: AudioState = field(default_factory=AudioState)
    prev_monitor_up: bool = False
    escalations_applied: int = 0
    blackout: bool = False
    cause: TerminationCause | None = None
    events: list[tuple[int, str]] = field(default_factory=list)

    @property
    def terminated(self) -> bool:
        """Whether the episode has ended."""
        return self.cause is not None

    def entity(self, entity: EntityId) -> DoorEntityState:
        """The state of a door entity. WARDEN and SPRINTER have their own state shapes."""
        if entity is EntityId.DRIFTER:
            return self.drifter
        if entity is EntityId.PROWLER:
            return self.prowler
        raise KeyError(f"{entity.name} does not have door-entity state")

    def record(self, event: str) -> None:
        """Note a named event at the current tick. Becomes the trace ``event`` field in v0.2."""
        self.events.append((self.tick, event))

    def signature(self) -> tuple[object, ...]:
        """A hashable summary of ground truth, for determinism tests.

        Power is included at full precision: identical inputs must reproduce identical bits, so this
        is a determinism check rather than a float comparison in the power model.
        """
        office = self.office
        return (
            self.tick,
            self.power_pct,
            office.door_left,
            office.door_right,
            office.light_left,
            office.light_right,
            office.monitor_up,
            int(office.selected_camera),
            office.jam_left,
            office.jam_right,
            tuple(self.ai_levels),
            int(self.warden.node),
            self.warden.path_index,
            self.warden.fire_count,
            self.warden.countdown_units,
            self.warden.office_kill_units,
            int(self.drifter.node),
            self.drifter.fire_count,
            self.drifter.invaded_at_tick,
            int(self.prowler.node),
            self.prowler.fire_count,
            self.prowler.invaded_at_tick,
            self.sprinter.stage,
            self.sprinter.fire_count,
            self.sprinter.immune_until_tick,
            self.sprinter.bang_count,
            self.sprinter.armed_at_tick,
            self.sprinter.attack_at_tick,
            self.blackout,
            self.cause,
        )
