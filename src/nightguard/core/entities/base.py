"""Entity protocol, the shared movement opportunity timer, and door-entity behaviour.

PROJECT.md 3.3 for the AI level system, 3.5 and 3.6 for the two door entities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from numpy.random import Generator

from ..clock import Clock
from ..config import AIConfig, OfficeConfig
from ..state import DoorEntityState, DoorSide, EntityId, SimState
from ..topology import Node


@dataclass(frozen=True)
class OpportunityTimer:
    """A per-entity movement opportunity countdown. PROJECT.md 3.3.

    The intervals (3.02, 4.97, 4.98, 5.01 s) are deliberately mutually non-commensurate so that the
    entities never synchronise, and none of them is a multiple of the 0.1 s sim tick. The schedule is
    therefore held in exact integer resolution units and the n-th firing tick is recomputed from the
    firing count, never accumulated: rounding the intervals to the tick grid would collapse 4.97 and
    4.98 s onto the same 50-tick period and synchronise DRIFTER with PROWLER permanently.

    Quantisation still puts the two on the same *tick* for roughly the first 50 s, until the 0.01 s
    difference accumulates past one tick. That is inherent to the tick size; the fixed resolution
    order of PROJECT.md 3.13 makes it deterministic rather than a race.
    """

    interval_units: int
    units_per_tick: int

    @classmethod
    def from_interval(
        cls, interval_s: float, clock: Clock, name: str = "interval_s"
    ) -> OpportunityTimer:
        """Build a timer for an interval given in seconds."""
        return cls(
            interval_units=clock.to_units(interval_s, name),
            units_per_tick=clock.units_per_tick,
        )

    def fire_tick(self, fire_count: int) -> int:
        """The tick on which opportunity number ``fire_count + 1`` fires.

        Rounded up, so an interval of 4.97 s first fires on the tick covering t = 4.97 s, i.e. tick
        50 at a 0.1 s tick.
        """
        total_units = (fire_count + 1) * self.interval_units
        return -(-total_units // self.units_per_tick)

    def fires_at(self, tick: int, fire_count: int) -> bool:
        """Whether the next opportunity is due at ``tick``."""
        return tick >= self.fire_tick(fire_count)


def opportunity_succeeds(rng: Generator, ai_level: int, config: AIConfig) -> bool:
    """Roll a movement opportunity. PROJECT.md 3.3.

    Draws uniformly over 1..20 inclusive and succeeds when ``ai_level >= roll``, so level 0 can never
    succeed, level 20 always does, and each level is worth exactly 5 percentage points.
    """
    roll = int(rng.integers(config.level_min + 1, config.level_max, endpoint=True))
    return ai_level >= roll


class Entity(Protocol):
    """The interface the simulator drives once per opportunity."""

    entity_id: EntityId
    timer: OpportunityTimer

    def resolve(self, state: SimState, rng: Generator) -> None:
        """Resolve one *successful* movement opportunity."""


class DoorEntity(ABC):
    """Shared behaviour for the entities that walk to a door corner and try the door.

    DRIFTER and PROWLER differ only in how they choose their next node; everything from the corner
    onwards is identical (PROJECT.md 3.6: "Door resolution at E_CORNER is identical to DRIFTER's at
    W_CORNER"), so it lives here.
    """

    def __init__(
        self,
        entity_id: EntityId,
        *,
        nodes: tuple[str, ...],
        start: str,
        corner: str,
        door: DoorSide,
        retreat_to: str,
        interval_s: float,
        clock: Clock,
        office_config: OfficeConfig,
    ) -> None:
        self.entity_id = entity_id
        self.nodes = tuple(Node[name] for name in nodes)
        self.start = Node[start]
        self.corner = Node[corner]
        self.door = door
        self.retreat_to = Node[retreat_to]
        self.timer = OpportunityTimer.from_interval(interval_s, clock)
        self.office_config = office_config

    @property
    def start_node(self) -> Node:
        """The node this entity begins the night on."""
        return self.start

    @abstractmethod
    def advance(self, entity: DoorEntityState, rng: Generator) -> Node:
        """Choose the next node from the current one, for an entity not at its door corner."""

    def resolve(self, state: SimState, rng: Generator) -> None:
        """Resolve one successful movement opportunity.

        At the door corner the opportunity resolves the door instead of moving:

        * Door closed: the entity retreats, regardless of the monitor.
        * Door open, monitor up: it enters the office and jams the door behind it.
        * Door open, monitor down: it waits at the corner, and the opportunity is spent.

        The monitor gate on entry is not in PROJECT.md 3.5, but without it that section is internally
        incoherent — being "killed the next time ``monitor_up`` becomes false" only means anything if
        entry happened while the monitor was up — and 8.2's night-1 assertion (do_nothing survives at
        least 80% of the time, "the single most diagnostic assertion in the suite") cannot hold,
        because a do-nothing policy leaves both doors open all night. See CHANGELOG.
        """
        entity = state.entity(self.entity_id)
        if entity.in_office:
            return

        if entity.node == self.corner:
            if state.office.door(self.door):
                entity.node = self.retreat_to
            elif state.office.monitor_up:
                self._invade(state, entity)
            return

        entity.node = self.advance(entity, rng)

    def _invade(self, state: SimState, entity: DoorEntityState) -> None:
        """Enter the office, jamming the door behind. PROJECT.md 3.5."""
        entity.node = Node.OFFICE
        entity.invaded_at_tick = state.tick
        if self.office_config.door_jam_on_invasion:
            state.office.set_jammed(self.door)
            state.record(f"door_jam_{self.door.name.lower()}")
        state.record(f"invasion_{self.entity_id.name.lower()}")
