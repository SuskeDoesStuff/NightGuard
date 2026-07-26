"""DRIFTER: uniform teleport within a pool. PROJECT.md 3.5.

The pool mode is deliberately chosen for the west wing because it makes observation nearly worthless
for prediction: seeing DRIFTER at W_CLOSET says almost nothing about where it will be five seconds
later. Contrast PROWLER, whose chain makes observation highly informative. Two different belief-decay
profiles in one environment is most of what makes the memory task non-trivial, so the asymmetry must
be preserved.
"""

from __future__ import annotations

from numpy.random import Generator

from ..clock import Clock
from ..config import DrifterConfig, OfficeConfig
from ..state import DoorEntityState, EntityId, door_side_from_name
from ..topology import Node
from .base import DoorEntity


class Drifter(DoorEntity):
    """The west-wing pool entity."""

    def __init__(self, config: DrifterConfig, clock: Clock, office_config: OfficeConfig) -> None:
        super().__init__(
            EntityId.DRIFTER,
            nodes=config.pool,
            start=config.start,
            corner=config.corner,
            door=door_side_from_name(config.door),
            retreat_to=config.retreat_to,
            interval_s=config.interval_s,
            clock=clock,
            office_config=office_config,
        )

    def advance(self, entity: DoorEntityState, rng: Generator) -> Node:
        """Teleport to a node drawn uniformly from the pool.

        The new node need not be adjacent to the current one. STAGE is the start node and is one-way:
        it is not in the pool, so once DRIFTER leaves it cannot return.
        """
        return self.nodes[int(rng.integers(len(self.nodes)))]
