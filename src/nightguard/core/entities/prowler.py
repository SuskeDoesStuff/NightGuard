"""PROWLER: uniform step to an adjacent node in a chain. PROJECT.md 3.6.

Movement within the chain is bidirectional, so PROWLER can retreat, and belief about its position
decays gradually rather than instantly. That is the deliberate contrast with DRIFTER.
"""

from __future__ import annotations

from numpy.random import Generator

from ..clock import Clock
from ..config import OfficeConfig, ProwlerConfig
from ..state import DoorEntityState, EntityId, door_side_from_name
from ..topology import Node
from .base import DoorEntity


class Prowler(DoorEntity):
    """The east-wing chain entity."""

    def __init__(self, config: ProwlerConfig, clock: Clock, office_config: OfficeConfig) -> None:
        super().__init__(
            EntityId.PROWLER,
            nodes=config.chain,
            start=config.start,
            corner=config.corner,
            door=door_side_from_name(config.door),
            retreat_to=config.retreat_to,
            interval_s=config.interval_s,
            clock=clock,
            office_config=office_config,
        )

    def advance(self, entity: DoorEntityState, rng: Generator) -> Node:
        """Step to a uniformly chosen neighbour in the chain.

        STAGE is one-way and is not a chain member, so leaving it always lands on the chain's head
        (COMMONS) with no roll consumed for the choice.
        """
        if entity.node == self.start:
            return self.nodes[0]
        index = self.nodes.index(entity.node)
        neighbours = []
        if index > 0:
            neighbours.append(self.nodes[index - 1])
        if index + 1 < len(self.nodes):
            neighbours.append(self.nodes[index + 1])
        return neighbours[int(rng.integers(len(neighbours)))]
