"""Node graph metadata. PROJECT.md 2.2 and 2.3.

Node IDs are stable integers and must never be reordered: from v0.2 they index the trace format and
from v1.0 they index the observation one-hots.

This module carries node *metadata* only, not edges. PROJECT.md 2.3 makes the route definitions in
section 3 the authority on the graph, so the DRIFTER pool, the PROWLER chain and the WARDEN path live
in the entity configs. Encoding them here as well would create a second source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import yaml


class Node(IntEnum):
    """The twelve nodes of PROJECT.md 2.2. Values are the normative stable IDs."""

    STAGE = 0
    COMMONS = 1
    COVE = 2
    W_BACKSTAGE = 3
    W_CLOSET = 4
    W_HALL = 5
    W_CORNER = 6
    E_RESTROOMS = 7
    E_KITCHEN = 8
    E_HALL = 9
    E_CORNER = 10
    OFFICE = 11


class TopologyError(ValueError):
    """Raised when a topology file disagrees with the normative node table."""


@dataclass(frozen=True)
class NodeSpec:
    """Metadata for a single node.

    Attributes:
        node: The node itself.
        wing: One of ``shared``, ``west``, ``east``, ``none``.
        selectable: Whether the node can be chosen as a camera (nodes 0-10).
        has_video: Whether selecting it yields a visual observation. E_KITCHEN is selectable but
            blind; its occupancy reaches the agent only through the ``kitchen`` audio signal.
    """

    node: Node
    wing: str
    selectable: bool
    has_video: bool


@dataclass(frozen=True)
class Topology:
    """The twelve node specs, indexed by node ID."""

    nodes: tuple[NodeSpec, ...]

    def spec(self, node: Node) -> NodeSpec:
        """Return the metadata for ``node``."""
        return self.nodes[int(node)]

    def has_video(self, node: Node) -> bool:
        """Whether ``node`` yields a visual observation when selected."""
        return self.nodes[int(node)].has_video

    def is_selectable(self, node: Node) -> bool:
        """Whether ``node`` can be chosen as a camera."""
        return self.nodes[int(node)].selectable

    def wing(self, node: Node) -> str:
        """The wing ``node`` belongs to."""
        return self.nodes[int(node)].wing

    @property
    def cameras(self) -> tuple[Node, ...]:
        """Every selectable camera node, in ID order."""
        return tuple(spec.node for spec in self.nodes if spec.selectable)


def load_topology(path: Path) -> Topology:
    """Load and validate a topology YAML file.

    The file must list all twelve nodes, in ID order, with names matching :class:`Node` exactly.
    Any deviation raises :class:`TopologyError` rather than silently permuting the ID basis.
    """
    with path.open(encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)

    if not isinstance(raw, dict) or "nodes" not in raw:
        raise TopologyError(f"{path}: expected a mapping with a 'nodes' key")

    entries = raw["nodes"]
    if not isinstance(entries, list):
        raise TopologyError(f"{path}: 'nodes' must be a list")
    if len(entries) != len(Node):
        raise TopologyError(f"{path}: expected {len(Node)} nodes, found {len(entries)}")

    specs: list[NodeSpec] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TopologyError(f"{path}: node {index} is not a mapping")
        name = str(entry.get("name", ""))
        if name not in Node.__members__:
            raise TopologyError(f"{path}: node {index} has unknown name {name!r}")
        node = Node[name]
        if int(entry.get("id", -1)) != int(node):
            raise TopologyError(f"{path}: node {name} must have id {int(node)}")
        if index != int(node):
            raise TopologyError(f"{path}: node {name} is out of order; IDs must not be reordered")
        specs.append(
            NodeSpec(
                node=node,
                wing=str(entry.get("wing", "none")),
                selectable=bool(entry.get("selectable", True)),
                has_video=bool(entry.get("has_video", True)),
            )
        )

    return Topology(nodes=tuple(specs))
