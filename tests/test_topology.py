"""Node metadata. PROJECT.md 2.2 and 2.3."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nightguard.core import Node, Topology, TopologyError, load_topology


def test_all_twelve_nodes_in_id_order(topology: Topology) -> None:
    assert len(topology.nodes) == 12
    assert [spec.node for spec in topology.nodes] == list(Node)


def test_cameras_are_nodes_zero_to_ten(topology: Topology) -> None:
    assert topology.cameras == tuple(Node)[:11]
    assert not topology.is_selectable(Node.OFFICE)


def test_kitchen_is_selectable_but_blind(topology: Topology) -> None:
    """E_KITCHEN's occupancy reaches the agent only through the audio channel. PROJECT.md 3.9."""
    assert topology.is_selectable(Node.E_KITCHEN)
    assert not topology.has_video(Node.E_KITCHEN)
    blind = [node for node in topology.cameras if not topology.has_video(node)]
    assert blind == [Node.E_KITCHEN]


def test_wings(topology: Topology) -> None:
    assert topology.wing(Node.STAGE) == "shared"
    assert topology.wing(Node.COMMONS) == "shared"
    assert topology.wing(Node.W_CORNER) == "west"
    assert topology.wing(Node.COVE) == "west"
    assert topology.wing(Node.E_CORNER) == "east"


def _write(tmp_path: Path, nodes: list[dict[str, object]]) -> Path:
    path = tmp_path / "topology.yaml"
    path.write_text(yaml.safe_dump({"nodes": nodes}), encoding="utf-8")
    return path


def _valid_nodes() -> list[dict[str, object]]:
    return [
        {"id": int(node), "name": node.name, "wing": "none", "selectable": True, "has_video": True}
        for node in Node
    ]


def test_reordered_ids_are_rejected(tmp_path: Path) -> None:
    """The ID basis is the trace and observation layout; a silent permutation would corrupt both."""
    nodes = _valid_nodes()
    nodes[0], nodes[1] = nodes[1], nodes[0]
    with pytest.raises(TopologyError):
        load_topology(_write(tmp_path, nodes))


def test_mismatched_id_is_rejected(tmp_path: Path) -> None:
    nodes = _valid_nodes()
    nodes[3]["id"] = 99
    with pytest.raises(TopologyError):
        load_topology(_write(tmp_path, nodes))


def test_unknown_node_name_is_rejected(tmp_path: Path) -> None:
    nodes = _valid_nodes()
    nodes[5]["name"] = "BASEMENT"
    with pytest.raises(TopologyError):
        load_topology(_write(tmp_path, nodes))


def test_missing_node_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TopologyError):
        load_topology(_write(tmp_path, _valid_nodes()[:-1]))
