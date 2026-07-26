"""PROWLER: adjacency-chain movement and door resolution. PROJECT.md 3.6."""

from __future__ import annotations

from collections.abc import Callable

from nightguard.core import Action, NightSim, Node, TerminationCause
from tests.conftest import step_until

CHAIN = (Node.COMMONS, Node.E_RESTROOMS, Node.E_KITCHEN, Node.E_HALL, Node.E_CORNER)

PROWLER_ONLY = [0, 0, 20, 0]
NOBODY = [0, 0, 0, 0]


def test_level_zero_never_succeeds(make_sim: Callable[..., NightSim]) -> None:
    """Escalation is disabled: on night 1 it would lift PROWLER to 2 by 4AM."""
    sim = make_sim(levels=NOBODY, seed=11, escalation=())
    sim.run()
    assert sim.state.prowler.node is Node.STAGE


def test_leaving_stage_is_deterministic(make_sim: Callable[..., NightSim]) -> None:
    """STAGE is one-way and is not a chain member, so the first move always lands on COMMONS."""
    for seed in range(8):
        sim = make_sim(levels=PROWLER_ONLY, seed=seed)
        assert step_until(sim, lambda s: s.state.prowler.node is not Node.STAGE, max_steps=12)
        assert sim.state.prowler.node is Node.COMMONS


def test_movement_stays_on_the_chain(make_sim: Callable[..., NightSim]) -> None:
    seen: set[Node] = set()
    for seed in range(40):
        sim = make_sim(levels=PROWLER_ONLY, seed=seed)
        for _ in range(60):
            if sim.state.terminated:
                break
            sim.step(Action.NOOP)
            seen.add(sim.state.prowler.node)
    assert set(CHAIN) <= seen
    assert seen <= set(CHAIN) | {Node.STAGE}


def test_steps_are_to_immediate_neighbours_only(make_sim: Callable[..., NightSim]) -> None:
    """Unlike DRIFTER's pool, a step is always to an adjacent node, which is why belief decays
    gradually in the east wing and instantly in the west."""
    for seed in range(30):
        sim = make_sim(levels=PROWLER_ONLY, seed=seed)
        previous = sim.state.prowler.node
        for _ in range(80):
            if sim.state.terminated:
                break
            sim.step(Action.NOOP)
            current = sim.state.prowler.node
            if current != previous and previous is not Node.STAGE:
                gap = abs(CHAIN.index(current) - CHAIN.index(previous))
                assert gap == 1, f"jumped from {previous.name} to {current.name}"
            previous = current


def test_movement_is_bidirectional(make_sim: Callable[..., NightSim]) -> None:
    """PROWLER can retreat, so it does not simply march to the door."""
    retreated = False
    for seed in range(30):
        sim = make_sim(levels=PROWLER_ONLY, seed=seed)
        previous = sim.state.prowler.node
        for _ in range(80):
            if sim.state.terminated:
                break
            sim.step(Action.NOOP)
            current = sim.state.prowler.node
            if (
                previous in CHAIN
                and current in CHAIN
                and CHAIN.index(current) < CHAIN.index(previous)
            ):
                retreated = True
            previous = current
    assert retreated


def _at_corner(make_sim: Callable[..., NightSim], seed: int = 1) -> NightSim:
    sim = make_sim(levels=PROWLER_ONLY, seed=seed)
    sim.state.prowler.node = Node.E_CORNER
    return sim


def test_closed_right_door_sends_it_back_to_commons(make_sim: Callable[..., NightSim]) -> None:
    """Door resolution at E_CORNER is identical to DRIFTER's at W_CORNER, on the right door."""
    sim = _at_corner(make_sim)
    sim.state.office.door_right = True
    assert step_until(sim, lambda s: s.state.prowler.node is Node.COMMONS, max_steps=12)
    assert not sim.state.office.jam_right


def test_open_right_door_with_the_monitor_up_invades_and_jams(
    make_sim: Callable[..., NightSim],
) -> None:
    sim = _at_corner(make_sim)
    assert step_until(
        sim, lambda s: s.state.prowler.in_office, max_steps=12, action=Action.SELECT_CAM_0
    )
    assert sim.state.office.jam_right
    assert not sim.state.office.jam_left

    sim.step(Action.MONITOR_DOWN)
    assert sim.state.cause is TerminationCause.KILLED_PROWLER


def test_left_door_does_not_stop_the_east_entity(make_sim: Callable[..., NightSim]) -> None:
    sim = _at_corner(make_sim)
    sim.state.office.door_left = True
    assert step_until(
        sim, lambda s: s.state.prowler.in_office, max_steps=12, action=Action.SELECT_CAM_0
    )
