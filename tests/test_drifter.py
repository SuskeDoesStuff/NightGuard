"""DRIFTER: pool teleport and door resolution. PROJECT.md 3.5."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from nightguard.core import Action, NightSim, Node, TerminationCause
from tests.conftest import step_until

POOL = {Node.COMMONS, Node.W_BACKSTAGE, Node.W_CLOSET, Node.W_HALL, Node.W_CORNER}

# WARDEN and SPRINTER are absent in v0.1, so only slots 1 (DRIFTER) and 2 (PROWLER) matter.
DRIFTER_ONLY = [0, 20, 0, 0]
NOBODY = [0, 0, 0, 0]


def test_level_zero_never_succeeds(make_sim: Callable[..., NightSim]) -> None:
    """Level 0 can never succeed, so DRIFTER sits on STAGE for the whole night.

    Escalation is disabled here: on night 1 it would lift DRIFTER to 3 by 4AM, at which point the
    entity is expected to move.
    """
    sim = make_sim(levels=NOBODY, seed=11, escalation=())
    sim.run()
    assert sim.state.drifter.node is Node.STAGE
    assert sim.state.drifter.fire_count > 100  # opportunities fired; they simply all failed


def test_escalation_wakes_a_level_zero_entity(make_sim: Callable[..., NightSim]) -> None:
    """The converse: with escalation on, night 1's DRIFTER does start moving after 2AM."""
    sim = make_sim(levels=NOBODY, seed=11)
    sim.run()
    assert sim.state.drifter.node is not Node.STAGE


def test_first_move_lands_in_the_pool(make_sim: Callable[..., NightSim]) -> None:
    sim = make_sim(levels=DRIFTER_ONLY, seed=3)
    assert step_until(sim, lambda s: s.state.drifter.node is not Node.STAGE, max_steps=12)
    assert sim.state.drifter.node in POOL


def test_stage_is_one_way(make_sim: Callable[..., NightSim]) -> None:
    """Once DRIFTER leaves STAGE it cannot return: STAGE is not in the pool."""
    sim = make_sim(levels=DRIFTER_ONLY, seed=5)
    left = False
    for _ in range(300):
        if sim.state.terminated:
            break
        sim.step(Action.NOOP)
        if sim.state.drifter.node is Node.STAGE:
            assert not left, "DRIFTER returned to STAGE"
        else:
            left = True
    assert left


def test_teleport_is_uniform_over_the_pool(make_sim: Callable[..., NightSim]) -> None:
    """Adjacency is irrelevant: any pool node can follow any other."""
    seen: set[Node] = set()
    for seed in range(40):
        sim = make_sim(levels=DRIFTER_ONLY, seed=seed)
        for _ in range(60):
            if sim.state.terminated:
                break
            sim.step(Action.NOOP)
            seen.add(sim.state.drifter.node)
    assert POOL <= seen
    assert seen <= POOL | {Node.STAGE}


def _at_corner(make_sim: Callable[..., NightSim], seed: int = 1) -> NightSim:
    sim = make_sim(levels=DRIFTER_ONLY, seed=seed)
    sim.state.drifter.node = Node.W_CORNER
    return sim


def test_closed_door_sends_it_back_to_commons(make_sim: Callable[..., NightSim]) -> None:
    sim = _at_corner(make_sim)
    sim.state.office.door_left = True
    assert step_until(sim, lambda s: s.state.drifter.node is Node.COMMONS, max_steps=12)
    assert not sim.state.office.jam_left


def test_open_door_with_the_monitor_down_is_a_stalemate(make_sim: Callable[..., NightSim]) -> None:
    """It camps at the door: it cannot enter unseen, and the agent is not spending power."""
    sim = _at_corner(make_sim)
    for _ in range(40):
        sim.step(Action.NOOP)
    assert sim.state.drifter.node is Node.W_CORNER
    assert not sim.state.drifter.in_office
    assert not sim.state.office.jam_left
    assert not sim.state.terminated


def test_open_door_with_the_monitor_up_invades_and_jams(make_sim: Callable[..., NightSim]) -> None:
    sim = _at_corner(make_sim)
    assert step_until(
        sim, lambda s: s.state.drifter.in_office, max_steps=12, action=Action.SELECT_CAM_0
    )
    assert sim.state.drifter.node is Node.OFFICE
    assert sim.state.office.jam_left
    assert not sim.state.office.jam_right

    sim.step(Action.TOGGLE_DOOR_LEFT)
    assert not sim.state.office.door_left, "a jammed door must not respond to its toggle"
    assert not sim.state.terminated


def test_invader_kills_when_the_monitor_comes_down(make_sim: Callable[..., NightSim]) -> None:
    sim = _at_corner(make_sim)
    step_until(sim, lambda s: s.state.drifter.in_office, max_steps=12, action=Action.SELECT_CAM_0)
    sim.step(Action.MONITOR_DOWN)
    assert sim.state.cause is TerminationCause.KILLED_DRIFTER


def test_invader_kills_after_the_timeout_even_with_the_monitor_up(
    make_sim: Callable[..., NightSim],
) -> None:
    """Holding the monitor up is not a way out; PROJECT.md 3.5 caps it at 30 s."""
    sim = _at_corner(make_sim)
    step_until(sim, lambda s: s.state.drifter.in_office, max_steps=12, action=Action.SELECT_CAM_0)
    entered = sim.state.drifter.invaded_at_tick
    assert entered is not None

    step_until(sim, lambda s: s.state.terminated, max_steps=80, action=Action.SELECT_CAM_0)
    assert sim.state.cause is TerminationCause.KILLED_DRIFTER
    assert sim.state.tick - entered >= 300  # 30.0 s at a 0.1 s tick


def test_death_and_invasion_are_recorded_as_events(make_sim: Callable[..., NightSim]) -> None:
    sim = _at_corner(make_sim)
    step_until(sim, lambda s: s.state.drifter.in_office, max_steps=12, action=Action.SELECT_CAM_0)
    sim.step(Action.MONITOR_DOWN)
    names = [name for _, name in sim.state.events]
    assert "invasion_drifter" in names
    assert "door_jam_left" in names
    assert "death_drifter" in names


def test_terminated_episode_cannot_be_stepped(make_sim: Callable[..., NightSim]) -> None:
    sim = _at_corner(make_sim)
    step_until(sim, lambda s: s.state.drifter.in_office, max_steps=12, action=Action.SELECT_CAM_0)
    sim.step(Action.MONITOR_DOWN)
    with pytest.raises(RuntimeError):
        sim.step(Action.NOOP)
