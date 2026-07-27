"""Shared fixtures and helpers for the v0.1 test suite."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace

import pytest

from nightguard.core import (
    Action,
    Clock,
    EntityId,
    LevelSpec,
    NightConfig,
    NightSim,
    Node,
    SimState,
    Topology,
    load_night_config,
    load_topology,
    with_levels,
)
from nightguard.core.config import EscalationEvent

# The idle drain table of PROJECT.md 3.10, as (night, night_divisor).
NIGHT_DIVISORS = {1: 9.6, 2: 6.0, 3: 5.0, 4: 4.0, 5: 3.0, 6: 3.0}

MAX_LEVEL = 20


@pytest.fixture(scope="session")
def topology() -> Topology:
    """The default node metadata, loaded once."""
    return load_topology(NightConfig().topology_path())


@pytest.fixture(scope="session")
def night1() -> NightConfig:
    """Night 1's preset."""
    return load_night_config(1)


@pytest.fixture(scope="session")
def clock(night1: NightConfig) -> Clock:
    """The default clock."""
    return Clock.from_config(night1.timing)


@pytest.fixture
def make_sim(topology: Topology) -> Callable[..., NightSim]:
    """Build a :class:`NightSim` for a night, optionally overriding the AI levels."""

    def factory(
        night: int = 1,
        seed: int = 0,
        levels: Sequence[LevelSpec] | None = None,
        escalation: Sequence[EscalationEvent] | None = None,
        only: Iterable[EntityId] | None = None,
    ) -> NightSim:
        config = load_night_config(night)
        if levels is not None:
            config = with_levels(config, levels)
        if escalation is not None:
            config = replace(config, ai=replace(config.ai, escalation=tuple(escalation)))
        if only is not None:
            config = with_only(config, only)
        return NightSim.from_seed(config, seed=seed, topology=topology)

    return factory


def with_only(config: NightConfig, enabled: Iterable[EntityId]) -> NightConfig:
    """Disable every entity except those named. PROJECT.md 8.3's "all other entities disabled"."""
    keep = set(enabled)
    entities = config.entities
    return replace(
        config,
        entities=replace(
            entities,
            warden=replace(entities.warden, enabled=EntityId.WARDEN in keep),
            drifter=replace(entities.drifter, enabled=EntityId.DRIFTER in keep),
            prowler=replace(entities.prowler, enabled=EntityId.PROWLER in keep),
            sprinter=replace(entities.sprinter, enabled=EntityId.SPRINTER in keep),
        ),
    )


def clear_stage(sim: NightSim) -> None:
    """Move both door entities off STAGE so WARDEN's stage lock does not hold.

    Disabled entities never move, so a WARDEN-only simulation would otherwise keep them parked on
    STAGE forever. PROJECT.md 8.3 calls this "the stage lock bypassed".
    """
    sim.state.drifter.node = Node.COMMONS
    sim.state.prowler.node = Node.COMMONS


def force_blackout_at(sim: NightSim, sim_tick: int) -> NightSim:
    """Zero power so the simulation enters blackout on ``sim_tick`` by its own rule. PROJECT.md 8.7.

    8.7 specifies "power forced to 0 at t = 500 s". Power is zeroed on the preceding tick and the
    crossing is then detected by 3.13 step 3, so onset happens in-band: the event stamp and
    ``phase_started_tick`` agree, and nothing mutates state between ticks. Calling ``apply_onset``
    from outside the tick loop instead would stamp the event with the already-written tick and make
    the trace show it one record late.
    """
    target = sim_tick - 1

    def zero_power(state: SimState, _action: Action | None) -> None:
        if state.tick == target:
            state.power_pct = 0.0

    sim.on_tick = zero_power
    while sim.state.tick < sim_tick and not sim.state.terminated:
        sim.step(Action.NOOP)
    sim.on_tick = None
    return sim


def step_until(
    sim: NightSim,
    predicate: Callable[[NightSim], bool],
    max_steps: int = 200,
    action: Action = Action.NOOP,
) -> bool:
    """Step with ``action`` until ``predicate`` holds, the episode ends, or the budget runs out."""
    for _ in range(max_steps):
        if predicate(sim):
            return True
        if sim.state.terminated:
            return False
        sim.step(action)
    return predicate(sim)
