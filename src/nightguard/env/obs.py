"""Observation encoding. PROJECT.md 6.2.

The design rule is the one that matters: **give the agent exactly what a human player perceives, and
nothing more.** Never leak a true entity position. That is what the `Oracle` wrapper is for, and a
leak here produces a policy that learns beautifully and means nothing — the learning curve looks
better, not worse, which is why 6.2's no-leak test is the most important in the milestone.

Layout, 100 dimensions:

===========  ====  ======================================================================
Block        Dims  Contents
===========  ====  ======================================================================
Resources       2  ``power / 99.0``, ``time_s / 535.0``
Office          7  door_left, door_right, **jam_left, jam_right**, light_left,
                   light_right, monitor_up
Camera         12  one-hot over cameras 0-10, plus a twelfth slot for "monitor down"
Belief x 4     56  14 per entity: 12-dim last-observed node, ``ticks_since_observed``,
                   ``visible_now``
Audio           4  footstep, kitchen, running, bang
Door proximity  2  left_occupied, right_occupied, non-zero **only** on a step in which
                   the corresponding light was flashed
Last action    17  one-hot
===========  ====  ======================================================================

The jam bits are an addition to 6.2's original 98-dim layout, inserted into the office block rather
than appended. A door toggle with invisible semantics is unlearnable: the agent presses the button,
nothing happens, and it cannot distinguish "jammed" from "I toggled twice in a row". Jam state is
office state, so it belongs in that block; every block after it shifts by two. See PROJECT.md 10.

Three details that are easy to get wrong, all of them load-bearing:

**SPRINTER's belief block is irregular.** It has no node, so its 12-dim slot encodes stage as a
one-hot over ``{unknown, 0, 1, 2, 3}`` in the first five positions with the rest zero. This is the
one asymmetry in an otherwise uniform layout.

**Observation runs on two independent channels**, joined by ``or`` in 6.2, and conflating them is a
bug that looks entirely plausible. An entity is observed if the monitor is up and its node is the
selected camera and that node has a video feed; **or** if it is at a door corner and the
corresponding light was flashed this step. Both corners carry video, so watching a corner camera
*is* an observation — treating the light as the only corner channel would make both corners
camera-invisible, teach the policy that peeking at a corner is pointless, and silently break the
wing asymmetry 2.3 depends on. The proximity block is the light-specific channel, not the only one.

**Belief is seeded at reset from the configured start state**, with staleness 0. The start never
varies and is publicly known, so a competent player begins with correct belief; modelling ignorance
nobody has is the less faithful choice. Seeding reads **config, not live state**, which is what
keeps the no-leak test exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from gymnasium.spaces import Box

from ..core.config import NightConfig
from ..core.sim import NightSim
from ..core.state import Action, DoorSide, EntityId, SimState
from ..core.topology import Node

NUM_NODES = len(Node)
NUM_ENTITIES = len(EntityId)

RESOURCE_DIMS = 2
OFFICE_DIMS = 7
CAMERA_DIMS = NUM_NODES  # 11 cameras plus a "monitor down" slot
BELIEF_DIMS_PER_ENTITY = NUM_NODES + 2  # node one-hot, staleness, visible_now
BELIEF_DIMS = BELIEF_DIMS_PER_ENTITY * NUM_ENTITIES
AUDIO_DIMS = 4
PROXIMITY_DIMS = 2
ACTION_DIMS = len(Action)

OBS_DIMS = (
    RESOURCE_DIMS
    + OFFICE_DIMS
    + CAMERA_DIMS
    + BELIEF_DIMS
    + AUDIO_DIMS
    + PROXIMITY_DIMS
    + ACTION_DIMS
)

# Block offsets, derived rather than written down, so inserting the jam bits could not desynchronise
# them from the widths above.
RESOURCE_START = 0
OFFICE_START = RESOURCE_START + RESOURCE_DIMS
CAMERA_START = OFFICE_START + OFFICE_DIMS
BELIEF_START = CAMERA_START + CAMERA_DIMS
AUDIO_START = BELIEF_START + BELIEF_DIMS
PROXIMITY_START = AUDIO_START + AUDIO_DIMS
ACTION_START = PROXIMITY_START + PROXIMITY_DIMS

# PROJECT.md 6.2: staleness saturates at 60 s, roughly twelve DRIFTER moves and well past the point
# where belief carries information.
STALENESS_CAP_TICKS = 600

# SPRINTER's irregular slot: one-hot over {unknown, 0, 1, 2, 3}.
SPRINTER_UNKNOWN_SLOT = 0
SPRINTER_STAGE_SLOT = 1


def observation_space() -> Box:
    """The `Box` of PROJECT.md 6.2, widened to 100 dims by the jam bits."""
    return Box(low=0.0, high=1.0, shape=(OBS_DIMS,), dtype=np.float32)


@dataclass
class EntityBelief:
    """What the agent last saw of one entity.

    Attributes:
        slot: Node index for the three positional entities, or stage for SPRINTER.
        last_seen_tick: Tick of the most recent observation, or ``None`` if never observed.
        ever_seen: Whether any observation has happened. False only under a configuration that
            does not seed belief at reset.
    """

    slot: int = 0
    last_seen_tick: int | None = None
    ever_seen: bool = False


@dataclass
class BeliefTracker:
    """Per-entity belief, updated once per decision step. PROJECT.md 6.2.

    This is the only stateful part of the encoder. It holds exactly what a player could have seen
    and never reads a hidden position: :meth:`update` consults live state only to decide whether an
    entity is *currently observable*, and copies its slot only when it is.
    """

    beliefs: dict[EntityId, EntityBelief] = field(default_factory=dict)

    @classmethod
    def seeded(cls, config: NightConfig) -> BeliefTracker:
        """Seed from the configured start state. PROJECT.md 3 and 10.

        Reads **config**, never live state: an agent that starts each night knowing where the roster
        begins is modelling public knowledge, but reading live state here would leak.
        """
        entities = config.entities
        start = {
            EntityId.WARDEN: int(Node[entities.warden.path[0]]),
            EntityId.DRIFTER: int(Node[entities.drifter.start]),
            EntityId.PROWLER: int(Node[entities.prowler.start]),
            EntityId.SPRINTER: 0,  # stage 0
        }
        return cls(
            beliefs={
                entity: EntityBelief(slot=slot, last_seen_tick=0, ever_seen=True)
                for entity, slot in start.items()
            }
        )

    def update(self, sim: NightSim, state: SimState) -> None:
        """Record any entity observable this step. PROJECT.md 6.2."""
        for entity in EntityId:
            if not _is_observed(sim, state, entity):
                continue
            belief = self.beliefs.setdefault(entity, EntityBelief())
            belief.slot = _true_slot(state, entity)
            belief.last_seen_tick = state.tick
            belief.ever_seen = True

    def staleness(self, entity: EntityId, tick: int) -> float:
        """Normalised ticks since the last observation; 1.0 if never observed.

        Never-seen reads as maximally stale, which is what it is. Zero would collide with "seen this
        instant" and make the feature jump upward on the first observation, inverting its meaning.
        """
        belief = self.beliefs.get(entity)
        if belief is None or belief.last_seen_tick is None:
            return 1.0
        elapsed = min(tick - belief.last_seen_tick, STALENESS_CAP_TICKS)
        return elapsed / STALENESS_CAP_TICKS


def _true_slot(state: SimState, entity: EntityId) -> int:
    """The value belief stores: a node index, or SPRINTER's stage."""
    if entity is EntityId.SPRINTER:
        return state.sprinter.stage
    if entity is EntityId.WARDEN:
        return int(state.warden.node)
    return int(state.entity(entity).node)


def _true_node(state: SimState, entity: EntityId) -> Node | None:
    """The entity's node, or ``None`` for SPRINTER, which has no position."""
    if entity is EntityId.SPRINTER:
        return None
    if entity is EntityId.WARDEN:
        return state.warden.node
    return state.entity(entity).node


def _is_observed(sim: NightSim, state: SimState, entity: EntityId) -> bool:
    """Whether the agent can see this entity this step. PROJECT.md 6.2.

    Two independent channels joined by ``or``. SPRINTER has no position and is never visually
    observable at all — it reaches the agent only through audio, which is the asymmetry 3.7 exists
    to create.
    """
    node = _true_node(state, entity)
    if node is None:
        return False

    office = state.office
    # Channel 1: the camera. E_KITCHEN has no feed, so selecting it never yields an observation.
    if office.monitor_up and office.selected_camera == node and sim.topology.has_video(node):
        return True

    # Channel 2: the door light, which reveals only the corner it illuminates.
    if node == sim.drifter.corner and office.light_left:
        return True
    return bool(node == sim.prowler.corner and office.light_right)


def _corner_occupied(sim: NightSim, state: SimState, side: DoorSide) -> bool:
    """Whether anything is at that door corner. Only ever read when the light is on."""
    if side is DoorSide.LEFT:
        return state.drifter.node == sim.drifter.corner
    return state.prowler.node == sim.prowler.corner or state.warden.node == sim.warden.corner


def encode(
    sim: NightSim, state: SimState, belief: BeliefTracker, last_action: Action
) -> np.ndarray:
    """Encode one observation. PROJECT.md 6.2.

    Returns:
        A ``float32`` vector of :data:`OBS_DIMS` values, every one inside ``[0, 1]``.
    """
    obs = np.zeros(OBS_DIMS, dtype=np.float32)
    office = state.office
    config = sim.config

    # Resources. Power is guaranteed non-negative by 3.13 step 3; time saturates at dawn.
    obs[RESOURCE_START] = min(1.0, max(0.0, state.power_pct / config.power.start_pct))
    obs[RESOURCE_START + 1] = min(1.0, state.tick / sim.clock.total_ticks)

    # Office, including the jam bits.
    obs[OFFICE_START : OFFICE_START + OFFICE_DIMS] = (
        float(office.door_left),
        float(office.door_right),
        float(office.jam_left),
        float(office.jam_right),
        float(office.light_left),
        float(office.light_right),
        float(office.monitor_up),
    )

    # Camera: one-hot over the selected node, or the trailing "monitor down" slot.
    if office.monitor_up:
        obs[CAMERA_START + int(office.selected_camera)] = 1.0
    else:
        obs[CAMERA_START + CAMERA_DIMS - 1] = 1.0

    # Belief, one 14-dim block per entity in EntityId order.
    for entity in EntityId:
        base = BELIEF_START + int(entity) * BELIEF_DIMS_PER_ENTITY
        record = belief.beliefs.get(entity)
        if record is not None and record.ever_seen:
            if entity is EntityId.SPRINTER:
                # Irregular: stage one-hot over {unknown, 0, 1, 2, 3} in the first five slots.
                obs[base + SPRINTER_STAGE_SLOT + record.slot] = 1.0
            else:
                obs[base + record.slot] = 1.0
        elif entity is EntityId.SPRINTER:
            obs[base + SPRINTER_UNKNOWN_SLOT] = 1.0
        obs[base + NUM_NODES] = belief.staleness(entity, state.tick)
        obs[base + NUM_NODES + 1] = float(_is_observed(sim, state, entity))

    # Audio, from the step-level channel: 3.9 emits for the decision step in which the event
    # occurred, and a tick is a fifth of a step.
    audio = state.step_audio
    obs[AUDIO_START : AUDIO_START + AUDIO_DIMS] = (
        float(audio.footstep),
        float(audio.kitchen),
        float(audio.running),
        float(audio.bang),
    )

    # Door proximity: the light-specific channel, non-zero only where the light fired this step.
    if office.light_left:
        obs[PROXIMITY_START] = float(_corner_occupied(sim, state, DoorSide.LEFT))
    if office.light_right:
        obs[PROXIMITY_START + 1] = float(_corner_occupied(sim, state, DoorSide.RIGHT))

    obs[ACTION_START + int(last_action)] = 1.0
    return obs
