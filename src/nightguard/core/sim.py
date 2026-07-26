"""``NightSim``: the transition function. PROJECT.md 3.13.

The resolution order within a tick is fixed and must never change; ambiguity there produces bugs that
look like fidelity failures. v0.1 implements steps 1-4 and 6-8, with step 5 (WARDEN's countdown)
arriving in v0.2 and step 9 (trace emission) in v0.2 as well.

Scope note: PROJECT.md 7 schedules office invasion for v0.2, but this implements it now. §3.5's rule
is confirmed faithful by the primary reference, and deferring it would leave an open door with no
consequence at all, making the door mechanic untestable. See CHANGELOG.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.random import Generator

from . import blackout, power
from .clock import Clock
from .config import EscalationEvent, LevelSpec, NightConfig, UniformChoice
from .entities.base import DoorEntity, opportunity_succeeds
from .entities.drifter import Drifter
from .entities.prowler import Prowler
from .rng import SimRng
from .state import (
    Action,
    DoorEntityState,
    DoorSide,
    EntityId,
    OfficeState,
    SimState,
    TerminationCause,
    camera_for_action,
    is_camera_action,
    killed_by,
)
from .topology import Topology, load_topology


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of one night.

    Attributes:
        cause: Why the episode ended. Always populated; it is the primary diagnostic when a policy
            plateaus.
        ticks: Sim ticks elapsed.
        time_s: Elapsed simulated time, in seconds.
        final_power_pct: Remaining power in percentage points, floored at 0 for reporting.
        ai_levels: End-of-night AI levels, indexed by :class:`EntityId`.
        escalations_applied: How many hour-boundary escalation events fired.
        events: ``(tick, name)`` pairs for every notable event.
    """

    cause: TerminationCause
    ticks: int
    time_s: float
    final_power_pct: float
    ai_levels: tuple[int, ...]
    escalations_applied: int
    events: tuple[tuple[int, str], ...]

    @property
    def survived(self) -> bool:
        """Whether the agent reached dawn."""
        return self.cause is TerminationCause.SURVIVED


class NightSim:
    """A single night of simulation, driven one decision step at a time.

    The injected generator is split into fixed per-consumer substreams (see :mod:`.rng`); nothing in
    this class touches global randomness.
    """

    def __init__(
        self,
        config: NightConfig,
        rng: Generator,
        topology: Topology | None = None,
        root: Path | None = None,
    ) -> None:
        self.config = config
        self.clock = Clock.from_config(config.timing)
        self.topology = (
            topology if topology is not None else load_topology(config.topology_path(root))
        )
        self.rng = SimRng.from_generator(rng)

        self.drifter = Drifter(config.entities.drifter, self.clock, config.office)
        self.prowler = Prowler(config.entities.prowler, self.clock, config.office)
        # Fixed resolution order of PROJECT.md 3.13: [WARDEN, DRIFTER, PROWLER, SPRINTER].
        # WARDEN and SPRINTER are absent in v0.1 but keep their slots in the ordering.
        self._entities: tuple[tuple[DoorEntity, Generator], ...] = (
            (self.drifter, self.rng.drifter),
            (self.prowler, self.rng.prowler),
        )

        self._escalations = self._build_escalation_schedule()
        self._invasion_timeout_ticks = self.clock.to_ticks(
            config.office.invasion_kill_timeout_s, "invasion_kill_timeout_s"
        )
        self.state = self._initial_state()

    @classmethod
    def from_seed(
        cls,
        config: NightConfig,
        seed: int,
        topology: Topology | None = None,
        root: Path | None = None,
    ) -> NightSim:
        """Build a simulation from an integer seed. PROJECT.md 7, v0.1 exit criterion 1."""
        return cls(config, np.random.default_rng(seed), topology=topology, root=root)

    # --- setup ---------------------------------------------------------------------------------

    def _build_escalation_schedule(self) -> dict[int, list[EscalationEvent]]:
        """Map the tick of each hour boundary to the escalation events that fire on it."""
        schedule: dict[int, list[EscalationEvent]] = {}
        for event in self.config.ai.escalation:
            if not 1 <= event.hour < self.clock.hours:
                raise ValueError(
                    f"escalation hour {event.hour} is not an interior hour boundary of the night"
                )
            tick = self.clock.hour_boundary_ticks[event.hour]
            schedule.setdefault(tick, []).append(event)
        return schedule

    def _resolve_level(self, spec: LevelSpec) -> int:
        """Resolve a configured AI level, rolling it if it is a distribution."""
        if isinstance(spec, UniformChoice):
            value = spec.values[int(self.rng.reset.integers(len(spec.values)))]
        else:
            value = spec
        return self._clamp_level(value)

    def _clamp_level(self, level: int) -> int:
        """Clamp an AI level to the configured range."""
        return min(max(level, self.config.ai.level_min), self.config.ai.level_max)

    def _initial_state(self) -> SimState:
        """Build the state at 12AM."""
        levels = self.config.ai.levels
        if len(levels) != len(EntityId):
            raise ValueError(f"ai.levels must have {len(EntityId)} entries, found {len(levels)}")
        return SimState(
            tick=0,
            power_pct=self.config.power.start_pct,
            office=OfficeState(),
            ai_levels=[self._resolve_level(spec) for spec in levels],
            drifter=DoorEntityState(node=self.drifter.start_node),
            prowler=DoorEntityState(node=self.prowler.start_node),
        )

    # --- driving -------------------------------------------------------------------------------

    def step(self, action: Action | int) -> SimState:
        """Advance one decision step (5 sim ticks by default) and return the live state.

        The action is applied on the first tick of the step only; momentary lights expire at the end
        of it (PROJECT.md 3.13 steps 1 and 8).
        """
        if self.state.terminated:
            raise RuntimeError("cannot step a terminated episode")

        resolved = Action(int(action))
        for offset in range(self.clock.ticks_per_decision_step):
            self._advance_tick(resolved if offset == 0 else None)
            if self.state.terminated:
                break

        if self.config.office.light_momentary:
            self.state.office.clear_lights()
        return self.state

    def run(self, actions: Iterable[Action | int] = (), pad: Action = Action.NOOP) -> EpisodeResult:
        """Run to termination, taking actions from ``actions`` and padding with ``pad``.

        Padding rather than stopping means a short script still plays the night out, which is what
        the exit-criteria checks want.
        """
        iterator = iter(actions)
        while not self.state.terminated:
            self.step(next(iterator, pad))
        return self.result()

    def result(self) -> EpisodeResult:
        """Summarise a finished episode."""
        state = self.state
        if state.cause is None:
            raise RuntimeError("episode has not terminated")
        return EpisodeResult(
            cause=state.cause,
            ticks=state.tick,
            time_s=self.clock.time_s(state.tick),
            final_power_pct=max(0.0, state.power_pct),
            ai_levels=tuple(state.ai_levels),
            escalations_applied=state.escalations_applied,
            events=tuple(state.events),
        )

    # --- the transition function ---------------------------------------------------------------

    def _advance_tick(self, action: Action | None) -> None:
        """One sim tick, in the fixed order of PROJECT.md 3.13."""
        state = self.state

        # 1. Apply the agent's action (decision-step boundaries only).
        if action is not None:
            self._apply_action(action)

        # 2. Advance the clock; escalate if an hour boundary was crossed.
        state.tick += 1
        self._apply_escalation(state.tick)

        # 3. Drain power; enter blackout at zero.
        active = power.active_units(state.office, self.config.power)
        state.power_pct -= power.drain_per_tick(active, self.config.power, self.clock.sim_tick_s)
        if state.power_pct <= 0.0 and not state.blackout and self.config.blackout.enabled:
            blackout.apply_onset(state)

        # 4. If in blackout, advance its state machine and check for a kill.
        if state.blackout:
            state.cause = blackout.resolve(state)
            return

        # 5. WARDEN's countdown lands in v0.2.

        # 6. Movement opportunities, in the fixed entity order.
        for entity, rng in self._entities:
            entity_state = state.entity(entity.entity_id)
            if not entity.timer.fires_at(state.tick, entity_state.fire_count):
                continue
            entity_state.fire_count += 1
            if opportunity_succeeds(rng, state.ai_levels[entity.entity_id], self.config.ai):
                entity.resolve(state, rng)

        # 7. Resolve pending kills.
        self._resolve_pending_kills()
        if state.terminated:
            return

        # 8. Momentary lights expire at the end of the decision step, in step().
        # 9. Trace emission lands in v0.2.

        if state.tick >= self.clock.total_ticks:
            state.cause = TerminationCause.SURVIVED

    def _apply_action(self, action: Action) -> None:
        """Apply one action to the office. PROJECT.md 3.2."""
        state = self.state
        office = state.office

        # During blackout all actions except NOOP are no-ops.
        if state.blackout or action is Action.NOOP:
            return

        if action is Action.TOGGLE_DOOR_LEFT:
            if not office.jammed(DoorSide.LEFT):
                office.door_left = not office.door_left
        elif action is Action.TOGGLE_DOOR_RIGHT:
            if not office.jammed(DoorSide.RIGHT):
                office.door_right = not office.door_right
        elif action is Action.FLASH_LIGHT_LEFT:
            office.light_left = True
        elif action is Action.FLASH_LIGHT_RIGHT:
            office.light_right = True
        elif action is Action.MONITOR_DOWN:
            office.monitor_up = False
        elif is_camera_action(action):
            node = camera_for_action(action)
            if not self.topology.is_selectable(node):
                raise ValueError(f"{node.name} is not a selectable camera")
            office.monitor_up = True
            office.selected_camera = node

    def _apply_escalation(self, tick: int) -> None:
        """Apply any escalation events due on ``tick``. PROJECT.md 3.3."""
        events = self._escalations.get(tick)
        if events is None:
            return
        state = self.state
        for event in events:
            for name in event.entities:
                entity_id = EntityId[name]
                state.ai_levels[entity_id] = self._clamp_level(
                    state.ai_levels[entity_id] + event.delta
                )
            state.escalations_applied += 1
            state.record(f"escalation_hour_{event.hour}")

    def _resolve_pending_kills(self) -> None:
        """Resolve office occupancy. PROJECT.md 3.5.

        An entity that has entered the office kills the next time the monitor is down, or after the
        invasion timeout, whichever comes first. If the monitor was already down when it entered,
        that condition is satisfied on the same tick.
        """
        state = self.state
        for entity_id in (EntityId.DRIFTER, EntityId.PROWLER):
            entity_state = state.entity(entity_id)
            if entity_state.invaded_at_tick is None:
                continue
            elapsed = state.tick - entity_state.invaded_at_tick
            if not state.office.monitor_up or elapsed >= self._invasion_timeout_ticks:
                state.cause = killed_by(entity_id)
                state.record(f"death_{entity_id.name.lower()}")
                return
