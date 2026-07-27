"""The Gymnasium environment. PROJECT.md 6.

A thin wrapper over the validated :class:`~nightguard.core.sim.NightSim`. Everything about *what
happens* lives in ``core/``; this layer only decides what the agent is told and what it is paid.

``info`` carries full ground truth so the v0.2 trace writer works unchanged inside an RL loop.
Ground truth in ``info`` is fine — it is a debugging channel the policy never sees. Ground truth in
``obs`` is the failure this milestone must not have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from ..core.config import NightConfig, load_night_config, load_preset
from ..core.sim import NightSim
from ..core.state import Action, EntityId, TerminationCause
from ..core.topology import Topology, load_topology
from . import obs as obs_mod
from .actions import ACTION_COUNT, decode
from .reward import RewardTracker

DEFAULT_NIGHT = 4


def _resolve_config(
    night: int, preset: str | None, config: NightConfig | None, root: Path | None
) -> NightConfig:
    """Pick a config from the three mutually exclusive ways of naming one."""
    if config is not None:
        return config
    if preset is not None:
        return load_preset(preset, root)
    return load_night_config(night, root)


class NightGuardEnv(gym.Env[np.ndarray, np.int64]):
    """A single night, as a Gymnasium environment. PROJECT.md 6.

    Args:
        night: Which night preset to load, 1-6.
        preset: A preset name under ``configs/nights/``, overriding ``night``. This is how the
            v1.1 curriculum reaches its custom third stage.
        config: A prebuilt config, overriding both ``night`` and ``preset``.
        topology: A preloaded topology, to avoid re-reading the file per environment.
        root: Repository root, for resolving the topology path.
    """

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012 - gym.Env declares this

    def __init__(
        self,
        night: int = DEFAULT_NIGHT,
        preset: str | None = None,
        config: NightConfig | None = None,
        topology: Topology | None = None,
        root: Path | None = None,
    ) -> None:
        self.night = night
        self.preset = preset
        self.config = _resolve_config(night, preset, config, root)
        self.topology = (
            topology if topology is not None else load_topology(self.config.topology_path(root))
        )

        self.action_space: Discrete[np.int64] = Discrete(ACTION_COUNT)
        self.observation_space: Box = obs_mod.observation_space()

        self._reward = RewardTracker(self.config)
        self._belief = obs_mod.BeliefTracker.seeded(self.config)
        self._last_action = Action.NOOP
        self._sim: NightSim | None = None

    # --- properties --------------------------------------------------------------------------

    @property
    def sim(self) -> NightSim:
        """The underlying simulation. Raises if :meth:`reset` has not been called."""
        if self._sim is None:
            raise RuntimeError("reset() must be called before accessing the simulation")
        return self._sim

    # --- gymnasium API -----------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new night. PROJECT.md 6."""
        super().reset(seed=seed)
        # Always spawn from ``self.np_random``, never from a second generator built here.
        # Gymnasium seeds ``self.np_random`` from ``SeedSequence(seed)``, which is bit-identical to
        # ``default_rng(seed)``, so ``reset(seed=k)`` still matches ``NightSim.from_seed(k)``. But
        # ``SimRng`` spawns *children* and advances the parent's spawn counter, so consecutive
        # unseeded resets get fresh substreams. Building a fresh generator on the seeded path meant
        # the next unseeded reset spawned children 0..5 a second time and replayed the seeded
        # episode verbatim -- which is exactly what SB3 does, once per run.
        self._sim = NightSim(self.config, self.np_random, topology=self.topology)
        self._reward.reset()
        # Seeded from config, never from live state: see obs.BeliefTracker.seeded.
        self._belief = obs_mod.BeliefTracker.seeded(self.config)
        self._last_action = Action.NOOP
        self._belief.update(self._sim, self._sim.state)
        return self._observe(), self._info()

    def step(self, action: np.int64 | int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one decision step. PROJECT.md 3.1, 6.3."""
        sim = self.sim
        resolved = decode(action)
        power_before = sim.state.power_pct

        sim.step(resolved)
        self._last_action = resolved
        self._belief.update(sim, sim.state)

        terminated = sim.state.terminated
        reward = self._reward.step_reward(sim.state, power_before, terminated)
        # The night is a fixed horizon that the simulator ends itself, so reaching dawn is a
        # terminal state rather than a time-limit truncation.
        return self._observe(), reward, terminated, False, self._info()

    # --- internals ---------------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        return obs_mod.encode(self.sim, self.sim.state, self._belief, self._last_action)

    def _info(self) -> dict[str, Any]:
        """Full ground truth, for the trace writer and for diagnostics. Never seen by the policy."""
        state = self.sim.state
        return {
            "tick": state.tick,
            "time_s": self.sim.clock.time_s(state.tick),
            "power_pct": state.power_pct,
            # Camera duty cycle is the fraction of decision steps with the monitor raised: the
            # quantity that both drains power and opens the office. PROJECT.md 7, v1.1 criterion 6.
            "monitor_up": state.office.monitor_up,
            "blackout": state.blackout,
            "blackout_phase": (
                None if state.blackout_state is None else state.blackout_state.phase.name
            ),
            "termination_cause": None if state.cause is None else state.cause.value,
            "ai_levels": tuple(state.ai_levels),
            "true_nodes": {
                EntityId.WARDEN.name: int(state.warden.node),
                EntityId.DRIFTER.name: int(state.drifter.node),
                EntityId.PROWLER.name: int(state.prowler.node),
            },
            "sprinter": {
                "stage": state.sprinter.stage,
                "armed": state.sprinter.armed,
                "bangs": state.sprinter.bang_count,
            },
            "jams": (state.office.jam_left, state.office.jam_right),
        }

    def survived(self) -> bool:
        """Whether the finished episode reached dawn."""
        return self.sim.state.cause is TerminationCause.SURVIVED
