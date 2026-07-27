"""Observation wrappers. PROJECT.md 6.4.

``Oracle`` is the non-negotiable one: it is how the partial-observability gap becomes a number
instead of an assumption. If the gap measures zero, the base observation is leaking and the leak
must be found before anything downstream is believed.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from ..core.state import Action, EntityId
from . import obs as obs_mod
from .actions import ACTION_COUNT

ORACLE_DIMS = len(EntityId)
MAX_NODE_INDEX = obs_mod.NUM_NODES - 1


class Oracle(gym.ObservationWrapper[np.ndarray, np.int64, np.ndarray]):
    """Append ground truth for all four entities. PROJECT.md 6.4.

    Three node indices plus SPRINTER's stage, each normalised to ``[0, 1]``. SPRINTER has no
    position, and its stage is the ground truth that actually predicts an attack; a constant would
    carry no information and would report a zero gap for the entity whose hidden state matters most.
    An ``armed`` flag is not included because it is exactly ``stage == stages_to_arm`` and could
    never disagree with the value beside it.

    **The measured gap is a lower bound.** SPRINTER's genuinely hidden state is larger than its
    stage: the immunity window (``immune_until_tick``) and the pending attack resolution tick are
    both invisible and both determine whether a peek is safe. This wrapper does not expose them, so
    an Oracle-versus-base comparison understates the full gap. v1.1 must not over-read it.
    """

    def __init__(self, env: gym.Env[np.ndarray, np.int64]) -> None:
        super().__init__(env)
        base = env.observation_space
        assert isinstance(base, Box)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(base.shape[0] + ORACLE_DIMS,), dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Append the ground-truth block."""
        state = self.env.unwrapped.sim.state  # type: ignore[attr-defined]
        stages_to_arm = self.env.unwrapped.config.entities.sprinter.stages_to_arm  # type: ignore[attr-defined]
        truth = np.array(
            [
                int(state.warden.node) / MAX_NODE_INDEX,
                int(state.drifter.node) / MAX_NODE_INDEX,
                int(state.prowler.node) / MAX_NODE_INDEX,
                min(1.0, state.sprinter.stage / stages_to_arm),
            ],
            dtype=np.float32,
        )
        return np.concatenate([observation, truth]).astype(np.float32)


class PreviousAction(gym.ObservationWrapper[np.ndarray, np.int64, np.ndarray]):
    """Append a one-hot of the previous action. PROJECT.md 6.4.

    Already folded into the base observation; this is the wrapper form, for POPGym-style
    comparability. Applying it to the base environment duplicates that block, which is intended:
    the point is protocol comparability, not information.
    """

    def __init__(self, env: gym.Env[np.ndarray, np.int64]) -> None:
        super().__init__(env)
        base = env.observation_space
        assert isinstance(base, Box)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(base.shape[0] + ACTION_COUNT,), dtype=np.float32
        )
        self._previous = Action.NOOP

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Clear the remembered action."""
        self._previous = Action.NOOP
        return super().reset(**kwargs)

    def step(self, action: np.int64) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Remember the action, then delegate."""
        observation, reward, terminated, truncated, info = super().step(action)
        self._previous = Action(int(action))
        return observation, float(reward), terminated, truncated, info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Append the one-hot."""
        one_hot = np.zeros(ACTION_COUNT, dtype=np.float32)
        one_hot[int(self._previous)] = 1.0
        return np.concatenate([observation, one_hot]).astype(np.float32)


class AudioMask(gym.ObservationWrapper[np.ndarray, np.int64, np.ndarray]):
    """Zero the audio block. PROJECT.md 6.4.

    The ablation for how much the environment leans on §3.9. Expect it to bite hardest against
    SPRINTER, whose only warning is the `running` cue inside a 0.5 s grace period.
    """

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Return a copy with the audio dims zeroed."""
        masked = observation.copy()
        masked[obs_mod.AUDIO_START : obs_mod.AUDIO_START + obs_mod.AUDIO_DIMS] = 0.0
        return masked


class FrameStack(gym.ObservationWrapper[np.ndarray, np.int64, np.ndarray]):
    """Concatenate the last N observations. PROJECT.md 6.4.

    Provided for completeness. Note the arithmetic before reaching for it: covering a 15 s WARDEN
    countdown at a 0.5 s decision step needs **30 frames**, which at 100 dims is a 3,000-dim
    observation carrying one bit of genuinely useful history. That is why recurrence is the intended
    approach for v1.1 and why §7 rules frame stacking out there.
    """

    def __init__(self, env: gym.Env[np.ndarray, np.int64], frames: int = 4) -> None:
        super().__init__(env)
        if frames < 1:
            raise ValueError(f"frames must be at least 1, found {frames}")
        base = env.observation_space
        assert isinstance(base, Box)
        self.frames = frames
        self._width = int(base.shape[0])
        self._buffer: deque[np.ndarray] = deque(maxlen=frames)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(self._width * frames,), dtype=np.float32
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Fill the buffer with the first observation rather than with zeros."""
        self._buffer.clear()
        return super().reset(**kwargs)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Append and flatten."""
        if not self._buffer:
            for _ in range(self.frames):
                self._buffer.append(observation)
        else:
            self._buffer.append(observation)
        return np.concatenate(list(self._buffer)).astype(np.float32)
