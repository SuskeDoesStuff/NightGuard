"""Gymnasium wrapper layer. PROJECT.md 6.

Spaces, observation encoding, reward and the environment itself. May import gymnasium and numpy;
must not import torch. Ground truth reaches the caller through ``info``, never through ``obs``.
"""

from .actions import ACTION_COUNT, action_space, decode, encode
from .nightguard_env import NightGuardEnv
from .obs import OBS_DIMS, BeliefTracker, observation_space
from .reward import RewardTracker, max_return
from .wrappers import AudioMask, FrameStack, Oracle, PreviousAction

__all__ = [
    "ACTION_COUNT",
    "OBS_DIMS",
    "AudioMask",
    "BeliefTracker",
    "FrameStack",
    "NightGuardEnv",
    "Oracle",
    "PreviousAction",
    "RewardTracker",
    "action_space",
    "decode",
    "encode",
    "max_return",
    "observation_space",
]
