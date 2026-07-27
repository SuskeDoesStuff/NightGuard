"""Gymnasium wrapper layer. PROJECT.md 6.

Spaces, observation encoding, reward and the environment itself. May import gymnasium and numpy;
must not import torch. Ground truth reaches the caller through ``info``, never through ``obs``.

Importing this package registers the env IDs of PROJECT.md 2.4 as a side effect, which is what makes
``gym.make("NightGuard-v0")`` work.
"""

from .actions import ACTION_COUNT, action_space, decode, encode
from .nightguard_env import NightGuardEnv
from .obs import OBS_DIMS, BeliefTracker, observation_space
from .registration import (
    BASE_ID,
    CUSTOM_MAX_ID,
    DEFERRED_IDS,
    REGISTERED_IDS,
    register_environments,
)
from .reward import RewardTracker, max_return
from .wrappers import AudioMask, FrameStack, Oracle, PreviousAction

register_environments()

__all__ = [
    "ACTION_COUNT",
    "BASE_ID",
    "CUSTOM_MAX_ID",
    "DEFERRED_IDS",
    "OBS_DIMS",
    "REGISTERED_IDS",
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
    "register_environments",
]
