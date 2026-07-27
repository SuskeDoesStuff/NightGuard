"""Non-recurrent PPO. PROJECT.md 1.1, and the memory ablation for v1.1.

Feed-forward PPO on the same observation, with no memory at all. It is not a baseline anyone expects
to win: the environment hides every entity position, so a policy that cannot remember where it last
looked is choosing under a much smaller information set than the LSTM has. That is the point --
the gap between this and ``RecurrentPPO`` says whether the task is genuinely a memory task, from the
opposite direction to the way the Oracle wrapper says it.

Deliberately thin. Everything shared -- environment construction, the evaluation callback, the
curve -- lives in :mod:`~nightguard.train.recurrent_ppo`, because duplicating it would let the two
arms drift apart and turn an ablation into a comparison of two harnesses.
"""

from __future__ import annotations

from dataclasses import replace

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from .config import TrainConfig
from .recurrent_ppo import build_model


def build_feedforward_model(
    venv: VecEnv,
    train: TrainConfig,
    tensorboard_log: str | None = None,
) -> BaseAlgorithm:
    """Build the feed-forward arm, whatever ``train.algorithm`` happens to say."""
    return build_model(venv, replace(train, algorithm="ppo"), tensorboard_log)
