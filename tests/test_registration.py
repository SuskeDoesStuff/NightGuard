"""Every ID PROJECT.md 2.4 declares must resolve. v1.1 criterion 1.

§2.4 declared ``NightGuard-v0`` from v0.1 onwards, but nothing ever called ``register``, so the ID
did not exist. The environment passed ``check_env`` when constructed directly, which is why the gap
survived v1.0's review: criterion 1 was satisfied by a path that never touched the declaration.
"""

from __future__ import annotations

import warnings

import gymnasium as gym
import pytest
from gymnasium.error import NameNotFound
from gymnasium.utils.env_checker import check_env

from nightguard.core.config import ConfigError, load_preset
from nightguard.env import DEFERRED_IDS, REGISTERED_IDS
from nightguard.env.obs import OBS_DIMS
from nightguard.env.registration import BASE_ID, CUSTOM_MAX_ID


@pytest.mark.parametrize("env_id", REGISTERED_IDS)
def test_gym_make_returns_a_working_env(env_id: str) -> None:
    env = gym.make(env_id)
    try:
        assert env.observation_space.shape == (OBS_DIMS,)
        observation, _ = env.reset(seed=11)
        assert env.observation_space.contains(observation)
        observation, _, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", REGISTERED_IDS)
def test_gym_make_passes_the_env_checker(env_id: str) -> None:
    env = gym.make(env_id)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_night_kwarg_reaches_the_config() -> None:
    env = gym.make(BASE_ID, night=6)
    try:
        assert env.unwrapped.config.ai.levels == (4, 10, 12, 16)  # type: ignore[attr-defined]
    finally:
        env.close()


def test_custom_max_id_loads_the_custom_preset() -> None:
    env = gym.make(CUSTOM_MAX_ID)
    try:
        config = env.unwrapped.config  # type: ignore[attr-defined]
        assert config.ai.levels == (20, 20, 20, 20)
        assert config.power.night_divisor == 3.0
    finally:
        env.close()


def test_no_time_limit_wrapper() -> None:
    """The night is a fixed horizon the simulator ends itself; truncation would misreport it."""
    env = gym.make(BASE_ID, night=1)
    try:
        assert "TimeLimit" not in str(env)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", DEFERRED_IDS)
def test_v2_ids_are_not_registered_yet(env_id: str) -> None:
    """A registered ID that resolves to something else is worse than a missing one."""
    with pytest.raises(NameNotFound):
        gym.make(env_id)


def test_registration_is_idempotent() -> None:
    from nightguard.env.registration import register_environments

    register_environments()
    register_environments()
    gym.make(BASE_ID).close()


def test_escalation_cannot_push_custom_max_above_the_ceiling() -> None:
    """20/20/20/20 plus 3.3's escalation events stays at 20: sim.py clamps to ai.level_max."""
    config = load_preset("custom_max")
    assert max(config.ai.levels) == config.ai.level_max  # type: ignore[type-var]


def test_preset_names_cannot_traverse_directories() -> None:
    with pytest.raises(ConfigError):
        load_preset("../topology/default")
