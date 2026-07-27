"""Gymnasium registration. PROJECT.md 2.4.

§2.4 declares the env IDs but nothing ever called :func:`gymnasium.register`, so
``gym.make("NightGuard-v0")`` raised ``NameNotFound`` while the environment itself passed
``check_env`` when constructed directly. Every config-driven workflow -- ``make_vec_env``, most
training tooling -- takes a string ID, so the declaration has to be real.

Two IDs, which is the minimum the v1.1 curriculum needs:

``NightGuard-v0``
    The six shipped nights, selected with ``env_kwargs={"night": n}``. Defaults to night 4.
``NightGuard-CustomMax-v0``
    Curriculum stage 3: ``configs/nights/custom_max.yaml``, every entity at ``ai.level_max``.

The three IDs §2.4 schedules for v2.0 are deliberately **not** registered. They name randomised
configurations that do not exist yet, and a registered ID that silently resolves to something else
is worse than a missing one.

**No ``max_episode_steps``.** The night is a fixed horizon the simulator ends itself (3.1), so
termination is terminal; a ``TimeLimit`` wrapper would convert it into truncation and change the
bootstrapping of the final value estimate.
"""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

ENTRY_POINT = "nightguard.env.nightguard_env:NightGuardEnv"

BASE_ID = "NightGuard-v0"
CUSTOM_MAX_ID = "NightGuard-CustomMax-v0"

REGISTERED_IDS: tuple[str, ...] = (BASE_ID, CUSTOM_MAX_ID)

# Declared in 2.4 for v2.0, and unregistered until the randomised configs exist.
DEFERRED_IDS: tuple[str, ...] = (
    "NightGuard-Easy-v0",
    "NightGuard-Hard-v0",
    "NightGuard-Random-v0",
)


def register_environments() -> None:
    """Register every ID this version ships. Idempotent, so re-import is harmless."""
    if BASE_ID not in registry:
        register(id=BASE_ID, entry_point=ENTRY_POINT)
    if CUSTOM_MAX_ID not in registry:
        register(id=CUSTOM_MAX_ID, entry_point=ENTRY_POINT, kwargs={"preset": "custom_max"})
