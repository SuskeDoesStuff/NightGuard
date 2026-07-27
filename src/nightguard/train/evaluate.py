"""Scoring a policy. PROJECT.md 7, v1.1 criteria 3, 4 and 6.

**One evaluation path for every kind of policy.** A learned policy and `rhythm` are scored by the
same function over the same seeds, because the milestone's substantive claim is a comparison between
them and a comparison run through two harnesses is a comparison of two harnesses.

Torch is imported lazily and only when a learned policy is passed, so a base install can still score
the reference policies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from ..core.config import NightConfig
from ..core.state import Action, TerminationCause
from ..core.topology import Topology
from ..env.nightguard_env import NightGuardEnv
from ..policies.base import Policy, observe


class Predictor(Protocol):
    """The subset of an SB3 model that evaluation needs."""

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Any]:
        """Choose an action, carrying recurrent state across calls."""


@dataclass
class EvalResult:
    """The outcome of an evaluation run.

    Attributes:
        episodes: How many nights were played.
        survival: Fraction reaching dawn. The headline number.
        mean_return: Mean undiscounted episode return under the configured reward.
        mean_steps: Mean decision steps survived. The dense-reward proxy, and the only signal that
            moves early on a night where nothing reaches dawn.
        camera_duty_cycle: Fraction of decision steps with the monitor raised. PROJECT.md 7's
            criterion 6. `rhythm` measures 0.0793; its design figure of 1/12 is slightly higher
            because the peek cycle spends extra steps closing doors first.
        causes: Termination-cause histogram. The primary diagnostic when a policy plateaus.
    """

    episodes: int
    survival: float
    mean_return: float
    mean_steps: float
    camera_duty_cycle: float
    causes: dict[str, int] = field(default_factory=dict)

    def standard_error(self) -> float:
        """Binomial standard error on ``survival``, for "outside sampling error" claims."""
        if self.episodes == 0:
            return 0.0
        return float((self.survival * (1.0 - self.survival) / self.episodes) ** 0.5)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, for the run summary."""
        return {
            "episodes": self.episodes,
            "survival": self.survival,
            "standard_error": self.standard_error(),
            "mean_return": self.mean_return,
            "mean_steps": self.mean_steps,
            "camera_duty_cycle": self.camera_duty_cycle,
            "causes": dict(self.causes),
        }


def eval_seeds(count: int, offset: int) -> list[int]:
    """The held-out evaluation seeds. Fixed, so two policies are scored on the same nights."""
    return [offset + index for index in range(count)]


def _collect(
    env: gym.Env[np.ndarray, np.int64],
    seeds: Iterable[int],
    choose: Callable[[np.ndarray, NightGuardEnv, bool], Action],
) -> EvalResult:
    """Run one episode per seed, accumulating the shared statistics.

    ``env`` may be wrapped -- the Oracle arm is scored through this same function -- so the
    simulator is reached through ``unwrapped`` rather than assumed to be at the top.
    """
    core: NightGuardEnv = env.unwrapped  # type: ignore[assignment]
    survived = 0
    returns: list[float] = []
    lengths: list[int] = []
    monitor_up = 0
    steps_total = 0
    causes: Counter[str] = Counter()

    played = 0
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        total = 0.0
        steps = 0
        first = True
        while True:
            action = choose(observation, core, first)
            first = False
            observation, reward, terminated, truncated, info = env.step(np.int64(action))
            total += float(reward)
            steps += 1
            monitor_up += bool(info["monitor_up"])
            if terminated or truncated:
                break
        played += 1
        steps_total += steps
        returns.append(total)
        lengths.append(steps)
        cause = core.sim.state.cause
        assert cause is not None
        causes[cause.value] += 1
        survived += cause is TerminationCause.SURVIVED

    return EvalResult(
        episodes=played,
        survival=survived / played if played else 0.0,
        mean_return=float(np.mean(returns)) if returns else 0.0,
        mean_steps=float(np.mean(lengths)) if lengths else 0.0,
        camera_duty_cycle=monitor_up / steps_total if steps_total else 0.0,
        causes=dict(causes),
    )


def evaluate_scripted(
    config: NightConfig,
    factory: Callable[[], Policy],
    seeds: Sequence[int],
    topology: Topology | None = None,
) -> EvalResult:
    """Score a scripted policy from ``policies/``, through the Gymnasium env.

    Scripted policies consume a :class:`~nightguard.policies.base.Percept` rather than the Box
    observation, so this builds one per step from the same simulator the learned policy sees. The
    env, the seeds, the reward and the statistics are otherwise identical.
    """
    env = NightGuardEnv(config=config, topology=topology)
    policy = factory()
    step_counter = {"n": 0}

    def choose(_observation: np.ndarray, env: NightGuardEnv, first: bool) -> Action:
        if first:
            policy.reset()
            step_counter["n"] = 0
        percept = observe(env.sim, env.sim.state, step_counter["n"])
        step_counter["n"] += 1
        return policy(percept)

    return _collect(env, seeds, choose)


def evaluate_model(
    env: gym.Env[np.ndarray, np.int64],
    model: Predictor,
    seeds: Sequence[int],
    deterministic: bool = True,
) -> EvalResult:
    """Score a learned policy.

    ``state`` carries the LSTM hidden state between steps and is cleared at every episode boundary;
    a recurrent policy evaluated without that reset is being scored on the previous night's memory.
    """
    recurrent: Any = None

    def choose(observation: np.ndarray, _env: NightGuardEnv, first: bool) -> Action:
        nonlocal recurrent
        if first:
            recurrent = None
        starts = np.array([first])
        action, recurrent = model.predict(
            observation[None, :],
            state=recurrent,
            episode_start=starts,
            deterministic=deterministic,
        )
        return Action(int(np.asarray(action).reshape(-1)[0]))

    return _collect(env, seeds, choose)
