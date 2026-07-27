"""``RecurrentPPO`` over NightGuard. PROJECT.md 7, v1.1.

The environment is a memory task by construction: entity positions are hidden, the audio channel is
a single-tick cue, and the only way to know where anything is is to have looked and remembered.
Frame stacking cannot cover that -- a 15 s WARDEN countdown is 30 frames at a 0.5 s decision step --
so recurrence is the intended approach and an LSTM carries the belief.

This module builds the model and the vector env, and logs the two things the milestone is measuring:
survival on held-out seeds, and camera duty cycle. It deliberately does **no** reward shaping and no
hyperparameter search; the prompt's working discipline is one configuration until a curve rises.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from ..core.config import NightConfig, load_night_config, load_preset
from ..core.topology import Topology, load_topology
from ..env.nightguard_env import NightGuardEnv
from ..env.wrappers import Oracle
from .config import AlgoConfig, EvalConfig, PolicyConfig, StageConfig, TrainConfig
from .evaluate import EvalResult, eval_seeds, evaluate_model

ALGORITHMS = ("recurrent_ppo", "ppo")


def stage_config(stage: StageConfig, sparse: bool) -> NightConfig:
    """Resolve a stage to a simulation config, applying the sparse-reward ablation."""
    config = load_preset(stage.preset) if stage.preset else load_night_config(stage.night)
    if sparse:
        config = replace(config, reward=replace(config.reward, sparse_mode=True))
    return config


def make_env(
    config: NightConfig,
    oracle: bool,
    topology: Topology | None = None,
    monitor_path: Path | None = None,
) -> NightGuardEnv | Any:
    """One environment, wrapped per the run's ablation settings."""
    env: Any = NightGuardEnv(config=config, topology=topology)
    if oracle:
        env = Oracle(env)
    if monitor_path is not None:
        env = Monitor(env, filename=str(monitor_path))
    return env


def make_vec_env(
    config: NightConfig,
    train: TrainConfig,
    topology: Topology | None = None,
) -> VecEnv:
    """The training vector env.

    ``DummyVecEnv`` rather than ``SubprocVecEnv``: the simulator runs at tens of thousands of steps
    per second per core, so process boundaries would cost more in serialisation than they recover.
    Whether that holds is measured rather than assumed -- see ``scripts/profile_training.py``.
    """
    shared = topology if topology is not None else load_topology(config.topology_path())
    venv: VecEnv = DummyVecEnv(
        [(lambda: make_env(config, train.oracle, shared)) for _ in range(train.algo.n_envs)]
    )
    if train.algo.normalize_observations or train.algo.normalize_reward:
        venv = VecNormalize(
            venv,
            norm_obs=train.algo.normalize_observations,
            norm_reward=train.algo.normalize_reward,
            gamma=train.algo.gamma,
        )
    return venv


def build_model(
    venv: VecEnv,
    train: TrainConfig,
    tensorboard_log: str | None = None,
) -> BaseAlgorithm:
    """Construct the algorithm named by ``train.algorithm``."""
    if train.algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {train.algorithm!r}; expected one of {ALGORITHMS}")
    algo: AlgoConfig = train.algo
    policy: PolicyConfig = train.policy
    # A linear decay to zero across the stage. SB3 passes remaining progress in [1, 0].
    rate: float | Callable[[float], float] = algo.learning_rate
    if algo.anneal_learning_rate:
        base_rate = algo.learning_rate

        def rate(progress: float) -> float:
            """SB3 passes remaining progress in ``[1, 0]``."""
            return base_rate * progress

    common: dict[str, Any] = {
        "target_kl": algo.target_kl,
        "learning_rate": rate,
        "n_steps": algo.n_steps,
        "batch_size": algo.batch_size,
        "n_epochs": algo.n_epochs,
        "gamma": algo.gamma,
        "gae_lambda": algo.gae_lambda,
        "clip_range": algo.clip_range,
        "ent_coef": algo.ent_coef,
        "vf_coef": algo.vf_coef,
        "max_grad_norm": algo.max_grad_norm,
        "seed": train.seed,
        "device": algo.device,
        "verbose": 0,
        "tensorboard_log": tensorboard_log,
    }
    if train.algorithm == "ppo":
        return PPO(
            "MlpPolicy",
            venv,
            policy_kwargs={"net_arch": list(policy.net_arch)},
            **common,
        )
    return RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        policy_kwargs={
            "net_arch": list(policy.net_arch),
            "lstm_hidden_size": policy.lstm_hidden_size,
            "n_lstm_layers": policy.n_lstm_layers,
            "shared_lstm": policy.shared_lstm,
            "enable_critic_lstm": policy.enable_critic_lstm,
        },
        **common,
    )


@dataclass
class CurvePoint:
    """One periodic evaluation during training. The learning curve is a list of these."""

    timesteps: int
    survival: float
    mean_return: float
    mean_steps: float
    camera_duty_cycle: float


class EvaluationCallback(BaseCallback):
    """Evaluate on held-out seeds every ``every_steps``, and log the curve.

    Survival is the headline, but on a night where nothing reaches dawn early in training it is a
    flat zero, so ``mean_steps`` is logged beside it as the signal that actually moves. Camera duty
    cycle is logged every time because PROJECT.md 7's criterion 6 asks for the trend either way --
    an agent that beats `rhythm` while peeking *less* is the interesting result; one that beats it by
    peeking more found a different and duller strategy.
    """

    def __init__(
        self,
        config: NightConfig,
        train: TrainConfig,
        topology: Topology | None,
        csv_path: Path,
    ) -> None:
        super().__init__(verbose=0)
        self.eval_config: EvalConfig = train.eval
        self.env = make_env(config, train.oracle, topology)
        self.seeds = eval_seeds(train.eval.curve_episodes, train.eval.seed_offset)
        self.csv_path = csv_path
        self.curve: list[CurvePoint] = []
        self.best: CurvePoint | None = None
        self.best_path = csv_path.parent / "best"
        self._next_at = train.eval.every_steps
        self._write_header()

    def _write_header(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ["timesteps", "survival", "mean_return", "mean_steps", "camera_duty_cycle"]
            )

    def _record(self, result: EvalResult) -> None:
        point = CurvePoint(
            timesteps=int(self.num_timesteps),
            survival=result.survival,
            mean_return=result.mean_return,
            mean_steps=result.mean_steps,
            camera_duty_cycle=result.camera_duty_cycle,
        )
        self.curve.append(point)
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    point.timesteps,
                    f"{point.survival:.4f}",
                    f"{point.mean_return:.4f}",
                    f"{point.mean_steps:.2f}",
                    f"{point.camera_duty_cycle:.4f}",
                ]
            )
        self.logger.record("eval/survival", point.survival)
        self.logger.record("eval/mean_steps", point.mean_steps)
        self.logger.record("eval/camera_duty_cycle", point.camera_duty_cycle)
        self._keep_if_best(point)

    def _keep_if_best(self, point: CurvePoint) -> None:
        """Checkpoint the best policy seen, not just the last one.

        v1.1 needed this the hard way. The `normalised` arm reached 0.1200 survival at 700,000
        steps and then collapsed; the stage saved its 2,000,000-step weights, which scored 0.0000,
        and the good policy was unrecoverable. On a run this unstable the final weights are an
        arbitrary sample of the oscillation, so survival is ranked first and ``mean_steps`` breaks
        ties -- the latter being the only thing that moves while survival is pinned at zero.
        """
        if self.model is None:
            return
        better = self.best is None or (point.survival, point.mean_steps) > (
            self.best.survival,
            self.best.mean_steps,
        )
        if better:
            self.best = point
            self.model.save(str(self.best_path))

    def evaluate_now(self) -> EvalResult:
        """Run one evaluation and append it to the curve."""
        assert self.model is not None
        result = evaluate_model(self.env, self.model, self.seeds)
        self._record(result)
        return result

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_at:
            self._next_at += self.eval_config.every_steps
            self.evaluate_now()
        return True


def final_evaluation(
    model: BaseAlgorithm,
    config: NightConfig,
    train: TrainConfig,
    topology: Topology | None = None,
) -> EvalResult:
    """Score the finished policy over the full ``eval.episodes`` held-out set."""
    env = make_env(config, train.oracle, topology)
    seeds = eval_seeds(train.eval.episodes, train.eval.seed_offset)
    return evaluate_model(env, model, seeds)
