"""Training hyperparameters as frozen dataclasses with YAML override. PROJECT.md 1.2, 1.3.

Same rule as the simulation: no constant lives in logic. A hyperparameter buried in a call to
``RecurrentPPO`` is a constant in logic, and it also cannot be hashed, which makes exit criterion 7
-- every run reproducible from a recorded SHA, config hash and seed -- unenforceable.

Torch-free by design, so the schema can be loaded, hashed and tested on a base ``[dev]`` install.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ..core.config import REPO_ROOT, ConfigError


@dataclass(frozen=True)
class PolicyConfig:
    """The recurrent policy network. PROJECT.md 7 (v1.1).

    Frame stacking is not viable here -- covering a 15 s WARDEN countdown needs 30 frames at a 0.5 s
    decision step, which at 100 dims is a 3,000-dim observation carrying one useful bit -- so the
    memory lives in an LSTM instead.

    Attributes:
        lstm_hidden_size: Units in the LSTM hidden state.
        n_lstm_layers: Stacked LSTM layers.
        net_arch: Hidden widths of the MLP feeding the LSTM, shared by actor and critic.
        shared_lstm: Whether actor and critic read one LSTM. False keeps the value function's
            memory from being shaped by the policy gradient.
        enable_critic_lstm: Whether the critic gets its own recurrence. Required when
            ``shared_lstm`` is False.
    """

    lstm_hidden_size: int = 128
    n_lstm_layers: int = 1
    net_arch: tuple[int, ...] = (128, 128)
    shared_lstm: bool = False
    enable_critic_lstm: bool = True


@dataclass(frozen=True)
class AlgoConfig:
    """PPO's own knobs.

    Attributes:
        n_envs: Parallel environments in the vector env.
        n_steps: Rollout length per environment, so the batch is ``n_envs * n_steps``.
        batch_size: Minibatch size for the update.
        n_epochs: Passes over each rollout.
        learning_rate: Adam step size.
        gamma: Discount. The default of 0.999 is chosen against the episode length, not by habit:
            a night is 1070 decision steps and the ``+10`` for reaching dawn has to survive being
            discounted back to the first one. At 0.99 it arrives worth 3e-5.
        gae_lambda: GAE trace decay.
        clip_range: PPO ratio clip.
        ent_coef: Entropy bonus. Non-zero because the action space is 17-way discrete and the
            reward for exploring the camera system arrives many steps later.
        vf_coef: Value-loss weight.
        max_grad_norm: Gradient clipping.
        device: ``auto``, ``cpu`` or ``cuda``. Recorded in the manifest either way.
    """

    n_envs: int = 8
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "auto"


@dataclass(frozen=True)
class EvalConfig:
    """How a policy is scored.

    Attributes:
        episodes: Episodes per evaluation. 500 is what PROJECT.md 7's v1.1 criteria specify.
        every_steps: Environment steps between periodic evaluations during training.
        curve_episodes: Episodes per periodic evaluation. Smaller than ``episodes``, because this
            one runs many times and only has to show a trend.
        seed_offset: Evaluation seeds are ``seed_offset + i``. Held away from the training seeds so
            that a policy is never scored on a night it was trained on.
    """

    episodes: int = 500
    every_steps: int = 50_000
    curve_episodes: int = 100
    seed_offset: int = 1_000_000


@dataclass(frozen=True)
class StageConfig:
    """One curriculum stage. PROJECT.md 10.

    Attributes:
        name: Stage label, used for the run directory.
        night: Which shipped night, 1-6. Ignored when ``preset`` is set.
        preset: A ``configs/nights/`` stem, overriding ``night``.
        total_steps: Environment steps budgeted for the stage.
        graduation: Mean survival needed to advance, or ``None`` for a report-only stage.
        reference: ``rhythm``'s measured survival on this config, as the bar to beat.
    """

    name: str
    night: int = 5
    preset: str | None = None
    total_steps: int = 2_000_000
    graduation: float | None = None
    reference: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    """A complete training specification.

    Attributes:
        algorithm: ``recurrent_ppo`` or ``ppo``. The non-recurrent variant is the memory ablation.
        seed: The run seed. Everything downstream derives from it.
        oracle: Whether to wrap the environment in :class:`~nightguard.env.wrappers.Oracle`.
        sparse: Whether to run with PROJECT.md 6.3's ``sparse_mode`` reward.
        policy: See :class:`PolicyConfig`.
        algo: See :class:`AlgoConfig`.
        eval: See :class:`EvalConfig`.
        stages: The curriculum, run in order.
    """

    algorithm: str = "recurrent_ppo"
    seed: int = 918442
    oracle: bool = False
    sparse: bool = False
    policy: PolicyConfig = PolicyConfig()
    algo: AlgoConfig = AlgoConfig()
    eval: EvalConfig = EvalConfig()
    stages: tuple[StageConfig, ...] = ()

    def reward_horizon_value(self, steps: int, terminal: float) -> float:
        """What a terminal reward ``steps`` away is worth at the first step, under ``algo.gamma``.

        The discount is not a habit here, it is an arithmetic constraint. A night is 1070 decision
        steps and reaching dawn pays ``+10`` on the last one; at ``gamma=0.99`` that arrives worth
        3e-5 and the policy is effectively optimising the dense terms alone.
        """
        return terminal * self.algo.gamma**steps

    def stage(self, name: str) -> StageConfig:
        """Look a stage up by name."""
        for candidate in self.stages:
            if candidate.name == name:
                return candidate
        known = ", ".join(s.name for s in self.stages)
        raise ConfigError(f"no stage named {name!r}; known stages: {known}")


# --- loading -------------------------------------------------------------------------------------


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Convert one YAML scalar or sequence to the annotated field type.

    ``from __future__ import annotations`` means ``dataclasses.fields`` hands back the annotation as
    a *string*, so this dispatches on the source text rather than on the type object. The schema is
    deliberately narrow -- five shapes, all present in the dataclasses above -- and an unrecognised
    annotation raises rather than falling through to a permissive default.
    """
    name = str(annotation).strip()
    if name == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, found {value!r}")
        return value
    if name == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, found {value!r}")
        return value
    if name == "float":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path}: expected a number, found {value!r}")
        return float(value)
    if name == "str":
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, found {value!r}")
        return value
    if name in {"str | None", "float | None"}:
        if value is None:
            return None
        return _coerce(value, name.split(" | ")[0], path)
    if name == "tuple[int, ...]":
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise ConfigError(f"{path}: expected a sequence, found {value!r}")
        return tuple(_coerce(item, "int", f"{path}[]") for item in value)
    raise ConfigError(f"{path}: unsupported field type {annotation!r}")


def _merge_dataclass(base: Any, data: Mapping[str, Any], path: str) -> Any:
    """Deep-merge a raw mapping onto a frozen dataclass instance, field by field."""
    updates: dict[str, Any] = {}
    known = {field.name: field for field in fields(base)}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"{path}{key}: unknown key")
        field = known[key]
        child = getattr(base, field.name)
        if is_dataclass(child) and not isinstance(child, type):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{path}{key}: expected a mapping")
            updates[key] = _merge_dataclass(child, value, f"{path}{key}.")
        else:
            updates[key] = _coerce(value, field.type, f"{path}{key}")
    return replace(base, **updates)


def _parse_stages(raw: Any) -> tuple[StageConfig, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ConfigError("stages: expected a list")
    stages: list[StageConfig] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping) or "name" not in entry:
            raise ConfigError(f"stages[{index}]: expected a mapping with a 'name'")
        blank = StageConfig(name=str(entry["name"]))
        stages.append(_merge_dataclass(blank, entry, f"stages[{index}]."))
    return tuple(stages)


def train_config_from_mapping(data: Mapping[str, Any]) -> TrainConfig:
    """Build a :class:`TrainConfig` from a raw mapping, deep-merged onto the defaults."""
    rest = {key: value for key, value in data.items() if key != "stages"}
    config: TrainConfig = _merge_dataclass(TrainConfig(), rest, "")
    if "stages" in data:
        config = replace(config, stages=_parse_stages(data["stages"]))
    return config


def load_train_config(path: Path | str, root: Path | None = None) -> TrainConfig:
    """Load a training config from YAML.

    ``path`` may be a filesystem path or a bare name resolved against ``configs/train/``.
    """
    base = REPO_ROOT if root is None else root
    candidate = Path(path)
    if not candidate.suffix:
        candidate = base / "configs" / "train" / f"{candidate}.yaml"
    if not candidate.is_file():
        raise ConfigError(f"no such training config: {candidate}")
    with candidate.open(encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{candidate}: expected a mapping at the top level")
    return train_config_from_mapping(raw)


# --- hashing -------------------------------------------------------------------------------------


def as_dict(config: Any) -> Any:
    """Recursively convert a frozen dataclass tree to plain JSON-serialisable data."""
    if is_dataclass(config) and not isinstance(config, type):
        return {field.name: as_dict(getattr(config, field.name)) for field in fields(config)}
    if isinstance(config, tuple | list):
        return [as_dict(item) for item in config]
    return config


def config_hash(*configs: Any) -> str:
    """A stable short hash over one or more config trees. Exit criterion 7.

    Canonical JSON with sorted keys, so the hash depends on the values and not on field order,
    whitespace or which YAML file the values arrived in. Two runs with the same hash and the same
    seed are the same experiment.
    """
    payload = json.dumps([as_dict(c) for c in configs], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
