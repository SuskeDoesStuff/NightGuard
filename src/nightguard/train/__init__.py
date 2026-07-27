"""Training. PROJECT.md 1.1, 7 (v1.1).

The only layer allowed to import torch. Everything here is optional: the package installs without
``[train]``, and nothing under ``core/``, ``env/``, ``trace/`` or ``policies/`` imports this.

``config`` and ``manifest`` are torch-free on purpose, so that reproducibility metadata and
hyperparameter schemas can be read, hashed and tested without a 3 GB dependency present.
"""

from .config import (
    AlgoConfig,
    EvalConfig,
    PolicyConfig,
    StageConfig,
    TrainConfig,
    config_hash,
    load_train_config,
)
from .evaluate import EvalResult
from .manifest import RunManifest

__all__ = [
    "AlgoConfig",
    "EvalConfig",
    "EvalResult",
    "PolicyConfig",
    "RunManifest",
    "StageConfig",
    "TrainConfig",
    "config_hash",
    "load_train_config",
]
