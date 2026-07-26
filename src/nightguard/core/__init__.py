"""Pure simulation. PROJECT.md 1.

This layer must never import gymnasium, torch, or any rendering library, and must never contain
global randomness. Observation encoding, reward and action decoding belong in ``env/``.
"""

from .clock import Clock
from .config import (
    AIConfig,
    ConfigError,
    LevelSpec,
    NightConfig,
    PowerConfig,
    UniformChoice,
    config_from_mapping,
    load_config,
    load_night_config,
    with_levels,
)
from .rng import SimRng
from .sim import EpisodeResult, NightSim
from .state import (
    Action,
    DoorSide,
    EntityId,
    OfficeState,
    SimState,
    TerminationCause,
    action_for_camera,
    camera_for_action,
)
from .topology import Node, Topology, TopologyError, load_topology

__all__ = [
    "AIConfig",
    "Action",
    "Clock",
    "ConfigError",
    "DoorSide",
    "EntityId",
    "EpisodeResult",
    "LevelSpec",
    "NightConfig",
    "NightSim",
    "Node",
    "OfficeState",
    "PowerConfig",
    "SimRng",
    "SimState",
    "TerminationCause",
    "Topology",
    "TopologyError",
    "UniformChoice",
    "action_for_camera",
    "camera_for_action",
    "config_from_mapping",
    "load_config",
    "load_night_config",
    "load_topology",
    "with_levels",
]
