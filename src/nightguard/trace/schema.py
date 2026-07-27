"""Trace record construction. PROJECT.md 5.

The trace ships in v0.2 and **never changes shape afterwards**: it is the seam that decouples the
viewer from everything else, and the primary debugging surface during fidelity work. Keys are
therefore always present, even when the producing layer does not exist yet.

Two blocks are produced by ``env/``, which does not exist until v1.0, and are emitted as ``null``
here with their keys in place:

* ``belief`` — the agent's last-observed node per entity, and how long ago.
* ``policy`` — action probabilities and the value estimate. Optional by 5's own wording, and
  omitted entirely for scripted policies.

``metrics`` is half-computable: ``belief_error`` needs ``belief`` and is ``null``, while
``cam_duty`` is pure core state and is emitted from v0.2.

Ground truth is written even though the agent never sees it. That is by design — the trace is a
debugging artifact, not an observation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from ..core.config import NightConfig
from ..core.sim import NightSim
from ..core.state import Action, SimState
from ..core.topology import Topology

TRACE_VERSION = "1.1"

# PROJECT.md 5's `event` field holds a single value, but several events genuinely co-occur on one
# tick — an invasion always jams its door on the same tick, and a death often follows the event
# that caused it. Co-occurring events are resolved by this priority, most significant first, so the
# viewer's scrubber marks the most informative one. The complete list stays in `SimState.events`.
#
# `door_jam_*` is therefore never emitted on its own: it is always coincident with an invasion, and
# a jammed door is already visible in the office panel.
EVENT_PRIORITY: tuple[str, ...] = (
    "death_",
    "blackout",
    "invasion_",
    "sprinter_attack",
    "bang",
    "sprinter_armed",
    "warden_retreat",
    "warden_countdown_start",
    "door_jam_",
    "escalation_",
)


def event_rank(name: str) -> int:
    """Priority index of an event name; unlisted events sort last."""
    for index, prefix in enumerate(EVENT_PRIORITY):
        if name.startswith(prefix):
            return index
    return len(EVENT_PRIORITY)


# Power and time are rounded for output only. The simulation keeps full precision; this keeps the
# file readable and byte-stable without changing any decision the simulator made.
POWER_DECIMALS = 6
TIME_DECIMALS = 6


def config_hash(config: NightConfig) -> str:
    """A stable ``sha256:`` digest of the full configuration.

    Hashes a canonical JSON dump of the config dataclasses with sorted keys, so two runs of the
    same config hash identically regardless of how it was assembled — file, defaults or overrides.
    """
    payload = json.dumps(_plain(config), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    """Convert dataclasses, tuples and enums into JSON-serialisable primitives."""
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def header_record(
    sim: NightSim,
    night: int,
    seed: int | None,
    topology: Topology,
    policy: str | None = None,
) -> dict[str, Any]:
    """The single header record. PROJECT.md 5."""
    return {
        "type": "header",
        "version": TRACE_VERSION,
        "night": night,
        "seed": seed,
        "config_hash": config_hash(sim.config),
        "topology": [
            {
                "id": int(spec.node),
                "name": spec.node.name,
                "wing": spec.wing,
                "selectable": spec.selectable,
                "has_video": spec.has_video,
            }
            for spec in topology.nodes
        ],
        "ai_levels_initial": list(sim.state.ai_levels),
        "policy": policy,
    }


def tick_record(
    sim: NightSim,
    state: SimState,
    action: Action | None,
    cam_duty: float,
    event: str | None,
) -> dict[str, Any]:
    """One record per sim tick. PROJECT.md 5."""
    office = state.office
    warden = state.warden
    sprinter = state.sprinter
    countdown_ticks = (
        None if warden.countdown_units is None else sim.clock.units_to_ticks(warden.countdown_units)
    )
    return {
        "type": "tick",
        "t": state.tick,
        "time_s": round(sim.clock.time_s(state.tick), TIME_DECIMALS),
        "hour": sim.clock.hour_at_tick(state.tick),
        "power": round(state.power_pct, POWER_DECIMALS),
        "doors": [office.door_left, office.door_right],
        # Added in trace 1.1. 9.1 requires a jammed door to read JAMMED rather than open, and the
        # blackout phase to be visible; the 1.0 shape carried neither, and blackout did not exist
        # when that shape was frozen. See PROJECT.md 5 and 10.
        "jams": [office.jam_left, office.jam_right],
        "blackout": (None if state.blackout_state is None else state.blackout_state.phase.name),
        "lights": [office.light_left, office.light_right],
        "monitor": {
            "up": office.monitor_up,
            "cam": int(office.selected_camera) if office.monitor_up else None,
        },
        "entities": {
            "warden": {
                "node": int(warden.node),
                "countdown_ticks": countdown_ticks,
                "in_office": warden.in_office,
            },
            "drifter": {
                "node": int(state.drifter.node),
                "at_door": state.drifter.node == sim.drifter.corner,
                "in_office": state.drifter.in_office,
            },
            "prowler": {
                "node": int(state.prowler.node),
                "at_door": state.prowler.node == sim.prowler.corner,
                "in_office": state.prowler.in_office,
            },
            "sprinter": {
                "stage": sprinter.stage,
                "immune_until": sprinter.immune_until_tick,
                "bangs": sprinter.bang_count,
                "armed": sprinter.armed,
            },
        },
        "audio": {
            "footstep": state.audio.footstep,
            "kitchen": state.audio.kitchen,
            "running": state.audio.running,
            "bang": state.audio.bang,
        },
        "belief": None,
        "action": None if action is None else int(action),
        "policy": None,
        "metrics": {"belief_error": None, "cam_duty": round(cam_duty, POWER_DECIMALS)},
        "event": event,
    }


def footer_record(
    sim: NightSim,
    state: SimState,
    cam_duty_mean: float,
) -> dict[str, Any]:
    """The single footer record. PROJECT.md 5.

    ``return`` is ``null``: reward is an ``env/`` concept and does not exist until v1.0.
    """
    return {
        "type": "footer",
        "terminated_at": state.tick,
        "cause": None if state.cause is None else state.cause.value,
        "final_power": round(state.power_pct, POWER_DECIMALS),
        "return": None,
        "cam_duty_mean": round(cam_duty_mean, POWER_DECIMALS),
    }
