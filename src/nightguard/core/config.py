"""Frozen configuration dataclasses and YAML loaders. PROJECT.md 4.

Every numeric constant in the simulation lives here with a default, and every default is overridable
from YAML (PROJECT.md 1.2, 1.3). Night files carry only what they change; the loader deep-merges them
onto these defaults, so ``configs/nights/custom_template.yaml`` documents the full schema in one place
without any file having to restate it.

Sections that v0.1 does not simulate (``warden``, ``sprinter``, ``blackout``, ``flags``, ``reward``)
are still parsed into dataclasses, so that v0.2 and v1.0 extend the schema rather than reshape it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

NIGHT_MIN = 1
NIGHT_MAX = 6


class ConfigError(ValueError):
    """Raised when a configuration file cannot be parsed into the schema."""


@dataclass(frozen=True)
class TimingConfig:
    """The three clocks of PROJECT.md 3.1.

    Attributes:
        sim_tick_s: Length of one simulation tick, in seconds.
        decision_step_s: Length of one ``env.step()``, in seconds.
        hour_durations_s: Length of each in-game hour, in seconds. Hour 0 is longer.
        time_units_per_second: Integer units per second of the exact scheduling grid, via
            ``units = round(seconds * time_units_per_second)``. Entity intervals and WARDEN's
            countdown are not multiples of the sim tick, so scheduling them in floats
            accumulates drift. 300 = lcm(60, 100) is the coarsest grid on which every timing
            constant in PROJECT.md 3 and 4 is an exact integer: the countdown table divides by
            60, the opportunity intervals are two-decimal. This is a resolution, not a frame
            rate, and nothing may treat it as one.
        conversion_tolerance: Numerical-hygiene epsilon for that conversion. Not a game constant.
    """

    sim_tick_s: float = 0.1
    decision_step_s: float = 0.5
    hour_durations_s: tuple[float, ...] = (90.0, 89.0, 89.0, 89.0, 89.0, 89.0)
    time_units_per_second: int = 300
    conversion_tolerance: float = 1e-9


@dataclass(frozen=True)
class TopologyConfig:
    """Where to find the node metadata file."""

    file: str = "configs/topology/default.yaml"


@dataclass(frozen=True)
class WardenConfig:
    """WARDEN. PROJECT.md 3.4. Not simulated in v0.1.

    The ``countdown_divisor`` of 60 reflects the original's frame counter and exists only to derive
    the seconds table in 3.4. It is not a frame rate and must not be treated as one elsewhere.
    """

    enabled: bool = True
    interval_s: float = 3.02
    path: tuple[str, ...] = (
        "STAGE",
        "COMMONS",
        "E_RESTROOMS",
        "E_KITCHEN",
        "E_HALL",
        "E_CORNER",
        "OFFICE",
    )
    countdown_numerator: float = 1000.0
    countdown_per_level: float = 100.0
    countdown_divisor: float = 60.0
    office_kill_prob_per_s: float = 0.25
    office_kill_interval_s: float = 1.0
    stage_lock: bool = True
    door: str = "right"


@dataclass(frozen=True)
class DrifterConfig:
    """DRIFTER: uniform teleport within a pool. PROJECT.md 3.5."""

    enabled: bool = True
    interval_s: float = 4.97
    mode: str = "pool"
    pool: tuple[str, ...] = ("COMMONS", "W_BACKSTAGE", "W_CLOSET", "W_HALL", "W_CORNER")
    start: str = "STAGE"
    corner: str = "W_CORNER"
    door: str = "left"
    retreat_to: str = "COMMONS"


@dataclass(frozen=True)
class ProwlerConfig:
    """PROWLER: uniform step to an adjacent node in a chain. PROJECT.md 3.6."""

    enabled: bool = True
    interval_s: float = 4.98
    mode: str = "chain"
    chain: tuple[str, ...] = ("COMMONS", "E_RESTROOMS", "E_KITCHEN", "E_HALL", "E_CORNER")
    start: str = "STAGE"
    corner: str = "E_CORNER"
    door: str = "right"
    retreat_to: str = "COMMONS"


@dataclass(frozen=True)
class SprinterConfig:
    """SPRINTER. PROJECT.md 3.7. Not simulated in v0.1."""

    enabled: bool = True
    interval_s: float = 5.01
    stages_to_arm: int = 3
    immunity_range_s: tuple[float, float] = (0.83, 16.67)
    forced_attack_after_s: float = 25.0
    grace_period_s: float = 0.5
    bang_base_pct: float = 1.0
    bang_increment_pct: float = 5.0
    reset_stage_choices: tuple[int, ...] = (0, 1)
    door: str = "left"


@dataclass(frozen=True)
class EntitiesConfig:
    """All four entity configurations."""

    warden: WardenConfig = WardenConfig()
    drifter: DrifterConfig = DrifterConfig()
    prowler: ProwlerConfig = ProwlerConfig()
    sprinter: SprinterConfig = SprinterConfig()


@dataclass(frozen=True)
class UniformChoice:
    """An AI level rolled uniformly at reset, e.g. night 4's WARDEN. PROJECT.md 3.3."""

    values: tuple[int, ...]


LevelSpec = int | UniformChoice


@dataclass(frozen=True)
class EscalationEvent:
    """A within-night AI escalation fired once when an hour boundary is crossed."""

    hour: int
    entities: tuple[str, ...]
    delta: int


@dataclass(frozen=True)
class AIConfig:
    """AI levels and within-night escalation. PROJECT.md 3.3.

    ``levels`` is ordered [WARDEN, DRIFTER, PROWLER, SPRINTER] throughout the project.
    """

    levels: tuple[LevelSpec, ...] = (0, 0, 0, 0)
    level_min: int = 0
    level_max: int = 20
    escalation: tuple[EscalationEvent, ...] = (
        EscalationEvent(hour=2, entities=("DRIFTER",), delta=1),
        EscalationEvent(hour=3, entities=("DRIFTER", "PROWLER", "SPRINTER"), delta=1),
        EscalationEvent(hour=4, entities=("DRIFTER", "PROWLER", "SPRINTER"), delta=1),
    )


@dataclass(frozen=True)
class PowerConfig:
    """The drain model. PROJECT.md 3.10.

    ``active`` is the count of true office controls clamped to [``active_min``, ``active_max``]. The
    floor of 1 is why an entirely idle office still drains, and why exactly one control active costs
    the same as none. Sources disagree on that last point; see CHANGELOG for the resolution.

    ``night_constant_numerator`` is the 0.1 in ``night_constant = 0.1 / D``, lifted out of logic per
    PROJECT.md 1.3.
    """

    start_pct: float = 99.0
    active_min: int = 1
    active_max: int = 4
    divisor: float = 10.0
    night_constant_numerator: float = 0.1
    night_divisor: float = 9.6


@dataclass(frozen=True)
class OfficeConfig:
    """Office rules. PROJECT.md 3.2 and 3.5."""

    invasion_kill_timeout_s: float = 30.0
    door_jam_on_invasion: bool = True
    light_momentary: bool = True


@dataclass(frozen=True)
class BlackoutPhaseConfig:
    """One phase of the three-phase blackout sequence. PROJECT.md 3.11."""

    interval_s: float
    prob: float
    max_s: float | None = None


@dataclass(frozen=True)
class BlackoutConfig:
    """Blackout. PROJECT.md 3.11. v0.2 still uses a placeholder that terminates immediately.

    There is deliberately no ``enabled`` flag. Blackout is unconditional: power reaching 0 always
    enters the absorbing state. A disable switch left the power model — the subsystem everything
    else is validated against — with an undefined corner where power went negative and nothing
    happened.
    """

    approach: BlackoutPhaseConfig = BlackoutPhaseConfig(interval_s=5.0, prob=0.2, max_s=20.0)
    song: BlackoutPhaseConfig = BlackoutPhaseConfig(interval_s=5.0, prob=0.2, max_s=20.0)
    kill: BlackoutPhaseConfig = BlackoutPhaseConfig(interval_s=2.0, prob=0.2, max_s=None)


@dataclass(frozen=True)
class AudioConfig:
    """The audio channel. PROJECT.md 3.9.

    ``footstep_nodes`` are the nodes near enough to the office for a completed move to be heard.
    The ``kitchen`` signal is not configured here: it is derived from the topology as "an entity
    is at a node with no video feed", so the blind node and its audio compensation cannot drift
    apart.
    """

    footstep_nodes: tuple[str, ...] = ("W_HALL", "W_CORNER", "E_HALL", "E_CORNER")


@dataclass(frozen=True)
class TraceConfig:
    """Trace emission. PROJECT.md 5.

    ``cam_duty_window_steps`` is 5's "fraction of the last 100 decision steps with the monitor
    up"; ``stride`` subsamples tick records for long batch runs, where 5 sim ticks per record is
    one record per decision step.
    """

    cam_duty_window_steps: int = 100
    stride: int = 1


@dataclass(frozen=True)
class FlagsConfig:
    """Ablation flags. PROJECT.md 3.8 and 4."""

    rare_noise_enabled: bool = False
    timing_jitter_enabled: bool = False
    timing_jitter_pct: float = 0.0


@dataclass(frozen=True)
class RewardConfig:
    """Reward terms. PROJECT.md 6.3. Consumed by env/, not by core/."""

    step_survival: float = 0.01
    reach_dawn: float = 10.0
    death: float = -10.0
    power_penalty_coeff: float = -0.002
    blackout_entry: float = -0.05
    sparse_mode: bool = False


@dataclass(frozen=True)
class NightConfig:
    """A complete simulation configuration."""

    timing: TimingConfig = TimingConfig()
    topology: TopologyConfig = TopologyConfig()
    entities: EntitiesConfig = EntitiesConfig()
    ai: AIConfig = AIConfig()
    power: PowerConfig = PowerConfig()
    office: OfficeConfig = OfficeConfig()
    blackout: BlackoutConfig = BlackoutConfig()
    audio: AudioConfig = AudioConfig()
    trace: TraceConfig = TraceConfig()
    flags: FlagsConfig = FlagsConfig()
    reward: RewardConfig = RewardConfig()

    def topology_path(self, root: Path | None = None) -> Path:
        """Resolve the topology file against the repository root."""
        base = REPO_ROOT if root is None else root
        return base / self.topology.file


# --- parsing helpers -------------------------------------------------------------------------


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key!r} must be a mapping, found {type(value).__name__}")
    return value


def _float(data: Mapping[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key!r} must be a number, found {value!r}")
    return float(value)


def _int(data: Mapping[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key!r} must be an integer, found {value!r}")
    return int(value)


def _bool(data: Mapping[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key!r} must be a boolean, found {value!r}")
    return bool(value)


def _str(data: Mapping[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key!r} must be a string, found {value!r}")
    return value


def _sequence(data: Mapping[str, Any], key: str) -> Sequence[Any] | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(f"{key!r} must be a list, found {value!r}")
    return value


def _str_tuple(data: Mapping[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _sequence(data, key)
    if raw is None:
        return default
    return tuple(str(item) for item in raw)


def _float_tuple(
    data: Mapping[str, Any], key: str, default: tuple[float, ...]
) -> tuple[float, ...]:
    raw = _sequence(data, key)
    if raw is None:
        return default
    return tuple(float(item) for item in raw)


def _int_tuple(data: Mapping[str, Any], key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = _sequence(data, key)
    if raw is None:
        return default
    return tuple(int(item) for item in raw)


def _pair(data: Mapping[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = _sequence(data, key)
    if raw is None:
        return default
    if len(raw) != 2:
        raise ConfigError(f"{key!r} must have exactly two entries")
    return (float(raw[0]), float(raw[1]))


def _parse_level(value: Any) -> LevelSpec:
    if isinstance(value, bool):
        raise ConfigError(f"AI level must be an integer or a choice, found {value!r}")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, Mapping) and "uniform_choice" in value:
        choices = value["uniform_choice"]
        if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
            raise ConfigError("'uniform_choice' must be a non-empty list")
        return UniformChoice(values=tuple(int(item) for item in choices))
    raise ConfigError(f"AI level must be an integer or {{uniform_choice: [...]}}, found {value!r}")


# --- section parsers -------------------------------------------------------------------------


def _parse_timing(data: Mapping[str, Any], base: TimingConfig) -> TimingConfig:
    return TimingConfig(
        sim_tick_s=_float(data, "sim_tick_s", base.sim_tick_s),
        decision_step_s=_float(data, "decision_step_s", base.decision_step_s),
        hour_durations_s=_float_tuple(data, "hour_durations_s", base.hour_durations_s),
        time_units_per_second=_int(data, "time_units_per_second", base.time_units_per_second),
        conversion_tolerance=_float(data, "conversion_tolerance", base.conversion_tolerance),
    )


def _parse_warden(data: Mapping[str, Any], base: WardenConfig) -> WardenConfig:
    return WardenConfig(
        enabled=_bool(data, "enabled", base.enabled),
        interval_s=_float(data, "interval_s", base.interval_s),
        path=_str_tuple(data, "path", base.path),
        countdown_numerator=_float(data, "countdown_numerator", base.countdown_numerator),
        countdown_per_level=_float(data, "countdown_per_level", base.countdown_per_level),
        countdown_divisor=_float(data, "countdown_divisor", base.countdown_divisor),
        office_kill_prob_per_s=_float(data, "office_kill_prob_per_s", base.office_kill_prob_per_s),
        office_kill_interval_s=_float(data, "office_kill_interval_s", base.office_kill_interval_s),
        stage_lock=_bool(data, "stage_lock", base.stage_lock),
        door=_str(data, "door", base.door),
    )


def _parse_drifter(data: Mapping[str, Any], base: DrifterConfig) -> DrifterConfig:
    return DrifterConfig(
        enabled=_bool(data, "enabled", base.enabled),
        interval_s=_float(data, "interval_s", base.interval_s),
        mode=_str(data, "mode", base.mode),
        pool=_str_tuple(data, "pool", base.pool),
        start=_str(data, "start", base.start),
        corner=_str(data, "corner", base.corner),
        door=_str(data, "door", base.door),
        retreat_to=_str(data, "retreat_to", base.retreat_to),
    )


def _parse_prowler(data: Mapping[str, Any], base: ProwlerConfig) -> ProwlerConfig:
    return ProwlerConfig(
        enabled=_bool(data, "enabled", base.enabled),
        interval_s=_float(data, "interval_s", base.interval_s),
        mode=_str(data, "mode", base.mode),
        chain=_str_tuple(data, "chain", base.chain),
        start=_str(data, "start", base.start),
        corner=_str(data, "corner", base.corner),
        door=_str(data, "door", base.door),
        retreat_to=_str(data, "retreat_to", base.retreat_to),
    )


def _parse_sprinter(data: Mapping[str, Any], base: SprinterConfig) -> SprinterConfig:
    return SprinterConfig(
        enabled=_bool(data, "enabled", base.enabled),
        interval_s=_float(data, "interval_s", base.interval_s),
        stages_to_arm=_int(data, "stages_to_arm", base.stages_to_arm),
        immunity_range_s=_pair(data, "immunity_range_s", base.immunity_range_s),
        forced_attack_after_s=_float(data, "forced_attack_after_s", base.forced_attack_after_s),
        grace_period_s=_float(data, "grace_period_s", base.grace_period_s),
        bang_base_pct=_float(data, "bang_base_pct", base.bang_base_pct),
        bang_increment_pct=_float(data, "bang_increment_pct", base.bang_increment_pct),
        reset_stage_choices=_int_tuple(data, "reset_stage_choices", base.reset_stage_choices),
        door=_str(data, "door", base.door),
    )


def _parse_entities(data: Mapping[str, Any], base: EntitiesConfig) -> EntitiesConfig:
    return EntitiesConfig(
        warden=_parse_warden(_mapping(data, "warden"), base.warden),
        drifter=_parse_drifter(_mapping(data, "drifter"), base.drifter),
        prowler=_parse_prowler(_mapping(data, "prowler"), base.prowler),
        sprinter=_parse_sprinter(_mapping(data, "sprinter"), base.sprinter),
    )


def _parse_ai(data: Mapping[str, Any], base: AIConfig) -> AIConfig:
    raw_levels = _sequence(data, "levels")
    levels = base.levels if raw_levels is None else tuple(_parse_level(v) for v in raw_levels)

    raw_escalation = _sequence(data, "escalation")
    if raw_escalation is None:
        escalation = base.escalation
    else:
        events: list[EscalationEvent] = []
        for entry in raw_escalation:
            if not isinstance(entry, Mapping):
                raise ConfigError(f"escalation entry must be a mapping, found {entry!r}")
            events.append(
                EscalationEvent(
                    hour=_int(entry, "hour", 0),
                    entities=_str_tuple(entry, "entities", ()),
                    delta=_int(entry, "delta", 1),
                )
            )
        escalation = tuple(events)

    return AIConfig(
        levels=levels,
        level_min=_int(data, "level_min", base.level_min),
        level_max=_int(data, "level_max", base.level_max),
        escalation=escalation,
    )


def _parse_power(data: Mapping[str, Any], base: PowerConfig) -> PowerConfig:
    return PowerConfig(
        start_pct=_float(data, "start_pct", base.start_pct),
        active_min=_int(data, "active_min", base.active_min),
        active_max=_int(data, "active_max", base.active_max),
        divisor=_float(data, "divisor", base.divisor),
        night_constant_numerator=_float(
            data, "night_constant_numerator", base.night_constant_numerator
        ),
        night_divisor=_float(data, "night_divisor", base.night_divisor),
    )


def _parse_office(data: Mapping[str, Any], base: OfficeConfig) -> OfficeConfig:
    return OfficeConfig(
        invasion_kill_timeout_s=_float(
            data, "invasion_kill_timeout_s", base.invasion_kill_timeout_s
        ),
        door_jam_on_invasion=_bool(data, "door_jam_on_invasion", base.door_jam_on_invasion),
        light_momentary=_bool(data, "light_momentary", base.light_momentary),
    )


def _parse_phase(data: Mapping[str, Any], base: BlackoutPhaseConfig) -> BlackoutPhaseConfig:
    max_s = base.max_s if "max_s" not in data else _float(data, "max_s", 0.0)
    return BlackoutPhaseConfig(
        interval_s=_float(data, "interval_s", base.interval_s),
        prob=_float(data, "prob", base.prob),
        max_s=max_s,
    )


def _parse_blackout(data: Mapping[str, Any], base: BlackoutConfig) -> BlackoutConfig:
    if "enabled" in data:
        raise ConfigError(
            "blackout.enabled was removed in v0.2: blackout is unconditional. See PROJECT.md 4."
        )
    return BlackoutConfig(
        approach=_parse_phase(_mapping(data, "approach"), base.approach),
        song=_parse_phase(_mapping(data, "song"), base.song),
        kill=_parse_phase(_mapping(data, "kill"), base.kill),
    )


def _parse_flags(data: Mapping[str, Any], base: FlagsConfig) -> FlagsConfig:
    return FlagsConfig(
        rare_noise_enabled=_bool(data, "rare_noise_enabled", base.rare_noise_enabled),
        timing_jitter_enabled=_bool(data, "timing_jitter_enabled", base.timing_jitter_enabled),
        timing_jitter_pct=_float(data, "timing_jitter_pct", base.timing_jitter_pct),
    )


def _parse_reward(data: Mapping[str, Any], base: RewardConfig) -> RewardConfig:
    return RewardConfig(
        step_survival=_float(data, "step_survival", base.step_survival),
        reach_dawn=_float(data, "reach_dawn", base.reach_dawn),
        death=_float(data, "death", base.death),
        power_penalty_coeff=_float(data, "power_penalty_coeff", base.power_penalty_coeff),
        blackout_entry=_float(data, "blackout_entry", base.blackout_entry),
        sparse_mode=_bool(data, "sparse_mode", base.sparse_mode),
    )


# --- public loaders --------------------------------------------------------------------------


def config_from_mapping(data: Mapping[str, Any], base: NightConfig | None = None) -> NightConfig:
    """Deep-merge a raw config mapping onto ``base`` (the dataclass defaults if omitted)."""
    current = NightConfig() if base is None else base
    return NightConfig(
        timing=_parse_timing(_mapping(data, "timing"), current.timing),
        topology=TopologyConfig(
            file=_str(_mapping(data, "topology"), "file", current.topology.file)
        ),
        entities=_parse_entities(_mapping(data, "entities"), current.entities),
        ai=_parse_ai(_mapping(data, "ai"), current.ai),
        power=_parse_power(_mapping(data, "power"), current.power),
        office=_parse_office(_mapping(data, "office"), current.office),
        blackout=_parse_blackout(_mapping(data, "blackout"), current.blackout),
        audio=AudioConfig(
            footstep_nodes=_str_tuple(
                _mapping(data, "audio"), "footstep_nodes", current.audio.footstep_nodes
            )
        ),
        trace=TraceConfig(
            cam_duty_window_steps=_int(
                _mapping(data, "trace"),
                "cam_duty_window_steps",
                current.trace.cam_duty_window_steps,
            ),
            stride=_int(_mapping(data, "trace"), "stride", current.trace.stride),
        ),
        flags=_parse_flags(_mapping(data, "flags"), current.flags),
        reward=_parse_reward(_mapping(data, "reward"), current.reward),
    )


def load_config(path: Path, base: NightConfig | None = None) -> NightConfig:
    """Load a YAML config file, deep-merged onto ``base``."""
    with path.open(encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return config_from_mapping(raw, base)


def load_night_config(night: int, root: Path | None = None) -> NightConfig:
    """Load the preset for ``night`` (1-6). PROJECT.md 3.3 and 3.10."""
    if not NIGHT_MIN <= night <= NIGHT_MAX:
        raise ConfigError(f"night must be in [{NIGHT_MIN}, {NIGHT_MAX}], found {night}")
    base = REPO_ROOT if root is None else root
    return load_config(base / "configs" / "nights" / f"night{night}.yaml")


def with_levels(config: NightConfig, levels: Sequence[LevelSpec]) -> NightConfig:
    """Return a copy of ``config`` with AI levels replaced. Useful for targeted tests."""
    return replace(config, ai=replace(config.ai, levels=tuple(levels)))
