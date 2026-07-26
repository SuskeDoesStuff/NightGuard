"""Trace format. PROJECT.md 5, and v0.2 exit criterion 2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from nightguard.core import Action, NightSim, Node, TerminationCause, load_night_config
from nightguard.trace import TRACE_VERSION, TraceWriter, config_hash, write_episode
from tests.conftest import step_until

SEEDS = 40
MIN_DISTINCT_HASHES = 10


def random_script(seed: int, length: int = 1200) -> list[Action]:
    """A reproducible random action script. Raises the monitor, so it is not vacuous."""
    rng = np.random.default_rng(seed)
    return [Action(int(value)) for value in rng.integers(0, len(Action), size=length)]


def trace_for(
    tmp_path: Path, night: int, seed: int, script: Sequence[Action], stride: int | None = None
) -> Path:
    sim = NightSim.from_seed(load_night_config(night), seed=seed)
    return write_episode(
        tmp_path / f"n{night}_s{seed}.jsonl",
        sim,
        night=night,
        actions=script,
        seed=seed,
        stride=stride,
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestShape:
    def test_header_then_ticks_then_footer(self, tmp_path: Path) -> None:
        rows = records(trace_for(tmp_path, 1, 3, random_script(3)))
        assert rows[0]["type"] == "header"
        assert rows[-1]["type"] == "footer"
        assert all(row["type"] == "tick" for row in rows[1:-1])
        assert len(rows) > 2

    def test_header_fields(self, tmp_path: Path) -> None:
        header = records(trace_for(tmp_path, 5, 11, random_script(11)))[0]
        assert header["version"] == TRACE_VERSION
        assert header["night"] == 5
        assert header["seed"] == 11
        assert header["config_hash"].startswith("sha256:")
        assert header["ai_levels_initial"] == [3, 5, 7, 5]
        assert len(header["topology"]) == len(Node)
        assert header["policy"] is None

    def test_tick_record_carries_ground_truth_and_stable_null_keys(self, tmp_path: Path) -> None:
        """PROJECT.md 5: the shape never changes, so env-produced blocks are null, not absent."""
        rows = records(trace_for(tmp_path, 4, 7, random_script(7)))
        tick = rows[1]
        assert set(tick) == {
            "type",
            "t",
            "time_s",
            "hour",
            "power",
            "doors",
            "lights",
            "monitor",
            "entities",
            "audio",
            "belief",
            "action",
            "policy",
            "metrics",
            "event",
        }
        assert set(tick["entities"]) == {"warden", "drifter", "prowler", "sprinter"}
        assert set(tick["audio"]) == {"footstep", "kitchen", "running", "bang"}
        assert tick["belief"] is None
        assert tick["policy"] is None
        assert tick["metrics"]["belief_error"] is None
        assert tick["metrics"]["cam_duty"] is not None

    def test_footer_fields(self, tmp_path: Path) -> None:
        footer = records(trace_for(tmp_path, 1, 3, random_script(3)))[-1]
        assert footer["cause"] in {cause.value for cause in TerminationCause}
        assert footer["return"] is None
        assert 0.0 <= footer["cam_duty_mean"] <= 1.0
        assert footer["final_power"] >= 0.0

    def test_ticks_are_contiguous_and_carry_the_action_only_on_boundaries(
        self, tmp_path: Path
    ) -> None:
        rows = records(trace_for(tmp_path, 1, 3, random_script(3)))
        ticks = [row for row in rows if row["type"] == "tick"]
        assert [row["t"] for row in ticks] == list(range(1, len(ticks) + 1))
        with_action = [row for row in ticks if row["action"] is not None]
        assert all(row["t"] % 5 == 1 for row in with_action)


class TestDeterminism:
    """v0.2 exit criterion 2, with the non-vacuity check PROJECT.md 8.0 requires."""

    def test_the_script_produces_distinct_traces_across_seeds(self, tmp_path: Path) -> None:
        """Run this first: an all-NOOP script would make the reproducibility test below vacuous."""
        hashes = {
            digest(trace_for(tmp_path, 4, seed, random_script(seed))) for seed in range(SEEDS)
        }
        assert len(hashes) >= MIN_DISTINCT_HASHES, f"only {len(hashes)} distinct traces"

    def test_same_seed_and_script_produce_a_byte_identical_trace(self, tmp_path: Path) -> None:
        for seed in (1, 2, 3):
            script = random_script(seed)
            first = trace_for(tmp_path / "a", 4, seed, script)
            second = trace_for(tmp_path / "b", 4, seed, script)
            assert first.read_bytes() == second.read_bytes()

    def test_an_all_noop_script_is_vacuous(self, tmp_path: Path) -> None:
        """Documents *why* the non-vacuity check exists, so nobody removes it later."""
        hashes = {digest(trace_for(tmp_path, 1, seed, [], stride=1)) for seed in range(8)}
        assert len(hashes) > 1, "seeds differ once SPRINTER can kill"
        causes = {records(trace_for(tmp_path, 1, seed, []))[-1]["cause"] for seed in range(8)}
        assert causes <= {"SURVIVED", "KILLED_SPRINTER"}, (
            "with the monitor never raised, only SPRINTER can reach the agent"
        )


class TestStride:
    def test_stride_subsamples_tick_records(self, tmp_path: Path) -> None:
        script = random_script(5)
        full = records(trace_for(tmp_path / "full", 1, 5, script, stride=1))
        strided = records(trace_for(tmp_path / "strided", 1, 5, script, stride=5))

        full_ticks = [row for row in full if row["type"] == "tick"]
        strided_ticks = [row for row in strided if row["type"] == "tick"]
        assert len(strided_ticks) < len(full_ticks)
        assert all(row["t"] % 5 == 0 or row is strided_ticks[-1] for row in strided_ticks)

    def test_the_terminal_tick_is_always_written(self, tmp_path: Path) -> None:
        """PROJECT.md 9.4: a subsampled trace must not break the timeline at the end."""
        script = random_script(9)
        strided = records(trace_for(tmp_path, 1, 9, script, stride=7))
        assert strided[-2]["t"] == strided[-1]["terminated_at"]

    def test_stride_must_be_positive(self, tmp_path: Path) -> None:
        sim = NightSim.from_seed(load_night_config(1), seed=0)
        with pytest.raises(ValueError):
            TraceWriter(tmp_path / "bad.jsonl", sim, night=1, stride=0)


class TestConfigHash:
    def test_identical_configs_hash_identically(self) -> None:
        assert config_hash(load_night_config(3)) == config_hash(load_night_config(3))

    def test_different_nights_hash_differently(self) -> None:
        """Nights 5 and 6 share a power divisor but differ in AI levels, so all six differ."""
        hashes = {config_hash(load_night_config(night)) for night in range(1, 7)}
        assert len(hashes) == 6

    def test_every_night_config_is_distinguishable(self) -> None:
        pairs = [(a, b) for a in range(1, 7) for b in range(1, 7) if a < b]
        for a, b in pairs:
            assert config_hash(load_night_config(a)) != config_hash(load_night_config(b))


def test_events_reach_the_trace(tmp_path: Path, make_sim: Callable[..., NightSim]) -> None:
    sim = make_sim(night=1, seed=1, levels=[0, 20, 0, 0], escalation=())
    sim.state.drifter.node = Node.W_CORNER
    with TraceWriter(tmp_path / "events.jsonl", sim, night=1, seed=1):
        step_until(
            sim, lambda s: s.state.drifter.in_office, max_steps=12, action=Action.SELECT_CAM_0
        )
        sim.step(Action.MONITOR_DOWN)
    names = {row.get("event") for row in records(tmp_path / "events.jsonl")}
    # invasion_drifter and door_jam_left land on the same tick; EVENT_PRIORITY keeps the invasion.
    assert "invasion_drifter" in names
    assert "death_drifter" in names
    assert "door_jam_left" not in names


def test_writer_does_not_mutate_the_simulation(tmp_path: Path) -> None:
    """PROJECT.md 1: trace never mutates state."""
    script = random_script(21)
    plain = NightSim.from_seed(load_night_config(4), seed=21).run(script)
    traced_sim = NightSim.from_seed(load_night_config(4), seed=21)
    write_episode(tmp_path / "t.jsonl", traced_sim, night=4, actions=script, seed=21)
    assert traced_sim.result() == plain
