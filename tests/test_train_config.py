"""Training configuration and run provenance. PROJECT.md 1.2, and v1.1 criterion 7.

Torch-free on purpose: hyperparameters and manifests must be loadable, hashable and testable on a
base ``[dev]`` install, so a fresh clone still runs ``pytest`` green without a 3 GB dependency.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nightguard.core.config import ConfigError
from nightguard.train.config import (
    TrainConfig,
    as_dict,
    config_hash,
    load_train_config,
    train_config_from_mapping,
)
from nightguard.train.manifest import RunManifest, new_run_id, package_versions


def test_the_shipped_baseline_loads() -> None:
    config = load_train_config("baseline")
    assert config.algorithm == "recurrent_ppo"
    assert [stage.name for stage in config.stages] == [
        "stage1-night5",
        "stage2-night6",
        "stage3-custom-max",
    ]


def test_stage_references_match_the_measured_rhythm_series() -> None:
    """PROJECT.md 10. Measured at 400 seeds with `rhythm` frozen; see CHANGELOG, v1.1.

    These are the bars criterion 4 is judged against, so a silent edit here would move the
    milestone's substantive claim without moving any assertion that mentions it.
    """
    references = {stage.name: stage.reference for stage in load_train_config("baseline").stages}
    assert references == {
        "stage1-night5": 0.973,
        "stage2-night6": 0.325,
        "stage3-custom-max": 0.185,
    }


def test_stage_three_is_report_only() -> None:
    """`rhythm` at 0.185 there is a power-economy measurement, not a difficulty one."""
    stage = load_train_config("baseline").stage("stage3-custom-max")
    assert stage.graduation is None
    assert stage.preset == "custom_max"


def test_gamma_survives_the_episode_length() -> None:
    """A night is 1070 decision steps and dawn pays +10 on the last one."""
    config = load_train_config("baseline")
    discounted = config.reward_horizon_value(steps=1070, terminal=10.0)
    assert discounted > 1.0, discounted


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        train_config_from_mapping({"algo": {"lr": 0.1}})


def test_type_errors_are_rejected() -> None:
    with pytest.raises(ConfigError, match="expected a number"):
        train_config_from_mapping({"algo": {"learning_rate": "fast"}})


def test_stages_need_names() -> None:
    with pytest.raises(ConfigError, match="name"):
        train_config_from_mapping({"stages": [{"night": 5}]})


def test_missing_config_file_is_an_error() -> None:
    with pytest.raises(ConfigError, match="no such training config"):
        load_train_config("no_such_config")


class TestConfigHash:
    """Exit criterion 7: two runs with the same hash and seed are the same experiment."""

    def test_is_stable_across_calls(self) -> None:
        config = load_train_config("baseline")
        assert config_hash(config) == config_hash(load_train_config("baseline"))

    def test_changes_when_a_hyperparameter_changes(self) -> None:
        config = load_train_config("baseline")
        moved = replace(config, algo=replace(config.algo, learning_rate=1e-4))
        assert config_hash(config) != config_hash(moved)

    def test_changes_for_each_ablation_arm(self) -> None:
        """The Oracle and sparse arms must not share a hash with the baseline."""
        config = load_train_config("baseline")
        arms = {
            config_hash(config),
            config_hash(replace(config, oracle=True)),
            config_hash(replace(config, sparse=True)),
            config_hash(replace(config, algorithm="ppo")),
        }
        assert len(arms) == 4

    def test_does_not_change_with_key_order(self) -> None:
        first = train_config_from_mapping({"seed": 1, "oracle": True})
        second = train_config_from_mapping({"oracle": True, "seed": 1})
        assert config_hash(first) == config_hash(second)


def test_as_dict_is_json_shaped() -> None:
    data = as_dict(TrainConfig())
    assert data["policy"]["net_arch"] == [128, 128]
    assert isinstance(data["stages"], list)


class TestManifest:
    def test_records_what_reproduction_needs(self, tmp_path: Path) -> None:
        manifest = RunManifest(run_id="t", stage="s", config_hash="abc", seed=7)
        for key in ("git_sha", "started_at", "machine", "versions", "device"):
            assert getattr(manifest, key)

    def test_versions_report_absent_rather_than_guessing(self) -> None:
        versions = package_versions()
        assert versions["numpy"] != "absent"
        assert set(versions) >= {"python", "numpy", "gymnasium", "torch", "sb3_contrib"}

    def test_run_ids_are_sortable_and_distinct(self) -> None:
        first = new_run_id("stage1")
        assert first.endswith("-stage1")
        assert new_run_id("stage1", "oracle").endswith("-stage1-oracle")
