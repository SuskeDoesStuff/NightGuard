"""The training pipeline runs end to end. PROJECT.md 7 (v1.1).

Marked ``slow`` and skipped without ``[train]`` installed: this is the one test that builds a torch
model. It asserts that the pipeline *runs and records*, not that it learns -- a few hundred steps
cannot show learning, and a test that claimed otherwise would be measuring noise.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nightguard.core.config import load_night_config
from nightguard.policies import Rhythm
from nightguard.train.config import load_train_config
from nightguard.train.evaluate import eval_seeds, evaluate_scripted

pytest.importorskip("sb3_contrib", reason="requires the [train] extra")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tiny_config():  # type: ignore[no-untyped-def]
    """A configuration small enough to run in a test, and otherwise the shipped baseline."""
    config = load_train_config("baseline")
    config = replace(
        config,
        algo=replace(config.algo, n_envs=2, n_steps=32, batch_size=32, n_epochs=1, device="cpu"),
        eval=replace(config.eval, episodes=3, curve_episodes=2, every_steps=64),
        stages=(replace(config.stage("stage1-night5"), total_steps=128, night=1),),
    )
    return config


def test_learn_and_evaluate_runs(tiny_config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from nightguard.train.curriculum import run_stage

    result = run_stage(tiny_config.stages[0], tiny_config, tmp_path)
    assert result.timesteps >= 128
    assert result.final.episodes == 3
    assert 0.0 <= result.final.survival <= 1.0
    assert 0.0 <= result.final.camera_duty_cycle <= 1.0
    assert (tmp_path / result.stage / "curve.csv").exists()
    assert Path(result.model_path).exists()
    assert result.curve, "no periodic evaluation was recorded"


def test_oracle_arm_builds_with_the_wider_observation(tiny_config) -> None:  # type: ignore[no-untyped-def]
    """The matched Oracle run must reach the model, not fail at the space check."""
    from nightguard.train.recurrent_ppo import build_model, make_vec_env, stage_config

    config = replace(tiny_config, oracle=True)
    night = stage_config(config.stages[0], sparse=False)
    venv = make_vec_env(night, config)
    try:
        assert venv.observation_space.shape == (104,)
        build_model(venv, config)
    finally:
        venv.close()


def test_a_learned_policy_is_scored_by_the_same_path_as_a_scripted_one(tiny_config) -> None:  # type: ignore[no-untyped-def]
    """Criterion 4 is a comparison, so both arms must run through one evaluation function."""
    from nightguard.train.evaluate import evaluate_model
    from nightguard.train.recurrent_ppo import build_model, make_env, make_vec_env, stage_config

    night = stage_config(tiny_config.stages[0], sparse=False)
    venv = make_vec_env(night, tiny_config)
    try:
        model = build_model(venv, tiny_config)
    finally:
        venv.close()

    seeds = eval_seeds(2, tiny_config.eval.seed_offset)
    learned = evaluate_model(make_env(night, oracle=False), model, seeds)
    scripted = evaluate_scripted(night, Rhythm, seeds)
    assert learned.episodes == scripted.episodes == 2
    assert set(learned.causes) <= {
        "SURVIVED",
        "KILLED_WARDEN",
        "KILLED_DRIFTER",
        "KILLED_PROWLER",
        "KILLED_SPRINTER",
        "KILLED_BLACKOUT",
    }


def test_evaluation_seeds_are_held_out_and_identical_across_policies() -> None:
    """Two policies must be scored on the same nights, or the comparison is between night sets."""
    config = load_train_config("baseline")
    seeds = eval_seeds(config.eval.episodes, config.eval.seed_offset)
    assert seeds[0] == 1_000_000
    assert len(set(seeds)) == config.eval.episodes


def test_scripted_evaluation_agrees_with_the_core_harness() -> None:
    """The Gymnasium path and `run_policy` must give the same number, or comparisons are unsound."""
    from nightguard.core import NightSim, TerminationCause, load_topology
    from nightguard.policies import run_policy

    config = load_night_config(6)
    topology = load_topology(config.topology_path())
    seeds = list(range(60))

    survived = 0
    for seed in seeds:
        sim = NightSim.from_seed(config, seed=seed, topology=topology)
        run_policy(sim, Rhythm())
        survived += sim.state.cause is TerminationCause.SURVIVED

    through_env = evaluate_scripted(config, Rhythm, seeds, topology)
    assert np.isclose(through_env.survival, survived / len(seeds))
