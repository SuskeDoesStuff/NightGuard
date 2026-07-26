"""Determinism. PROJECT.md 7 v0.1 exit criterion 1, and 8.5.

v0.1 has no trace format, so the byte-identical-trace assertion of 8.5 is approximated here with the
full ground-truth state signature at every decision step. The trace-level version arrives in v0.2.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from nightguard.core import Action, NightConfig, NightSim, Topology, load_night_config

EPISODES = 100
SCRIPT_LENGTH = 400
SEEDS = 40
MIN_DISTINCT_OUTCOMES = 10


def random_script(seed: int, length: int = SCRIPT_LENGTH) -> list[Action]:
    """A reproducible random action script, generated outside the simulation's own streams."""
    rng = np.random.default_rng(seed)
    return [Action(int(value)) for value in rng.integers(0, len(Action), size=length)]


def signatures(
    config: NightConfig, seed: int, script: Sequence[Action], topology: Topology
) -> list[tuple[object, ...]]:
    """Every decision step's ground-truth signature, from reset to termination."""
    sim = NightSim.from_seed(config, seed=seed, topology=topology)
    trace = [sim.state.signature()]
    for action in script:
        if sim.state.terminated:
            break
        sim.step(action)
        trace.append(sim.state.signature())
    return trace


def test_same_seed_and_script_reproduce_the_episode(topology: Topology) -> None:
    """100 episodes with random seeds and random action scripts, re-run identically. PROJECT.md 8.5."""
    config = load_night_config(4)
    for episode in range(EPISODES):
        seed = 1_000 + episode
        script = random_script(seed)
        assert signatures(config, seed, script, topology) == signatures(
            config, seed, script, topology
        )


def test_from_seed_twice_is_identical(topology: Topology) -> None:
    """PROJECT.md 7 v1.0 criterion 5 in embryo: reset(seed=k) twice, same trajectory."""
    config = load_night_config(6)
    script = random_script(77)
    first = NightSim.from_seed(config, seed=77, topology=topology).run(script)
    second = NightSim.from_seed(config, seed=77, topology=topology).run(script)
    assert first == second


def test_an_injected_generator_is_equivalent_to_a_seed(topology: Topology) -> None:
    config = load_night_config(5)
    script = random_script(5)
    from_seed = NightSim.from_seed(config, seed=31337, topology=topology).run(script)
    injected = NightSim(config, np.random.default_rng(31337), topology=topology).run(script)
    assert from_seed == injected


def test_different_seeds_diverge(topology: Topology) -> None:
    """PROJECT.md 8.0's non-vacuity check for the reproducibility test above.

    Keyed on the full ground-truth signature rather than the tick count alone, and held to a
    threshold comparable to `test_trace.py`'s: a test satisfied by two distinct outcomes in thirty
    is nearly as vacuous as the thing 8.0 exists to prevent.
    """
    config = load_night_config(6)
    script = random_script(9)
    outcomes = {
        NightSim.from_seed(config, seed=seed, topology=topology).run(script).ticks
        for seed in range(SEEDS)
    }
    signatures = set()
    for seed in range(SEEDS):
        sim = NightSim.from_seed(config, seed=seed, topology=topology)
        sim.run(script)
        signatures.add(sim.state.signature())

    assert len(signatures) >= MIN_DISTINCT_OUTCOMES, (
        f"only {len(signatures)} distinct final states over {SEEDS} seeds"
    )
    assert len(outcomes) > 1, f"only {len(outcomes)} distinct end ticks"


def test_padding_is_deterministic_and_terminates(topology: Topology) -> None:
    """A short script still plays the night out, and does so reproducibly."""
    config = load_night_config(1)
    first = NightSim.from_seed(config, seed=8, topology=topology).run([Action.NOOP])
    second = NightSim.from_seed(config, seed=8, topology=topology).run([Action.NOOP])
    assert first == second
    assert first.ticks > 0


def test_result_is_unavailable_before_termination(topology: Topology) -> None:
    sim = NightSim.from_seed(load_night_config(1), seed=1, topology=topology)
    with pytest.raises(RuntimeError):
        sim.result()
