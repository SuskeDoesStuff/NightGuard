"""Reference policies and the survival assertions of PROJECT.md 8.2.

Sample sizes here are smaller than 8.2's 10,000 so the suite stays fast; `scripts/validate.py`
runs the full-size versions. Every threshold records its measured value, per 8.0.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import pytest

from nightguard.core import (
    Action,
    EntityId,
    NightConfig,
    NightSim,
    Node,
    TerminationCause,
    load_night_config,
    load_topology,
)
from nightguard.policies import DoNothing, MonitorDown, Rhythm, run_policy
from tests.conftest import step_until, with_only

# PROJECT.md 8.2's night-1 figure, derived in CHANGELOG from 3.1, 3.3 and 3.7 before SPRINTER was
# written. Do not adjust this to match a measurement; if they disagree, that is the finding.
# No longer provisional: SPRINTER exists from v0.2, so this assertion can fail in both directions.
DERIVED_NIGHT_1_SURVIVAL = 0.2397

EPISODES = 250
NIGHT_1_EPISODES = 2000
SIGMA_TOLERANCE = 4.0


@pytest.fixture(scope="module")
def topology():
    return load_topology(NightConfig().topology_path())


def survival(night: int, factory, episodes: int, topology) -> float:
    """Fraction of ``episodes`` seeded nights the policy survives."""
    config = load_night_config(night)
    survived = 0
    for seed in range(episodes):
        sim = NightSim.from_seed(config, seed=seed, topology=topology)
        run_policy(sim, factory())
        survived += sim.state.cause is TerminationCause.SURVIVED
    return survived / episodes


def binomial_sigma(p: float, n: int) -> float:
    return (p * (1.0 - p) / n) ** 0.5


@pytest.mark.slow
class TestDoNothing:
    """8.2's night-1 agreement test, which can fail in both directions."""

    def test_night_one_matches_the_analytic_derivation(self, topology) -> None:
        measured = survival(1, DoNothing, NIGHT_1_EPISODES, topology)
        sigma = binomial_sigma(DERIVED_NIGHT_1_SURVIVAL, NIGHT_1_EPISODES)
        deviation = abs(measured - DERIVED_NIGHT_1_SURVIVAL) / sigma
        assert deviation <= SIGMA_TOLERANCE, (
            f"measured {measured:.4f}, derived {DERIVED_NIGHT_1_SURVIVAL:.4f}, "
            f"{deviation:.1f} sigma at n={NIGHT_1_EPISODES}"
        )

    def test_survival_is_non_increasing_across_nights(self, topology) -> None:
        """Ties are permitted: nights 2 to 6 are all near zero by design."""
        rates = [survival(night, DoNothing, EPISODES, topology) for night in range(1, 7)]
        assert all(a >= b for a, b in pairwise(rates)), rates
        assert rates[1] <= 0.05, rates

    def test_sprinter_is_the_only_kill_path(self, topology) -> None:
        """Under the monitor gates, nothing else can reach a policy that never looks up."""
        causes = set()
        for seed in range(200):
            sim = NightSim.from_seed(load_night_config(3), seed=seed, topology=topology)
            run_policy(sim, DoNothing())
            causes.add(sim.state.cause)
        assert causes <= {TerminationCause.SURVIVED, TerminationCause.KILLED_SPRINTER}, causes


@pytest.mark.slow
class TestRhythm:
    """8.2: rhythm must beat do_nothing on nights 3 through 6."""

    @pytest.mark.parametrize("night", [3, 4, 5, 6])
    def test_rhythm_beats_do_nothing(self, night: int, topology) -> None:
        rhythm = survival(night, Rhythm, EPISODES, topology)
        nothing = survival(night, DoNothing, EPISODES, topology)
        assert rhythm > nothing, f"night {night}: rhythm {rhythm:.3f} vs do_nothing {nothing:.3f}"

    def test_rhythm_survives_the_easy_nights(self, topology) -> None:
        for night in (1, 2):
            rate = survival(night, Rhythm, 200, topology)
            assert rate >= 0.9, f"night {night}: {rate:.3f}"


@pytest.mark.slow
class TestMonitorDownProbe:
    """v0.2 exit criterion 7. A reported finding, not a pass/fail gate."""

    def test_the_probe_is_strong_early_and_power_limited_late(self, topology) -> None:
        early = survival(2, MonitorDown, 200, topology)
        late = survival(6, MonitorDown, 200, topology)
        assert early > 0.5, f"night 2: {early:.3f}"
        assert late == 0.0, f"night 6: {late:.3f}"

    def test_the_probe_dies_to_blackout_not_to_entities(self, topology) -> None:
        """The escalating bang drain is what stops it, which is the mechanic doing its job."""
        causes = set()
        for seed in range(120):
            sim = NightSim.from_seed(load_night_config(5), seed=seed, topology=topology)
            run_policy(sim, MonitorDown())
            causes.add(sim.state.cause)
        assert TerminationCause.KILLED_BLACKOUT in causes
        assert TerminationCause.KILLED_WARDEN not in causes
        assert TerminationCause.KILLED_DRIFTER not in causes
        assert TerminationCause.KILLED_PROWLER not in causes


def test_all_four_entities_can_kill(topology) -> None:
    """v0.2 exit criterion 1: every entity can produce a kill, verified by seeded scenarios.

    Each entity is isolated and placed in the state its own section describes as immediately
    dangerous, then driven to its kill. A policy sweep is the wrong vehicle: the tuned `rhythm`
    survives most seeds, so an absent kill would prove nothing, and with the full roster enabled a
    faster entity reaches the agent first.
    """
    base = load_night_config(6)
    seen: set[TerminationCause] = set()

    def isolated(entity: EntityId, levels: list[int]) -> NightSim:
        config = replace(base, ai=replace(base.ai, levels=tuple(levels), escalation=()))
        return NightSim.from_seed(with_only(config, [entity]), seed=0, topology=topology)

    # WARDEN: in the office with the monitor down, 25% per second (3.4).
    sim = isolated(EntityId.WARDEN, [20, 0, 0, 0])
    sim.state.warden.node = Node.OFFICE
    sim.state.warden.path_index = len(sim.warden.path) - 1
    step_until(sim, lambda s: s.state.terminated, max_steps=200)
    seen.add(sim.state.cause)

    # DRIFTER and PROWLER: enter while the monitor is up, killed the next time it comes down.
    for entity, levels, name in (
        (EntityId.DRIFTER, [0, 20, 0, 0], "drifter"),
        (EntityId.PROWLER, [0, 0, 20, 0], "prowler"),
    ):
        sim = isolated(entity, levels)
        getattr(sim.state, name).node = getattr(sim, name).corner
        entered = step_until(
            sim,
            lambda s, n=name: getattr(s.state, n).in_office,
            max_steps=30,
            action=Action.SELECT_CAM_0,
        )
        assert entered, f"{name} never entered the office"
        sim.step(Action.MONITOR_DOWN)
        seen.add(sim.state.cause)

    # SPRINTER: armed, forced attack after 25 s, left door open (3.7).
    sim = isolated(EntityId.SPRINTER, [0, 0, 0, 20])
    sim.state.sprinter.stage = sim.config.entities.sprinter.stages_to_arm
    sim.state.sprinter.armed_at_tick = sim.state.tick
    step_until(sim, lambda s: s.state.terminated, max_steps=200)
    seen.add(sim.state.cause)

    wanted = {
        TerminationCause.KILLED_WARDEN,
        TerminationCause.KILLED_DRIFTER,
        TerminationCause.KILLED_PROWLER,
        TerminationCause.KILLED_SPRINTER,
    }
    assert wanted <= seen, f"never observed {sorted(c.value for c in wanted - seen)}"
