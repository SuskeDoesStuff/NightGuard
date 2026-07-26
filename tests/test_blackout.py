"""The three-phase blackout. PROJECT.md 3.11 and 8.7."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import pytest

from nightguard import derivations
from nightguard.core import Action, NightConfig, NightSim, TerminationCause, load_night_config
from nightguard.core.blackout import BlackoutPhase, apply_onset
from tests.conftest import force_blackout_at

# PROJECT.md 8.7: power forced to 0 at t = 500 s of a 535 s night.
ONSET_TICK = 5000
BUDGET_S = 35.0
EPISODES = 10_000
FAST_EPISODES = 1200
SIGMA_TOLERANCE = 4.0


def derived_survival(budget_s: float = BUDGET_S) -> float:
    """The analytic target. Derived in CHANGELOG from PROJECT.md 3.11; never measured."""
    return derivations.blackout_survival(NightConfig(), budget_s)


class TestOnset:
    """PROJECT.md 3.11's onset rules."""

    def test_onset_opens_doors_kills_lights_and_forces_the_monitor_down(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        sim = make_sim(night=1, seed=0, only=())
        sim.state.office.door_left = True
        sim.state.office.door_right = True
        sim.state.office.jam_left = True
        sim.state.office.monitor_up = True
        sim.state.office.light_left = True

        apply_onset(sim.state)

        assert not sim.state.office.door_left
        assert not sim.state.office.door_right
        assert not sim.state.office.light_left
        assert not sim.state.office.monitor_up
        assert sim.state.blackout
        assert sim.state.office.jam_left, "jams are irrelevant, not cleared"

    def test_all_actions_become_no_ops(self, make_sim: Callable[..., NightSim]) -> None:
        sim = make_sim(night=1, seed=0, only=())
        apply_onset(sim.state)
        for action in (Action.TOGGLE_DOOR_LEFT, Action.SELECT_CAM_3, Action.FLASH_LIGHT_RIGHT):
            sim.step(action)
            assert not sim.state.office.door_left
            assert not sim.state.office.monitor_up
            assert not sim.state.office.light_right

    def test_phase_one_starts_from_the_onset_tick(self, make_sim: Callable[..., NightSim]) -> None:
        sim = make_sim(night=1, seed=0, only=())
        for _ in range(40):
            sim.step(Action.NOOP)
        apply_onset(sim.state)
        assert sim.state.blackout_state is not None
        assert sim.state.blackout_state.phase_started_tick == sim.state.tick
        assert sim.state.blackout_state.phase is BlackoutPhase.APPROACH


class TestSequence:
    """The roll schedule settled in PROJECT.md 3.11 and 10."""

    def test_a_phase_cannot_complete_before_one_interval(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        """The first roll lands one interval in, so nothing resolves at t=0 of a phase."""
        interval_ticks = 0
        for seed in range(30):
            sim = make_sim(night=1, seed=seed, only=())
            apply_onset(sim.state)
            interval_ticks = sim.blackout.interval_ticks(BlackoutPhase.APPROACH)
            for _ in range(interval_ticks // sim.clock.ticks_per_decision_step - 1):
                sim.step(Action.NOOP)
            assert sim.state.blackout_state is not None
            assert sim.state.blackout_state.phase is BlackoutPhase.APPROACH
        assert interval_ticks == 50  # 5.0 s

    def test_a_capped_phase_always_completes_by_its_cap(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        """The 20 s guarantee replaces the roll at 20 s rather than following one."""
        for seed in range(40):
            sim = make_sim(night=1, seed=seed, only=())
            apply_onset(sim.state)
            cap = sim.blackout.cap_ticks(BlackoutPhase.APPROACH)
            assert cap is not None
            for _ in range(cap // sim.clock.ticks_per_decision_step):
                if sim.state.terminated:
                    break
                sim.step(Action.NOOP)
            assert sim.state.blackout_state is not None
            assert sim.state.blackout_state.phase is not BlackoutPhase.APPROACH

    def test_the_kill_phase_is_reachable_and_uncapped(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        phases = Counter()
        for seed in range(60):
            sim = force_blackout_at(make_sim(night=1, seed=seed, only=()), ONSET_TICK)
            sim.run()
            assert sim.state.blackout_state is not None
            phases[sim.state.blackout_state.phase] += 1
        assert phases[BlackoutPhase.KILL] > 0
        assert set(phases) <= {BlackoutPhase.SONG, BlackoutPhase.KILL, BlackoutPhase.APPROACH}


class TestSurvivability:
    """PROJECT.md 8.7, replacing "strictly between 0 and 1" with an agreement test."""

    def _measure(self, make_sim: Callable[..., NightSim], episodes: int) -> tuple[float, Counter]:
        causes: Counter = Counter()
        for seed in range(episodes):
            sim = force_blackout_at(make_sim(night=1, seed=seed, only=()), ONSET_TICK)
            sim.run()
            causes[sim.state.cause] += 1
        return causes[TerminationCause.SURVIVED] / episodes, causes

    def test_blackout_is_both_survivable_and_lethal(
        self, make_sim: Callable[..., NightSim]
    ) -> None:
        """The non-vacuity check (8.0): both outcomes must occur before agreement means anything."""
        _, causes = self._measure(make_sim, 200)
        assert causes[TerminationCause.SURVIVED] > 0
        assert causes[TerminationCause.KILLED_BLACKOUT] > 0
        assert set(causes) == {TerminationCause.SURVIVED, TerminationCause.KILLED_BLACKOUT}

    def test_survival_agrees_with_the_derivation(self, make_sim: Callable[..., NightSim]) -> None:
        measured, _ = self._measure(make_sim, FAST_EPISODES)
        derived = derived_survival()
        sigma = (derived * (1 - derived) / FAST_EPISODES) ** 0.5
        deviation = abs(measured - derived) / sigma
        assert deviation <= SIGMA_TOLERANCE, (
            f"measured {measured:.4f}, derived {derived:.4f}, {deviation:.2f} sigma "
            f"at n={FAST_EPISODES}"
        )

    @pytest.mark.slow
    def test_survival_agrees_at_full_sample_size(self, make_sim: Callable[..., NightSim]) -> None:
        """8.7's stated 10,000 runs."""
        measured, _ = self._measure(make_sim, EPISODES)
        derived = derived_survival()
        sigma = (derived * (1 - derived) / EPISODES) ** 0.5
        deviation = abs(measured - derived) / sigma
        assert deviation <= SIGMA_TOLERANCE, (
            f"measured {measured:.4f}, derived {derived:.4f}, {deviation:.2f} sigma at n={EPISODES}"
        )


def test_reaching_dawn_mid_sequence_is_survived(make_sim: Callable[..., NightSim]) -> None:
    """PROJECT.md 3.11: blackout is not a separate outcome, only being killed during it is."""
    sim = force_blackout_at(make_sim(night=1, seed=3, only=()), sim_tick=5340)
    result = sim.run()
    assert result.cause is TerminationCause.SURVIVED
    assert sim.state.blackout


def test_phase_mass_matches_the_derivation() -> None:
    """The exact completion mass recorded in PROJECT.md 3.11."""
    mass = derivations.blackout_phase_mass(NightConfig(), "approach")
    assert {seconds: float(weight) for seconds, weight in mass.items()} == pytest.approx(
        {5: 0.2, 10: 0.16, 15: 0.128, 20: 0.512}
    )


def test_the_derived_value_is_sensitive_to_the_budget() -> None:
    """8.7's assertion is sharp: a 5 s change in budget moves it far beyond sampling error."""
    config = load_night_config(1)
    curve = {b: derivations.blackout_survival(config, b) for b in (20, 25, 30, 35, 40, 45)}
    assert curve[35] == pytest.approx(0.6148, abs=5e-4)
    assert all(a > b for a, b in zip(list(curve.values()), list(curve.values())[1:], strict=False))
