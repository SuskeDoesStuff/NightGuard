"""Within-night AI escalation. PROJECT.md 3.3, v0.1 exit criterion 4, and 8.6."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from nightguard.core import EntityId, NightSim, UniformChoice, load_night_config, with_levels
from nightguard.core.config import AIConfig
from nightguard.core.entities import opportunity_succeeds

# PROJECT.md 8.6 asks for all six nights; 3.3 names nights 1 and 6 as the sanity check.
END_OF_NIGHT_LEVELS = {
    1: (0, 3, 2, 2),
    2: (0, 6, 3, 3),
    3: (1, 3, 7, 4),
    5: (3, 8, 9, 7),
    6: (4, 13, 14, 18),
}

EXPECTED_ESCALATION_EVENTS = 3


class TestOpportunityRoll:
    """The 1-20 roll. PROJECT.md 3.3."""

    ai = AIConfig()

    def test_level_zero_can_never_succeed(self) -> None:
        rng = np.random.default_rng(0)
        assert not any(opportunity_succeeds(rng, 0, self.ai) for _ in range(1000))

    def test_level_twenty_always_succeeds(self) -> None:
        rng = np.random.default_rng(1)
        assert all(opportunity_succeeds(rng, 20, self.ai) for _ in range(1000))

    @pytest.mark.parametrize("level", [1, 5, 10, 15, 19])
    def test_each_level_is_worth_five_percentage_points(self, level: int) -> None:
        rng = np.random.default_rng(level)
        trials = 40_000
        rate = sum(opportunity_succeeds(rng, level, self.ai) for _ in range(trials)) / trials
        assert rate == pytest.approx(level / 20, abs=0.01)


@pytest.mark.parametrize("night", sorted(END_OF_NIGHT_LEVELS))
def test_end_of_night_levels(night: int, make_sim: Callable[..., NightSim]) -> None:
    result = make_sim(night=night, seed=99).run()
    assert result.ai_levels == END_OF_NIGHT_LEVELS[night]


def test_night_four_rolls_wardens_starting_level(make_sim: Callable[..., NightSim]) -> None:
    """Night 4's WARDEN starts at 1 or 2, uniformly, rolled at reset. WARDEN never escalates."""
    outcomes = set()
    for seed in range(60):
        result = make_sim(night=4, seed=seed).run()
        assert result.ai_levels[1:] == (5, 6, 8)
        outcomes.add(result.ai_levels[EntityId.WARDEN])
    assert outcomes == {1, 2}


def test_escalation_fires_exactly_once_per_boundary(make_sim: Callable[..., NightSim]) -> None:
    """v0.1 exit criterion 4."""
    sim = make_sim(night=1, seed=5)
    result = sim.run()

    assert result.escalations_applied == EXPECTED_ESCALATION_EVENTS
    escalations = [(tick, name) for tick, name in result.events if name.startswith("escalation")]
    assert [tick for tick, _ in escalations] == [1790, 2680, 3570]
    assert [name for _, name in escalations] == [
        "escalation_hour_2",
        "escalation_hour_3",
        "escalation_hour_4",
    ]


def test_warden_never_escalates_within_a_night(make_sim: Callable[..., NightSim]) -> None:
    for night in sorted(END_OF_NIGHT_LEVELS):
        config = load_night_config(night)
        start = config.ai.levels[EntityId.WARDEN]
        result = make_sim(night=night, seed=1).run()
        assert result.ai_levels[EntityId.WARDEN] == start


def test_levels_are_clamped_to_the_maximum(make_sim: Callable[..., NightSim]) -> None:
    result = make_sim(night=1, seed=1, levels=[20, 20, 20, 20]).run()
    assert result.ai_levels == (20, 20, 20, 20)


def test_configured_levels_are_clamped_at_reset() -> None:
    config = with_levels(load_night_config(1), [99, -5, UniformChoice((25,)), 0])
    sim = NightSim.from_seed(config, seed=0)
    assert sim.state.ai_levels == [20, 0, 20, 0]


def test_rolling_a_level_does_not_disturb_the_entity_streams() -> None:
    """The reset stream is separate, so a rolled level cannot shift DRIFTER's or PROWLER's draws."""
    fixed = with_levels(load_night_config(4), [1, 2, 4, 6])
    rolled = load_night_config(4)

    fixed_sim = NightSim.from_seed(fixed, seed=2024)
    rolled_sim = NightSim.from_seed(rolled, seed=2024)
    fixed_sim.run()
    rolled_sim.run()

    assert fixed_sim.state.drifter.node == rolled_sim.state.drifter.node
    assert fixed_sim.state.prowler.node == rolled_sim.state.prowler.node
