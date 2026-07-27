"""Reward. PROJECT.md 6.3.

::

    +0.01   per decision step survived
    +10.0   reaching 6AM
    -10.0   any death
    -0.002 x (power drained this step - idle drain this step)
    -0.05   on entering blackout, once

The survival term and the power term are already in tension, and that tension *is* the game. An
always-doors-closed policy dies to power depletion; a never-doors-closed policy dies to entities.
The optimum is a peek-then-react loop, and the interesting result is watching the policy discover it
unaided.

**No shaping terms.** Not for camera coverage, not for door discipline, not for anything. A shaping
term that encodes the peek-then-react loop is the result being assumed rather than found.

``sparse_mode`` zeroes the step-survival and power terms and leaves only the terminal signals. That
variant is the actually-hard benchmark and v2.0 reports it alongside the dense one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.config import NightConfig
from ..core.power import drain_per_second, idle_drain_per_second
from ..core.state import SimState, TerminationCause


@dataclass
class RewardTracker:
    """Accumulates the per-step reward. PROJECT.md 6.3.

    Stateful only in that the blackout penalty fires once per episode.
    """

    config: NightConfig
    blackout_charged: bool = False

    def reset(self) -> None:
        """Clear per-episode state."""
        self.blackout_charged = False

    def step_reward(self, state: SimState, power_before: float, terminated: bool) -> float:
        """Reward for one decision step.

        Args:
            state: State after the step.
            power_before: Power at the start of the step, for the drain term.
            terminated: Whether the episode ended on this step.
        """
        settings = self.config.reward
        reward = 0.0

        if not settings.sparse_mode:
            reward += settings.step_survival
            reward += settings.power_penalty_coeff * self._excess_drain(state, power_before)

        if not self.blackout_charged and state.blackout:
            self.blackout_charged = True
            reward += settings.blackout_entry

        if terminated:
            reward += (
                settings.reach_dawn if state.cause is TerminationCause.SURVIVED else settings.death
            )
        return reward

    def _excess_drain(self, state: SimState, power_before: float) -> float:
        """Power spent above what an idle office would have spent over the same step.

        Never negative. During blackout the drain is skipped entirely (§3.13 step 3), so this is
        naturally zero there — which is correct: there is nothing left to spend.
        """
        drained = power_before - state.power_pct
        idle = idle_drain_per_second(self.config.power) * self.config.timing.decision_step_s
        return max(0.0, drained - idle)


def max_return(config: NightConfig) -> float:
    """Approximate best achievable return, for sanity-checking a learning curve.

    PROJECT.md 6.3 quotes ~20.7: 1070 steps of survival plus reaching dawn, with no power spent
    above idle.
    """
    if config.reward.sparse_mode:
        return config.reward.reach_dawn
    steps = round(sum(config.timing.hour_durations_s) / config.timing.decision_step_s)
    return steps * config.reward.step_survival + config.reward.reach_dawn


def drain_for_active(config: NightConfig, active: int) -> float:
    """Drain per decision step at a given active-control count. Used by tests and diagnostics."""
    return drain_per_second(active, config.power) * config.timing.decision_step_s
