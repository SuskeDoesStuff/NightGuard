"""The Gymnasium environment. PROJECT.md 6, and the v1.0 exit criteria."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from nightguard.core import (
    Action,
    EntityId,
    NightSim,
    Node,
    TerminationCause,
    load_night_config,
)
from nightguard.env import NightGuardEnv
from nightguard.env import obs as obs_mod
from nightguard.env.wrappers import AudioMask, FrameStack, Oracle, PreviousAction

NIGHTS = range(1, 7)
LEAK_SEEDS = 12


@pytest.fixture
def env() -> NightGuardEnv:
    return NightGuardEnv(night=4)


def rollout(env: NightGuardEnv, seed: int, steps: int | None = None) -> list[np.ndarray]:
    """Drive a random episode, returning every observation."""
    observation, _ = env.reset(seed=seed)
    seen = [observation]
    rng = np.random.default_rng(seed)
    while True:
        observation, _, terminated, truncated, _ = env.step(int(rng.integers(17)))
        seen.append(observation)
        if terminated or truncated or (steps is not None and len(seen) > steps):
            return seen


# --- criterion 1 --------------------------------------------------------------------------------


def test_check_env_passes_without_warnings() -> None:
    """v1.0 exit criterion 1."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(NightGuardEnv(night=4), skip_render_check=True)
    assert not caught, [str(w.message) for w in caught]


# --- criterion 3: the no-leak test ---------------------------------------------------------------


def hide_everything(env: NightGuardEnv) -> None:
    """Put the office in a state where nothing is observable: monitor down, both lights off."""
    office = env.sim.state.office
    office.monitor_up = False
    office.light_left = False
    office.light_right = False


def encode_now(env: NightGuardEnv) -> np.ndarray:
    """Re-encode the current state without advancing it."""
    return obs_mod.encode(env.sim, env.sim.state, env._belief, Action.NOOP)


def set_position(env: NightGuardEnv, entity: EntityId, value: int) -> None:
    """Move an entity, or set SPRINTER's stage."""
    state = env.sim.state
    if entity is EntityId.SPRINTER:
        state.sprinter.stage = value
    elif entity is EntityId.WARDEN:
        state.warden.node = Node(value)
    else:
        state.entity(entity).node = Node(value)


class TestNoPositionLeak:
    """v1.0 exit criterion 3 — the most important test in the milestone.

    A leak here produces a policy that learns beautifully and means nothing, and the learning curve
    looks *better*, not worse, so nothing downstream would flag it.
    """

    @pytest.mark.parametrize("entity", list(EntityId))
    @pytest.mark.parametrize("seed", range(4))
    def test_moving_a_hidden_entity_does_not_change_the_observation(
        self, entity: EntityId, seed: int
    ) -> None:
        env = NightGuardEnv(night=6)
        env.reset(seed=seed)
        for _ in range(20):
            env.step(Action.NOOP)
        hide_everything(env)

        baseline = encode_now(env)
        values = range(4) if entity is EntityId.SPRINTER else [int(n) for n in Node]
        for value in values:
            set_position(env, entity, value)
            assert np.array_equal(encode_now(env), baseline), (
                f"{entity.name} at {value} changed the observation while hidden"
            )

    @pytest.mark.parametrize("entity", [EntityId.DRIFTER, EntityId.PROWLER])
    def test_no_leak_while_an_entity_is_in_the_office(self, entity: EntityId) -> None:
        """In-office occupancy is hidden state too: the player sees a jammed door, not the entity."""
        env = NightGuardEnv(night=6)
        env.reset(seed=1)
        state = env.sim.state
        state.entity(entity).node = Node.OFFICE
        state.entity(entity).invaded_at_tick = state.tick
        hide_everything(env)

        baseline = encode_now(env)
        for value in [int(n) for n in Node]:
            set_position(env, entity, value)
            assert np.array_equal(encode_now(env), baseline)

    def test_no_leak_with_doors_jammed(self) -> None:
        """The jam bits report the door, not who jammed it or where they are now."""
        env = NightGuardEnv(night=6)
        env.reset(seed=2)
        env.sim.state.office.jam_left = True
        env.sim.state.office.jam_right = True
        hide_everything(env)

        baseline = encode_now(env)
        for entity in EntityId:
            values = range(4) if entity is EntityId.SPRINTER else [int(n) for n in Node]
            for value in values:
                set_position(env, entity, value)
                assert np.array_equal(encode_now(env), baseline)

    @pytest.mark.parametrize("entity", [EntityId.DRIFTER, EntityId.PROWLER, EntityId.WARDEN])
    def test_no_leak_through_the_proximity_bits_with_a_light_on(self, entity: EntityId) -> None:
        """The proximity bits read live state, so they are the one place a leak could hide.

        They are gated on the light *and* on the entity being at that corner, so moving it among
        non-corner nodes with both lights burning must change nothing.
        """
        env = NightGuardEnv(night=6)
        env.reset(seed=4)
        office = env.sim.state.office
        office.monitor_up = False
        office.light_left = True
        office.light_right = True

        corners = {env.sim.drifter.corner, env.sim.prowler.corner, env.sim.warden.corner}
        elsewhere = [n for n in Node if n not in corners and n is not Node.OFFICE]
        set_position(env, entity, int(elsewhere[0]))
        baseline = encode_now(env)
        for node in elsewhere:
            set_position(env, entity, int(node))
            assert np.array_equal(encode_now(env), baseline), (
                f"{entity.name} at {node.name} leaked through the proximity bits"
            )

    def test_proximity_leak_test_can_fail(self) -> None:
        """The converse: at the corner with the light on, the bit must fire. Otherwise the above
        would pass on an encoder that ignored proximity entirely."""
        env = NightGuardEnv(night=6)
        env.reset(seed=4)
        env.sim.state.office.light_left = True
        env.sim.state.drifter.node = Node.W_BACKSTAGE
        assert encode_now(env)[obs_mod.PROXIMITY_START] == 0.0
        env.sim.state.drifter.node = env.sim.drifter.corner
        assert encode_now(env)[obs_mod.PROXIMITY_START] == 1.0

    def test_the_leak_test_can_fail(self) -> None:
        """PROJECT.md 8.0. If nothing can ever change the observation, the tests above prove nothing.

        With the monitor up on DRIFTER's corner, moving DRIFTER onto and off that node *must* change
        the observation — that is the camera channel doing its job.
        """
        env = NightGuardEnv(night=6)
        env.reset(seed=3)
        office = env.sim.state.office
        office.monitor_up = True
        office.selected_camera = env.sim.drifter.corner

        env.sim.state.drifter.node = env.sim.drifter.corner
        env._belief.update(env.sim, env.sim.state)
        visible = encode_now(env)

        env.sim.state.drifter.node = Node.W_BACKSTAGE
        env._belief.update(env.sim, env.sim.state)
        assert not np.array_equal(encode_now(env), visible)


# --- criterion 5 and 6 ---------------------------------------------------------------------------


def test_reset_with_the_same_seed_reproduces_the_trajectory() -> None:
    """v1.0 exit criterion 5."""
    env = NightGuardEnv(night=5)
    script = [int(v) for v in np.random.default_rng(0).integers(0, 17, size=400)]

    def run() -> list[tuple[float, float]]:
        observation, _ = env.reset(seed=99)
        out = [(float(observation.sum()), 0.0)]
        for action in script:
            observation, reward, terminated, truncated, _ = env.step(action)
            out.append((float(observation.sum()), reward))
            if terminated or truncated:
                break
        return out

    assert run() == run()


def _outcome(env: NightGuardEnv) -> tuple[int, str, float]:
    return (env.sim.state.tick, env.sim.state.cause.value, round(env.sim.state.power_pct, 6))


def _play_out(env: NightGuardEnv) -> tuple[int, str, float]:
    while not env.sim.state.terminated:
        env.step(int(Action.NOOP))
    return _outcome(env)


def test_unseeded_resets_do_not_replay_the_seeded_episode() -> None:
    """Consecutive resets must draw fresh substreams.

    ``reset`` used to build its own ``default_rng(seed)`` on the seeded path while falling back to
    ``self.np_random`` on the unseeded one. Gymnasium seeds ``self.np_random`` from
    ``SeedSequence(seed)``, which is bit-identical, so the *first* unseeded reset spawned children
    0..5 for a second time and replayed the seeded episode exactly. SB3 seeds once and then resets
    without a seed for the rest of the run, so every training run duplicated its first episode.
    """
    env = NightGuardEnv(night=6)
    env.reset(seed=7)
    outcomes = [_play_out(env)]
    for _ in range(4):
        env.reset()
        outcomes.append(_play_out(env))
    assert len(set(outcomes)) > 1, f"every episode identical: {outcomes[0]}"
    assert outcomes[0] != outcomes[1], "the first unseeded reset replayed the seeded episode"


def test_seeded_reset_still_agrees_with_the_core_simulator() -> None:
    """The fix must not move any seeded fixture: reset(seed=k) == NightSim.from_seed(k)."""
    config = load_night_config(6)
    for seed in (0, 7, 918442):
        env = NightGuardEnv(config=config)
        env.reset(seed=seed)
        sim = NightSim.from_seed(config, seed=seed)
        sim.run([Action.NOOP])
        assert _play_out(env) == (
            sim.state.tick,
            sim.state.cause.value,
            round(sim.state.power_pct, 6),
        )


@pytest.mark.parametrize("night", NIGHTS)
def test_observations_stay_inside_the_declared_box(night: int) -> None:
    """v1.0 exit criterion 6, asserted directly rather than trusted to check_env."""
    env = NightGuardEnv(night=night)
    for seed in range(4):
        for observation in rollout(env, seed):
            assert env.observation_space.contains(observation), observation[
                ~((observation >= 0.0) & (observation <= 1.0))
            ]


@pytest.mark.slow
@pytest.mark.parametrize("night", NIGHTS)
def test_many_random_episodes_complete_without_exception(night: int) -> None:
    """v1.0 exit criterion 2, at a reduced count; validate.py runs the full 10,000."""
    env = NightGuardEnv(night=night)
    causes = set()
    for seed in range(200):
        rollout(env, seed)
        causes.add(env.sim.state.cause)
    assert causes <= set(TerminationCause)
    assert len(causes) > 1, f"night {night} produced only {causes}"


# --- observation layout --------------------------------------------------------------------------


class TestLayout:
    def test_dimensions_and_block_offsets(self) -> None:
        assert obs_mod.OBS_DIMS == 100
        assert (obs_mod.RESOURCE_START, obs_mod.OFFICE_START, obs_mod.CAMERA_START) == (0, 2, 9)
        assert (obs_mod.BELIEF_START, obs_mod.AUDIO_START) == (21, 77)
        assert (obs_mod.PROXIMITY_START, obs_mod.ACTION_START) == (81, 83)

    def test_jam_bits_sit_inside_the_office_block(self, env: NightGuardEnv) -> None:
        env.reset(seed=0)
        base = encode_now(env)
        assert base[obs_mod.OFFICE_START + 2] == 0.0
        env.sim.state.office.jam_left = True
        assert encode_now(env)[obs_mod.OFFICE_START + 2] == 1.0

    def test_belief_is_seeded_from_the_start_state(self, env: NightGuardEnv) -> None:
        """PROJECT.md 10: the start never varies and is publicly known."""
        observation, _ = env.reset(seed=0)
        for entity in (EntityId.WARDEN, EntityId.DRIFTER, EntityId.PROWLER):
            base = obs_mod.BELIEF_START + int(entity) * obs_mod.BELIEF_DIMS_PER_ENTITY
            assert observation[base + int(Node.STAGE)] == 1.0
            assert observation[base + obs_mod.NUM_NODES] == 0.0  # staleness 0, not 1

    def test_sprinter_belief_encodes_stage_not_a_node(self, env: NightGuardEnv) -> None:
        """The one irregularity in an otherwise uniform layout."""
        observation, _ = env.reset(seed=0)
        base = obs_mod.BELIEF_START + int(EntityId.SPRINTER) * obs_mod.BELIEF_DIMS_PER_ENTITY
        assert observation[base + obs_mod.SPRINTER_STAGE_SLOT + 0] == 1.0
        assert observation[base + 5 : base + obs_mod.NUM_NODES].sum() == 0.0

    def test_staleness_saturates_and_never_exceeds_one(self, env: NightGuardEnv) -> None:
        env.reset(seed=0)
        for _ in range(300):
            if env.sim.state.terminated:
                break
            env.step(Action.NOOP)
        observation = encode_now(env)
        for entity in EntityId:
            base = obs_mod.BELIEF_START + int(entity) * obs_mod.BELIEF_DIMS_PER_ENTITY
            assert 0.0 <= observation[base + obs_mod.NUM_NODES] <= 1.0

    def test_never_observed_reads_as_maximally_stale(self, env: NightGuardEnv) -> None:
        """Unreachable under the shipped seeded start, but defined for v2.0's randomised starts."""
        env.reset(seed=0)
        tracker = obs_mod.BeliefTracker()
        assert tracker.staleness(EntityId.DRIFTER, tick=0) == 1.0


class TestObservationChannels:
    """PROJECT.md 6.2's two channels, joined by `or`. Conflating them is a plausible-looking bug."""

    def test_the_camera_channel_sees_a_corner(self, env: NightGuardEnv) -> None:
        """Both corners carry video; treating the light as the only corner channel is the bug."""
        env.reset(seed=0)
        env.sim.state.drifter.node = env.sim.drifter.corner
        env.sim.state.office.monitor_up = True
        env.sim.state.office.selected_camera = env.sim.drifter.corner
        assert obs_mod._is_observed(env.sim, env.sim.state, EntityId.DRIFTER)

    def test_the_light_channel_sees_a_corner(self, env: NightGuardEnv) -> None:
        env.reset(seed=0)
        env.sim.state.drifter.node = env.sim.drifter.corner
        env.sim.state.office.monitor_up = False
        env.sim.state.office.light_left = True
        assert obs_mod._is_observed(env.sim, env.sim.state, EntityId.DRIFTER)

    def test_the_blind_node_is_never_visually_observed(self, env: NightGuardEnv) -> None:
        """E_KITCHEN reaches the agent only through the `kitchen` audio signal."""
        env.reset(seed=0)
        env.sim.state.prowler.node = Node.E_KITCHEN
        env.sim.state.office.monitor_up = True
        env.sim.state.office.selected_camera = Node.E_KITCHEN
        assert not obs_mod._is_observed(env.sim, env.sim.state, EntityId.PROWLER)

    def test_sprinter_is_never_visually_observed(self, env: NightGuardEnv) -> None:
        """It has no position at all; audio is its only channel, which is the 3.7 asymmetry."""
        env.reset(seed=0)
        for camera in (Node.COVE, Node.STAGE, Node.W_CORNER):
            env.sim.state.office.monitor_up = True
            env.sim.state.office.selected_camera = camera
            assert not obs_mod._is_observed(env.sim, env.sim.state, EntityId.SPRINTER)

    def test_proximity_bits_fire_only_with_the_light(self, env: NightGuardEnv) -> None:
        env.reset(seed=0)
        env.sim.state.drifter.node = env.sim.drifter.corner
        env.sim.state.office.light_left = False
        assert encode_now(env)[obs_mod.PROXIMITY_START] == 0.0
        env.sim.state.office.light_left = True
        assert encode_now(env)[obs_mod.PROXIMITY_START] == 1.0


# --- reward --------------------------------------------------------------------------------------


class TestReward:
    def test_survival_and_dawn(self) -> None:
        """PROJECT.md 6.3's approximate maximum return, ~20.7."""
        from nightguard.env import max_return

        assert max_return(load_night_config(1)) == pytest.approx(1070 * 0.01 + 10.0)

    def test_death_is_penalised(self, env: NightGuardEnv) -> None:
        env.reset(seed=0)
        total = 0.0
        while True:
            _, reward, terminated, _, info = env.step(Action.NOOP)
            total += reward
            if terminated:
                break
        assert info["termination_cause"] != "SURVIVED"
        assert total < 0.0

    def test_sparse_mode_leaves_only_terminal_signals(self) -> None:
        from dataclasses import replace

        config = load_night_config(4)
        sparse = replace(config, reward=replace(config.reward, sparse_mode=True))
        env = NightGuardEnv(night=4, config=sparse)
        env.reset(seed=0)
        rewards = []
        while True:
            _, reward, terminated, _, _ = env.step(Action.NOOP)
            rewards.append(reward)
            if terminated:
                break
        assert all(r == 0.0 for r in rewards[:-1]), "sparse mode paid a non-terminal reward"
        assert rewards[-1] != 0.0

    def test_the_blackout_penalty_fires_once(self) -> None:
        env = NightGuardEnv(night=6)
        env.reset(seed=0)
        env.sim.state.office.door_left = True
        env.sim.state.office.door_right = True
        charged = 0
        while True:
            was = env._reward.blackout_charged
            _, _, terminated, _, _ = env.step(Action.NOOP)
            if env._reward.blackout_charged and not was:
                charged += 1
            if terminated:
                break
        assert charged <= 1


# --- wrappers --------------------------------------------------------------------------------------


class TestWrappers:
    def test_oracle_appends_ground_truth_and_widens_the_space(self) -> None:
        env = Oracle(NightGuardEnv(night=6))
        observation, _ = env.reset(seed=0)
        assert observation.shape == (obs_mod.OBS_DIMS + 4,)
        assert env.observation_space.contains(observation)

    def test_oracle_actually_reveals_position(self) -> None:
        """The gap must be non-zero, or the wrapper is measuring nothing."""
        env = Oracle(NightGuardEnv(night=6))
        env.reset(seed=0)
        inner = env.unwrapped
        inner.sim.state.office.monitor_up = False
        inner.sim.state.drifter.node = Node.W_BACKSTAGE
        first = env.observation(inner._observe())
        inner.sim.state.drifter.node = Node.W_CORNER
        second = env.observation(inner._observe())
        assert not np.array_equal(first, second)
        assert np.array_equal(first[: obs_mod.OBS_DIMS], second[: obs_mod.OBS_DIMS])

    def test_audio_mask_zeroes_only_the_audio_block(self) -> None:
        env = AudioMask(NightGuardEnv(night=6))
        observation, _ = env.reset(seed=0)
        block = observation[obs_mod.AUDIO_START : obs_mod.AUDIO_START + obs_mod.AUDIO_DIMS]
        assert not block.any()

    def test_previous_action_and_frame_stack_shapes(self) -> None:
        previous = PreviousAction(NightGuardEnv(night=4))
        observation, _ = previous.reset(seed=0)
        assert observation.shape == (obs_mod.OBS_DIMS + 17,)

        stacked = FrameStack(NightGuardEnv(night=4), frames=3)
        observation, _ = stacked.reset(seed=0)
        assert observation.shape == (obs_mod.OBS_DIMS * 3,)

    def test_frame_stack_rejects_a_zero_width(self) -> None:
        with pytest.raises(ValueError):
            FrameStack(NightGuardEnv(night=4), frames=0)
