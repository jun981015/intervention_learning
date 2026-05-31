from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box, Dict

from il.algo.bc.flow import BCFlowAgent, get_config as get_bc_flow_config
from il.algo.bc.mlp import BCMLPAgent, get_config as get_bc_mlp_config
from il.algo.rl.residual_rlpd import ResidualRLPDAgent, get_config as get_residual_rlpd_config
from il.algo.rl.residual_td3 import ResidualTD3Agent, get_config as get_residual_td3_config
from il.algo.rl.rlpd import ACRLPDAgent, get_config as get_rlpd_config
from il.buffers.mixed import MixedReplaySampler, MixedSamplingSpec, ReplayBufferCollection
from il.buffers.replay_buffer import ReplayBuffer
from il.buffers.routing import route_episode_to_buffers
from il.buffers.schema import make_replay_example, step_record_to_transition
from il.builders.components import _cache_residual_base_actions, build_buffers, infer_env_spec
from il.builders.config import new_schema_to_legacy_recipe
from il.gating.expert_q_gap import ExpertQGapGate
from il.gating.random_gate import RandomGate
from il.envs.robomimic_lowdim import RobomimicLowdimWrapper
from il.logger import MetricLogger
from il.loops.rollout import choose_rollout_action, prepare_next_base_action, reset_rollout_state, sample_base_action
from il.loops.updates import _assert_residual_metadata
from il.policies.bc_flow import BCFlowPolicy
from il.policies.rlpd import RLPDPolicy
from il.utils.config import TrainingConfig
from il.utils.flax_utils import save_agent
from il.utils.updates import sample_rl_update_batch
from il.utils.types import PolicyOutput, StepRecord


@dataclass
class ConstantPolicy:
    action: np.ndarray
    log_prob: float = 0.0

    def sample_action(self, observation: np.ndarray, *, rng) -> PolicyOutput:
        del observation, rng
        return PolicyOutput(action=self.action.copy(), log_prob=self.log_prob)


class FakeRobomimicEnv:
    """Tiny Robomimic-like env used to test wrapper-only behavior."""

    action_dimension = 2

    def __init__(self, *, success: bool):
        self.success = success

    def get_observation(self):
        return {
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            "object": np.zeros(10, dtype=np.float32),
        }

    def reset(self):
        return None

    def seed(self, seed=None):
        return [seed]

    def step(self, action):
        del action
        return self.get_observation(), 123.0, False, {"is_success": {"task": self.success}}

    def render(self, *args, **kwargs):
        del args, kwargs
        return np.zeros((8, 8, 3), dtype=np.uint8)


def smoke_reward_transform() -> None:
    """Check wrapper-level reward scale/shift without launching a simulator."""
    failure_env = RobomimicLowdimWrapper(
        FakeRobomimicEnv(success=False),
        reward_scale=1.0,
        reward_shift=-1.0,
        max_episode_length=1,
    )
    _, reward, terminated, truncated, info = failure_env.step(np.zeros(2, dtype=np.float32))
    assert reward == -1.0
    assert not terminated
    assert truncated
    assert info["task_reward"] == 0.0
    assert info["reward_shift"] == -1.0

    success_env = RobomimicLowdimWrapper(
        FakeRobomimicEnv(success=True),
        reward_scale=1.0,
        reward_shift=-1.0,
    )
    _, reward, terminated, truncated, info = success_env.step(np.zeros(2, dtype=np.float32))
    assert reward == 0.0
    assert terminated
    assert not truncated
    assert info["task_reward"] == 1.0
    assert info["success"] == 1


def choose_smoke_action(
    *,
    step: int,
    observation: np.ndarray,
    learner: ConstantPolicy,
    expert: ConstantPolicy,
    gate: RandomGate,
    learner_rng,
    expert_rng,
    gate_rng: np.random.Generator,
):
    """Small simulator-free helper for smoke tests only."""
    learner_output = learner.sample_action(observation, rng=learner_rng)
    expert_output = expert.sample_action(observation, rng=expert_rng)
    decision = gate.decide(
        step=step,
        observation=observation,
        learner=learner_output,
        expert=expert_output,
        rng=gate_rng,
    )
    action = expert_output.action if decision.use_expert else learner_output.action
    return np.asarray(action, dtype=np.float32), learner_output, expert_output, decision


class ChunkPolicy:
    """Policy stub that emits an action chunk once and then relies on rollout queueing."""

    def __init__(self, chunk: np.ndarray, *, use_info_chunk: bool = True):
        self.chunk = np.asarray(chunk, dtype=np.float32)
        self.use_info_chunk = use_info_chunk
        self.calls = 0

    def sample_action(self, observation: np.ndarray, *, rng) -> PolicyOutput:
        del observation, rng
        self.calls += 1
        if self.use_info_chunk:
            return PolicyOutput(
                action=self.chunk[0].copy(),
                log_prob=0.0,
                info={"full_action_chunk": self.chunk.copy()},
            )
        return PolicyOutput(action=self.chunk.reshape(-1).copy(), log_prob=0.0)


class FailingPolicy:
    """Policy stub used to assert a rollout path does not query a policy."""

    def sample_action(self, observation: np.ndarray, *, rng) -> PolicyOutput:
        del observation, rng
        raise AssertionError("policy should not be queried")


def smoke_base_action_chunk_queue() -> None:
    """Check ResFit-style base action chunks are popped one primitive action at a time."""
    chunk = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    policy = ChunkPolicy(chunk)
    context = SimpleNamespace(
        base=SimpleNamespace(policy=policy),
        action_dim=2,
        env_spec=SimpleNamespace(state_key=None),
        rollout_state={},
    )
    observation = np.zeros(5, dtype=np.float32)

    first = sample_base_action(context, observation, rng=None)
    second_for_target = prepare_next_base_action(context, observation, rng=None)
    second_for_rollout = sample_base_action(context, observation, rng=None)
    third = sample_base_action(context, observation, rng=None)

    assert policy.calls == 1
    assert np.allclose(first.action, chunk[0])
    assert np.allclose(second_for_target.action, chunk[1])
    assert np.allclose(second_for_rollout.action, chunk[1])
    assert np.allclose(third.action, chunk[2])
    assert first.info["base_chunk_index"] == 0
    assert second_for_target.info["base_chunk_index"] == 1
    assert third.info["base_chunk_index"] == 2

    reset_rollout_state(context)
    first_after_reset = sample_base_action(context, observation, rng=None)
    assert policy.calls == 2
    assert np.allclose(first_after_reset.action, chunk[0])

    flat_policy = ChunkPolicy(chunk, use_info_chunk=False)
    flat_context = SimpleNamespace(
        base=SimpleNamespace(policy=flat_policy),
        action_dim=2,
        env_spec=SimpleNamespace(state_key=None),
        rollout_state={},
    )
    assert np.allclose(sample_base_action(flat_context, observation, rng=None).action, chunk[0])
    assert np.allclose(sample_base_action(flat_context, observation, rng=None).action, chunk[1])
    assert flat_policy.calls == 1


def smoke_residual_base_noise_warmup() -> None:
    """Check residual rollout can collect base+noise warmup without querying learner."""
    base_action = np.asarray([0.2, -0.1], dtype=np.float32)
    context = SimpleNamespace(
        base=SimpleNamespace(
            kind="bc_flow",
            checkpoint_path=None,
            policy=ConstantPolicy(base_action),
        ),
        learner=SimpleNamespace(
            config={"residual_scale": 0.1},
            policy=FailingPolicy(),
        ),
        expert=None,
        action_dim=2,
        env_spec=SimpleNamespace(state_key=None),
        rollout_state={},
        rng=jax.random.PRNGKey(0),
        config={
            "rollout": {
                "execute": "residual",
                "sample_expert": False,
                "residual_warmup_steps": 5,
                "warmup_noise_scale": 0.0,
                "use_base_policy_for_warmup": True,
            }
        },
    )
    action, learner_output, expert_output, decision = choose_rollout_action(
        context,
        np.zeros(4, dtype=np.float32),
        step=1,
    )

    assert np.allclose(action, base_action)
    assert np.allclose(learner_output.action, base_action)
    assert np.allclose(learner_output.info["residual_action"], np.zeros_like(base_action))
    assert learner_output.info["residual_warmup"] == 1
    assert np.isnan(expert_output.action).all()
    assert not decision.use_expert


def smoke_gate_and_replay() -> None:
    observation = np.zeros(5, dtype=np.float32)
    learner = ConstantPolicy(np.full(2, -0.5, dtype=np.float32), log_prob=-1.0)
    expert = ConstantPolicy(np.full(2, 0.5, dtype=np.float32), log_prob=-0.5)
    gate = RandomGate(expert_probability=1.0)
    action, learner_output, expert_output, decision = choose_smoke_action(
        step=0,
        observation=observation,
        learner=learner,
        expert=expert,
        gate=gate,
        learner_rng=None,
        expert_rng=None,
        gate_rng=np.random.default_rng(0),
    )
    assert decision.use_expert
    assert np.allclose(action, expert_output.action)
    assert not np.allclose(action, learner_output.action)

    example = make_replay_example(observation, action)
    replay = ReplayBuffer.create(example, size=16)
    for i in range(8):
        transition = {key: np.array(value, copy=True) for key, value in example.items()}
        transition["rewards"] = np.asarray(float(i), dtype=np.float32)
        transition["terminals"] = np.asarray(1.0 if i == 4 else 0.0, dtype=np.float32)
        transition["masks"] = np.asarray(0.0 if i == 4 else 1.0, dtype=np.float32)
        replay.add_transition(transition)

    batch = replay.sample_sequence(batch_size=4, sequence_length=3, discount=0.9)
    assert batch["observations"].shape == (4, 5)
    assert batch["actions"].shape == (4, 3, 2)
    assert batch["expert_actions"].shape == (4, 3, 2)
    assert batch["valid"].shape == (4, 3)


def make_mock_episode(gate_probs: list[float], *, success: bool, episode_id: int = -1) -> list[dict]:
    """Build a deterministic intervention rollout without a simulator."""
    observation_dim = 5
    learner = ConstantPolicy(np.full(2, -0.5, dtype=np.float32), log_prob=-1.0)
    expert = ConstantPolicy(np.full(2, 0.5, dtype=np.float32), log_prob=-0.5)
    gate_rng = np.random.default_rng(42)
    episode = []

    for step, expert_probability in enumerate(gate_probs):
        observation = np.full(observation_dim, step, dtype=np.float32)
        next_observation = np.full(observation_dim, step + 1, dtype=np.float32)
        action, learner_output, expert_output, decision = choose_smoke_action(
            step=step,
            observation=observation,
            learner=learner,
            expert=expert,
            gate=RandomGate(expert_probability=expert_probability),
            learner_rng=None,
            expert_rng=None,
            gate_rng=gate_rng,
        )
        is_last = step == len(gate_probs) - 1
        record = StepRecord(
            observation=observation,
            learner=learner_output,
            expert=expert_output,
            decision=decision,
            action=action,
            reward=float(success and is_last),
            terminated=is_last,
            truncated=False,
            next_observation=next_observation,
            episode_id=episode_id,
            episode_step=step,
        )
        transition = step_record_to_transition(record)
        transition["_success"] = success if is_last else False
        transition["_step"] = step
        episode.append(transition)

    return episode


def smoke_intervention_routing() -> None:
    """Check learner/expert metadata storage and demo/intervention routing."""
    observation = np.zeros(5, dtype=np.float32)
    action = np.zeros(2, dtype=np.float32)
    example = make_replay_example(observation, action)
    rl_buffer = ReplayBuffer.create(example, size=64)
    demo_buffer = ReplayBuffer.create(example, size=64)
    intervention_buffer = ReplayBuffer.create(example, size=64)

    autonomous_success = make_mock_episode([0.0, 0.0, 0.0], success=True)
    for transition in autonomous_success:
        rl_buffer.add_transition({key: value for key, value in transition.items() if not key.startswith("_")})
    counts = route_episode_to_buffers(
        autonomous_success,
        demo_buffer=demo_buffer,
        intervention_buffer=intervention_buffer,
        include_failed_interventions=False,
    )
    assert counts == {
        "demo_added": 3,
        "demo_removed": 0,
        "demo_skipped": 0,
        "intervention_added": 0,
        "failed_intervention_seen": 0,
    }
    assert demo_buffer.size == 3
    assert intervention_buffer.size == 0

    intervention_success = make_mock_episode([0.0, 0.0, 1.0, 1.0], success=True)
    for transition in intervention_success:
        rl_buffer.add_transition({key: value for key, value in transition.items() if not key.startswith("_")})
    counts = route_episode_to_buffers(
        intervention_success,
        demo_buffer=demo_buffer,
        intervention_buffer=intervention_buffer,
        include_failed_interventions=False,
    )
    assert counts == {
        "demo_added": 0,
        "demo_removed": 0,
        "demo_skipped": 0,
        "intervention_added": 2,
        "failed_intervention_seen": 0,
    }
    assert demo_buffer.size == 3
    assert intervention_buffer.size == 2

    first_intervention = intervention_success[2]
    assert int(first_intervention["interventions"]) == 1
    assert np.allclose(first_intervention["actions"], first_intervention["expert_actions"])
    assert not np.allclose(first_intervention["actions"], first_intervention["learner_actions"])

    failed_intervention = make_mock_episode([0.0, 1.0, 1.0], success=False)
    counts = route_episode_to_buffers(
        failed_intervention,
        demo_buffer=demo_buffer,
        intervention_buffer=intervention_buffer,
        include_failed_interventions=False,
    )
    assert counts == {
        "demo_added": 0,
        "demo_removed": 0,
        "demo_skipped": 0,
        "intervention_added": 0,
        "failed_intervention_seen": 2,
    }
    assert intervention_buffer.size == 2

    counts = route_episode_to_buffers(
        failed_intervention,
        demo_buffer=demo_buffer,
        intervention_buffer=intervention_buffer,
        include_failed_interventions=True,
    )
    assert counts == {
        "demo_added": 0,
        "demo_removed": 0,
        "demo_skipped": 0,
        "intervention_added": 2,
        "failed_intervention_seen": 0,
    }
    assert intervention_buffer.size == 4

    batch = rl_buffer.sample_sequence(batch_size=4, sequence_length=2, discount=0.99)
    assert batch["observations"].shape == (4, 5)
    assert batch["actions"].shape == (4, 2, 2)

    counts = route_episode_to_buffers(
        autonomous_success,
        demo_buffer=demo_buffer,
        intervention_buffer=intervention_buffer,
        include_failed_interventions=False,
        demo_insert_mode="none",
    )
    assert counts == {
        "demo_added": 0,
        "demo_removed": 0,
        "demo_skipped": 1,
        "intervention_added": 0,
        "failed_intervention_seen": 0,
    }
    assert demo_buffer.size == 3


def smoke_demo_episode_replacement() -> None:
    """Check demo pool replacement by shortest successful horizon."""
    observation = np.zeros(5, dtype=np.float32)
    action = np.zeros(2, dtype=np.float32)
    demo_buffer = ReplayBuffer.create(make_replay_example(observation, action), size=10)
    intervention_buffer = ReplayBuffer.create(make_replay_example(observation, action), size=8)

    long_success = make_mock_episode([0.0] * 5, success=True, episode_id=10)
    counts = route_episode_to_buffers(
        long_success,
        demo_buffer=demo_buffer,
        intervention_buffer=intervention_buffer,
        include_failed_interventions=False,
        demo_insert_mode="replace_longest_if_better",
    )
    assert counts["demo_added"] == 5
    assert counts["demo_removed"] == 0
    assert demo_buffer.episode_lengths() == {10: 5}
    assert demo_buffer.episode_worst_order == [10]

    shorter_success = make_mock_episode([0.0] * 4, success=True, episode_id=11)
    counts = route_episode_to_buffers(
        shorter_success,
        demo_buffer=demo_buffer,
        intervention_buffer=intervention_buffer,
        include_failed_interventions=False,
        demo_insert_mode="replace_longest_if_better",
    )
    assert counts["demo_added"] == 4
    assert counts["demo_removed"] == 5
    assert counts["demo_skipped"] == 0
    assert demo_buffer.size == 4
    assert demo_buffer.episode_lengths() == {11: 4}
    assert demo_buffer.worst_episode()[1] == 4

    longer_success = make_mock_episode([0.0] * 6, success=True, episode_id=12)
    counts = route_episode_to_buffers(
        longer_success,
        demo_buffer=demo_buffer,
        intervention_buffer=intervention_buffer,
        include_failed_interventions=False,
        demo_insert_mode="replace_longest_if_better",
    )
    assert counts["demo_added"] == 0
    assert counts["demo_removed"] == 0
    assert counts["demo_skipped"] == 1
    assert demo_buffer.size == 4
    assert demo_buffer.episode_lengths() == {11: 4}


def make_constant_reward_buffer(reward: float, *, obs_dim: int, action_dim: int, size: int) -> ReplayBuffer:
    """Create a replay buffer whose samples can be identified by reward value."""
    example = make_replay_example(np.zeros(obs_dim, dtype=np.float32), np.zeros(action_dim, dtype=np.float32))
    replay = ReplayBuffer.create(example, size=size)
    for i in range(size):
        transition = make_replay_example(
            np.full(obs_dim, i, dtype=np.float32),
            np.full(action_dim, reward, dtype=np.float32),
        )
        transition["rewards"] = np.asarray(reward, dtype=np.float32)
        transition["expert_actions"] = np.full(action_dim, reward, dtype=np.float32)
        replay.add_transition(transition)
    return replay


def smoke_mixed_replay_sampling() -> None:
    """Check flexible ratio sampling across online, intervention, and demo buffers."""
    collection = ReplayBufferCollection(
        online=make_constant_reward_buffer(1.0, obs_dim=5, action_dim=2, size=8),
        intervention=make_constant_reward_buffer(2.0, obs_dim=5, action_dim=2, size=8),
        demo=make_constant_reward_buffer(3.0, obs_dim=5, action_dim=2, size=8),
    )
    spec = MixedSamplingSpec({"online": 0.5, "intervention": 0.25, "demo": 0.25})
    assert spec.counts(8) == {"online": 4, "intervention": 2, "demo": 2}

    sampler = MixedReplaySampler(collection, spec, shuffle=False)
    batch = sampler.sample(batch_size=8)
    assert batch["observations"].shape == (8, 5)
    assert batch["actions"].shape == (8, 2)
    assert np.allclose(batch["rewards"][:4], 1.0)
    assert np.allclose(batch["rewards"][4:6], 2.0)
    assert np.allclose(batch["rewards"][6:], 3.0)

    sequence_batch = sampler.sample_sequence(batch_size=8, sequence_length=1, discount=0.99)
    assert sequence_batch["observations"].shape == (8, 5)
    assert sequence_batch["actions"].shape == (8, 1, 2)
    assert np.allclose(sequence_batch["rewards"][:4, 0], 1.0)
    assert np.allclose(sequence_batch["rewards"][4:6, 0], 2.0)
    assert np.allclose(sequence_batch["rewards"][6:, 0], 3.0)

    update_batch = sample_rl_update_batch(
        collection,
        TrainingConfig(
            batch_size=4,
            utd_ratio=2,
            horizon_length=1,
            discount=0.99,
            sampling_fractions={"online": 0.5, "intervention": 0.25, "demo": 0.25},
        ),
    )
    assert update_batch["observations"].shape == (2, 4, 5)
    assert update_batch["actions"].shape == (2, 4, 1, 2)


def smoke_residual_base_action_cache() -> None:
    """Check offline prefill can cache base and next-base actions for residual RL."""
    chunk = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    policy = ChunkPolicy(chunk)
    base_actor = SimpleNamespace(policy=policy)
    env_spec = SimpleNamespace(obs_dim=5, action_dim=2, state_key=None)
    dataset = {
        "observations": np.zeros((3, 5), dtype=np.float32),
        "next_observations": np.ones((3, 5), dtype=np.float32),
        "actions": np.full((3, 2), 0.5, dtype=np.float32),
        "episode_ids": np.zeros(3, dtype=np.int64),
    }

    cached = _cache_residual_base_actions(dataset, base_actor=base_actor, env_spec=env_spec, seed=0)

    assert np.allclose(cached["base_actions"], chunk)
    assert np.allclose(cached["next_base_actions"][:2], chunk[1:])
    assert np.allclose(cached["residual_actions"], cached["actions"] - cached["base_actions"])
    assert policy.calls == 2


def smoke_replay_prefill() -> None:
    """Check optional initial replay prefill, including image side storage."""

    class DummyEnv:
        observation_space = Dict(
            {
                "state": Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "agentview": Box(low=0, high=255, shape=(4, 4, 3), dtype=np.uint8),
            }
        )
        action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    env_spec = infer_env_spec(DummyEnv())
    source = ReplayBuffer.create(
        make_replay_example(env_spec.observation_example, env_spec.action_example),
        size=8,
    )
    for i in range(3):
        observation = {
            "state": np.full(2, i, dtype=np.float32),
            "agentview": np.full((4, 4, 3), i + 1, dtype=np.uint8),
        }
        next_observation = {
            "state": np.full(2, i + 1, dtype=np.float32),
            "agentview": np.full((4, 4, 3), i + 2, dtype=np.uint8),
        }
        transition = make_replay_example(observation, np.asarray([i], dtype=np.float32))
        transition["next_observations"] = next_observation
        transition["episode_ids"] = np.asarray(0, dtype=np.int64)
        transition["episode_steps"] = np.asarray(i, dtype=np.int32)
        source.add_transition(transition)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "demo_replay.npz"
        source.save_npz(path)
        buffers = build_buffers(
            {
                "replay": {
                    "frame_stack": 1,
                    "online_size": 8,
                    "intervention_size": 8,
                    "demo_size": 8,
                    "prefill": {"demo": {"path": str(path), "format": "npz"}},
                }
            },
            env_spec=env_spec,
        )

    assert buffers.online.size == 0
    assert buffers.demo.size == 3
    assert buffers.intervention.size == 0
    assert buffers.demo.image_data is not None
    assert list(buffers.demo.image_data) == ["agentview"]

    batch = buffers.demo.get_subset(np.asarray([0, 1, 2]))
    assert batch["observations"]["state"].shape == (3, 2)
    assert batch["observations"]["agentview"].shape == (3, 4, 4, 3)
    assert batch["next_observations"]["agentview"].shape == (3, 4, 4, 3)
    assert np.all(batch["observations"]["agentview"][:, 0, 0, 0] == np.asarray([1, 2, 3]))
    assert np.all(batch["next_observations"]["agentview"][:, 0, 0, 0] == np.asarray([2, 3, 0]))
    assert np.allclose(batch["image_next_valid"], np.asarray([1.0, 1.0, 0.0], dtype=np.float32))


def make_update_batch(batch_size: int, obs_dim: int, action_dim: int, horizon: int) -> dict:
    observations = np.random.default_rng(0).normal(size=(batch_size, obs_dim)).astype(np.float32)
    actions = np.random.default_rng(1).uniform(
        low=-0.5,
        high=0.5,
        size=(batch_size, horizon, action_dim),
    ).astype(np.float32)
    next_observations = np.random.default_rng(2).normal(
        size=(batch_size, horizon, obs_dim),
    ).astype(np.float32)
    return {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": np.zeros((batch_size, horizon), dtype=np.float32),
        "masks": np.ones((batch_size, horizon), dtype=np.float32),
        "terminals": np.zeros((batch_size, horizon), dtype=np.float32),
        "valid": np.ones((batch_size, horizon), dtype=np.float32),
    }


def smoke_rlpd(obs_dim: int, action_dim: int) -> None:
    config = get_rlpd_config()
    config.horizon_length = 1
    config.action_chunking = False
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.batch_size = 4
    config.num_qs = 2
    config.target_q_agg = "min"
    config.grad_clip_norm = None

    ex_observations = jnp.zeros((config.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((config.batch_size, action_dim), dtype=jnp.float32)
    agent = ACRLPDAgent.create(0, ex_observations, ex_actions, config)
    batch = make_update_batch(config.batch_size, obs_dim, action_dim, config.horizon_length)
    agent, info = agent.update(batch)
    action, log_prob = agent.sample_actions_with_log_prob(
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        rng=jax.random.PRNGKey(1),
    )
    assert action.shape == (1, action_dim)
    assert log_prob.shape == (1,)
    assert np.isfinite(float(info["critic/critic_loss"]))


def smoke_rlpd_td_sequence_length(obs_dim: int, action_dim: int) -> None:
    """Check TD backup length comes from sampled replay sequence, not actor horizon."""
    config = get_rlpd_config()
    config.horizon_length = 5
    config.action_chunking = False
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.batch_size = 4
    config.num_qs = 2
    config.target_q_agg = "min"
    config.grad_clip_norm = None

    ex_observations = jnp.zeros((config.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((config.batch_size, action_dim), dtype=jnp.float32)
    agent = ACRLPDAgent.create(5, ex_observations, ex_actions, config)
    batch = make_update_batch(config.batch_size, obs_dim, action_dim, horizon=3)
    agent, info = agent.update(batch)

    assert np.isfinite(float(info["critic/critic_loss"]))
    assert float(info["critic/td_n_step"]) == 3.0


def smoke_residual_transition_schema(obs_dim: int, action_dim: int) -> None:
    """Check residual rollout metadata is stored in canonical replay keys."""
    observation = np.zeros(obs_dim, dtype=np.float32)
    next_observation = np.ones(obs_dim, dtype=np.float32)
    base_action = np.full(action_dim, 0.2, dtype=np.float32)
    residual_action = np.full(action_dim, -0.05, dtype=np.float32)
    next_base_action = np.full(action_dim, 0.1, dtype=np.float32)
    action = np.clip(base_action + residual_action, -1.0, 1.0)
    record = StepRecord(
        observation=observation,
        learner=PolicyOutput(action=residual_action),
        expert=PolicyOutput(action=np.full(action_dim, np.nan, dtype=np.float32)),
        decision=RandomGate(expert_probability=0.0).decide(
            step=0,
            observation=observation,
            learner=PolicyOutput(action=residual_action),
            expert=PolicyOutput(action=np.full(action_dim, np.nan, dtype=np.float32)),
            rng=np.random.default_rng(0),
        ),
        action=action,
        reward=0.0,
        terminated=False,
        truncated=False,
        next_observation=next_observation,
        base_action=base_action,
        residual_action=residual_action,
        next_base_action=next_base_action,
    )
    transition = step_record_to_transition(record)
    assert np.allclose(transition["actions"], action)
    assert np.allclose(transition["base_actions"], base_action)
    assert np.allclose(transition["residual_actions"], residual_action)
    assert np.allclose(transition["next_base_actions"], next_base_action)


def smoke_residual_rlpd(obs_dim: int, action_dim: int) -> None:
    """Check residual RLPD updates with base-action augmented observations."""
    config = get_residual_rlpd_config()
    config.horizon_length = 1
    config.action_chunking = False
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.batch_size = 4
    config.num_qs = 2
    config.target_q_agg = "min"
    config.grad_clip_norm = None
    config.residual_scale = 0.1
    config.residual_action_l2 = 0.01
    config.base_obs_dim = obs_dim
    config.bc_alpha = 0.0

    ex_observations = jnp.zeros((config.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((config.batch_size, action_dim), dtype=jnp.float32)
    agent = ResidualRLPDAgent.create(10, ex_observations, ex_actions, config)
    batch = make_update_batch(config.batch_size, obs_dim, action_dim, config.horizon_length)
    batch["base_actions"] = np.full((config.batch_size, config.horizon_length, action_dim), 0.2, dtype=np.float32)
    batch["next_base_actions"] = np.full((config.batch_size, config.horizon_length, action_dim), 0.1, dtype=np.float32)
    agent, critic_info = agent.update_critic_only(batch)
    assert np.isfinite(float(critic_info["critic/critic_loss"]))
    assert float(critic_info["actor/update_actor"]) == 0.0
    agent, info = agent.update(batch)

    residual_obs = jnp.zeros((1, obs_dim + action_dim), dtype=jnp.float32)
    action, log_prob = agent.sample_actions_with_log_prob(residual_obs, rng=jax.random.PRNGKey(11))
    assert action.shape == (1, action_dim)
    assert log_prob.shape == (1,)
    assert np.isfinite(float(info["critic/critic_loss"]))
    assert np.isfinite(float(info["actor/residual_l2"]))



def smoke_residual_td3(obs_dim: int, action_dim: int) -> None:
    """Check residual TD3 uses state+base_action actor inputs and state/action critic inputs."""
    config = get_residual_td3_config()
    config.horizon_length = 1
    config.action_chunking = False
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.batch_size = 4
    config.num_qs = 2
    config.target_q_agg = "min"
    config.grad_clip_norm = None
    config.residual_scale = 0.1
    config.base_obs_dim = obs_dim
    config.target_policy_noise = False
    config.exploration_noise = 0.0

    ex_observations = jnp.zeros((config.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((config.batch_size, action_dim), dtype=jnp.float32)
    agent = ResidualTD3Agent.create(13, ex_observations, ex_actions, config)

    batch = make_update_batch(config.batch_size, obs_dim, action_dim, config.horizon_length)
    batch["base_actions"] = np.full((config.batch_size, config.horizon_length, action_dim), 0.2, dtype=np.float32)
    batch["next_base_actions"] = np.full((config.batch_size, config.horizon_length, action_dim), 0.1, dtype=np.float32)
    agent, info = agent.update(batch)

    q = agent.evaluate_q(
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        jnp.zeros((1, action_dim), dtype=jnp.float32),
    )
    residual_obs = jnp.zeros((1, obs_dim + action_dim), dtype=jnp.float32)
    action, log_prob = agent.sample_actions_with_log_prob(residual_obs, rng=jax.random.PRNGKey(13))

    assert q.shape == (1,)
    assert action.shape == (1, action_dim)
    assert log_prob.shape == (1,)
    assert np.isfinite(float(info["critic/critic_loss"]))
    assert np.isfinite(float(info["actor/residual_l2"]))

def smoke_residual_rlpd_with_bc_aux(obs_dim: int, action_dim: int) -> None:
    """Check residual BC regularization can train against cached demo base actions."""
    config = get_residual_rlpd_config()
    config.horizon_length = 1
    config.action_chunking = False
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.batch_size = 4
    config.num_qs = 2
    config.target_q_agg = "min"
    config.grad_clip_norm = None
    config.residual_scale = 1.0
    config.residual_action_l2 = 0.0
    config.base_obs_dim = obs_dim
    config.bc_alpha = 0.1

    ex_observations = jnp.zeros((config.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((config.batch_size, action_dim), dtype=jnp.float32)
    agent = ResidualRLPDAgent.create(12, ex_observations, ex_actions, config)

    batch = make_update_batch(config.batch_size, obs_dim, action_dim, config.horizon_length)
    batch["base_actions"] = np.full((config.batch_size, config.horizon_length, action_dim), 0.2, dtype=np.float32)
    batch["next_base_actions"] = np.full((config.batch_size, config.horizon_length, action_dim), 0.1, dtype=np.float32)
    batch["bc_observations"] = np.zeros((config.batch_size, obs_dim), dtype=np.float32)
    batch["bc_actions"] = np.full((config.batch_size, 1, action_dim), 0.5, dtype=np.float32)
    batch["bc_base_actions"] = np.full((config.batch_size, 1, action_dim), 0.2, dtype=np.float32)

    agent, info = agent.update(batch)

    assert np.isfinite(float(info["critic/critic_loss"]))
    assert np.isfinite(float(info["actor/bc_loss"]))
    assert float(info["actor/bc_loss"]) != 0.0


def smoke_residual_metadata_validation(obs_dim: int, action_dim: int) -> None:
    """Check residual updates fail fast when base-action metadata is missing."""

    @dataclass
    class Target:
        name: str
        config: dict

    batch = make_update_batch(batch_size=4, obs_dim=obs_dim, action_dim=action_dim, horizon=1)
    batch["base_actions"] = np.full((4, 1, action_dim), 0.2, dtype=np.float32)
    batch["next_base_actions"] = np.full((4, 1, action_dim), 0.1, dtype=np.float32)
    target = Target(name="residual", config={"residual_policy": True, "bc_alpha": 0.0})
    _assert_residual_metadata(target, {"name": "residual"}, batch)

    bad_batch = dict(batch)
    bad_batch["base_actions"] = np.array(batch["base_actions"], copy=True)
    bad_batch["base_actions"][0, 0, 0] = np.nan
    try:
        _assert_residual_metadata(target, {"name": "residual"}, bad_batch)
    except ValueError as exc:
        assert "base_actions" in str(exc)
    else:
        raise AssertionError("NaN residual base_actions should fail before JAX update.")

    bc_batch = dict(batch)
    bc_batch["bc_actions"] = np.array(batch["actions"], copy=True)
    bc_target = Target(name="residual_bc", config={"residual_policy": True, "bc_alpha": 0.1})
    try:
        _assert_residual_metadata(bc_target, {"name": "residual_bc"}, bc_batch)
    except KeyError as exc:
        assert "bc_base_actions" in str(exc)
    else:
        raise AssertionError("Auxiliary residual BC batch without bc_base_actions should fail.")


def smoke_rlpd_policy_checkpoint(obs_dim: int, action_dim: int) -> None:
    """Check source-agnostic RLPD checkpoint restore through the policy adapter."""
    config = get_rlpd_config()
    config.horizon_length = 1
    config.action_chunking = False
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.batch_size = 4
    config.num_qs = 2
    config.target_q_agg = "min"
    config.grad_clip_norm = None

    ex_observations = jnp.zeros((config.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((config.batch_size, action_dim), dtype=jnp.float32)
    agent = ACRLPDAgent.create(0, ex_observations, ex_actions, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_agent(agent, tmpdir, 0)
        policy = RLPDPolicy.from_checkpoint(
            Path(tmpdir) / "params_0.pkl",
            config=config.to_dict(),
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=0,
        )
        output = policy.sample_action(
            np.zeros(obs_dim, dtype=np.float32),
            rng=jax.random.PRNGKey(4),
        )

    assert output.action.shape == (action_dim,)
    assert np.isfinite(output.log_prob)


def smoke_bc_mlp(obs_dim: int, action_dim: int) -> None:
    """Check deterministic MLP BC updates from replay expert labels."""
    config = get_bc_mlp_config()
    config.horizon_length = 1
    config.action_chunking = False
    config.actor_hidden_dims = (32, 32)
    config.batch_size = 4
    config.grad_clip_norm = None

    ex_observations = jnp.zeros((config.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((config.batch_size, action_dim), dtype=jnp.float32)
    agent = BCMLPAgent.create(0, ex_observations, ex_actions, config)

    observation = np.zeros(obs_dim, dtype=np.float32)
    action = np.zeros(action_dim, dtype=np.float32)
    replay = ReplayBuffer.create(make_replay_example(observation, action), size=16)
    for i in range(8):
        transition = make_replay_example(
            np.full(obs_dim, i, dtype=np.float32),
            np.zeros(action_dim, dtype=np.float32),
        )
        transition["expert_actions"] = np.full(action_dim, 0.25, dtype=np.float32)
        transition["learner_actions"] = np.full(action_dim, -0.25, dtype=np.float32)
        transition["actions"] = transition["expert_actions"].copy()
        transition["interventions"] = np.asarray(1, dtype=np.int8)
        replay.add_transition(transition)

    batch = replay.sample(config.batch_size)
    agent, info = agent.update(batch)
    action, log_prob = agent.sample_actions_with_log_prob(
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        rng=jax.random.PRNGKey(3),
    )
    assert action.shape == (1, action_dim)
    assert log_prob.shape == (1,)
    assert np.isfinite(float(info["actor/bc_loss"]))


def smoke_bc_flow(obs_dim: int, action_dim: int) -> None:
    """Check flow-matching BC actor updates."""
    config = get_bc_flow_config()
    config.horizon_length = 1
    config.action_chunking = False
    config.actor_hidden_dims = (32, 32)
    config.batch_size = 4
    config.flow_steps = 2
    config.grad_clip_norm = None

    ex_observations = jnp.zeros((config.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((config.batch_size, action_dim), dtype=jnp.float32)
    agent = BCFlowAgent.create(0, ex_observations, ex_actions, config)
    batch = make_update_batch(config.batch_size, obs_dim, action_dim, config.horizon_length)
    agent, info = agent.update(batch)
    action, log_prob = agent.sample_actions_with_log_prob(
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        rng=jax.random.PRNGKey(2),
    )
    assert action.shape == (1, action_dim)
    assert log_prob.shape == (1,)
    assert np.isfinite(float(info["actor/bc_flow_loss"]))

    dagger_batch = {
        "observations": batch["observations"],
        "expert_actions": np.full((config.batch_size, action_dim), 0.25, dtype=np.float32),
    }
    config.target_action_key = "expert_actions"
    agent = BCFlowAgent.create(1, ex_observations, ex_actions, config)
    agent, info = agent.update(dagger_batch)
    assert np.isfinite(float(info["actor/bc_flow_loss"]))


def smoke_bc_agents_with_aux_critic(obs_dim: int, action_dim: int) -> None:
    """Check optional BC critics update and expose Q diagnostics without changing actor objective."""
    for agent_cls, get_config, actor_name in (
        (BCMLPAgent, get_bc_mlp_config, "bc_mlp"),
        (BCFlowAgent, get_bc_flow_config, "bc_flow"),
    ):
        config = get_config()
        config.horizon_length = 1
        config.action_chunking = False
        config.actor_hidden_dims = (32, 32)
        config.value_hidden_dims = (32, 32)
        config.batch_size = 4
        config.train_critic = True
        config.num_qs = 2
        config.target_q_agg = "min"
        config.grad_clip_norm = None
        if actor_name == "bc_flow":
            config.flow_steps = 2
        if actor_name == "bc_mlp":
            config.target_action_key = "expert_actions"

        ex_observations = jnp.zeros((config.batch_size, obs_dim), dtype=jnp.float32)
        ex_actions = jnp.zeros((config.batch_size, action_dim), dtype=jnp.float32)
        agent = agent_cls.create(9, ex_observations, ex_actions, config)
        batch = make_update_batch(config.batch_size, obs_dim, action_dim, config.horizon_length)
        batch["expert_actions"] = batch["actions"].copy()

        agent, info = agent.update(batch)
        assert np.isfinite(float(info["critic/critic_loss"]))
        assert np.isfinite(float(info["critic/q_mean"]))
        if actor_name == "bc_flow":
            assert np.isfinite(float(info["actor/bc_flow_loss"]))
        else:
            assert np.isfinite(float(info["actor/bc_loss"]))

        q_heads = agent.evaluate_q_heads(
            jnp.zeros((1, obs_dim), dtype=jnp.float32),
            jnp.zeros((1, action_dim), dtype=jnp.float32),
        )
        q_value = agent.evaluate_q(
            jnp.zeros((1, obs_dim), dtype=jnp.float32),
            jnp.zeros((1, action_dim), dtype=jnp.float32),
            q_agg="min",
        )
        assert q_heads.shape == (2, 1)
        assert q_value.shape == (1,)


def smoke_bc_flow_chunk_valid_handling(obs_dim: int, action_dim: int) -> None:
    """Check chunked BCFlow accepts valid single-step labels and rejects mismatched chunks."""
    config = get_bc_flow_config()
    config.horizon_length = 1
    config.action_chunking = True
    config.actor_hidden_dims = (32, 32)
    config.batch_size = 4
    config.flow_steps = 2
    config.grad_clip_norm = None
    config.target_action_key = "expert_actions"

    ex_observations = jnp.zeros((config.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((config.batch_size, action_dim), dtype=jnp.float32)
    agent = BCFlowAgent.create(2, ex_observations, ex_actions, config)
    single_step_batch = {
        "observations": np.zeros((config.batch_size, obs_dim), dtype=np.float32),
        "expert_actions": np.full((config.batch_size, action_dim), 0.1, dtype=np.float32),
    }
    agent, info = agent.update(single_step_batch)
    assert np.isfinite(float(info["actor/bc_flow_loss"]))
    assert float(info["actor/flow_valid_fraction"]) == 1.0

    config = get_bc_flow_config()
    config.horizon_length = 3
    config.action_chunking = True
    config.actor_hidden_dims = (32, 32)
    config.batch_size = 4
    config.flow_steps = 2
    config.grad_clip_norm = None
    config.target_action_key = "expert_actions"
    agent = BCFlowAgent.create(3, ex_observations, ex_actions, config)
    sequence_batch = {
        "observations": np.zeros((config.batch_size, obs_dim), dtype=np.float32),
        "expert_actions": np.full(
            (config.batch_size, config.horizon_length, action_dim),
            0.1,
            dtype=np.float32,
        ),
    }
    agent, info = agent.update(sequence_batch)
    assert np.isfinite(float(info["actor/bc_flow_loss"]))
    assert float(info["actor/flow_valid_fraction"]) == 1.0

    bad_batch = {
        "observations": np.zeros((config.batch_size, obs_dim), dtype=np.float32),
        "expert_actions": np.full((config.batch_size, action_dim), 0.1, dtype=np.float32),
    }
    try:
        agent.update(bad_batch)
    except ValueError as exc:
        assert "action_chunking=True" in str(exc)
    else:
        raise AssertionError("chunked BCFlow should reject primitive single-step targets for horizon > 1")


def smoke_config_sequence_length_semantics() -> None:
    """Check public replay sampling sequence length does not overwrite actor chunk horizon."""
    recipe = new_schema_to_legacy_recipe(
        {
            "experiment": {"name": "smoke", "seed": 0},
            "env": {"kind": "robomimic", "name": "square-mh-low_dim"},
            "actors": {
                "learner": {
                    "kind": "bc_flow",
                    "network": {"horizon_length": 5, "action_chunking": True},
                },
            },
            "training": {"total_steps": 1},
            "replay": {"sampling": {"bc": {"source": "online", "sequence_length": 1}}},
            "intervention": {"enabled": False, "gate": {"kind": "always_off"}},
        }
    )
    assert recipe["learner"]["config"]["horizon_length"] == 5
    assert recipe["updates"][0]["sequence_length"] == 1
    assert "horizon_length" not in recipe["updates"][0]


def smoke_expert_query_semantics() -> None:
    """Check intervention configs query expert proposals by default."""

    def make_public_config(intervention):
        return {
            "experiment": {"name": "smoke", "seed": 0},
            "env": {"kind": "robomimic", "name": "square-mh-low_dim"},
            "actors": {
                "learner": {"kind": "bc_flow"},
                "expert": {"kind": "rlpd", "pretrained_path": "dummy"},
            },
            "training": {"total_steps": 1},
            "replay": {"sampling": {"bc": {"source": "online"}}},
            "intervention": intervention,
        }

    recipe = new_schema_to_legacy_recipe(
        make_public_config({"enabled": True, "gate": {"kind": "expert_q_gap", "threshold": 0.5}})
    )
    assert recipe["rollout"]["expert_query"] == "always"
    assert recipe["rollout"]["sample_expert"] is True

    recipe = new_schema_to_legacy_recipe(
        make_public_config(
            {"enabled": True, "expert_query": "never", "gate": {"kind": "expert_q_gap", "threshold": 0.5}}
        )
    )
    assert recipe["rollout"]["expert_query"] == "never"
    assert recipe["rollout"]["sample_expert"] is False


class NonFiniteQAgent:
    config = {"action_chunking": False, "horizon_length": 1}

    def evaluate_q(self, observations, actions, *, q_agg="min"):
        del observations, actions, q_agg
        return jnp.asarray([jnp.nan], dtype=jnp.float32)


def smoke_expert_q_gap_nonfinite_fails() -> None:
    """Check missing/invalid proposals do not silently become no-intervention."""
    gate = ExpertQGapGate(threshold=0.5, intervention_prob=1.0)
    learner = PolicyOutput(action=np.zeros(2, dtype=np.float32))
    expert = PolicyOutput(action=np.full(2, np.nan, dtype=np.float32), info={"missing": "expert_not_sampled"})
    try:
        gate.decide(
            step=0,
            observation=np.zeros(5, dtype=np.float32),
            learner=learner,
            expert=expert,
            rng=np.random.default_rng(0),
            expert_agent=NonFiniteQAgent(),
            action_dim=2,
        )
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("expert_q_gap should fail on non-finite Q values")

def smoke_metric_logger_routing_aggregation() -> None:
    """Check averages, routing interval sums, and routing total counters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricLogger(
            run_dir=Path(tmpdir),
            config={"run": {}},
            stdout_interval=0,
            jsonl_enabled=True,
            csv_enabled=True,
            wandb_enabled=False,
        )
        logger.record(
            {
                "loss": 1.0,
                "routing/demo_added": 0.0,
                "routing/demo_added_total": 0.0,
                "gate/expert_execute_steps": 1.0,
                "gate/expert_execute_steps_total": 1.0,
                "gate/intervention_started_count": 1.0,
                "gate/intervention_started_total": 1.0,
            },
            step=1,
        )
        logger.record(
            {
                "loss": 3.0,
                "routing/demo_added": 2.0,
                "routing/demo_added_total": 2.0,
                "gate/expert_execute_steps": 0.0,
                "gate/expert_execute_steps_total": 1.0,
                "gate/intervention_started_count": 0.0,
                "gate/intervention_started_total": 1.0,
            },
            step=2,
        )
        logger.record(
            {
                "loss": 5.0,
                "routing/demo_added": 0.0,
                "routing/demo_added_total": 2.0,
                "gate/expert_execute_steps": 1.0,
                "gate/expert_execute_steps_total": 2.0,
                "gate/intervention_started_count": 0.0,
                "gate/intervention_started_total": 1.0,
            },
            step=3,
            force_flush=True,
        )
        logger.close()
        rows = [json.loads(line) for line in (Path(tmpdir) / "metrics.jsonl").read_text().splitlines()]

    assert len(rows) == 1
    assert rows[0]["loss"] == 3.0
    assert rows[0]["routing/demo_added"] == 2.0
    assert rows[0]["routing/demo_added_total"] == 2.0
    assert rows[0]["gate/expert_execute_steps"] == 2.0
    assert rows[0]["gate/expert_execute_steps_total"] == 2.0
    assert rows[0]["gate/intervention_started_count"] == 1.0
    assert rows[0]["gate/intervention_started_total"] == 1.0

def smoke_bc_flow_policy_checkpoint(obs_dim: int, action_dim: int) -> None:
    """Check source-agnostic BC flow checkpoint restore through the policy adapter."""
    config = get_bc_flow_config()
    config.horizon_length = 1
    config.action_chunking = False
    config.actor_hidden_dims = (32, 32)
    config.batch_size = 4
    config.flow_steps = 2
    config.grad_clip_norm = None

    ex_observations = jnp.zeros((config.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((config.batch_size, action_dim), dtype=jnp.float32)
    agent = BCFlowAgent.create(0, ex_observations, ex_actions, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_agent(agent, tmpdir, 0)
        policy = BCFlowPolicy.from_checkpoint(
            Path(tmpdir) / "params_0.pkl",
            config=config.to_dict(),
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=0,
        )
        output = policy.sample_action(
            np.zeros(obs_dim, dtype=np.float32),
            rng=jax.random.PRNGKey(5),
        )

    assert output.action.shape == (action_dim,)
    assert np.isnan(output.log_prob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-dim", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=2)
    args = parser.parse_args()

    print(f"jax={jax.__version__}")
    print(f"devices={jax.devices()}")
    smoke_reward_transform()
    print("reward transform smoke ok")
    smoke_gate_and_replay()
    print("gate/replay smoke ok")
    smoke_base_action_chunk_queue()
    print("base action chunk queue smoke ok")
    smoke_residual_base_noise_warmup()
    print("residual base+noise warmup smoke ok")
    smoke_config_sequence_length_semantics()
    print("config sequence length semantics smoke ok")
    smoke_expert_query_semantics()
    print("expert query semantics smoke ok")
    smoke_expert_q_gap_nonfinite_fails()
    print("expert q-gap non-finite smoke ok")
    smoke_metric_logger_routing_aggregation()
    print("metric logger routing aggregation smoke ok")
    smoke_intervention_routing()
    print("intervention routing smoke ok")
    smoke_demo_episode_replacement()
    print("demo episode replacement smoke ok")
    smoke_mixed_replay_sampling()
    print("mixed replay sampling smoke ok")
    smoke_replay_prefill()
    print("replay prefill smoke ok")
    smoke_residual_base_action_cache()
    print("residual base action cache smoke ok")
    smoke_rlpd(args.obs_dim, args.action_dim)
    print("rlpd smoke ok")
    smoke_rlpd_td_sequence_length(args.obs_dim, args.action_dim)
    print("rlpd td sequence length smoke ok")
    smoke_residual_transition_schema(args.obs_dim, args.action_dim)
    print("residual transition schema smoke ok")
    smoke_residual_rlpd(args.obs_dim, args.action_dim)
    print("residual rlpd smoke ok")
    smoke_residual_td3(args.obs_dim, args.action_dim)
    print("residual td3 smoke ok")
    smoke_residual_rlpd_with_bc_aux(args.obs_dim, args.action_dim)
    print("residual rlpd bc aux smoke ok")
    smoke_residual_metadata_validation(args.obs_dim, args.action_dim)
    print("residual metadata validation smoke ok")
    smoke_rlpd_policy_checkpoint(args.obs_dim, args.action_dim)
    print("rlpd policy checkpoint smoke ok")
    smoke_bc_mlp(args.obs_dim, args.action_dim)
    print("bc mlp smoke ok")
    smoke_bc_flow(args.obs_dim, args.action_dim)
    print("bc flow smoke ok")
    smoke_bc_agents_with_aux_critic(args.obs_dim, args.action_dim)
    print("bc agents auxiliary critic smoke ok")
    smoke_bc_flow_chunk_valid_handling(args.obs_dim, args.action_dim)
    print("bc flow chunk valid handling smoke ok")
    smoke_bc_flow_policy_checkpoint(args.obs_dim, args.action_dim)
    print("bc flow policy checkpoint smoke ok")


if __name__ == "__main__":
    main()
