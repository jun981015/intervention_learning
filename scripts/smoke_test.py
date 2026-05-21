from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box, Dict

from il.algo.bc.flow import BCFlowAgent, get_config as get_bc_flow_config
from il.algo.bc.mlp import BCMLPAgent, get_config as get_bc_mlp_config
from il.algo.rl.rlpd import ACRLPDAgent, get_config as get_rlpd_config
from il.buffers.mixed import MixedReplaySampler, MixedSamplingSpec, ReplayBufferCollection
from il.buffers.replay_buffer import ReplayBuffer
from il.buffers.routing import route_episode_to_buffers
from il.buffers.schema import make_replay_example, step_record_to_transition
from il.builders.components import build_buffers, infer_env_spec
from il.gating.random_gate import RandomGate
from il.loops.online import choose_action
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


def smoke_gate_and_replay() -> None:
    observation = np.zeros(5, dtype=np.float32)
    learner = ConstantPolicy(np.full(2, -0.5, dtype=np.float32), log_prob=-1.0)
    expert = ConstantPolicy(np.full(2, 0.5, dtype=np.float32), log_prob=-0.5)
    gate = RandomGate(expert_probability=1.0)
    action, learner_output, expert_output, decision = choose_action(
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
        action, learner_output, expert_output, decision = choose_action(
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
    smoke_gate_and_replay()
    print("gate/replay smoke ok")
    smoke_intervention_routing()
    print("intervention routing smoke ok")
    smoke_demo_episode_replacement()
    print("demo episode replacement smoke ok")
    smoke_mixed_replay_sampling()
    print("mixed replay sampling smoke ok")
    smoke_replay_prefill()
    print("replay prefill smoke ok")
    smoke_rlpd(args.obs_dim, args.action_dim)
    print("rlpd smoke ok")
    smoke_rlpd_policy_checkpoint(args.obs_dim, args.action_dim)
    print("rlpd policy checkpoint smoke ok")
    smoke_bc_mlp(args.obs_dim, args.action_dim)
    print("bc mlp smoke ok")
    smoke_bc_flow(args.obs_dim, args.action_dim)
    print("bc flow smoke ok")
    smoke_bc_flow_policy_checkpoint(args.obs_dim, args.action_dim)
    print("bc flow policy checkpoint smoke ok")


if __name__ == "__main__":
    main()
