from __future__ import annotations

"""Canonical replay transition schema for intervention rollouts."""

import numpy as np

from il.utils.types import ControllerId, StepRecord


def nan_like_action(action: np.ndarray) -> np.ndarray:
    """Return an action-shaped float array for missing policy outputs."""
    return np.full_like(np.asarray(action, dtype=np.float32), np.nan, dtype=np.float32)


def tree_asarray(value):
    """Convert arrays inside a nested dict observation to numpy arrays."""
    if isinstance(value, dict):
        return {key: tree_asarray(item) for key, item in value.items()}
    return np.asarray(value)


def step_record_to_transition(record: StepRecord) -> dict:
    """Convert one rollout step into the canonical replay transition schema.

    `terminals` tracks any episode boundary, including timeout/truncation.
    `masks` tracks true environment termination only, so timeout handling can
    still bootstrap from `next_observations` when the learner uses it.
    """
    controller_id = int(record.decision.controller_id)
    return {
        "observations": tree_asarray(record.observation),
        "actions": np.asarray(record.action, dtype=np.float32),
        "learner_actions": np.asarray(record.learner.action, dtype=np.float32),
        "expert_actions": np.asarray(record.expert.action, dtype=np.float32),
        "rewards": np.asarray(record.reward, dtype=np.float32),
        "terminals": np.asarray(float(record.done), dtype=np.float32),
        "masks": np.asarray(1.0 - float(record.terminated), dtype=np.float32),
        "next_observations": tree_asarray(record.next_observation),
        "controller_ids": np.asarray(controller_id, dtype=np.int8),
        "episode_ids": np.asarray(record.episode_id, dtype=np.int64),
        "episode_steps": np.asarray(record.episode_step, dtype=np.int32),
        "gating_reasons": np.asarray(int(record.decision.reason), dtype=np.int16),
        "gating_scores": np.asarray(record.decision.score, dtype=np.float32),
        "learner_action_log_probs": np.asarray(record.learner.log_prob, dtype=np.float32),
        "expert_action_log_probs": np.asarray(record.expert.log_prob, dtype=np.float32),
        "interventions": np.asarray(int(record.decision.controller_id == ControllerId.EXPERT), dtype=np.int8),
    }


def make_replay_example(observation: np.ndarray, action: np.ndarray) -> dict:
    """Create a scalar example transition for replay-buffer initialization.

    The example fixes key names, shapes, and dtypes for all later inserts.
    """
    action = np.asarray(action, dtype=np.float32)
    return {
        "observations": tree_asarray(observation),
        "actions": action,
        "learner_actions": action.copy(),
        "expert_actions": action.copy(),
        "rewards": np.asarray(0.0, dtype=np.float32),
        "terminals": np.asarray(0.0, dtype=np.float32),
        "masks": np.asarray(1.0, dtype=np.float32),
        "next_observations": tree_asarray(observation),
        "controller_ids": np.asarray(0, dtype=np.int8),
        "episode_ids": np.asarray(-1, dtype=np.int64),
        "episode_steps": np.asarray(-1, dtype=np.int32),
        "gating_reasons": np.asarray(0, dtype=np.int16),
        "gating_scores": np.asarray(0.0, dtype=np.float32),
        "learner_action_log_probs": np.asarray(np.nan, dtype=np.float32),
        "expert_action_log_probs": np.asarray(np.nan, dtype=np.float32),
        "interventions": np.asarray(0, dtype=np.int8),
    }
