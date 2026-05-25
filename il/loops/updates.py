from __future__ import annotations

"""Configured update execution for trainable actors."""

from typing import Any

import jax
import numpy as np

from il.buffers.mixed import MixedReplaySampler, MixedSamplingSpec
from il.builders.types import ActorBundle, TrainContext
from il.utils.config import TrainingConfig


def _tree_to_float_dict(tree: dict) -> dict[str, float]:
    """Convert scalar metric leaves to plain Python floats."""
    out = {}
    for key, value in tree.items():
        arr = np.asarray(value)
        if arr.size == 1:
            out[str(key)] = float(arr.reshape(()))
    return out


def _select_update_source(context: TrainContext, spec: dict[str, Any], train_cfg: TrainingConfig):
    """Return the replay source requested by an update spec."""
    if train_cfg.sampling_fractions is not None:
        return MixedReplaySampler(context.buffers, MixedSamplingSpec(train_cfg.sampling_fractions))
    return context.buffers.get(spec.get("source", "online"))


def _make_update_config(context: TrainContext, target: ActorBundle, spec: dict[str, Any]) -> TrainingConfig:
    """Resolve update sampling knobs from recipe + target actor config.

    `sequence_length` is the replay/TD horizon. It is intentionally separate
    from an actor's `horizon_length`, which may mean action chunk length.
    """
    sequence_length = spec.get("sequence_length", spec.get("horizon_length", target.config.get("td_n_step", 1)))
    return TrainingConfig(
        batch_size=int(spec.get("batch_size", context.config["train"]["batch_size"])),
        utd_ratio=int(spec.get("utd_ratio", spec.get("utd", 1))),
        horizon_length=int(sequence_length),
        discount=float(spec.get("discount", target.config.get("discount", 0.99))),
        sampling_fractions=spec.get("sampling_fractions"),
    )


def _sample_update_batch(source, train_cfg: TrainingConfig) -> dict:
    """Sample a sequence batch and stack it for optional UTD updates."""
    batch = source.sample_sequence(
        train_cfg.batch_size * train_cfg.utd_ratio,
        sequence_length=train_cfg.horizon_length,
        discount=train_cfg.discount,
    )
    if train_cfg.utd_ratio == 1:
        return batch
    return jax.tree_util.tree_map(
        lambda x: x.reshape((train_cfg.utd_ratio, train_cfg.batch_size) + x.shape[1:]),
        batch,
    )


def _target_bundle(context: TrainContext, name: str) -> ActorBundle:
    """Return the actor bundle named by an update spec."""
    if name == "learner":
        return context.learner
    if name == "expert" and context.expert is not None:
        return context.expert
    raise KeyError(f"Unknown update target {name!r}.")


def _set_bundle_agent(bundle: ActorBundle, agent) -> None:
    """Keep the trainable agent and its policy view synchronized."""
    bundle.agent = agent
    if bundle.policy is not None and hasattr(bundle.policy, "agent"):
        bundle.policy.agent = agent


def _assert_finite_target_actions(batch: dict, key: str, *, update_name: str) -> None:
    """Fail fast if an update target action key contains missing labels."""
    if key not in batch:
        raise KeyError(f"target_action_key {key!r} is missing from sampled batch for {update_name!r}.")
    targets = np.asarray(batch[key])
    finite = np.isfinite(targets)
    if bool(finite.all()):
        return
    bad_count = int(targets.size - finite.sum())
    raise ValueError(
        f"target_action_key {key!r} for update {update_name!r} contains "
        f"{bad_count}/{targets.size} NaN or Inf values. This usually means the requested "
        "action labels were not queried or stored before training."
    )


def _assert_finite_batch_key(batch: dict, key: str, *, update_name: str, purpose: str) -> None:
    """Fail fast if a required residual metadata array is missing or non-finite."""
    if key not in batch:
        raise KeyError(f"{purpose} key {key!r} is missing from sampled batch for {update_name!r}.")
    values = np.asarray(batch[key])
    finite = np.isfinite(values)
    if bool(finite.all()):
        return
    bad_count = int(values.size - finite.sum())
    raise ValueError(
        f"{purpose} key {key!r} for update {update_name!r} contains "
        f"{bad_count}/{values.size} NaN or Inf values. Residual RLPD requires base policy "
        "actions to be stored before this batch can be used."
    )


def _assert_residual_metadata(target: ActorBundle, spec: dict[str, Any], batch: dict) -> None:
    """Validate residual RLPD metadata before entering JAX update code."""
    if not bool(target.config.get("residual_policy", False)):
        return

    update_name = spec.get("name") or target.name
    _assert_finite_batch_key(batch, "base_actions", update_name=update_name, purpose="residual current base action")
    _assert_finite_batch_key(batch, "next_base_actions", update_name=update_name, purpose="residual next base action")

    if float(target.config.get("bc_alpha", 0.0)) == 0.0:
        return
    if "bc_actions" in batch and "bc_base_actions" not in batch:
        raise KeyError(
            f"Residual BC update {update_name!r} received auxiliary bc_actions without bc_base_actions. "
            "Prefill/cache base policy actions for demo data before enabling residual BC."
        )
    bc_base_key = "bc_base_actions" if "bc_base_actions" in batch else "base_actions"
    _assert_finite_batch_key(batch, bc_base_key, update_name=update_name, purpose="residual BC base action")


def _prepare_target_action_batch(target: ActorBundle, spec: dict[str, Any], batch: dict) -> dict:
    """Alias requested action labels to the target actor's configured label key and validate them."""
    requested_key = spec.get("target_action_key")
    configured_key = target.config.get("target_action_key")
    update_name = spec.get("name") or target.name
    target_key = requested_key or configured_key
    if target_key is not None:
        _assert_finite_target_actions(batch, target_key, update_name=update_name)
    if requested_key and configured_key and requested_key != configured_key:
        batch = dict(batch)
        batch[configured_key] = batch[requested_key]
    return batch


def _add_aux_batches(context: TrainContext, target: ActorBundle, spec: dict[str, Any], batch: dict, train_cfg: TrainingConfig) -> dict:
    """Sample optional auxiliary batches and attach them with a name prefix."""
    aux_batches = spec.get("aux_batches") or {}
    if not aux_batches:
        return batch
    batch = dict(batch)
    for aux_name, aux_spec in aux_batches.items():
        aux_spec = dict(aux_spec)
        aux_spec.setdefault("utd_ratio", train_cfg.utd_ratio)
        aux_cfg = _make_update_config(context, target, aux_spec)
        aux_source = _select_update_source(context, aux_spec, aux_cfg)
        aux_batch = _sample_update_batch(aux_source, aux_cfg)
        for key, value in aux_batch.items():
            batch[f"{aux_name}_{key}"] = value
    return batch


def run_update_spec(context: TrainContext, spec: dict[str, Any]) -> dict[str, float]:
    """Run one configured learner update and return scalar metrics."""
    target = _target_bundle(context, spec.get("target", "learner"))
    if target.agent is None:
        raise ValueError(f"Update target {target.name!r} has no trainable agent.")

    train_cfg = _make_update_config(context, target, spec)
    source = _select_update_source(context, spec, train_cfg)
    batch = _sample_update_batch(source, train_cfg)
    batch = _prepare_target_action_batch(target, spec, batch)
    batch = _add_aux_batches(context, target, spec, batch, train_cfg)
    _assert_residual_metadata(target, spec, batch)

    if train_cfg.utd_ratio > 1:
        agent, info = target.agent.batch_update(batch)
    else:
        agent, info = target.agent.update(batch)
    _set_bundle_agent(target, agent)
    prefix = spec.get("name") or target.name
    return {f"{prefix}/{key}": value for key, value in _tree_to_float_dict(info).items()}
