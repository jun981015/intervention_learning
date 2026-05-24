from __future__ import annotations

import copy
from functools import partial

import jax
import jax.numpy as jnp

from il.networks import Ensemble, MLP, StateActionValue


TARGET_NUM_QS = 2


def critic_enabled(config) -> bool:
    """Return whether a BC agent should instantiate and train an auxiliary critic."""
    return bool(config.get("train_critic", False))


def make_critic_network_defs(config):
    """Create online and target critic definitions using the RLPD critic architecture."""
    num_qs = int(config.get("num_qs", TARGET_NUM_QS))
    if num_qs < TARGET_NUM_QS:
        raise ValueError(f"num_qs must be >= {TARGET_NUM_QS} when train_critic=True.")

    critic_base_cls = partial(
        MLP,
        hidden_dims=config["value_hidden_dims"],
        activate_final=True,
        use_layer_norm=config["layer_norm"],
    )
    critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
    critic_def = Ensemble(critic_cls, num=num_qs)
    return critic_def, copy.deepcopy(critic_def)


def aggregate_q_heads(q_heads, q_agg: str):
    """Aggregate ensemble Q heads into one scalar per batch item."""
    if q_agg == "min":
        return q_heads.min(axis=0)
    if q_agg == "mean":
        return q_heads.mean(axis=0)
    if q_agg == "max":
        return q_heads.max(axis=0)
    raise ValueError(f"Unsupported q_agg: {q_agg}")


def aggregate_target_qs(target_qs, config):
    """Aggregate the first target critics for TD backup computation."""
    return aggregate_q_heads(target_qs[:TARGET_NUM_QS], config.get("target_q_agg", "min"))


def select_critic_actions(batch, config):
    """Return environment actions in the critic input shape."""
    actions = jnp.asarray(batch["actions"])
    action_dim = int(config["action_dim"])
    horizon = int(config["horizon_length"])

    if actions.ndim == 2:
        expected_dim = action_dim * horizon if config["action_chunking"] else action_dim
        if actions.shape[-1] != expected_dim:
            raise ValueError(f"BC critic expected action dim {expected_dim}, got shape {actions.shape}.")
        return actions

    if actions.ndim != 3:
        raise ValueError(f"BC critic expected actions with shape [B, A] or [B, H, A], got {actions.shape}.")
    if actions.shape[-1] != action_dim:
        raise ValueError(f"BC critic expected primitive action dim {action_dim}, got shape {actions.shape}.")

    if config["action_chunking"]:
        if actions.shape[1] != horizon:
            raise ValueError(f"BC critic expected horizon {horizon}, got action shape {actions.shape}.")
        return jnp.reshape(actions, (actions.shape[0], horizon * action_dim))
    return actions[:, 0, :]


def select_last_next_observations(batch):
    """Return the bootstrap observations at the final sampled sequence step."""
    if "next_observations" not in batch:
        raise KeyError("train_critic=True requires next_observations in the update batch.")
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[:, -1, ...], batch["next_observations"])


def sequence_last(batch, key: str, *, default=None):
    """Return the last sequence value for a scalar batch field."""
    if key not in batch:
        if default is None:
            raise KeyError(f"train_critic=True requires {key!r} in the update batch.")
        return default
    value = jnp.asarray(batch[key])
    if value.ndim == 1:
        return value
    return value[:, -1]


def critic_td_loss(network, config, batch, grad_params, next_actions):
    """Compute an auxiliary TD critic loss without using Q to update the actor."""
    batch_actions = select_critic_actions(batch, config)
    next_observations = select_last_next_observations(batch)

    target_qs = network.select("target_critic")(next_observations, next_actions)
    next_q = aggregate_target_qs(target_qs, config)

    rewards = sequence_last(batch, "rewards")
    masks = sequence_last(batch, "masks")
    valid = sequence_last(batch, "valid", default=jnp.ones_like(rewards))
    target_q = rewards + (float(config["discount"]) ** int(config["horizon_length"])) * masks * next_q
    target_q = jax.lax.stop_gradient(target_q)

    q = network.select("critic")(batch["observations"], batch_actions, params=grad_params)
    td_error = q - target_q
    squared_error = jnp.square(td_error) * valid
    normalizer = jnp.maximum(jnp.sum(valid) * q.shape[0], 1.0)
    critic_loss = jnp.sum(squared_error) / normalizer

    return critic_loss, {
        "critic_loss": critic_loss,
        "q_mean": q.mean(),
        "q_max": q.max(),
        "q_min": q.min(),
        "target_q_mean": target_q.mean(),
        "td_error_abs_mean": jnp.mean(jnp.abs(td_error)),
        "valid_fraction": jnp.mean(valid),
    }


def polyak_update_target_critic(network, config):
    """Polyak-average online critic parameters into target critic parameters."""
    tau = float(config["tau"])
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1.0 - tau),
        network.params["modules_critic"],
        network.params["modules_target_critic"],
    )
    network.params["modules_target_critic"] = new_target_params
