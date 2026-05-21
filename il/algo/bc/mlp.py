from __future__ import annotations

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from functools import partial

from il.distributions import TanhDeterministic
from il.networks import MLP
from il.utils.flax_utils import ModuleDict, TrainState, nonpytree_field


class BCMLPAgent(flax.struct.PyTreeNode):
    """Deterministic MLP behavior-cloning policy.

    This is intentionally simple: it does not implement an RL objective.  It
    trains a learner actor to match `expert_actions` collected at visited states.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _select_targets(self, batch):
        """Return the action labels used for BC."""
        targets = batch[self.config["target_action_key"]]
        if targets.ndim == 3:
            if self.config["action_chunking"]:
                return jnp.reshape(targets, (targets.shape[0], -1))
            return targets[:, 0, :]
        return targets

    def _select_weights(self, batch, targets):
        """Return optional per-sample BC weights."""
        weight_key = self.config["sample_weight_key"]
        if not weight_key or weight_key not in batch:
            return jnp.ones(targets.shape[0], dtype=targets.dtype)
        weights = batch[weight_key]
        if weights.ndim > 1:
            weights = weights.reshape((weights.shape[0], -1))[:, 0]
        return weights.astype(targets.dtype)

    def bc_loss(self, batch, grad_params):
        """Compute weighted MSE between actor actions and expert labels."""
        targets = self._select_targets(batch)
        weights = self._select_weights(batch, targets)
        pred_actions = self.network.select("actor")(batch["observations"], params=grad_params)
        squared_error = jnp.square(pred_actions - targets).mean(axis=-1)
        normalizer = jnp.maximum(weights.sum(), 1.0)
        bc_loss = (squared_error * weights).sum() / normalizer
        action_error = jnp.sqrt(jnp.maximum(squared_error, 0.0))
        return bc_loss, {
            "bc_loss": bc_loss,
            "action_rmse": jnp.mean(action_error),
            "pred_action_mean": pred_actions.mean(),
            "target_action_mean": targets.mean(),
            "weight_mean": weights.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute MLP BC loss and actor metrics."""
        del rng
        loss, info = self.bc_loss(batch, grad_params)
        return loss, {f"actor/{key}": value for key, value in info.items()}

    @staticmethod
    def _update(agent, batch):
        """Apply one MLP BC gradient update."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            """Closure passed to TrainState for differentiating BC loss."""
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """Run one JIT-compiled MLP BC update."""
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Run multiple MLP BC updates with `lax.scan` over a stacked batch."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        """Return deterministic tanh-bounded actor actions."""
        del rng
        return jnp.clip(self.network.select("actor")(observations), -1, 1)

    @jax.jit
    def sample_actions_with_log_prob(self, observations, rng=None):
        """Return deterministic actions and NaN log-probs."""
        actions = self.sample_actions(observations, rng=rng)
        log_probs = jnp.full(actions.shape[:-1], jnp.nan)
        return actions, log_probs

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Initialize the deterministic actor from example observation/action shapes."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        actor_base_cls = partial(
            MLP,
            hidden_dims=config["actor_hidden_dims"],
            activate_final=True,
            use_layer_norm=config["actor_layer_norm"],
        )
        actor_def = TanhDeterministic(actor_base_cls, full_action_dim)

        network_def = ModuleDict({"actor": actor_def})
        network_params = network_def.init(init_rng, actor=(ex_observations,))["params"]
        if config["weight_decay"] > 0.0:
            tx = optax.adamw(learning_rate=config["lr"], weight_decay=config["weight_decay"])
        else:
            tx = optax.adam(learning_rate=config["lr"])
        network = TrainState.create(network_def, network_params, tx=tx, grad_clip_norm=config["grad_clip_norm"])

        config["action_dim"] = action_dim
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return default MLP BC hyperparameters."""
    return ml_collections.ConfigDict(
        dict(
            agent_name="bc_mlp",
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=True,
            horizon_length=1,
            action_chunking=False,
            target_action_key="expert_actions",
            sample_weight_key="",
            weight_decay=0.0,
            grad_clip_norm=None,
            action_dim=ml_collections.config_dict.placeholder(int),
        )
    )
