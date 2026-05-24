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
from il.algo.bc.critic import (
    aggregate_q_heads,
    critic_enabled,
    critic_td_loss,
    make_critic_network_defs,
    polyak_update_target_critic,
    select_last_next_observations,
)


class BCMLPAgent(flax.struct.PyTreeNode):
    """Deterministic MLP behavior-cloning policy.

    This is intentionally simple: it does not implement an RL objective.  It
    trains a learner actor to match `expert_actions` collected at visited states.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _critic_enabled(self) -> bool:
        """Return whether this BC actor also trains an auxiliary critic."""
        return critic_enabled(self.config)

    def _sample_next_actions_for_critic(self, batch):
        """Compute deterministic bootstrap actions without actor-loss coupling."""
        next_observations = select_last_next_observations(batch)
        return self.sample_actions(next_observations)

    def critic_loss(self, batch, grad_params):
        """Compute auxiliary TD critic loss; actor loss remains pure BC."""
        next_actions = self._sample_next_actions_for_critic(batch)
        return critic_td_loss(self.network, self.config, batch, grad_params, next_actions)

    @jax.jit
    def evaluate_q_heads(self, observations, actions):
        """Evaluate all auxiliary critic heads for diagnostics or gates."""
        if not self._critic_enabled():
            raise ValueError("BCMLPAgent was created with train_critic=False.")
        return self.network.select("critic")(observations, actions)

    def evaluate_q(self, observations, actions, *, q_agg: str = "min"):
        """Evaluate aggregated auxiliary Q values for arbitrary action proposals."""
        return aggregate_q_heads(self.evaluate_q_heads(observations, actions), q_agg)

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
        """Compute MLP BC loss plus optional critic diagnostics loss."""
        del rng
        actor_loss, actor_info = self.bc_loss(batch, grad_params)
        info = {f"actor/{key}": value for key, value in actor_info.items()}
        loss = actor_loss

        if self._critic_enabled():
            critic_loss, critic_info = self.critic_loss(batch, grad_params)
            info.update({f"critic/{key}": value for key, value in critic_info.items()})
            loss = loss + float(self.config["critic_loss_coef"]) * critic_loss

        return loss, info

    @staticmethod
    def _update(agent, batch):
        """Apply one MLP BC gradient update."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            """Closure passed to TrainState for differentiating BC loss."""
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        if agent._critic_enabled():
            polyak_update_target_critic(new_network, agent.config)
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

        network_info = {"actor": (actor_def, (ex_observations,))}
        if critic_enabled(config):
            critic_def, target_critic_def = make_critic_network_defs(config)
            network_info["critic"] = (critic_def, (ex_observations, full_actions))
            network_info["target_critic"] = (target_critic_def, (ex_observations, full_actions))
        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)["params"]
        if critic_enabled(config):
            network_params["modules_target_critic"] = network_params["modules_critic"]
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
            train_critic=False,
            critic_loss_coef=1.0,
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            num_qs=2,
            target_q_agg="min",
        )
    )
