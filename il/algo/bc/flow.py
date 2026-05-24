from __future__ import annotations

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from il.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from il.networks.flow import ActorVectorField
from il.algo.bc.critic import (
    aggregate_q_heads,
    critic_enabled,
    critic_td_loss,
    make_critic_network_defs,
    polyak_update_target_critic,
    select_last_next_observations,
)


class BCFlowAgent(flax.struct.PyTreeNode):
    """Behavior-cloning flow-matching actor.

    This intentionally omits FQL/QC-FQL critic logic.  It is meant to be
    used as a BC policy component or future intervention-learning actor.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _critic_enabled(self) -> bool:
        """Return whether this BC actor also trains an auxiliary critic."""
        return critic_enabled(self.config)

    def _sample_next_actions_for_critic(self, batch, rng):
        """Sample bootstrap actions for critic targets without actor-loss coupling."""
        next_observations = select_last_next_observations(batch)
        return self.sample_actions(next_observations, rng=rng)

    def critic_loss(self, batch, grad_params, rng):
        """Compute auxiliary TD critic loss; actor loss remains pure BC."""
        next_actions = self._sample_next_actions_for_critic(batch, rng)
        return critic_td_loss(self.network, self.config, batch, grad_params, next_actions)

    @jax.jit
    def evaluate_q_heads(self, observations, actions):
        """Evaluate all auxiliary critic heads for diagnostics or gates."""
        if not self._critic_enabled():
            raise ValueError("BCFlowAgent was created with train_critic=False.")
        return self.network.select("critic")(observations, actions)

    def evaluate_q(self, observations, actions, *, q_agg: str = "min"):
        """Evaluate aggregated auxiliary Q values for arbitrary action proposals."""
        return aggregate_q_heads(self.evaluate_q_heads(observations, actions), q_agg)

    def _flatten_batch_actions(self, batch):
        """Return action targets in the actor output shape and validate chunk semantics."""
        target_key = self.config.get("target_action_key", "actions")
        actions = jnp.asarray(batch[target_key])
        action_dim = int(self.config["action_dim"])
        horizon = int(self.config["horizon_length"])
        if actions.ndim == 2:
            if self.config["action_chunking"]:
                full_action_dim = action_dim * horizon
                if actions.shape[-1] != full_action_dim:
                    raise ValueError(
                        "BCFlow action_chunking=True requires flat chunk targets "
                        f"with last dim {full_action_dim} or sequence targets "
                        f"[batch, {horizon}, {action_dim}], got shape {actions.shape}."
                    )
                return actions
            if actions.shape[-1] != action_dim:
                raise ValueError(
                    f"BCFlow expected primitive action dim {action_dim}, got shape {actions.shape}."
                )
            return actions
        if actions.ndim != 3:
            raise ValueError(
                "BCFlow action targets must have shape [batch, action_dim] "
                f"or [batch, horizon, action_dim], got shape {actions.shape}."
            )
        if actions.shape[-1] != action_dim:
            raise ValueError(f"BCFlow expected action dim {action_dim}, got shape {actions.shape}.")
        if self.config["action_chunking"]:
            if actions.shape[1] != horizon:
                raise ValueError(
                    f"BCFlow expected chunk horizon {horizon}, got target shape {actions.shape}."
                )
            return jnp.reshape(actions, (actions.shape[0], horizon * action_dim))
        return actions[:, 0, :]

    def _chunk_valid_mask(self, batch, *, batch_size: int, dtype):
        """Return a `[batch, horizon]` validity mask for chunked BC targets."""
        horizon = int(self.config["horizon_length"])
        valid = batch.get("valid")
        if valid is None:
            return jnp.ones((batch_size, horizon), dtype=dtype)
        valid = jnp.asarray(valid, dtype=dtype)
        if valid.ndim == 1:
            if horizon != 1:
                raise ValueError(
                    f"BCFlow valid mask must be [batch, {horizon}], got shape {valid.shape}."
                )
            valid = valid[:, None]
        if valid.ndim != 2 or valid.shape[1] != horizon:
            raise ValueError(f"BCFlow valid mask must be [batch, {horizon}], got shape {valid.shape}.")
        return valid

    def bc_flow_loss(self, batch, grad_params, rng):
        """Compute flow-matching behavior-cloning loss against dataset actions."""
        batch_actions = self._flatten_batch_actions(batch)
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_bc_flow")(batch["observations"], x_t, t, params=grad_params)

        if self.config["action_chunking"]:
            per_dim_loss = jnp.reshape(
                (pred - vel) ** 2,
                (batch_size, self.config["horizon_length"], self.config["action_dim"]),
            )
            valid = self._chunk_valid_mask(batch, batch_size=batch_size, dtype=per_dim_loss.dtype)
            masked_loss = per_dim_loss * valid[..., None]
            normalizer = jnp.maximum(jnp.sum(valid) * self.config["action_dim"], 1.0)
            flow_loss = jnp.sum(masked_loss) / normalizer
            valid_fraction = jnp.mean(valid)
        else:
            flow_loss = jnp.mean((pred - vel) ** 2)
            valid_fraction = jnp.asarray(1.0, dtype=flow_loss.dtype)

        return flow_loss, {
            "bc_flow_loss": flow_loss,
            "flow_pred_mean": pred.mean(),
            "flow_vel_mean": vel.mean(),
            "flow_valid_fraction": valid_fraction,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute BC flow loss plus optional critic diagnostics loss."""
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        actor_loss, actor_info = self.bc_flow_loss(batch, grad_params, actor_rng)
        info = {f"actor/{key}": value for key, value in actor_info.items()}
        loss = actor_loss

        if self._critic_enabled():
            critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
            info.update({f"critic/{key}": value for key, value in critic_info.items()})
            loss = loss + float(self.config["critic_loss_coef"]) * critic_loss

        return loss, info

    @staticmethod
    def _update(agent, batch):
        """Apply one flow-BC gradient update."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            """Closure passed to TrainState for differentiating flow-BC loss."""
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        if agent._critic_enabled():
            polyak_update_target_critic(new_network, agent.config)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """Run one JIT-compiled flow-BC update."""
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Run multiple flow-BC updates with `lax.scan` over a UTD batch."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        """Generate actions by integrating the learned flow or using one-step flow."""
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        noises = jax.random.normal(
            rng,
            (*observations.shape[: -len(self.config["ob_dims"])], action_dim),
        )
        if self.config["actor_type"] == "onestep":
            actions = self.network.select("actor_onestep_flow")(observations, noises)
        else:
            actions = self.compute_flow_actions(observations, noises)
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def sample_actions_with_log_prob(self, observations, rng=None):
        """Generate actions and return NaN log-probs because flow sampling is implicit."""
        actions = self.sample_actions(observations, rng=rng)
        log_probs = jnp.full(actions.shape[:-1], jnp.nan)
        return actions, log_probs

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        """Euler-integrate the actor vector field from Gaussian noise to actions."""
        actions = noises
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            vels = self.network.select("actor_bc_flow")(observations, actions, t)
            actions = actions + vels / self.config["flow_steps"]
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Initialize flow-matching actor modules from example observation/action shapes."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]
        ex_times = ex_actions[..., :1]

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            use_layer_norm=config["actor_layer_norm"],
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            use_layer_norm=config["actor_layer_norm"],
        )

        network_info = {
            "actor_bc_flow": (actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            "actor_onestep_flow": (actor_onestep_flow_def, (ex_observations, full_actions)),
        }
        if critic_enabled(config):
            critic_def, target_critic_def = make_critic_network_defs(config)
            network_info["critic"] = (critic_def, (ex_observations, full_actions))
            network_info["target_critic"] = (target_critic_def, (ex_observations, full_actions))
        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.0:
            tx = optax.adamw(learning_rate=config["lr"], weight_decay=config["weight_decay"])
        else:
            tx = optax.adam(learning_rate=config["lr"])
        params = network_def.init(init_rng, **network_args)["params"]
        if critic_enabled(config):
            params["modules_target_critic"] = params["modules_critic"]
        network = TrainState.create(network_def, params, tx=tx, grad_clip_norm=config["grad_clip_norm"])

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return default flow-BC actor hyperparameters."""
    return ml_collections.ConfigDict(
        dict(
            agent_name="bc_flow",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=True,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            flow_steps=10,
            actor_type="flow",
            target_action_key="actions",
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.0,
            grad_clip_norm=None,
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
