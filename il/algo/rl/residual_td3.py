from __future__ import annotations

"""TD3-style residual actor-critic.

This is closer to the original ResFiT implementation than `residual_rlpd`:
the residual actor is deterministic, actor updates are delayed, target policy
smoothing is used for TD targets, and there is no SAC entropy temperature.
"""

import copy
from functools import partial
from typing import Any, Type

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from il.algo.rl.residual_rlpd import ResidualRLPDAgent
from il.algo.rl.rlpd import TARGET_NUM_QS, get_config as get_rlpd_config
from il.networks import Ensemble, MLP, StateActionValue, default_init
from il.utils.flax_utils import ModuleDict, TrainState, nonpytree_field


def _output_init(scale: float):
    """Return output initialization, supporting exact zero init for ResFiT."""
    if scale == 0.0:
        return nn.initializers.zeros
    if scale == 1.0:
        return default_init()
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class DeterministicTanhActor(nn.Module):
    """Deterministic tanh actor returning raw residual actions in [-1, 1]."""

    base_cls: Type[nn.Module]
    action_dim: int
    final_fc_init_scale: float = 0.0

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> jnp.ndarray:
        x = self.base_cls()(inputs, *args, **kwargs)
        x = nn.Dense(
            self.action_dim,
            kernel_init=_output_init(float(self.final_fc_init_scale)),
            name="OutputDense",
        )(x)
        return jnp.tanh(x)


class ResidualTD3Agent(ResidualRLPDAgent):
    """ResFiT-style residual TD3 agent with deterministic residual actor."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @classmethod
    def actor_distribution_def(cls, actor_base_cls, action_dim, config):
        """Build the deterministic residual actor."""
        return DeterministicTanhActor(
            actor_base_cls,
            action_dim,
            final_fc_init_scale=float(config.get("actor_final_fc_init_scale", 0.0)),
        )

    def _target_actor_actions(self, next_observations, next_base_actions, rng):
        """Compute TD3 target actions with optional target policy smoothing."""
        raw_next_actions = self.network.select("target_actor")(next_observations)
        residual_scale = float(self.config.get("residual_scale", 1.0))
        residual_actions = residual_scale * raw_next_actions

        if bool(self.config.get("target_policy_noise", True)):
            noise_std = float(self.config.get("target_noise_std", 0.05))
            noise_clip = float(self.config.get("target_noise_clip", 0.3))
            noise = jax.random.normal(rng, residual_actions.shape) * noise_std
            noise = jnp.clip(noise, -noise_clip, noise_clip)
            residual_actions = residual_actions + noise

        return jnp.clip(jax.lax.stop_gradient(next_base_actions) + residual_actions, -1.0, 1.0)

    def critic_loss(self, batch, grad_params, rng):
        """Compute n-step TD3 critic loss using target actor smoothing."""
        batch_actions = batch["actions"][..., 0, :]
        observations = self._current_observations(batch)
        next_observations = self._next_observations(batch)
        next_base_actions = self._sequence_last_action(batch["next_base_actions"])

        next_actions = self._target_actor_actions(next_observations, next_base_actions, rng)
        next_qs = self.network.select("target_critic")(next_observations, next_actions)
        next_q = self.aggregate_target_qs(next_qs)

        td_n_step = batch["rewards"].shape[-1]
        target_q = batch["rewards"][..., -1] + (self.config["discount"] ** td_n_step) * batch["masks"][..., -1] * next_q

        q = self.network.select("critic")(observations, batch_actions, params=grad_params)
        td_error = q - target_q
        critic_loss = (jnp.square(td_error) * batch["valid"][..., -1]).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "target_q_mean": target_q.mean(),
            "td_error_abs_mean": jnp.abs(td_error).mean(),
            "td_n_step": jnp.asarray(td_n_step, dtype=jnp.float32),
        }

    def _actor_q(self, qs):
        """Aggregate critic heads for deterministic policy gradients."""
        q_agg = self.config.get("actor_q_agg", "mean")
        if q_agg == "mean":
            return jnp.mean(qs, axis=0)
        if q_agg == "min":
            return jnp.min(qs[:TARGET_NUM_QS], axis=0)
        if q_agg == "q1":
            return qs[0]
        raise ValueError(f"Unsupported actor_q_agg: {q_agg}")

    def actor_loss(self, batch, grad_params, rng):
        """Compute deterministic policy gradient plus optional residual BC."""
        del rng
        observations = self._current_observations(batch)
        raw_actions = self.network.select("actor")(observations, params=grad_params)
        base_actions = self._sequence_first_action(batch["base_actions"])
        actions_for_q = self._compose_residual_action(base_actions, raw_actions)

        qs = self.network.select("critic")(observations, actions_for_q)
        q = self._actor_q(qs)
        actor_loss_base = -q.mean()

        residual_scale = float(self.config.get("residual_scale", 1.0))
        residual_actions = residual_scale * raw_actions
        residual_l2 = jnp.mean(jnp.sum(jnp.square(residual_actions), axis=-1))
        actor_loss = actor_loss_base + float(self.config.get("residual_action_l2", 0.0)) * residual_l2

        bc_alpha = float(self.config.get("bc_alpha", 0.0))
        if bc_alpha != 0.0:
            bc_observations = batch.get("bc_observations", batch["observations"])
            bc_raw_actions = batch.get("bc_actions", batch["actions"])
            bc_base_raw = batch.get("bc_base_actions", batch["base_actions"])
            bc_actions = self._sequence_first_action(bc_raw_actions)
            bc_base_actions = self._sequence_first_action(bc_base_raw)
            bc_aug_observations = self._augment_residual_observations(bc_observations, bc_base_actions)
            bc_delta_targets = bc_actions - jax.lax.stop_gradient(bc_base_actions)
            bc_raw_pred = self.network.select("actor")(bc_aug_observations, params=grad_params)
            bc_delta_pred = residual_scale * bc_raw_pred
            bc_loss_unscaled = jnp.mean(jnp.sum(jnp.square(bc_delta_pred - bc_delta_targets), axis=-1))
        else:
            bc_loss_unscaled = jnp.asarray(0.0, dtype=actor_loss.dtype)
        bc_loss = bc_alpha * bc_loss_unscaled
        total_loss = actor_loss + bc_loss

        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "actor_loss_base": actor_loss_base,
            "bc_loss": bc_loss,
            "bc_loss_unscaled": bc_loss_unscaled,
            "q": q.mean(),
            "residual_l2": residual_l2,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, update_actor=1.0):
        """Compute critic loss and optionally apply actor loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for key, value in critic_info.items():
            info[f"critic/{key}"] = value

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        update_actor = jnp.asarray(update_actor, dtype=critic_loss.dtype)
        for key, value in actor_info.items():
            info[f"actor/{key}"] = value
        info["actor/update_actor"] = update_actor

        return critic_loss + update_actor * actor_loss, info

    def _conditional_target_update(self, network, module_name, should_update):
        """Soft-update a target module only when `should_update` is true."""
        should_update = jnp.asarray(should_update, dtype=bool)
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: jnp.where(
                should_update,
                p * self.config["tau"] + tp * (1 - self.config["tau"]),
                tp,
            ),
            network.params[f"modules_{module_name}"],
            network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @staticmethod
    def _update(agent, batch, update_actor):
        """Apply one TD3 update and update target networks."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng, update_actor=update_actor)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        agent._conditional_target_update(new_network, "actor", update_actor)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """Run one full TD3 update."""
        return self._update(self, batch, jnp.asarray(1.0, dtype=jnp.float32))

    @jax.jit
    def batch_update(self, batch):
        """Run UTD updates with delayed actor updates inside the scan."""
        actor_interval = int(self.config.get("actor_update_interval", 2))

        def scan_update(carry, xs):
            agent, index = carry
            update_actor = ((index + 1) % actor_interval == 0).astype(jnp.float32)
            agent, info = ResidualTD3Agent._update(agent, xs, update_actor)
            return (agent, index + 1), info

        (agent, _), infos = jax.lax.scan(scan_update, (self, jnp.asarray(0)), batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        """Sample TD3 rollout actions as deterministic actor plus exploration noise."""
        raw_actions = self.network.select("actor")(observations)
        noise_std = float(self.config.get("exploration_noise", 0.0))
        if rng is not None and noise_std > 0:
            noise = jax.random.normal(rng, raw_actions.shape) * noise_std
            noise_clip = float(self.config.get("exploration_noise_clip", 0.3))
            raw_actions = raw_actions + jnp.clip(noise, -noise_clip, noise_clip)
        return jnp.clip(raw_actions, -1.0, 1.0)

    @jax.jit
    def sample_actions_with_log_prob(self, observations, rng=None):
        """Return deterministic actions with undefined log-probabilities."""
        actions = self.sample_actions(observations, rng=rng)
        log_probs = jnp.full(actions.shape[:-1], jnp.nan, dtype=actions.dtype)
        return actions, log_probs

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Initialize deterministic actor, target actor, critic, and target critic."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            raise NotImplementedError("residual_td3 v0 supports primitive actions only; set action_chunking=False.")
        if config["num_qs"] < TARGET_NUM_QS:
            raise ValueError(f"num_qs must be >= {TARGET_NUM_QS}.")

        critic_base_cls = partial(
            MLP,
            hidden_dims=config["value_hidden_dims"],
            activate_final=True,
            use_layer_norm=config["layer_norm"],
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=config["num_qs"])

        actor_base_cls = partial(
            MLP,
            hidden_dims=config["actor_hidden_dims"],
            activate_final=True,
            use_layer_norm=config["actor_layer_norm"],
        )
        actor_def = cls.actor_distribution_def(actor_base_cls, action_dim, config)

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations,)),
        )
        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)["params"]
        network_params["modules_target_critic"] = network_params["modules_critic"]
        network_params["modules_target_actor"] = network_params["modules_actor"]

        labels = {}
        for key, value in network_params.items():
            if key == "modules_actor":
                label = "actor"
            elif key == "modules_critic":
                label = "critic"
            else:
                label = "target"
            labels[key] = jax.tree_util.tree_map(lambda _: label, value)
        tx = optax.multi_transform(
            {
                "actor": optax.adam(learning_rate=float(config.get("actor_lr", config["lr"]))),
                "critic": optax.adam(learning_rate=float(config.get("critic_lr", config["lr"]))),
                "target": optax.set_to_zero(),
            },
            labels,
        )
        network = TrainState.create(network_def, network_params, tx=tx, grad_clip_norm=config["grad_clip_norm"])
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return default residual TD3 hyperparameters."""
    config = get_rlpd_config()
    config.agent_name = "residual_td3"
    config.residual_policy = True
    config.residual_scale = 0.2
    config.residual_action_l2 = 0.0
    config.critic_warmup_steps = 10_000
    config.actor_update_interval = 4
    config.actor_final_fc_init_scale = 0.0
    config.actor_lr = 1e-6
    config.critic_lr = 1e-4
    config.bc_alpha = 0.0
    config.actor_q_agg = "mean"
    config.target_policy_noise = True
    config.target_noise_std = 0.05
    config.target_noise_clip = 0.3
    config.exploration_noise = 0.05
    config.exploration_noise_clip = 0.3
    config.base_obs_dim = ml_collections.config_dict.placeholder(int)
    config.target_entropy = None
    config.action_chunking = False
    return config
