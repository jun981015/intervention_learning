from __future__ import annotations

"""ResFiT-style residual RLPD agent.

This class intentionally keeps residual-specific behavior out of the plain
ACRLPDAgent. The frozen base policy is handled by rollout/builders; this agent
only learns the residual actor-critic update once base actions are present in
the replay batch.
"""

import copy
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from il.algo.rl.rlpd import ACRLPDAgent, TARGET_NUM_QS, Temperature, get_config as get_rlpd_config
from il.distributions import TanhNormal
from il.networks import Ensemble, MLP, StateActionValue
from il.utils.flax_utils import ModuleDict, TrainState


class ResidualRLPDAgent(ACRLPDAgent):
    """RLPD/SAC-style residual actor-critic for ResFiT expert training."""

    @classmethod
    def actor_distribution_def(cls, actor_base_cls, action_dim, config):
        """Build a residual actor with a small output-head initialization."""
        return TanhNormal(
            actor_base_cls,
            action_dim,
            final_fc_init_scale=float(config.get("actor_final_fc_init_scale", 1e-2)),
        )

    def _sequence_first_action(self, value):
        """Return the first primitive action from `[..., H, A]` or `[..., A]`."""
        value = jnp.asarray(value)
        return value[..., 0, :] if value.ndim >= 3 else value

    def _sequence_last_action(self, value):
        """Return the last primitive action from `[..., H, A]` or `[..., A]`."""
        value = jnp.asarray(value)
        return value[..., -1, :] if value.ndim >= 3 else value

    @staticmethod
    def _augment_residual_observations(observations, base_actions):
        """Append stop-gradient base actions for residual actor inputs."""
        return jnp.concatenate(
            [jnp.asarray(observations), jax.lax.stop_gradient(jnp.asarray(base_actions))],
            axis=-1,
        )

    def _current_actor_observations(self, batch):
        """Return current actor observations with base actions appended."""
        base_actions = self._sequence_first_action(batch["base_actions"])
        return self._augment_residual_observations(batch["observations"], base_actions)

    def _next_actor_observations(self, batch):
        """Return bootstrap actor observations with next base actions appended."""
        next_observations = batch["next_observations"][..., -1, :]
        next_base_actions = self._sequence_last_action(batch["next_base_actions"])
        return self._augment_residual_observations(next_observations, next_base_actions)

    def _current_critic_observations(self, batch):
        """Return current critic observations without residual actor features."""
        return batch["observations"]

    def _next_critic_observations(self, batch):
        """Return bootstrap critic observations without residual actor features."""
        return batch["next_observations"][..., -1, :]

    def _compose_residual_action(self, base_actions, raw_residual_actions):
        """Convert raw residual actor output into the executed action space."""
        residual_scale = float(self.config.get("residual_scale", 1.0))
        return jnp.clip(
            jax.lax.stop_gradient(base_actions) + residual_scale * raw_residual_actions,
            -1.0,
            1.0,
        )

    def critic_loss(self, batch, grad_params, rng):
        """Compute TD loss where Q learns the combined executed action."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)
        observations = self._current_critic_observations(batch)
        next_observations = self._next_critic_observations(batch)
        next_actor_observations = self._next_actor_observations(batch)

        next_dist = self.network.select("actor")(next_actor_observations)
        raw_next_actions = next_dist.sample(seed=sample_rng)
        next_base_actions = self._sequence_last_action(batch["next_base_actions"])
        next_actions = self._compose_residual_action(next_base_actions, raw_next_actions)

        next_qs = self.network.select("target_critic")(next_observations, next_actions)
        next_q = self.aggregate_target_qs(next_qs)

        td_n_step = batch["rewards"].shape[-1]
        target_q = batch["rewards"][..., -1] + (self.config["discount"] ** td_n_step) * batch["masks"][..., -1] * next_q

        q = self.network.select("critic")(observations, batch_actions, params=grad_params)
        valid = batch["valid"][..., -1]
        squared_error = jnp.square(q - target_q) * valid
        normalizer = jnp.maximum(jnp.sum(valid) * q.shape[0], 1.0)
        critic_loss = jnp.sum(squared_error) / normalizer

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "td_n_step": jnp.asarray(td_n_step, dtype=jnp.float32),
        }

    @jax.jit
    def critic_only_loss(self, batch, grad_params, rng=None):
        """Compute critic-only loss for ResFiT critic warmup."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng = jax.random.split(rng)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for key, value in critic_info.items():
            info[f"critic/{key}"] = value
        info["actor/update_actor"] = jnp.asarray(0.0, dtype=jnp.float32)
        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute residual actor, temperature, residual L2, and optional BC losses."""
        critic_observations = self._current_critic_observations(batch)
        actor_observations = self._current_actor_observations(batch)
        dist = self.network.select("actor")(actor_observations, params=grad_params)
        raw_actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(raw_actions)

        base_actions = self._sequence_first_action(batch["base_actions"])
        actions_for_q = self._compose_residual_action(base_actions, raw_actions)

        qs = self.network.select("critic")(critic_observations, actions_for_q)
        q = jnp.mean(qs, axis=0)
        actor_loss = (log_probs * self.network.select("alpha")() - q).mean()

        residual_scale = float(self.config.get("residual_scale", 1.0))
        residual_l2 = jnp.mean(jnp.sum((residual_scale * raw_actions) ** 2, axis=-1))
        actor_loss = actor_loss + float(self.config.get("residual_action_l2", 0.0)) * residual_l2

        alpha = self.network.select("alpha")(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha * (entropy - self.config["target_entropy"])).mean()

        bc_alpha = float(self.config["bc_alpha"])
        if bc_alpha != 0.0:
            bc_observations = batch.get("bc_observations", batch["observations"])
            bc_raw_actions = batch.get("bc_actions", batch["actions"])
            bc_base_raw = batch.get("bc_base_actions", batch["base_actions"])
            bc_actions = self._sequence_first_action(bc_raw_actions)
            bc_base_actions = self._sequence_first_action(bc_base_raw)
            bc_aug_observations = self._augment_residual_observations(bc_observations, bc_base_actions)
            bc_delta_targets = (bc_actions - jax.lax.stop_gradient(bc_base_actions)) / max(residual_scale, 1e-6)
            bc_delta_targets = jnp.clip(bc_delta_targets, -1 + 1e-5, 1 - 1e-5)
            bc_dist = self.network.select("actor")(bc_aug_observations, params=grad_params)
            bc_loss_unscaled = -bc_dist.log_prob(bc_delta_targets).mean()
        else:
            bc_loss_unscaled = jnp.asarray(0.0, dtype=actor_loss.dtype)
        bc_loss = bc_loss_unscaled * bc_alpha

        total_loss = actor_loss + alpha_loss + bc_loss

        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "bc_loss": bc_loss,
            "bc_loss_unscaled": bc_loss_unscaled,
            "alpha": alpha,
            "entropy": -log_probs.mean(),
            "q": q.mean(),
            "residual_l2": residual_l2,
            "update_actor": jnp.asarray(1.0, dtype=jnp.float32),
        }

    @staticmethod
    def _critic_only_update(agent, batch):
        """Apply one critic-only gradient update and then update the target critic."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            """Closure passed to TrainState for differentiating only critic loss."""
            return agent.critic_only_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_critic_only(self, batch):
        """Run one JIT-compiled critic-only update."""
        return self._critic_only_update(self, batch)

    @jax.jit
    def batch_update_critic_only(self, batch):
        """Run multiple critic-only updates with `lax.scan` over a UTD batch."""
        agent, infos = jax.lax.scan(self._critic_only_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)


    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Initialize residual actor on state+base_action and critic on state/action."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            raise NotImplementedError("residual_rlpd v0 supports primitive actions only; set action_chunking=False.")
        if config["target_entropy"] is None:
            config["target_entropy"] = -config["target_entropy_multiplier"] * action_dim
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
        alpha_def = Temperature(config["init_temp"])

        ex_actor_observations = cls._augment_residual_observations(ex_observations, ex_actions)
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_actor_observations,)),
            alpha=(alpha_def, ()),
        )
        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx, grad_clip_norm=config["grad_clip_norm"])
        network.params["modules_target_critic"] = network.params["modules_critic"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))



def get_config():
    """Return default config for the ResFiT-style residual RLPD agent."""
    config = get_rlpd_config()
    config.agent_name = "residual_rlpd"
    config.residual_policy = True
    config.residual_scale = 1.0
    config.residual_action_l2 = 0.0
    config.critic_warmup_steps = 0
    config.actor_final_fc_init_scale = 1e-2
    config.base_obs_dim = ml_collections.config_dict.placeholder(int)
    return config
