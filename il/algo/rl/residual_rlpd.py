from __future__ import annotations

"""ResFiT-style residual RLPD agent.

This class intentionally keeps residual-specific behavior out of the plain
ACRLPDAgent. The frozen base policy is handled by rollout/builders; this agent
only learns the residual actor-critic update once base actions are present in
the replay batch.
"""

import jax
import jax.numpy as jnp
import ml_collections

from il.algo.rl.rlpd import ACRLPDAgent, get_config as get_rlpd_config
from il.distributions import TanhNormal


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

    def _augment_residual_observations(self, observations, base_actions):
        """Append stop-gradient base actions to low-dim observations."""
        return jnp.concatenate(
            [jnp.asarray(observations), jax.lax.stop_gradient(jnp.asarray(base_actions))],
            axis=-1,
        )

    def _current_observations(self, batch):
        """Return current observations with base actions appended."""
        base_actions = self._sequence_first_action(batch["base_actions"])
        return self._augment_residual_observations(batch["observations"], base_actions)

    def _next_observations(self, batch):
        """Return bootstrap observations with next base actions appended."""
        next_observations = batch["next_observations"][..., -1, :]
        next_base_actions = self._sequence_last_action(batch["next_base_actions"])
        return self._augment_residual_observations(next_observations, next_base_actions)

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
        observations = self._current_observations(batch)
        next_observations = self._next_observations(batch)

        next_dist = self.network.select("actor")(next_observations)
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
        observations = self._current_observations(batch)
        dist = self.network.select("actor")(observations, params=grad_params)
        raw_actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(raw_actions)

        base_actions = self._sequence_first_action(batch["base_actions"])
        actions_for_q = self._compose_residual_action(base_actions, raw_actions)

        qs = self.network.select("critic")(observations, actions_for_q)
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
