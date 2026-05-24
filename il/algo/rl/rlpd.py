import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import linen as nn

from il.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from il.distributions import TanhNormal
from il.networks import Ensemble, StateActionValue, MLP

from functools import partial


TARGET_NUM_QS = 2


class Temperature(nn.Module):
    """Learnable SAC temperature parameter stored in log space."""

    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        """Return the positive temperature scalar `alpha`."""
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


class ACRLPDAgent(flax.struct.PyTreeNode):
    """Soft actor-critic (SAC) agent with action chunking.

    This agent can also be used for reinforcement learning with prior data (RLPD).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def aggregate_target_qs(self, target_qs):
        """Aggregate the first target critics for TD backup computation."""
        target_qs = target_qs[:TARGET_NUM_QS]
        if self.config["target_q_agg"] == "min":
            return target_qs.min(axis=0)
        if self.config["target_q_agg"] == "mean":
            return target_qs.mean(axis=0)
        raise ValueError(f"Unsupported target_q_agg: {self.config['target_q_agg']}")

    @jax.jit
    def evaluate_q_heads(self, observations, actions):
        """Evaluate all action-value heads for an arbitrary action proposal."""
        return self.network.select('critic')(observations, actions)

    def evaluate_q(self, observations, actions, *, q_agg: str = "min"):
        """Evaluate a scalar Q value using the requested ensemble aggregation."""
        q_heads = self.evaluate_q_heads(observations, actions)
        if q_agg == "min":
            return q_heads.min(axis=0)
        if q_agg == "mean":
            return q_heads.mean(axis=0)
        if q_agg == "max":
            return q_heads.max(axis=0)
        raise ValueError(f"Unsupported q_agg: {q_agg}")

    def critic_loss(self, batch, grad_params, rng):
        """Compute masked n-step TD loss for the critic ensemble."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action

        rng, sample_rng = jax.random.split(rng)

        next_dist = self.network.select('actor')(batch['next_observations'][..., -1, :])
        next_actions = next_dist.sample(seed=sample_rng)

        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
        next_q = self.aggregate_target_qs(next_qs)

        target_q = batch['rewards'][..., -1] + (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q
        
        q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute SAC policy, temperature, and optional BC regularization losses."""
        bc_observations = batch.get("bc_observations", batch["observations"])
        bc_raw_actions = batch.get("bc_actions", batch["actions"])
        if self.config["action_chunking"]:
            bc_actions = jnp.reshape(bc_raw_actions, (bc_raw_actions.shape[0], -1))
        else:
            bc_actions = bc_raw_actions[..., 0, :] # take the first action

        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(actions)

        # Actor loss.
        qs = self.network.select('critic')(batch['observations'], actions)
        q = jnp.mean(qs, axis=0)

        actor_loss = (log_probs * self.network.select('alpha')() - q).mean()

        # Entropy loss.
        alpha = self.network.select('alpha')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha * (entropy - self.config['target_entropy'])).mean()

        # BC loss. If an auxiliary `bc_*` batch exists, it is typically sampled
        # from demos/expert data; otherwise this preserves the old in-batch BC.
        bc_dist = self.network.select('actor')(bc_observations, params=grad_params)
        bc_loss_unscaled = -bc_dist.log_prob(jnp.clip(bc_actions, -1 + 1e-5, 1 - 1e-5)).mean()
        bc_loss = bc_loss_unscaled * self.config["bc_alpha"]

        total_loss = actor_loss + alpha_loss + bc_loss

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'bc_loss': bc_loss,
            'bc_loss_unscaled': bc_loss_unscaled,
            'alpha': alpha,
            'entropy': -log_probs.mean(),
            'q': q.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute joint critic/actor loss and prefix metrics by module."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Polyak-average one online module into its target module."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(self, batch):
        """Apply one gradient update and then update the target critic."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            """Closure passed to TrainState for differentiating total loss."""
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info
    
    @jax.jit
    def update(self, batch):
        """Run one JIT-compiled learner update on one batch."""
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Run multiple updates with `lax.scan` over a UTD-stacked batch."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        """Sample clipped actions from the stochastic actor."""
        dist = self.network.select('actor')(observations)
        actions = dist.sample(seed=rng)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_actions_with_log_prob(
        self,
        observations,
        rng=None,
    ):
        """Sample clipped actions and their actor log-probabilities."""
        dist = self.network.select('actor')(observations)
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(jnp.clip(actions, -1 + 1e-6, 1 - 1e-6))
        actions = jnp.clip(actions, -1, 1)
        return actions, log_probs

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Initialize actor, critic ensemble, target critic, and temperature modules.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * full_action_dim
        if config["num_qs"] < TARGET_NUM_QS:
            raise ValueError(f"num_qs must be >= {TARGET_NUM_QS}.")

        # Define networks
        critic_base_cls = partial(
            MLP,
            hidden_dims=config['value_hidden_dims'],
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
        actor_def = TanhNormal(actor_base_cls, full_action_dim)

        # Define the dual alpha variable.
        alpha_def = Temperature(config["init_temp"])

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor=(actor_def, (ex_observations,)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx, grad_clip_norm=config["grad_clip_norm"])

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
    
def get_config():
    """Return default RLPD/SAC hyperparameters for this project."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='acrlpd',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            num_qs=2,
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None for automatic tuning).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            alpha=1.0,
            bc_alpha=0.0,
            target_q_agg='min',  # Aggregation for the first two target critics in TD backup.
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,
            init_temp=1.0,
            grad_clip_norm=None,
        )
    )
    return config
