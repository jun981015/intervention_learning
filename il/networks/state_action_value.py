import flax.linen as nn
import jax.numpy as jnp

from il.networks import default_init


class StateActionValue(nn.Module):
    """Q-value head that maps concatenated `(observation, action)` to a scalar."""

    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        """Return scalar Q-values for each observation-action pair."""
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)


class StateActionFeature(nn.Module):
    """Feature head for observation-action pairs before scalar value projection."""

    base_cls: nn.Module
    feature_dim: int
    default_init: nn.initializers.Initializer = nn.initializers.xavier_uniform

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        """Return learned features for each observation-action pair."""
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        feature = nn.Dense(self.feature_dim, kernel_init=self.default_init())(outputs)

        return feature
