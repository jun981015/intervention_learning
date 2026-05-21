from typing import Type

import flax.linen as nn
import jax.numpy as jnp

from il.networks import default_init


class TanhDeterministic(nn.Module):
    """Deterministic tanh-squashed action head."""

    base_cls: Type[nn.Module]
    action_dim: int

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> jnp.ndarray:
        """Return a tanh-bounded action vector."""
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(), name="OutputDenseMean"
        )(x)

        means = nn.tanh(means)

        return means
