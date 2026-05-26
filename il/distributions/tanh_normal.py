import functools
from typing import Optional, Type

import tensorflow_probability

from il.distributions.tanh_transformed import TanhTransformedDistribution

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

import flax.linen as nn
import jax.numpy as jnp

from il.networks import default_init


def output_init(scale: float):
    """Return the actor output-head initializer, preserving Xavier at scale=1."""
    if scale == 1.0:
        return default_init()
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class Normal(nn.Module):
    """Gaussian policy head with optional tanh squashing."""

    base_cls: Type[nn.Module]
    action_dim: int
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        """Return a diagonal Gaussian or tanh-transformed Gaussian distribution."""
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=output_init(self.final_fc_init_scale), name="OutputDenseMean"
        )(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=output_init(self.final_fc_init_scale), name="OutputDenseLogStd"
            )(x)
        else:
            log_stds = self.param(
                "OutpuLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32
            )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)
