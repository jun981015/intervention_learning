from typing import Any, Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

default_init = nn.initializers.xavier_uniform


class MLP(nn.Module):
    """Configurable feed-forward MLP used by actors and critics."""

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Optional[Callable[..., Any]] = None
    use_layer_norm: bool = False
    layer_norm_after_activation: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    use_pnorm: bool = False
    sow_intermediate_feature: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply dense layers, optional normalization/dropout, and activation."""

        for i, size in enumerate(self.hidden_dims):
            is_final_layer = i + 1 == len(self.hidden_dims)
            if self.kernel_init is not None:
                kernel_init = self.kernel_init
            elif is_final_layer and self.scale_final is not None:
                kernel_init = default_init(self.scale_final)
            else:
                kernel_init = default_init()
            x = nn.Dense(size, kernel_init=kernel_init)(x)

            should_activate = not is_final_layer or self.activate_final
            if should_activate:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.use_layer_norm and not self.layer_norm_after_activation:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.use_layer_norm and self.layer_norm_after_activation:
                    x = nn.LayerNorm()(x)
            if self.sow_intermediate_feature and i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        if self.use_pnorm:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10)
        return x
