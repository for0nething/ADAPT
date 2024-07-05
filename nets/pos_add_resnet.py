

from flax import linen as nn  # The Linen API
from typing import Sequence, Callable, List
import jax
import jax.numpy as jnp
Array = jnp.ndarray

def sin_pe_func(pe_op, pe_t, pe_alpha, pe_ratio, n_hidden):
    # T: 0.5, 1.0, 2.0, 4.0, 8.0, 32.0
    indx = jnp.arange(n_hidden) / n_hidden
    T = pe_t
    mask = jnp.sin(2.0 * jnp.pi * indx * T)


    assert pe_op == "add"

    if pe_op == "add":
        mask = pe_alpha * mask
    elif pe_op == "mul":
        mask = pe_alpha * mask + 1.0
    else:
        pass

    # mask ratio
    n = int(pe_ratio * n_hidden)

    if pe_op == "add":
        mask = jnp.concatenate([mask[:n], jnp.zeros_like(mask[n:])])
    elif pe_op == "mul":
        mask = jnp.concatenate([mask[:n], jnp.ones_like(mask[n:])])
    else:
        pass


    mask = mask.reshape((1, -1))

    return mask

class PosAddResidualBlock(nn.Module):
    features: Sequence[int]
    # activation: Callable = nn.tanh
    loc_alpha: float = 1.0
    activation: Callable = nn.relu
    use_bias: bool = True
    init_weight_scale: float = 1e-2
    kernel_i: Callable = jax.nn.initializers.variance_scaling
    pre_masks: [Array] = None

    def setup(self):
        self.layers = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"),
            )
            for feat in self.features
        ]


    def __call__(self, x):
        input = x
        x = self.activation(x)
        for l, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = x + self.pre_masks[l]
            x = self.activation(x)
        x = self.layers[-1](x)
        x = x + self.pre_masks[-1]
        return x + input