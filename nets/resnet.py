from flax import linen as nn  # The Linen API
from typing import Sequence, Callable, List
import jax

class ResidualBlock(nn.Module):
    features: Sequence[int]
    # activation: Callable = nn.tanh
    activation: Callable = nn.relu
    use_bias: bool = True
    init_weight_scale: float = 1e-2
    kernel_i: Callable = jax.nn.initializers.variance_scaling

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
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x + input