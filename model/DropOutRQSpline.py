"""
    2024-02-26
    RQSpline with dropout

"""

from typing import Sequence, Callable, List
import numpy as np
import jax
import jax.numpy as jnp
import distrax
# from nets import *
# from nets import ResidualBlock
from flax import linen as nn  # The Linen API
from transform import Permute
from functools import partial
from typing import Sequence, Tuple
Array = jnp.ndarray
PRNGKey = Array


class DropOutResidualBlock(nn.Module):
    features: Sequence[int]
    # activation: Callable = nn.tanh
    activation: Callable = nn.relu
    use_bias: bool = True
    init_weight_scale: float = 1e-2
    kernel_i: Callable = jax.nn.initializers.variance_scaling
    training : bool = False
    def setup(self):
        self.layers = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"),
            )
            for feat in self.features
        ]
    @nn.compact
    def __call__(self, x):
        input = x
        x = self.activation(x)
        for l, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = nn.Dropout(rate=0.4, deterministic=not self.training)(x)
            x = self.activation(x)
            # x = self.activation(layer(x))

        x = self.layers[-1](x)
        return x + input
class Reshape(nn.Module):
    shape: Sequence[int]

    def __call__(self, x):
        return jnp.reshape(x.T, self.shape)


class Conditioner(nn.Module):
    n_features: int
    hidden_size: Sequence[int]
    num_bijector_params: int
    init_weight_scale: float = 1e-2
    kernel_i: Callable = jax.nn.initializers.variance_scaling
    training:bool = False
    def setup(self):
        self.conditioner = nn.Sequential(
            [
                # 【 Original version 】
                nn.Dense(
                    self.hidden_size[0],
                    use_bias=True,
                    kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"),
                ),

                # 【MLP Edition 】
                # MLP(list(self.hidden_size)),
                # MLP(list(self.hidden_size)),

                # [resnet] Same performance as MLP； note: The input of ResidualBlock is activated once, so there is no need to add activation function between the two blocks
                # ResidualBlock(list(self.hidden_size)),
                # ResidualBlock(list(self.hidden_size)),

                DropOutResidualBlock(list(self.hidden_size)),
                DropOutResidualBlock(list(self.hidden_size)),

                nn.relu,
                nn.Dense(
                    self.n_features * self.num_bijector_params,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                ),
                Reshape((self.n_features, self.num_bijector_params)),
            ]
        )

    def __call__(self, x):
        return self.conditioner(x)



class Scalar(nn.Module):
    n_features: int

    def setup(self):
        self.shift = self.param(
            "shifts", lambda rng, shape: jnp.zeros(shape), (self.n_features)
        )
        self.scale = self.param(
            "scales", lambda rng, shape: jnp.ones(shape), (self.n_features)
        )


    def __call__(self, x):
        return self.scale, self.shift

def scalar_affine(params: jnp.ndarray):
    return distrax.ScalarAffine(scale=params[0], shift=params[1])


class DropOutRQSpline(nn.Module):

    """
    Rational quadratic spline normalizing flow model using distrax.

    Parameters
    ----------
    n_features : int
        Number of features in the data.
    num_layers : int
        Number of layers in the flow.
    num_bins : int
        Number of bins in the spline.
    hidden_size : Sequence[int]
        Size of the hidden layers in the conditioner.
    spline_range : Sequence[float]
        Range of the spline.
    """

    n_features: int
    num_layers: int
    hidden_size: Sequence[int]
    num_bins: int
    spline_range: Sequence[float]
    loc_alpha : float = 1.0   # RQSpline do not use this parameter
    training: bool=False
    def setup(self):
        conditioner = []
        scalar = []
        for i in range(self.num_layers):
            conditioner.append(
                Conditioner(self.n_features, self.hidden_size, 3 * self.num_bins + 1)
            )
            scalar.append(Scalar(self.n_features))

        self.conditioner = conditioner
        self.scalar = scalar

        self.base_mean = self.variable(
            "variables", "base_mean", jnp.zeros, ((self.n_features))
        )
        self.base_cov = self.variable(
            "variables", "base_cov", jnp.eye, (self.n_features)
        )

        # self.vmap_call = jax.jit(jax.vmap(self.__call__))

        def bijector_fn(params: jnp.ndarray):
            return distrax.RationalQuadraticSpline(
                params, range_min=self.spline_range[0], range_max=self.spline_range[1]#, boundary_slopes='identity'      # 【TODO】 这里的identity之后得改，现在这样比较影响准确度
            )

        self.bijector_fn = bijector_fn

    def make_flow(self):
        mask = (jnp.arange(0, self.n_features) % 2).astype(bool)
        mask_all = (jnp.zeros(self.n_features)).astype(bool)
        layers = []
        rng_key = jax.random.PRNGKey(0)
        for i in range(self.num_layers):
            permutation = jax.random.choice(rng_key, jnp.arange(self.n_features), shape=(self.n_features,), replace=False)
            rng_key, _ = jax.random.split(rng_key)
            # layers.append(Permute(jax.random.permutation(rng_key, self.n_features)))
            layers.append(Permute(permutation))
            layers.append(
                distrax.MaskedCoupling(
                    mask=mask_all, bijector=scalar_affine, conditioner=self.scalar[i]
                )
            ),
            layers.append(
                distrax.MaskedCoupling(
                    mask=mask,
                    bijector=self.bijector_fn,
                    conditioner=self.conditioner[i],
                )
            )
            mask = jnp.logical_not(mask)
        flow = distrax.Chain(layers[::-1])
        base_dist = distrax.Independent(
            distrax.MultivariateNormalFullCovariance(
                loc=jnp.zeros(self.n_features),
                covariance_matrix=jnp.eye(self.n_features),
            )
        )

        return base_dist, flow

    # def __call__(self, x: jnp.array) -> jnp.array:
    #     # x = (x-self.base_mean.value)/jnp.sqrt(jnp.diag(self.base_cov.value))
    #     base_dist, flow = self.make_flow()
    #     transformed_x, log_det = flow.forward_and_log_det(x)
    #
    #     return base_dist.log_prob(transformed_x) + log_det
    #
    # def sample(self, rng: jax.random.PRNGKey, num_samples: int) -> jnp.array:
    #     base_dist, flow = self.make_flow()
    #     samples = base_dist.sample(
    #         seed=rng, sample_shape=(num_samples)
    #     )
    #     # Note: Make sure add vmap like this, or  get an error
    #     transformed_samples, log_det = jax.jit(jax.vmap(flow.inverse_and_log_det))(samples)
    #     return transformed_samples

        # @jax.jit # Adding this annotation will result in an error why?
    @partial(jax.vmap, in_axes=(None, 0))
    def __call__(self, x: jnp.array) -> jnp.array:
        # x = (x-self.base_mean.value)/jnp.sqrt(jnp.diag(self.base_cov.value))
        base_dist, flow = self.make_flow()
        transformed_x, log_det = flow.forward_and_log_det(x)

        return base_dist.log_prob(transformed_x) + log_det

        # Measured is faster than sample
    @partial(jax.vmap, in_axes=(None, 0))
    def inverse(self, x: Array) -> Tuple[Array, Array]:
        """ From latent space to data space"""
        base_dist, flow = self.make_flow()
        transformed_x, log_det = flow.inverse_and_log_det(x)
        return transformed_x

    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        samples = self.base_dist.sample(
            seed=rng_key, sample_shape=(n_samples)
        )
        # transformed_samples, log_det = flow.inverse_and_log_det(samples)
        transformed_samples = self.inverse(samples)
        return transformed_samples

    def log_prob(self, x: jnp.array) -> jnp.array:
        # return self.vmap_call(x)
        return self.__call__(x)
