"""
    location-aware rqSpline
"""

from typing import Sequence, Callable, List
import numpy as np
import jax
import jax.numpy as jnp
import distrax
from nets import LocMLP, LocResidualBlock
from flax import linen as nn  # The Linen API
from transform import Permute
Array = jnp.ndarray
PRNGKey = Array



class Reshape(nn.Module):
    shape: Sequence[int]

    def __call__(self, x):
        return jnp.reshape(x.T, self.shape)


class LocConditioner(nn.Module):
    n_features: int
    hidden_size: Sequence[int]
    num_bijector_params: int
    loc_alpha : float=1.0
    init_weight_scale: float = 1e-2
    kernel_i: Callable = jax.nn.initializers.variance_scaling

    def setup(self):
        self.conditioner = nn.Sequential(
            [
                # # 1st layer loc
                # LocMLP([self.hidden_size[0]]),
                # 1st layer non-loc
                nn.Dense(
                    self.hidden_size[0],
                    use_bias=True,
                    kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"),
                ),
                nn.relu,

                # # 【Location MLP】
                LocResidualBlock(list(self.hidden_size), loc_alpha=self.loc_alpha),
                nn.relu,    # new
                LocResidualBlock(list(self.hidden_size), loc_alpha=self.loc_alpha),
                # LocMLP(list(self.hidden_size)),

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
        #
        # mask = sin_pe_func(pe_op="add", pe_t=1.0, pe_alpha=0.05,
        #                          pe_ratio=1.0, n_hidden=self.n_features)
        # self.mask = mask


    def __call__(self, x):
        return self.scale, self.shift

        # return self.scale + self.mask, self.shift + self.mask


def scalar_affine(params: jnp.ndarray):
    return distrax.ScalarAffine(scale=params[0], shift=params[1])


class LocRQSpline(nn.Module):

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
    loc_alpha : float = 1.0
    dummy_input: Sequence[Array] = None

    def setup(self):
        conditioner = []
        scalar = []
        for i in range(self.num_layers):
            # if i!=5:
            #     conditioner.append(
            #         LocConditioner(self.n_features, self.hidden_size, 3 * self.num_bins + 1, loc_alpha=self.loc_alpha)
            #     )
            # else:
            #     conditioner.append(
            #         LocConditioner(self.n_features, self.hidden_size, 3 * self.num_bins + 1, loc_alpha=0)
            #     )
            conditioner.append(
                LocConditioner(self.n_features, self.hidden_size, 3 * self.num_bins + 1, loc_alpha=self.loc_alpha)
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

        self.vmap_call = jax.jit(jax.vmap(self.__call__))

        def bijector_fn(params: jnp.ndarray):
            return distrax.RationalQuadraticSpline(
                params, range_min=self.spline_range[0], range_max=self.spline_range[1]
            )

        self.bijector_fn = bijector_fn

    def make_flow(self):
        mask = (jnp.arange(0, self.n_features) % 2).astype(bool)        # Odd numbers are masked
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
        flow = distrax.Inverse(distrax.Chain(layers))
        base_dist = distrax.Independent(
            distrax.MultivariateNormalFullCovariance(
                loc=jnp.zeros(self.n_features),
                covariance_matrix=jnp.eye(self.n_features),
            )
        )

        return base_dist, flow

    def __call__(self, x: jnp.array) -> jnp.array:
        # x = (x-self.base_mean.value)/jnp.sqrt(jnp.diag(self.base_cov.value))
        base_dist, flow = self.make_flow()
        return distrax.Transformed(base_dist, flow).log_prob(x)

    def sample(self, rng: jax.random.PRNGKey, num_samples: int) -> jnp.array:
        base_dist, flow = self.make_flow()
        samples = base_dist.sample(
            seed=rng, sample_shape=(num_samples)
        )

        transformed_samples, log_det = jax.jit(jax.vmap(flow.inverse_and_log_det))(samples)
        return transformed_samples
    def log_prob(self, x: jnp.array) -> jnp.array:
        return self.vmap_call(x)
