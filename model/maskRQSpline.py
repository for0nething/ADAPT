"""
use a mask to represent the location
"""

from typing import Sequence, Callable, List
import numpy as np
import jax
import jax.numpy as jnp
import distrax
from nets import LocMLP, LocResidualBlock, MaskLocResidualBlock, MaskDense, computer_zero_mask
from flax import linen as nn  # The Linen API
from transform import Permute
# from nets import MaskDense
Array = jnp.ndarray
PRNGKey = Array
from typing import Sequence, Tuple
from functools import partial

class Reshape(nn.Module):
    shape: Sequence[int]

    def __call__(self, x):
        return jnp.reshape(x.T, self.shape)

# original
# def random_k_0_each_row_mask(d_in, d_out, mask_lim=0.005, rng_seed=0):
#     # mask = jnp.ones((d_out, d_in))
#     mask = jnp.ones((d_in, d_out))
#     k_th = int(np.ceil(mask_lim * d_in))
#
#     # Seed for the initial RNG
#     rng = jax.random.PRNGKey(rng_seed)
#     rng, *row_rngs = jax.random.split(rng, d_in + 1)
#
#     for i in range(d_in):
#         # Select columns to be zero
#         zero_cols = jax.random.choice(row_rngs[i], d_out, shape=[k_th], replace=False)
#         # Set the selected columns to zero
#         mask = mask.at[i, zero_cols].set(0.0)
#         # print("see zeros cols")
#         # print(zero_cols)
#     # best_dis = compute_min_pairwise_hamming_distance(mat)
#     # return best_dis, mat
#     return mask

# np version
def mask_selection(d_in, d_out, mask_lim=0.005, rng_seed=0):

    # d_out  n —— number of neurons
    # d_in   m —— input dimension

    # in total:  d_out * d_in * (1-mask_lim)

    mask = np.ones((d_in, d_out))
    rng = np.random.RandomState(rng_seed)


    K = int(np.ceil(mask_lim * d_in))
    total = d_out * d_in * mask_lim
    total = int(total)

    avg = int(np.floor(total/d_in))
    num_avg_add_1 = total % d_in
    num_avg = d_in - num_avg_add_1

    # random select num_avg_add_1 columns from the total d_in  rows
    random_rows = rng.choice(d_in, num_avg_add_1, replace=False)

    # compute remain_columns
    remain_rows = []
    for i in range(d_in):
        if i not in random_rows:
            remain_rows.append(i)

    available_cols = np.arange(d_out)

    col_sum = np.zeros(d_out)   # maintain the number of 0 of all columns


    for i in range(d_in):
        # random select K columns from the current available columns
        if i in random_rows:
            random_cols = rng.choice(available_cols, avg + 1, replace=False)
        else:
            random_cols = rng.choice(available_cols, avg, replace=False)


        mask[i,random_cols] = 0

        # update available_cols
        col_sum[random_cols] += 1
        # compute new available_cols, whose  col_sum is less than K
        new_available_cols = np.where(col_sum < K)[0]
        # print("see new available cols")
        # print(len(new_available_cols), avg)
        # print(new_available_cols)
        available_cols = new_available_cols

    mask = jnp.array(mask)

    return mask


# jnp version
# def random_k_0_each_row_mask(d_in, d_out, mask_lim=0.005, rng_seed=0):
#
#     # d_out  n —— number of neurons
#     # d_in   m —— input dimension
#
#     # in total:  d_out * d_in * (1-mask_lim)
#
#     mask = jnp.ones((d_in, d_out))
#     rng = jax.random.PRNGKey(rng_seed)
#
#     K = int(np.ceil(mask_lim * d_in))
#     total = d_out * d_in * (mask_lim)
#     total = int(total)
#
#
#     avg = int(np.floor(total/d_in))
#     num_avg_add_1 = total % d_in
#     num_avg = d_in - num_avg_add_1
#
#     # random select num_avg_add_1 columns from the total d_in  rows
#     rng, subkey = jax.random.split(rng)
#     random_rows = jax.random.choice(subkey, d_in, (num_avg_add_1,), replace=False)
#
#
#     # compute remain_columns
#     remain_rows = jnp.setdiff1d(jnp.arange(d_in), random_rows)
#
#     available_cols = jnp.arange(d_out)
#
#     col_sum = jnp.zeros(d_out)   # maintain the number of 0 of all columns
#
#     # for i in range(d_in):
#     #     # random select K columns from the current available columns
#     #     subkey, key_col = jax.random.split(rng)
#     #     if i in random_rows:
#     #         random_cols = jax.random.choice(key_col, available_cols, (avg + 1,), replace=False)
#     #     else:
#     #         random_cols = jax.random.choice(key_col, available_cols, (avg,), replace=False)
#     #
#     #     # mask = jax.ops.index_update(mask, (i, random_cols), 0)
#     #     mask = mask.at[i, random_cols].set(0)
#     #
#     #     # update available_cols
#     #     col_sum = col_sum.at[random_cols].add(1)
#     #     # compute new available_cols, whose  col_sum is less than K
#     #     new_available_cols = jnp.where(col_sum < K)[0]
#     #     available_cols = new_available_cols
#
#
#     for i in remain_rows:
#         # random select K columns from the current available columns
#         subkey, key_col = jax.random.split(rng)
#         random_cols = jax.random.choice(key_col, available_cols, (avg ,), replace=False)
#
#         # mask = jax.ops.index_update(mask, (i, random_cols), 0)
#         mask = mask.at[i, random_cols].set(0)
#
#         # update available_cols
#         col_sum = col_sum.at[random_cols].add(1)
#         # compute new available_cols, whose  col_sum is less than K
#         new_available_cols = jnp.where(col_sum < K)[0]
#         available_cols = new_available_cols
#
#
#     return mask




def p_mask(d_in, d_out, p=11, rng_seed=0):

    tmp_masks = jnp.zeros((d_in, d_out))
    rng = np.random.RandomState(rng_seed)
    for i in range(tmp_masks.shape[0]):
        tmp_masks = tmp_masks.at[i].set(rng.randint(p, size=tmp_masks.shape[1]) / p)
    return tmp_masks

# def p_mask():
#     # tmp_masks = jnp.zeros((256, 6))
#     tmp_masks = jnp.zeros((6, 256))
#     p = 11
#     rng = np.random.RandomState(0)
#     # generate a random matrix each has a int value from [0,11) with shape (6,256)
#     # for i in range(6):
#     for i in range(tmp_masks.shape[0]):
#         tmp_masks = tmp_masks.at[i].set(rng.randint(p, size=tmp_masks.shape[1]) / p)
# 
#     # tmp_masks = jnp.array(tmp_masks)
#     return tmp_masks


class LocConditioner(nn.Module):
    n_features: int
    hidden_size: Sequence[int]
    num_bijector_params: int
    loc_alpha : float=1.0
    init_weight_scale: float = 1e-2
    kernel_i: Callable = jax.nn.initializers.variance_scaling
    pre_masks : [Array] = None
    p_mask : Array = None

    # 【new for when layers_4 also use mask】
    # p_mask_out : Array = None


    def setup(self):

        self.conditioner = nn.Sequential(
            [

                # LocMLP([self.hidden_size[0]],loc_alpha=self.loc_alpha),
                # LocMLP([self.hidden_size[0]],loc_alpha=1),



                MaskDense(
                    self.hidden_size[0],
                    use_bias=True,
                    kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"),
                    mask_init=self.p_mask
                    # mask= computer_zero_mask(self.n_features, self.hidden_size[0], 0.2)
                ),


                # nn.Dense(
                #     self.hidden_size[0],
                #     use_bias=True,
                #     kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"),
                #     # mask= computer_zero_mask(self.n_features, self.hidden_size[0], 0.2)
                # ),


                # # 【Location MLP】
                MaskLocResidualBlock(list(self.hidden_size), loc_alpha=self.loc_alpha, pre_masks=self.pre_masks),
                # nn.relu,    # new
                MaskLocResidualBlock(list(self.hidden_size), loc_alpha=self.loc_alpha, pre_masks=self.pre_masks),
                # LocMLP(list(self.hidden_size)),

                nn.relu,

                nn.Dense(
                    self.n_features * self.num_bijector_params,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                ),
                #
                # MaskDense(
                #     self.n_features * self.num_bijector_params,
                #     kernel_init=jax.nn.initializers.zeros,
                #     bias_init=jax.nn.initializers.zeros,
                #     mask_init=self.p_mask_out
                # ),
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


class MaskRQSpline(nn.Module):

    """
    Rational quadratic spline normalizing flow model using distrax.
    【The location information is leveraged using masks】

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
    pre_masks: Sequence[Array] = None

    def setup(self):

        """ new: Put all of the mask calculations here to avoid multiple repetitions. mask """
        # todo: Parameter setting & passing may need to be optimized
        # pre_masks = []
        # for i in range(2):
        #     pre_masks.append(
        #         # random_k_0_each_row_mask(self.n_features, self.n_features, mask_lim=self.loc_alpha, rng_seed=2333 + i)
        #         random_k_0_each_row_mask(self.hidden_size[i], self.hidden_size[i], mask_lim=self.loc_alpha, rng_seed=2333 + i)
        #     )
        #
        #
        # self.pre_masks = pre_masks

        p_m = p_mask(d_in=self.n_features, d_out=self.hidden_size[0], p=11, rng_seed=0)
        self.p_mask = p_m

        # 【new for when layers_4 also use mask】
        # p_mask_out = random_k_0_each_row_mask(256, 150, mask_lim=self.loc_alpha, rng_seed=2333)
        # self.p_mask_out = p_mask_out

        conditioner = []
        scalar = []
        for i in range(self.num_layers):

            conditioner.append(
                # LocConditioner(self.n_features, self.hidden_size, 3 * self.num_bins + 1, loc_alpha=self.loc_alpha)
                LocConditioner(self.n_features, self.hidden_size, 3 * self.num_bins + 1, loc_alpha=self.loc_alpha,
                               pre_masks=self.pre_masks,
                               p_mask=self.p_mask,

                               # 【new for when layers_4 also use mask】
                               # p_mask_out=self.p_mask_out
                               )
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
                params, range_min=self.spline_range[0], range_max=self.spline_range[1]
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
                    conditioner=self.conditioner[i],          # conditioner not shared
                    # conditioner=self.conditioner[0],        # layer share nn   n_conditioner=0
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



    @partial(jax.vmap, in_axes=(None, 0))
    def __call__(self, x: jnp.array) -> jnp.array:
        # x = (x-self.base_mean.value)/jnp.sqrt(jnp.diag(self.base_cov.value))
        base_dist, flow = self.make_flow()
        transformed_x, log_det = flow.forward_and_log_det(x)

        return base_dist.log_prob(transformed_x) + log_det


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
