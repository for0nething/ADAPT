

from flax import linen as nn  # The Linen API
from typing import Sequence, Callable, List
import jax
import jax.numpy as jnp
from .maskDense import MaskDense
Array = jnp.ndarray

def sin_pe_func(pe_op, pe_t, pe_alpha, pe_ratio, n_hidden):
    # T: 0.5, 1.0, 2.0, 4.0, 8.0, 32.0
    indx = jnp.arange(n_hidden) / n_hidden
    T = pe_t

    # mask = jnp.sin(2.0 * jnp.pi * indx * T)

    # if pe_op == "add":
    #     mask = (indx + 1) * 2 - 1
    # else:
    mask = indx

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

    # if pe_op == "add":
    #     mask[n:] = 0.0
    # elif pe_op == "mul":
    #     mask[n:] = 1.0
    # else:
    #     pass

    mask = mask.reshape((1, -1))

    return mask

def computer_zero_mask(d_in, d_out, mask_lim=0.005):

    mask = jax.random.uniform(jax.random.PRNGKey(0), (d_in, d_out))
    mask = jnp.where(mask < mask_lim, 0.0, 1.0)


    return mask

import numpy as np
def random_k_0_each_row_mask(d_in, d_out, mask_lim=0.005, rng_seed=0):
    mask = jnp.ones((d_out, d_in))

    k_th = int(np.ceil(mask_lim * d_in))

    # Seed for the initial RNG
    rng = jax.random.PRNGKey(rng_seed)
    rng, *row_rngs = jax.random.split(rng, d_out + 1)

    for i in range(d_out):
        # Select columns to be zero
        zero_cols = jax.random.choice(row_rngs[i], d_in, shape=[k_th], replace=False)
        # Set the selected columns to zero
        mask = mask.at[i, zero_cols].set(0.0)
        # print("see zeros cols")
        # print(zero_cols)
    # best_dis = compute_min_pairwise_hamming_distance(mat)
    # return best_dis, mat
    return mask


# def random_k_0_each_row_mask(d_in, d_out, mask_lim=0.005, rng_seed=0):
#     k_th = int(jnp.ceil(mask_lim * d_in))
#     rng = jax.random.PRNGKey(rng_seed)
#
#     # Generate random indices for setting elements to 0
#     zero_indices = jax.random.randint(rng, (d_out, k_th), minval=0, maxval=d_in)
#
#     # Create the mask
#     mask = jnp.where(jnp.arange(d_in)[:, None] < zero_indices, 0.0, 1.0)
#
#     return mask

# pre_masks = []
# for i in range(2):
#     pre_masks.append(
#         random_k_0_each_row_mask(256, 256, mask_lim=0.05, rng_seed=2333+i)
#     )


class MaskLocResidualBlock(nn.Module):
    features: Sequence[int]
    # activation: Callable = nn.tanh
    loc_alpha: float = 0.005
    activation: Callable = nn.relu
    use_bias: bool = True
    init_weight_scale: float = 1e-2
    kernel_i: Callable = jax.nn.initializers.variance_scaling
    pre_masks: [Array] = None

    def setup(self):
        seeds = [i + 2333 for i in range(len(self.features))]
        """ Use a precomputed mask """
        self.layers = [
            # Diff: Use MaskDense instead of nn.Dense
            MaskDense(
                feat,
                use_bias=self.use_bias,
                kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"),
                # mask_init=computer_zero_mask(feat, feat, self.loc_alpha)        # TODO: make this more general
                # mask_init=random_k_0_each_row_mask(feat, feat, mask_lim=self.loc_alpha, rng_seed=seed)        # TODO: make this more general
                # mask_init=p_mask  # TODO: make this more general
                mask_init=p_mask  # TODO: make this more general
            )
            for feat, p_mask in zip(self.features, self.pre_masks)
        ]



        # self.layers = [
        #     # Diff: Use MaskDense instead of nn.Dense
        #     MaskDense(
        #         feat,
        #         use_bias=self.use_bias,
        #         kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"),
        #         # mask_init=computer_zero_mask(feat, feat, self.loc_alpha)        # TODO: make this more general
        #         # mask_init=random_k_0_each_row_mask(feat, feat, mask_lim=self.loc_alpha, rng_seed=seed)        # TODO: make this more general
        #         mask_init=jnp.ones((feat, feat), dtype=bool)        # TODO: make this more general
        #     )
        #     for feat,seed in zip(self.features, seeds)
        # ]




        # mask = []
        #
        # for l, layer in enumerate(self.layers):
        #     layer_mask = sin_pe_func(pe_op="mul", pe_t=1.0, pe_alpha=1,
        #                              pe_ratio=1.0, n_hidden=self.features[l])
        #     mask.append(layer_mask)
        # self.mask = mask

    def __call__(self, x):
        input = x
        x = self.activation(x)
        for l, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            # x = x * self.mask[l]
            x = self.activation(x)
        x = self.layers[-1](x)
        # x = x * self.mask[-1]
        return x + input