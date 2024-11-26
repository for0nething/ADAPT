import abc
import dataclasses
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

from flax.linen.initializers import lecun_normal
from flax.linen.initializers import variance_scaling
from flax.linen.initializers import zeros
from flax.linen.module import compact
from flax.linen.module import Module
from flax.linen.dtypes import promote_dtype
from jax import eval_shape
from jax import lax
from jax import ShapedArray
import jax.numpy as jnp
import numpy as np


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]

default_kernel_init = lecun_normal()



class MaskDense(Module):
  """A linear transformation applied over the last dimension of the input.

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  features: int
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  mask_init: Optional[Array] = None


  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.param('kernel',
                        self.kernel_init,
                        (jnp.shape(inputs)[-1], self.features),
                        self.param_dtype)
    # todo: Here it is better to create the variable in a higher layer e.g. maskRQSpline layer to reduce duplicate creation
    if self.mask_init is not None:
        # mask = self.param('mask', lambda *_: self.mask_init, self.mask_init.shape, self.mask_init.dtype)
        # mask = self.variable('mask', 'constant', lambda *_: self.mask_init)
        mask = self.variable("variables", 'mask', lambda *_: self.mask_init, self.mask_init.shape, self.mask_init.dtype)



    # Use mask to modify kernel
    # if mask is not None and mask.shape != kernel.shape:
    if mask is not None and mask.value.shape != kernel.shape:
      raise ValueError('Mask needs to have the same shape as weights. '
                       f'Shapes are: {mask.value.shape}, {kernel.shape}')


    if mask is not None:

        kernel *= mask.value

        # kernel = jnp.where(mask.value, kernel, 0)


      #   kernel = lax.select(
      #       mask, kernel, jnp.zeros_like(kernel))


    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,),
                        self.param_dtype)
    else:
      bias = None



    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),
                        precision=self.precision)
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


  # def modify_mask(self):
  #     self.mask = self.mask.at[0,0].set(self.mask[0,0]+1)





