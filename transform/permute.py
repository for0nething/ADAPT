import jax
import jax.numpy as jnp
import distrax
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple
Array = jnp.ndarray
class Permute(distrax.Bijector):
    def __init__(self, permutation: Array, axis: int = -1):

        super().__init__(event_ndims_in=1)

        self.permutation = jnp.array(permutation)
        self.axis = axis

    def permute_along_axis(self, x: Array, permutation: Array, axis: int = -1) -> Array:
        x = jnp.moveaxis(x, axis, 0)
        x = x[permutation, ...]
        x = jnp.moveaxis(x, 0, axis)
        return x

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        y = self.permute_along_axis(x, self.permutation, axis=self.axis)
        return y, jnp.zeros(x.shape[: -self.event_ndims_in])

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        inv_permutation = jnp.zeros_like(self.permutation)
        inv_permutation = inv_permutation.at[self.permutation].set(jnp.arange(len(self.permutation)))
        x = self.permute_along_axis(y, inv_permutation)
        return x, jnp.zeros(y.shape[: -self.event_ndims_in])