import jax
import jax.numpy as jnp
import dp_accounting
import warnings

def compute_epsilon(steps, batch_size, num_data, noise_multiplier, target_delta=1e-5, orders=None):

    if num_data * target_delta > 1.:
        warnings.warn('Your delta might be too high.')
    q = batch_size / float(num_data)
    # Each order can correspond to an epsilon in DP. Finally, you need to compute an epsilon for each order in orders and take the smallest epsilon
    if orders is None:
        orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(dp_accounting.PoissonSampledDpEvent(
      q, dp_accounting.GaussianDpEvent(noise_multiplier)), steps)
    return accountant.get_epsilon(target_delta)