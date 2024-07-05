import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import to_backend_dtype

from integration.vegas_mul_map import VEGASMultiMap

from helper_functions import setup_test_for_backend
import torch

# n, N_intervals, dim, backend, dtype, alpha=0.5

# VEGASMultiMap(n=200, N_intervals=1000, dim=6, backend="torch", dtype=torch.cuda.FloatTensor)
vmap = VEGASMultiMap(n=200, N_intervals=1000, dim=6, backend="torch", dtype=torch.float)


# rng_y = torch.rand((200, 6, 2000))
rng_y = torch.rand((6, 200, 2000))

# print(rng_y.min(), rng_y.max())
# (dim, n, bla)
rng_x = vmap.get_X(rng_y)
print(rng_x.shape)

""" 22.9.2 add 
    Check get_y
"""
ret_y = vmap.get_y(rng_x, torch.arange(200))
# ret_y2 = vmap.get_y(torch.repeat_interleave(rng_x,2,dim=1), torch.repeat_interleave(torch.arange(200),2) )
print("???")
delta = torch.abs(ret_y - rng_y)
delta[delta < 0.00001] = 0
total_delta = delta.sum()
if total_delta ==0:
    print("Multi map get_y passed test!")
else:
    print("Warning!!! multi map get_y failed test!")
print(delta.sum())
# print(ret_y==rng_y)

# assert (ret_y == rng_y).sum() == ()
jf2 = torch.rand_like(rng_y)

vmap.accumulate_weight(rng_y, jf2)


VEGASMultiMap._smooth_map(vmap.weights, vmap.counts, 0.5)

vmap.update_map()




