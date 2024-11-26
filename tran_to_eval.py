import jax
import jax.numpy as jnp  # JAX NumPy
import jax.random as random  # JAX random
import optax  # Optimizers
import flax  # Deep Learning library for JAX
import numpy as np
import os
import pandas as pd
import datasets
from DP.accounting import *
from model import RQSpline, LocRQSpline, MaskRQSpline
from model.utils import make_training_loop, sample_nf, create_train_state, create_schedule_train_state
# from flowMC.nfmodel.utils import make_training_loop, sample_nf
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import model as my_model
import pickle
from commonSetting import PROJECT_PATH
from model.utils_nightly import make_training_loop_unsupervised
os.environ['CUDA_VISIBLE_DEVICES']='2'

# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jiayi/disk/TensorRT-8.5.3.1/lib;



# dataset_name = "moon"
# dataset_name = "power"
# dataset_name = "BJAQ"
dataset_name = "flights"
# dataset_name = "imdbfull"

if_test_NLL = False # whether using test data for evaluation by test NLL

# model path
dir = ""

IF_DPSGD = False
# IF_DPSGD = True

loc_alpha = 0.01




# Optimization parameters
num_epochs = 40

batch_size = 512
# set learning rate
if IF_DPSGD == True:
    # learning_rate = 0.0005
    # learning_rate = 0.001
    learning_rate = 0.01
else:
    # learning_rate = 0.00001     # for
    learning_rate = 0.0005     # for
momentum = 0.9
# Privacy parameters
l2_norm_clip = 5        # l2 norm clip in DPSGD
noise_multiplier = 0.2
seed = 1337

print(f"Current learning rate is [{learning_rate}]")

# small model
# n_layers = 6
# n_hiddens = [108, 108]
# n_bins = 8



# Model parameters (In NSF paper)
n_layers = 10
n_hiddens = [256, 256]
# n_hiddens = [64, 64]

# n_layers = 4
# n_hiddens = [128, 128]

n_bins = 8


# Model parameters (Stable)
tail_bound = [-3, 3]
grad_norm_clip_value = 5.0  # l2 norm clip without DPSGD


read_data_name = dataset_name + "Split"
data, num_data, n_dim =datasets.__dict__[read_data_name]("train")
val_data, val_num, val_dim =datasets.__dict__[read_data_name]("val")
test_data, test_num, test_dim =datasets.__dict__[read_data_name]("test")

steps_per_epoch = int(len(data)//batch_size)

# data, num_data, n_dim= datasets.__dict__[dataset_name]()


key, rng_model, rng_init, rng_train, rng_nf_sample = jax.random.split(
    jax.random.PRNGKey(0), 5
)


model_name = "RQSpline"
model = my_model.__dict__[model_name](n_dim, n_layers, n_hiddens, n_bins, tail_bound,
                                           loc_alpha)
# model = RQSpline(n_dim, n_layers, n_hiddens, n_bins, tail_bound)
# model = MaskRQSpline(n_dim, n_layers, n_hiddens, n_bins, tail_bound, loc_alpha=0)
# model = LocRQSpline(n_dim, n_layers, n_hiddens, n_bins, tail_bound, loc_alpha=loc_alpha)
variables = model.init(rng_model, jnp.ones((1, n_dim)))["variables"]
variables = variables.unfreeze()
variables["base_mean"] = jnp.mean(data, axis=0)
variables["base_cov"] = jnp.cov(data.T)
variables = flax.core.freeze(variables)


steps = num_epochs * (num_data//batch_size)
print("steps is ", steps)


# state = create_train_state(rng_init, learning_rate, momenqtum)



# state = create_train_state(model, n_dim, rng_init, learning_rate, momentum,
#                            IF_DPSGD=IF_DPSGD,
#                            l2_norm_clip=l2_norm_clip,
#                            noise_multiplier=noise_multiplier,
#                            seed=seed)


state = create_schedule_train_state(model, n_dim, rng_init, learning_rate,
                                    momentum = momentum,
                                    IF_DPSGD=IF_DPSGD,
                                    l2_norm_clip=l2_norm_clip,
                                    noise_multiplier=noise_multiplier,
                                    seed=seed,
                                    total_steps=steps,
                                    grad_norm_clip_value=grad_norm_clip_value,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=num_epochs)



train_flow, train_epoch, train_step, evaluate = make_training_loop(model, IF_DPSGD=IF_DPSGD)


print()


rng_key = jax.random.PRNGKey(0)
permutation_list = []

for i in range(n_layers):
    permutation = jax.random.choice(rng_key, jnp.arange(n_dim), shape=(n_dim,), replace=False)
    rng_key, _ = jax.random.split(rng_key)
    print(f"Permutation {i}")
    print(permutation)
    permutation_list.append(np.array(permutation))
permutation_list = np.array(permutation_list)

print("see permutation list")
# print(permutation_list)
print(np.array2string(permutation_list, separator=', '))
print()






# ========================== read by pickle =============================

import pickle


load_state = pickle.load(open(os.path.join(dir, "params.pkl"), "rb"))
params= load_state["params"]
print("use loaded params!")


# =========================== Calculate NLL using the read model ===========================

if if_test_NLL == True:
    train_flow_unsup, train_epoch_unsup, train_step_unsup, evaluate_unsup = make_training_loop_unsupervised(model,
                                                                                                                IF_DPSGD=False)
    print("[Test NLL] ", evaluate_unsup(params, test_data, variables))



# change the items in params as np.array
params = jax.tree_map(lambda x: np.array(x), params)

correct_params = params

# check if correct_params and params have the same shape
# for key in params.keys():
#     print("key is ", key)
#     # note that correct_params[key] and params[key] may still be dict
#     if isinstance(correct_params[key], dict):
#         for sub_key in correct_params[key].keys():
#             print("sub_key is ", sub_key)
#             print("correct_params[key][sub_key].shape is ", correct_params[key][sub_key].shape)
#             print("params[key][sub_key].shape is ", params[key][sub_key].shape)

def check(x, y):
    flag = True
    if isinstance(x, flax.core.frozen_dict.FrozenDict) or isinstance(x, dict):
        for key in x.keys():
            flag = flag and check(x[key], y[key])
    else:
        flag = x.shape == y.shape
    return flag

check(correct_params, params)




# Base distribution
print(variables["base_mean"])
print(variables["base_cov"])




# scalar
# [scalar_{i}] shifts / scales

# conditioners

# [conditioner_{i}]["conditioner"] ["layers_0"]   kernel / bias
#                                  ["layers_2"]   [layers_0  layers_1]    kernel / bias
#                                  ["layers_4"]   [layers_0  layers_1]    kernel / bias
#                                  ["layers_4"]   kernel / bias




# eval_result = evaluate(params, test_data, variables)
# print("Evaluated result is ", eval_result)


# res = model.apply({"params": params, "variables": variables}, jnp.array([[0,0,0,0,0,0]]), method=model.log_prob)
# print(f"see log prob result at zeron point [{res}]")

""" 
    simple test
"""
test_few_data = test_data[:3, :]
# test_few_data = jnp.array( [[-5]*6])
print("see input data")
print(test_few_data)
res = model.apply({"params": params, "variables": variables}, test_few_data, method=model.log_prob)
print("see log_prob")
print(res)
print(" [For reference]  The correct log_prob should be around 3! Please check manually!")
if res[0] < -10:
    # output an error
    raise ValueError("The log_prob is too small! The transform is probably wrong! Please check manually!")
# test_few_data = test_few_data[0,:]
# base_dist, flow = res = model.apply({"params": params, "variables": variables}, method=model.make_flow)
#
# # flow.forward_and_log_det(test_few_data)
# x, log_det = flow._bijectors[-1].forward_and_log_det(test_few_data)
# jax.debug.print("See x after permutation")
# jax.debug.print("{x}", x=x)
# jax.debug.print("See log_det")
# jax.debug.print("{log_det}", log_det=log_det)
# cnt = 0
# for bijector in reversed(flow._bijectors[:-1]):
#     last_x = x  #
#     x, ld = bijector.forward_and_log_det(x)
#     log_det += ld
#
#     jax.debug.print("See x after {bijector}", bijector=bijector)
#     jax.debug.print("{x}", x=x)
#     jax.debug.print("See log_det")
#     jax.debug.print("{log_det}", log_det=log_det)
#
#     if cnt == 1:
#         jax.debug.print("See conditioner detail")
#         # intermediate_results = []
#         # last_x
#         last_x = jnp.where(bijector._event_mask, last_x, 0.)
#         jax.debug.print("The original x to this conditioenr is ")
#         jax.debug.print("{x}", x=last_x)
#         for layer in bijector.conditioner.conditioner.layers:
#             # jax.debug.print("  - Layer weight:")
#             # jax.debug.print("  - Layer param:")
#             # jax.debug.print("{param}", param=layer.param)
#             # jax.debug.print("{kernel}", kernel=layer["kernel"])
#             # jax.debug.print("  - Layer bias:")
#             # jax.debug.print("{bias}", bias=layer["bias"])
#
#             last_x = layer(last_x)
#             jax.debug.print("After this conditioner.{layer} ", layer=layer)
#             jax.debug.print("{outputs}", outputs=last_x)
#         last_x = last_x[jnp.logical_not(bijector._event_mask),:]
#         print("After masked")
#         print(last_x)
#
#     cnt = cnt + 1
#
# jax.debug.print("Final See x")
# jax.debug.print("{x}", x=x)
# jax.debug.print("Final See log_det")
# jax.debug.print("{log_det}", log_det=log_det)
#
#
# print("test_data[:3,:] is ")
# print(test_data[:3,:])
# print(f"see log prob result at test_data[:3,:]")
# print(res)      # [2.6500282 3.0751648 3.7129478]


# Build a dict to save the model parameters
model_dict = {}
mask = (jnp.arange(0, n_dim) % 2).astype(bool)

# Save the model parameters to model_dict
for i in range(n_layers):
    # print("#" * 30)
    # print(f"Layer {i}")
    # print("#" * 30)

    # scalar
    scalar_string = "scalar_{}".format(i)
    # print(scalar_string)

    shifts = params[scalar_string]["shifts"]
    scales = params[scalar_string]["scales"]
    # print("shifts: ", shifts)
    # print("scales: ", scales)
    model_dict[scalar_string] = {"shifts": shifts, "scales": scales}


    # conditioner
    conditioner_string = "conditioner_{}".format(i)
    conditioner = params[conditioner_string]["conditioner"]
    # # todo: support more general network

    # initial_layer
    layers_0 = conditioner["layers_0"]
    # print("layers_0: ")
    kernel = layers_0["kernel"]
    bias = layers_0["bias"]

    # mask=True denotes the invariant dimension, which is used as input to the conditioner
    kernel = kernel[mask, :]

    model_dict[conditioner_string] = {"layers_0": {"kernel": kernel, "bias": bias}}

    print("Initial layer: ")
    print("kernel: ", kernel)
    print("bias: ", bias)

    # 1st MLP
    layers_1 = conditioner["layers_1"]
    # print("layers_1: ")
    layers_1_0 = layers_1["layers_0"]
    layers_1_1 = layers_1["layers_1"]

    # print("layers_1_0 MLP: ")
    kernel = layers_1_0["kernel"]
    bias = layers_1_0["bias"]
    # print("kernel: ", kernel)
    # print("bias: ", bias)

    model_dict[conditioner_string]["layers_1"] = {"layers_0": {"kernel": kernel, "bias": bias}}

    # print("layers_1_1 MLP: ")
    kernel = layers_1_1["kernel"]
    bias = layers_1_1["bias"]
    # print("kernel: ", kernel)
    # print("bias: ", bias)

    model_dict[conditioner_string]["layers_1"]["layers_1"] = {"kernel": kernel, "bias": bias}

    # 2nd MLP
    layers_2 = conditioner["layers_2"]
    # print("layers_2: ")
    layers_2_0 = layers_2["layers_0"]
    layers_2_1 = layers_2["layers_1"]

    # print("layers_2_0: ")
    kernel = layers_2_0["kernel"]
    bias = layers_2_0["bias"]
    # print("kernel: ", kernel)
    # print("bias: ", bias)

    model_dict[conditioner_string]["layers_2"] = {"layers_0": {"kernel": kernel, "bias": bias}}

    # print("layers_2_1: ")
    kernel = layers_2_1["kernel"]
    bias = layers_2_1["bias"]
    # print("kernel: ", kernel)
    # print("bias: ", bias)

    model_dict[conditioner_string]["layers_2"]["layers_1"] = {"kernel": kernel, "bias": bias}


    # Last layer
    # mask=False denotes the layer to change, which is the output of the conditioner. The output of the last layer is [the number of mask=False] * (3*n_bins-1) derivative to remove

    x = np.zeros(n_dim * (n_bins * 3 + 1), dtype=bool)

    for i, value in enumerate(mask):
        start_index = (n_bins * 3 + 1) * i
        end_index = (n_bins * 3 + 1) * i + (n_bins * 2)
        if value:
            x[start_index:end_index] = False
        else:
            x[start_index:end_index] = True
            # x[end_index + 1 : end_index + n_bins] = True        # Maybe this is right. We should skip the first and last derivative
            x[end_index  : end_index + n_bins + 1] = True        # Maybe this is right. We should skip the first and last derivative
        # x[start_index:end_index] = not value


    layers_4 = conditioner["layers_4"]
    # print("layers_4: ")
    kernel = layers_4["kernel"]
    kernel = kernel[:, x]
    bias = layers_4["bias"]
    bias = bias[x]
    # print("kernel: ", kernel)
    # print("bias: ", bias)

    model_dict[conditioner_string]["layers_4"] = {"kernel": kernel, "bias": bias}

    # print("see current mask")
    # print(mask)
    mask = jnp.logical_not(mask)

# print(model_dict)




# use pickle to save model_dict
import pickle
pickle_dir = PROJECT_PATH

# save the model_dict
with open(os.path.join(pickle_dir, "model_dict.pickle"), "wb") as f:
    pickle.dump(model_dict, f)

print("Save Model to :     ", os.path.join(pickle_dir, "model_dict.pickle"))

# re-load the model_dict to validate whether saved correctly
with open(os.path.join(pickle_dir, "model_dict.pickle"), "rb") as f:
    model_dict = pickle.load(f)
# print(model_dict)

