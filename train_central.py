import jax
import jax.numpy as jnp  # JAX NumPy
import jax.random as random  # JAX random
import optax  # Optimizers
import flax  # Deep Learning library for JAX
import numpy as np
import os
import sys
sys.path.append("..")
import pandas as pd
import datasets
from DP.accounting import *
from model import RQSpline, LocRQSpline, MaskRQSpline
from model import mask_selection
# from model.utils import make_training_loop, sample_nf, restore_checkpoint, save_checkpoint, create_train_state, create_schedule_train_state
from model.utils_nightly import make_training_loop_unsupervised, create_train_state, create_schedule_train_state
# from flowMC.nfmodel.utils import make_training_loop, sample_nf
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import model as my_model
from model.utils import process_model_dict_by_model_type
import pickle
import optuna
import logging
from datetime import datetime
from commonSetting import PROJECT_PATH

target_epsilon = 10


def main_uci(trial):
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    day = datetime.now().strftime("%m-%d-%H:%M")

    """ Parameters to tune """
    l2_norm_clip = trial.suggest_float("l2_norm_clip", 0.2, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 5e-1, log=False)



    # dataset_name = "power"

    # dataset_name = "imdb"

    # dataset_name = "BJAQ"
    dataset_name = "flights"
    # dataset_name = "imdbfull"

    model_name = "RQSpline"
    # model_name = "MaskRQSpline"

    # IF_DPSGD = True
    IF_DPSGD = False

    # 【Optimization parameters】
    num_epochs = 40
    batch_size = 512

    if IF_DPSGD == True:
        learning_rate = learning_rate
    else:
        learning_rate = learning_rate
    momentum = 0.9


    loc_alpha = 0.05


    pruneParams = None


    # Privacy parameters
    l2_norm_clip = l2_norm_clip        # l2 norm clip in DPSGD


    # do not need to set anymore
    # noise_multiplier = 0.45     # for power
    # noise_multiplier = 0.54     # for BJAQ
    # noise_multiplier = 0.43     # for flights
    # noise_multiplier = 0.44     # for imdbfull

    seed = 1337

    print(f"Current learning rate is [{learning_rate}]")
    print(f"Current l2_norm_clip is [{l2_norm_clip}]")

    # Model parameters (In NSF paper)
    # n_layers = 5
    # n_hiddens = [64, 64]

    n_layers = 10
    n_hiddens = [256, 256]

    n_bins = 8

    # n_layers = 4
    # n_hiddens = [128, 128]
    # # n_hiddens = [64, 64]
    # n_bins = 8



    # Model parameters (Stable)
    tail_bound = [-3, 3]
    grad_norm_clip_value = 5.0  # l2 norm clip without DPSGD
    # if IF_DPSGD:
    #     grad_norm_clip_value = l2_norm_clip





    read_data_name = dataset_name + "Split"
    data, num_data, n_dim =datasets.__dict__[read_data_name]("train")
    val_data, val_num, val_dim =datasets.__dict__[read_data_name]("val")
    test_data, test_num, test_dim =datasets.__dict__[read_data_name]("test")

    print("see len of data", len(data))
    steps_per_epoch = int(len(data)//batch_size)
    print("steps_per_epoch", steps_per_epoch)

    # data, num_data, n_dim= datasets.__dict__[dataset_name]()
    key, rng_model, rng_init, rng_train, rng_nf_sample = jax.random.split(
        jax.random.PRNGKey(0), 5
    )


    # model = RQSpline(n_dim, n_layers, n_hiddens, n_bins, tail_bound)
    # model = MaskRQSpline(n_dim, n_layers, n_hiddens, n_bins, tail_bound, loc_alpha=loc_alpha)
    # model = LocRQSpline(n_dim, n_layers, n_hiddens, n_bins, tail_bound, loc_alpha=loc_alpha)

    pre_mask_list = []
    for i in range(2):
        pre_mask = mask_selection(256, 256, mask_lim=loc_alpha, rng_seed=2333 + i)
        pre_mask_list.append(pre_mask)


    # model = my_model.__dict__[model_name](n_dim, n_layers, n_hiddens, n_bins, tail_bound,
    #                                                loc_alpha)

    model = my_model.__dict__[model_name](n_dim, n_layers, n_hiddens, n_bins, tail_bound,
                                                   loc_alpha, pre_mask_list)

    variables = model.init(rng_model, jnp.ones((1, n_dim)))["variables"]
    variables = variables.unfreeze()
    variables["base_mean"] = jnp.mean(data, axis=0)
    variables["base_cov"] = jnp.cov(data.T)
    variables = flax.core.freeze(variables)
    print("see variables")
    print(variables["base_mean"])
    print(variables["base_cov"])


    steps = num_epochs * (num_data//batch_size)
    print("steps is ", steps)


    # delta_list = [1e-5, 1e-6, 1e-7]
    # for delta in delta_list:
    #     EPSILON = compute_epsilon(steps=steps,
    #                               batch_size=batch_size,
    #                               num_data=num_data,
    #                               noise_multiplier=noise_multiplier,
    #                               target_delta=delta)
    #     print(f"Delta = [{delta}]  EPSILON is {EPSILON}")



    # new: search the target noise_multiplier
    delta = 1.0 / num_data
    noise_l = 0
    noise_r = 1
    while noise_r - noise_l > 1e-3:
        noise_m = (noise_l + noise_r) / 2
        EPSILON = compute_epsilon(steps=steps,
                                  batch_size=batch_size,
                                  num_data=num_data,
                                  noise_multiplier=noise_m,
                                  target_delta=delta)
        if EPSILON > target_epsilon:
            noise_l = noise_m
        else:
            noise_r = noise_m
    noise_multiplier = (noise_l + noise_r) / 2
    print("Searched noise_multiplier is ---  [{}]".format(noise_multiplier))




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



    # train_flow, train_epoch, train_step, evaluate = make_training_loop(model, IF_DPSGD=IF_DPSGD)
    train_flow, train_epoch, train_step, evaluate = make_training_loop_unsupervised(model, IF_DPSGD=IF_DPSGD)

    rng, state, loss_values, gradient_norms, state_list, variables = train_flow(
        rng_train, state, variables, data, num_epochs, batch_size, val_data, pruneParams=pruneParams
    )

    # print(variables)
    # print(state.params)




    workdir = os.path.join(PROJECT_PATH, f"output/{dataset_name}/")
    if IF_DPSGD == False:
        subdir = f"{day}-{model_name}-epoch-{num_epochs}-batch-{batch_size}-lr-{learning_rate}-DP-{IF_DPSGD}-layers-{n_layers}"
    else:
        subdir = f"{day}-{model_name}-epoch-{num_epochs}-batch-{batch_size}-lr-{learning_rate}-DP-{IF_DPSGD}-clip-{l2_norm_clip}-noise-{noise_multiplier}-epsion-{EPSILON:.2f}-layers-{n_layers}"

    try:

        # best_state.client_states = None
        # save_checkpoint(best_state, workdir + subdir)

        to_save_state = {"params": state.params}

        to_save_state = process_model_dict_by_model_type(to_save_state,
                                                         model_name,
                                                         n_layers,
                                                         variables)

        # first create the workdir + subdir
        if not os.path.exists(workdir + subdir):
            os.makedirs(workdir + subdir)
        pickle.dump(to_save_state, open(workdir + subdir + "/params.pkl", "wb"))
        # save_checkpoint(best_state, workdir + subdir)



    except Exception as e:
        if isinstance(e, FileExistsError):
            print("[Error] A checkpoint with the same hyper-parameters already exists.")
        else:
            print(f"[Error] An error occurred when saving the checkpoint: {str(e)}")
    print("CKPT saved to ", workdir + subdir)




    test_loss = evaluate(state.params, test_data, variables)
    print("Test log likelihood is ", test_loss)
    return test_loss

# search_space = {
#     'l2_norm_clip': [0.5],
#     'learning_rate': [0.09]
# }
#
# search_space = {
#     'l2_norm_clip': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1, 2, 3, 4],
#     'learning_rate': [0.01 * i for i in range(2, 10)] + [0.1, 0.2, 0.3, 0.4, 0.5]
# }

# For DP
search_space = {


    'l2_norm_clip': [0.6],
    # 'learning_rate': [0.07]

    'learning_rate': [0.0005]


    # 'learning_rate': [0.005]
    # 'learning_rate': [0.001]
}


# for non-DP
# search_space = {
#     'l2_norm_clip': [0.6],
#     # 'learning_rate': [0.0005, 0.0001, 0.001]
#     'learning_rate': [0.0005]
# }

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
# study = optuna.create_study(pruner=optuna.pruners.SuccessiveHalvingPruner())

study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))


study.optimize(main_uci, n_trials=20000, timeout=3600*120)
print("Best params:")
print(study.best_params)
print("Best value:")
print(study.best_value)

