
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
# from model.utils import restore_checkpoint, save_checkpoint
# from model.utils_nightly import restore_checkpoint, save_checkpoint
from model import mask_selection
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import fedjax
import FL.fast_stateful_fed_avg as fast_stateful_fed_avg
from absl.testing import absltest
import time
import datasets.data_util as data_util
import model as my_model
from config.default_config import default_args
from collections import namedtuple
from tqdm import trange, tqdm
import sys
import optuna
import logging
from model.utils_nightly import process_model_dict_by_model_type
import pickle
from datetime import datetime
from commonSetting import PROJECT_PATH


# dataset_name = "power"
# dataset_name = "imdb"
# dataset_name = "flights"
# dataset_name = "imdbfull"
dataset_name = "BJAQ"

target_epsilon = 10

def learning_rate_schedule(init_lr, steps_per_epoch, epochs):
    @jax.jit
    def schedule(count):
        epoch = count // steps_per_epoch  # Assuming you have a variable to keep track of the current training step
        return init_lr * jnp.cos(epoch / epochs * jnp.pi / 2)
        # return init_lr / (2 ** epoch)  # Reduce the learning rate by a factor of 2 each epoch
    return schedule

    # schedule = jax.jit(schedule)


def main_fed(trial):
    day = datetime.now().strftime("%m-%d-%H:%M")


    l2_norm_clip = trial.suggest_float("l2_norm_clip", 0.2, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 5e-1, log=False)



    args = default_args[dataset_name]

    # args["model_name"] = "RQSpline"
    # args["model_name"] = "MaskRQSpline"
    args["model_name"] = "PosMulRQSpline"
    # args["model_name"] = "PosAddRQSpline"
    args["cuda"] = 3

    args["non_iid_degree"] = 0


    args["learning_rate"] = learning_rate


    args["loc_alpha"] = 0.05



    args["client_steps"] = None
    args["client_epochs"] = 1
    args["check_interval"] = 1

    # args["client_steps"] = 100
    # args["client_epochs"] = None
    # args["check_interval"] = 8

    # args["num_epochs"] = 60
    args["num_epochs"] = 40
    # args["num_epochs"] = 1


    args["IF_DPSGD"] = True

    args["l2_norm_clip"] = l2_norm_clip
    args["grad_norm_clip"] = 5

    args["n_client"] = 4



    # args["n_layers"] = 5
    # args["n_hiddens"] = [64, 64]
    args["n_layers"] = 10
    args["n_hiddens"] = [256, 256]

    args["orthogonal_regu"] = 1e-3  # orthogonal regularization


    args["nsample_clients"] = int(args["n_client"])


    param_names = args.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**args)

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.cuda}'
    data_rng = np.random.RandomState(args.data_rng_seed)
    print("#" * 50)
    print(" " * 20, "See args: ")
    print(args)


    read_data_name = dataset_name + "Split"
    data, num_data, n_dim = datasets.__dict__[read_data_name]("train")
    val_data, val_num, val_dim = datasets.__dict__[read_data_name]("val")
    test_data, test_num, test_dim = datasets.__dict__[read_data_name]("test")


    print("train_data.shape", data.shape)
    print("test_data.shape", test_data.shape)


    fed_num_data = int(num_data / args.n_client)
    print("fed_num_data ", fed_num_data)


    client_steps = args.client_steps
    if client_steps is None:
        assert args.client_epochs is not None

        client_steps = (args.client_epochs * (fed_num_data // args.batch_size))
    steps = args.num_epochs * (num_data // args.batch_size)
    fed_steps = int(np.ceil(steps / (client_steps * args.n_client)))

    print("Total steps : ", steps)
    print("Each local train, client steps ", client_steps)

    key, rng_model, rng_init, rng_train, rng_nf_sample = jax.random.split(
        jax.random.PRNGKey(0), 5
    )

    pre_mask_list = []
    for i in range(2):
        pre_mask = mask_selection(256, 256, mask_lim=args.loc_alpha, rng_seed=2333 + i)
        pre_mask_list.append(pre_mask)

    model = my_model.__dict__[args.model_name](n_dim, args.n_layers, args.n_hiddens, args.n_bins, args.tail_bound,
                                               args.loc_alpha, pre_mask_list)

    variables = model.init(rng_model, jnp.ones((1, n_dim)))["variables"]
    variables = variables.unfreeze()
    variables["base_mean"] = jnp.mean(data, axis=0)
    variables["base_cov"] = jnp.cov(data.T)
    variables = flax.core.freeze(variables)


    # delta_list = [1e-5, 1e-6, 1e-7]
    # for delta in delta_list:

    delta = 1.0 / fed_num_data
    # each_client_nsteps = steps
    # each_client_nsteps = int(args.epo * args.nsample_clients / args.n_client)
    each_client_nsteps = int(
        args.num_epochs * fed_num_data / args.batch_size * args.nsample_clients / args.n_client)


    # new: search the target noise_multiplier
    noise_l = 0
    noise_r = 1
    while noise_r - noise_l > 1e-3:
        noise_m = (noise_l + noise_r) / 2
        EPSILON = compute_epsilon(steps=each_client_nsteps,
                                  batch_size=args.batch_size,
                                  num_data=fed_num_data,
                                  noise_multiplier=noise_m,
                                  target_delta=delta)
        if EPSILON > target_epsilon:
            noise_l = noise_m
        else:
            noise_r = noise_m
    noise_multiplier = (noise_l + noise_r) / 2
    print("Searched noise_multiplier is ---  [{}]".format(noise_multiplier))
    # EPSILON = compute_epsilon(steps=each_client_nsteps,
    #                           batch_size=args.batch_size,
    #                           # num_data=num_data,
    #                           num_data=fed_num_data,
    #                           noise_multiplier=args.noise_multiplier,
    #                           target_delta=delta)
    print(f"Delta = [{delta}]  EPSILON is {EPSILON}")

    def loss(params, batch, rng):
        del rng
        log_det = model.apply(
            {"params": params, "variables": variables}, batch, method=model.log_prob
        )
        # ==================== original loss ====================
        return -jnp.mean(log_det)         # original loss

        # ==================== orthogonal loss ====================
        # # todo: add orthogonal regularization, i.e. computing the |WW^T - I| for each `kernel`
        # # 【2024-01-16】  simply changed the loss with adding orthogonal_regularization
        # #   But not sure if current implementation is correct, since no mask is used here
        # leaves = jax.tree_util.tree_leaves(params)
        # orthogonal_regularization = 0
        # reg_coef = args.orthogonal_regu
        #
        # first_mask = variables["conditioner_0"]["conditioner"]["layers_0"]["mask"]
        # second_mask = variables["conditioner_0"]["conditioner"]["layers_1"]["layers_0"]["mask"]
        #
        # for x in leaves:
        #     if x.ndim == 2:
        #         # tmp = jnp.matmul(x.T, x)
        #
        #         # # jax.debug.print("weight shape {y}", y=x.shape)
        #         if x.shape == first_mask.shape:
        #             # jax.debug.print("use 1st mask!")
        #             masked_weight = jnp.multiply(first_mask, x)
        #         elif x.shape == second_mask.shape:
        #             # jax.debug.print("use 2nd mask!")
        #             masked_weight = jnp.multiply(second_mask, x)
        #         else:
        #             # jax.debug.print("not use mask!")
        #             masked_weight = x
        #         tmp = jnp.matmul(masked_weight.T, masked_weight)
        #
        #         ones = jnp.ones_like(tmp)
        #         eyes = jnp.eye(tmp.shape[0])
        #         tmp = jnp.multiply(tmp, (ones - eyes))
        #         orthogonal_regularization = orthogonal_regularization + jnp.sum(jnp.square(tmp))
        # # jax.debug.print("log_det    loss: {x}",x=-jnp.mean(log_det))
        # # jax.debug.print("orthogonal loss: {x}",x=reg_coef * orthogonal_regularization)
        # return -jnp.mean(log_det) + reg_coef * orthogonal_regularization


    grad_fn = jax.grad(loss)

    if args.IF_DPSGD == True:

        grad_fn = jax.vmap(grad_fn, in_axes=(None, 0, None))

    grad_fn = jax.jit(grad_fn)

    def evaluate_batch(params, batch, variables):
        log_det = model.apply(
            {"params": params, "variables": variables}, batch, method=model.log_prob
        )
        nll = -jnp.mean(log_det)
        return nll

    evaluate_batch = jax.jit(evaluate_batch)

    test_batch_size = 1601 * 16

    def evaluate(params, evalData, variables):
        num_test_steps = int(np.ceil(evalData.shape[0] / test_batch_size))
        nll_values = jnp.zeros(num_test_steps)
        for i in range(num_test_steps):
            batch_data = evalData[i * test_batch_size:(i + 1) * test_batch_size, :]
            nll_values = nll_values.at[i].set(evaluate_batch(params, batch_data, variables))
        return jnp.mean(nll_values)

    evaluate = jax.jit(evaluate)


    scheduler = optax.cosine_decay_schedule(args.learning_rate, decay_steps=steps, alpha=0)
    # scheduler = optax.exponential_decay(args.learning_rate, transition_steps=5000, decay_rate=0.1, end_value=0.000001)
    clip_scheduler = optax.cosine_decay_schedule(args.l2_norm_clip, decay_steps=steps, alpha=0.5)

    params = model.init(rng_init, jnp.ones((1, n_dim)))["params"]

    # Build client optimizer
    if args.IF_DPSGD == False:
        gradient_transform = optax.chain(
            optax.clip_by_global_norm(args.grad_norm_clip_value),

            optax.scale_by_adam(),  # Use the updates from adam.


            # (optax.trace(decay=args.momentum, nesterov=False)
            #  if args.momentum is not None else optax.identity()),

            optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1.0)
        )


    else:

        nesterov = False
        # Combining gradient transforms using `optax.chain`.
        gradient_transform = optax.chain(
            # optax.clip_by_global_norm(args.grad_norm_clip_value),
            # optax.differentially_private_aggregate(l2_norm_clip=args.l2_norm_clip,

            # optax.differentially_private_aggregate(l2_norm_clip=clip_scheduler,
            optax.differentially_private_aggregate(l2_norm_clip=args.l2_norm_clip,
                                                   noise_multiplier=noise_multiplier,
                                                   seed=args.seed),  # Use the updates from adam.
            # optax.scale_by_adam(),

            # DP_changeC.differentially_private_aggregate(
            #                                        noise_multiplier=noise_multiplier,
            #                                        C_fn=clip_scheduler,
            #                                        seed=args.seed),  # Use the updates from adam.

            # momentum
            (optax.trace(decay=args.momentum, nesterov=nesterov)
             if args.momentum is not None else optax.identity()),

            # optax.scale_by_adam(),

            # 【if using scheduler】
            optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1.0)

            # 【if do not use scheduler】
            # optax.scale(-args.learning_rate)
        )

    client_optimizer = fedjax.optimizers.create_optimizer_from_optax(gradient_transform)

    # Build server optimizer
    # gradient_transform_server = optax.chain(
    #     # optax.clip_by_global_norm(grad_norm_clip_value),
    #     # optax.scale_by_adam(b1=momentum),  # Use the updates from adam.
    #     # optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
    #     # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    #     # optax.scale(-1.0)
    #     optax.scale_by_
    # )
    # server_optimizer = fedjax.optimizers.create_optimizer_from_optax(gradient_transform_server)
    # # server_optimizer = fedjax.optimizers.create_optimizer_from_optax(gradient_transform)

    # Standard server optimizer [used for test correctness]
    server_optimizer = fedjax.optimizers.sgd(learning_rate=1.0)

    # client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(
    #     batch_size=args.batch_size, num_steps=args.client_steps, seed=rng_train, num_epochs=None)


    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(
        batch_size=args.batch_size,

        num_steps=args.client_steps,
        num_epochs=args.client_epochs
        # num_steps=None,
        # num_epochs=1

        , seed=rng_train)
        # , seed=1234)
    #
    # algorithm = fed_avg.federated_averaging(grad_fn, client_optimizer,
    #                                        server_optimizer,
    #                                        client_batch_hparams)

    # algorithm = fast_fed_avg.federated_averaging(grad_fn, client_optimizer,
    #                                        server_optimizer,
    #                                        client_batch_hparams)
    algorithm = fast_stateful_fed_avg.federated_averaging(grad_fn, client_optimizer,
                                                          server_optimizer,
                                                          client_batch_hparams)

    state = algorithm.init(params)




    tmp_data = data

    client_data, data_ids = data_util.split_data_follow_non_iid_degree(tmp_data, args.non_iid_degree, args.n_client,
                                                                       data_rng)
    data_util.analyze_non_iid_ness(client_data, data_ids, num_data)



    client_id_list = [f'cid{i}'.encode() for i in range(args.n_client)]


    clients = [
        (client_id_list[i], fedjax.ClientDataset({'data': jnp.asarray(client_data[i])}),
         jax.random.PRNGKey(i)) for i in range(args.n_client)
    ]


    best_state = state
    best_loss = 1e9

    num_check = int(np.ceil(fed_steps / args.check_interval))
    print(f"Total number of check: [{num_check}]")
    loss_values = jnp.zeros(num_check)

    pbar = trange(fed_steps, desc="Federated Training NF ", miniters=args.check_interval)

    client_id_and_data = {clients[i][0]: {'data': client_data[i]} for i in range(args.n_client)}
    # print(client_id_and_data)
    client_id_and_data = fedjax.InMemoryFederatedData(client_id_and_data)

    # for itr in range(fed_steps):
    for itr in pbar:

        efficient_sampler = fedjax.client_samplers.UniformShuffledClientSampler(
            # client_id_and_data, num_clients=1)
            client_id_and_data.shuffled_clients(buffer_size=100, seed=0), num_clients=args.nsample_clients)
        sampled_clients_with_data = efficient_sampler.sample()
        state, client_diagnostics = algorithm.apply(state, sampled_clients_with_data)


        if itr % args.check_interval == 0:
            st_time = time.time()

            # nll = evaluate(state.params, test_data, variables)
            nll = evaluate(state.params, val_data, variables)
            if nll > 100 or jnp.isnan(nll):
                print("nll is ", nll)
                return 100000
            value = nll
            loss_values = loss_values.at[int(itr // args.check_interval)].set(value)

            if value < best_loss:
                best_state = state
                best_loss = value

            # print(f"[{itr}]Test Loss (NLL): ", nll)
            print("")
            print(f"[{itr}]Val Loss (NLL): ", nll)
            print(f"[{itr}]Best Val Loss (NLL): ", best_loss)
            print("-- Evaluation Time used: ", time.time() - st_time)
            print("All loss values are ", loss_values)

            # print(client_diagnostics)


    workdir = os.path.join(PROJECT_PATH, f"output/{args.dataset_name}/")
    if args.IF_DPSGD == False:
        subdir = f"{args.model_name}-{day}-nc-{args.n_client}-epoch-{args.num_epochs}-batch-{args.batch_size}-lr-{args.learning_rate}-nl-{args.n_layers}-hidden-{args.n_hiddens[0]}-DP-{args.IF_DPSGD}"
    else:
        subdir = f"{args.model_name}-{day}-nc-{args.n_client}-epoch-{args.num_epochs}-batch-{args.batch_size}-lr-{args.learning_rate}-nl-{args.n_layers}-hidden-{args.n_hiddens[0]}-DP-{args.IF_DPSGD}-clip-{args.l2_norm_clip}-noise-{noise_multiplier}-epsion-{EPSILON:.2f}"

    try:
        # best_state.client_states = None
        # save_checkpoint(best_state, workdir + subdir)

        to_save_state = {"params":best_state.params}
        to_save_state = process_model_dict_by_model_type(to_save_state,
                                                         args.model_name,
                                                         args.n_layers,
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


    test_loss = evaluate(best_state.params, test_data, variables)
    print("Test log likelihood is ", test_loss)


    # new_state = restore_checkpoint(state, workdir)

    # sleep for a long time
    # import time
    # time.sleep(3600*24*7)

    return test_loss



search_space = {
        'l2_norm_clip': [1],
        # 'learning_rate': [0.06] ,
        'learning_rate': [0.08] ,
    # 'l2_norm_clip': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # # 'learning_rate': [0.001 * i for i in range(1, 10, 2)] + [0.01 * i for i in range(1, 10, 2)] + [0.1, 0.2, 0.3, 0.4, 0.5]
    # 'learning_rate': [0.001 * i for i in range(1, 10, 2)] + [0.01 * i for i in range(1, 10)] + [0.1, 0.2, 0.3, 0.4, 0.5]    #
}


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
# study = optuna.create_study(pruner=optuna.pruners.SuccessiveHalvingPruner())

# ====================  grid sampler  ====================
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
# ====================  Tree sampler  ====================
# study = optuna.create_study(sampler=optuna.samplers.TPESampler(search_space))

# study = optuna.create_study(
#
# )
study.optimize(main_fed, n_trials=20000, timeout=3600*120)
print("Best params:")
print(study.best_params)
print("Best value:")
print(study.best_value)

