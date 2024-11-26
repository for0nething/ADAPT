import jax
import jax.numpy as jnp  # JAX NumPy
import jax.random as random  # JAX random
import optax  # Optimizers
import flax  # Deep Learning library for JAX
import numpy as np
import os
import pandas as pd
import sys
sys.path.append("..")

import datasets
from DP.accounting import *
# from model.utils import restore_checkpoint, save_checkpoint
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import fedjax
import FL.fast_stateful_fed_avg as fast_stateful_fed_avg
from absl.testing import absltest
import time
import datasets.data_util as data_util
import model as my_model
from model import mask_selection
import my_tree_util
from config.default_config import default_args
from collections import namedtuple
from tqdm import trange, tqdm
# from model.utils import process_model_dict_by_model_type
import pickle
# from fast_stateful_fed_avg_more_nightly import ServerState
from FL.fast_stateful_fed_avg import ServerState
from model.utils_nightly import make_training_loop_unsupervised, make_training_loop_supervised,   create_train_state, create_schedule_train_state,process_model_dict_by_model_type
import optuna
import logging
from datetime import datetime
from flax.training import train_state  # Useful dataclass to keep train state
from commonSetting import PROJECT_PATH


# dataset_name = "power"
# dataset_name = "imdb"
# dataset_name = "BJAQ"
# dataset_name = "flights"
dataset_name = "imdbfull"
target_epsilon = 10

Array = jnp.ndarray

def learning_rate_schedule(init_lr, steps_per_epoch, epochs):
    @jax.jit
    def schedule(count):
        epoch = count // steps_per_epoch  # Assuming you have a variable to keep track of the current training step
        return init_lr * jnp.cos(epoch / epochs * jnp.pi / 2)
        # return init_lr / (2 ** epoch)  # Reduce the learning rate by a factor of 2 each epoch
    return schedule

    # schedule = jax.jit(schedule)


def main_fed(trial):


    """ Parameters to tune """
    l2_norm_clip = trial.suggest_float("l2_norm_clip", 0.2, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 5e-1, log=False)
    sup_learning_rate = trial.suggest_float("sup_learning_rate", 1e-9, 1e-7, log=False)
    sup_batch_size = trial.suggest_int("sup_batch_size", 1, 1024, log=False)
    sup_itr_num = trial.suggest_int("sup_itr_num", 1, 5000, log=False)
    num_sup_epochs = trial.suggest_int("num_sup_epochs", 1, 5000, log=False)


    args = default_args[dataset_name]

    day = datetime.now().strftime("%m-%d-%H:%M")

    args["save_ckpt_in_training"] = False

    # args["model_name"] = "LocRQSpline"
    args["model_name"] = "MaskRQSpline"
    # args["model_name"] = "RQSpline"
    args["cuda"] = 3

    args["non_iid_degree"] = 0
    args["learning_rate"] = learning_rate
    args["sup_learning_rate"] = sup_learning_rate       # 【new】
    args["sup_batch_size"] = sup_batch_size             # 【new】
    args["loc_alpha"] = 0.05

    args["client_steps"] = None
    args["client_epochs"] = 1
    args["check_interval"] = 1

    # args["client_steps"] = 100
    # args["client_epochs"] = None
    # args["check_interval"] = 8


    args["num_epochs"] = 40


    args["IF_DPSGD"] = True

    args["l2_norm_clip"] = l2_norm_clip
    args["grad_norm_clip"] = 5.


    # # BJAQ  4 clients
    # args["noise_multiplier"] = 0.62
    # args["n_client"] = 4

    # power  4 clients
    # args["noise_multiplier"] = 0.52
    # args["n_client"] = 4

    # IMDB  4 clients
    # args["noise_multiplier"] = 0.5
    # args["n_client"] = 4

    # args["noise_multiplier"] = 0.56
    # args["n_client"] = 8

    # BJAQ 4 clients
    # args["noise_multiplier"] = 0.65   # for BJAQ
    # args["n_client"] = 4

    # args["noise_multiplier"] = 0.48 # for flights
    # args["n_client"] = 4

    # args["noise_multiplier"] = 0.5  # for imdbfull
    # args["n_client"] = 4



    # 【power 4 clients】
    # args["noise_multiplier"] = 0.52
    # args["n_client"] = 4

    # 【imdb 4 clients】
    # args["noise_multiplier"] = 0.48
    # args["n_client"] = 4

    # 【vary number of clients】

    # imdbfull
    # n_client = 128
    # args["noise_multiplier"] = 0.79
    # args["n_client"] = 128

    # n_client = 64
    # args["noise_multiplier"] = 0.7
    # args["n_client"] = 64


    args["n_hiddens"] = [256, 256]
    # args["n_hiddens"] = [264, 264]
    args["n_layers"] = 10

    # args["nsample_clients"] = int(args["n_client"] / 4)
    args["nsample_clients"] = int(args["n_client"])

    # new
    args["sup_itr_num"] = sup_itr_num
    args["orthogonal_regu"] = 1e-4


    param_names = args.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**args)

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.cuda}'
    data_rng = np.random.RandomState(args.data_rng_seed)
    print("#" * 50)
    print(" " * 20, "See args: ")
    print(args)

    # data, num_data, n_dim= datasets.__dict__[dataset_name]()
    #  train / val / test
    read_data_name = dataset_name + "Split"
    data, num_data, n_dim = datasets.__dict__[read_data_name]("train")
    val_data, val_num, val_dim = datasets.__dict__[read_data_name]("val")      # (184435, 6)
    test_data, test_num, test_dim = datasets.__dict__[read_data_name]("test")

    print("train_data.shape", data.shape)
    print("val_data.shape", val_data.shape)
    print("test_data.shape", test_data.shape)


    fed_num_data = int(num_data / args.n_client)
    print("fed_num_data ", fed_num_data)


    client_steps = args.client_steps
    if client_steps is None:
        assert args.client_epochs is not None
        client_steps = (args.client_epochs * (fed_num_data // args.batch_size))
    steps = args.num_epochs * (num_data // args.batch_size)  # total step num
    fed_steps = int(np.ceil(steps / (client_steps * args.n_client)))
    # fed_steps = int(args.num_epochs)
    print("Total steps : ", steps)
    print("Each local train, client steps ", client_steps)

    key, rng_model, rng_init, rng_train, rng_nf_sample = jax.random.split(
        jax.random.PRNGKey(0), 5
    )

    pre_mask_list = []
    for i in range(2):
        pre_mask = mask_selection(256, 256, mask_lim=args.loc_alpha, rng_seed=2333 + i)
        pre_mask_list.append(pre_mask)

    # construct model
    model = my_model.__dict__[args.model_name](n_dim, args.n_layers, args.n_hiddens, args.n_bins, args.tail_bound,
                                               args.loc_alpha, pre_mask_list)
    # model = RQSpline(n_dim, n_layers, n_hiddens, n_bins, tail_bound)
    # model = LocRQSpline(n_dim, n_layers, n_hiddens, n_bins, tail_bound)

    variables = model.init(rng_model, jnp.ones((1, n_dim)))["variables"]
    variables = variables.unfreeze()
    variables["base_mean"] = jnp.mean(data, axis=0)
    variables["base_cov"] = jnp.cov(data.T)
    variables = flax.core.freeze(variables)


    import distrax
    base_dist = distrax.Independent(
        distrax.MultivariateNormalFullCovariance(
            loc=jnp.zeros(n_dim),
            covariance_matrix=jnp.eye(n_dim),
        )
    )

    # unsupervise
    train_flow_unsup, train_epoch_unsup, train_step_unsup, evaluate_unsup = make_training_loop_unsupervised(model,
                                                                                                            IF_DPSGD=False)
    # supervise
    train_flow_sup, train_epoch_sup, train_step_sup, evaluate_sup, compute_expect_log_prob, get_samples, fast_train_flow_sup = make_training_loop_supervised(model,
                                                                                                  IF_DPSGD=False)



    delta = 1.0 / fed_num_data
    # each_client_nsteps = steps
    # each_client_nsteps = int(args.epo * args.nsample_clients / args.n_client)
    each_client_nsteps = int(
        args.num_epochs * fed_num_data / args.batch_size * args.nsample_clients / args.n_client)

    # EPSILON = compute_epsilon(steps=each_client_nsteps,
    #                           batch_size=args.batch_size,
    #                           # num_data=num_data,
    #                           num_data=fed_num_data,
    #                           noise_multiplier=args.noise_multiplier,
    #                           target_delta=delta)
    # print(f"Delta = [{delta}]  EPSILON is {EPSILON}")

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
    print(f"Delta = [{delta}]  EPSILON is {EPSILON}")



    def loss(params, batch, rng):
        del rng
        log_det = model.apply(
            {"params": params, "variables": variables}, batch, method=model.log_prob
        )
        # return -jnp.mean(log_det)         # original loss

        # todo: add orthogonal regularization, i.e. computing the |WW^T - I| for each `kernel`
        # 【2024-01-16】  simply changed the loss with adding orthogonal_regularization
        #   But not sure if current implementation is correct, since no mask is used here
        leaves = jax.tree_util.tree_leaves(params)
        orthogonal_regularization = 0
        reg_coef = args.orthogonal_regu

        first_mask = variables["conditioner_0"]["conditioner"]["layers_0"]["mask"]
        second_mask = variables["conditioner_0"]["conditioner"]["layers_1"]["layers_0"]["mask"]

        for x in leaves:
            if x.ndim == 2:

                tmp = jnp.matmul(x.T, x)


                # if x.shape == first_mask.shape:
                #     masked_weight = jnp.multiply(first_mask, x)
                # elif x.shape == second_mask.shape:
                #     masked_weight = jnp.multiply(second_mask, x)
                # else:
                #     masked_weight = x
                # tmp = jnp.matmul(masked_weight.T, masked_weight)

                ones = jnp.ones_like(tmp)
                eyes = jnp.eye(tmp.shape[0])
                tmp = jnp.multiply(tmp,  (ones-eyes))
                orthogonal_regularization = orthogonal_regularization + jnp.sum(jnp.square(tmp))
        return -jnp.mean(log_det) + reg_coef * orthogonal_regularization


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

    # class FedAvgTest(absltest.TestCase):


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
            #                                        noise_multiplier=args.noise_multiplier,
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
        # , seed=None)
    #

    algorithm = fast_stateful_fed_avg.federated_averaging(grad_fn, client_optimizer,
                                                          server_optimizer,
                                                          client_batch_hparams)

    state = algorithm.init(params)



    # np.random.shuffle(data)
    #
    # tmp_data = data[np.argsort(data[:, 1])]
    tmp_data = data

    client_data, data_ids = data_util.split_data_follow_non_iid_degree(tmp_data, args.non_iid_degree, args.n_client,
                                                                       data_rng)
    data_util.analyze_non_iid_ness(client_data, data_ids, num_data)


    # client_id_list = [b'cid0', b'cid1', b'cid2', b'cid3', b'cid4', b'cid5', b'cid6', b'cid7', b'cid8', b'cid9',
    #                   b'cid10', b'cid11', b'cid12', b'cid13', b'cid14', b'cid15']

    client_id_list = [f'cid{i}'.encode() for i in range(args.n_client)]

    rng_init, *client_prngs = random.split(key, args.n_client + 1)
    clients = [
        (client_id_list[i], fedjax.ClientDataset({'data': jnp.asarray(client_data[i])}),
         client_prngs[i]) for i in range(args.n_client)
    ]


    # save_path = "/home/jiayi/disk/FedAnalytic/datasets/"
    # for i in range(args.n_client):
    #     # save client_data[i] as a csv file
    #     f = f"{args.dataset_name}-{i}of{args.n_client}.csv"
    #
    #     pd.DataFrame(client_data[i]).to_csv(save_path + f, index=True, header=True)
    #
    # print("save finished!")
    # exit(0)



    best_state = state
    best_loss = 1e9

    num_check = int(np.ceil(fed_steps / args.check_interval))
    print(f"Total number of check: [{num_check}]")
    loss_values = jnp.zeros(num_check)

    pbar = trange(fed_steps, desc="Federated Training NF ", miniters=args.check_interval)

    client_id_and_data = {clients[i][0]: {'data': client_data[i]} for i in range(args.n_client)}
    # print(client_id_and_data)
    client_id_and_data = fedjax.InMemoryFederatedData(client_id_and_data)


    # ====================  new: for 【supervised traing】  ====================
    # Set the optimizer for supervised training here, by default, use adam
    # tx_sup = optax.adam(args.sup_learning_rate)

    # 2023-12-09 UPD: Add scheduler for sup
    scheduler_sup = optax.cosine_decay_schedule(args.sup_learning_rate, decay_steps=args.sup_itr_num, alpha=0)
    tx_sup = optax.chain(
        optax.clip_by_global_norm(args.grad_norm_clip_value),
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler_sup),
        optax.scale(-1)
        # optax.scale(-1.0 * args.sup_learning_rate)

    )


    sup_better_test = 0
    sup_better_val = 0
    total_sup_cnt = 0


    for itr in pbar:

        print("\n [Time] ", datetime.now(), "Start this fed itr!")


        efficient_sampler = fedjax.client_samplers.UniformShuffledClientSampler(
            client_id_and_data.shuffled_clients(buffer_size=100, seed=0), num_clients=args.nsample_clients)
        sampled_clients_with_data = efficient_sampler.sample()
        
        # print("see sampled:")
        # for client_id, client_data, client_rng in sampled_clients_with_data:
        #     print(client_id)

        # ------------------- used to reconstruct each client's model --------------------
        params_server_past = state.params

        state, client_diagnostics = algorithm.apply(state, sampled_clients_with_data)

        # ====================  【Below are newly added code for optimize server model】  ====================
        # ====================  Reconstruct each client's model  ====================
        params_server = state.params
        params_server_new = state.params


        # ====================  Generate samples using server state  ====================
        # n_samples = 512 * 1000
        n_samples = args.sup_batch_size * args.sup_itr_num

        print(datetime.now(), "Start to generate (inversely transform) samples!")


        rng_train, rng_key = jax.random.split(
            rng_train, 2
        )


        base_samples = base_dist.sample(seed=rng_key, sample_shape=(n_samples))
        samples = get_samples(base_samples, params_server, variables)
        samples = samples.reshape((n_samples, n_dim))
        jax.debug.print("samples : {x}", x=samples[0,:])

        # samples = val_data[:184320]
        print(datetime.now(), "generate finished! samples shape is ", samples.shape)


        client_params = []
        for c in client_diagnostics.keys():
            client_params.append(client_diagnostics[c]['model_params'])

        log_probs_array = jnp.zeros((args.n_client, n_samples))
        log_probs_sum = compute_expect_log_prob(samples, client_params, variables, log_probs_array)
        log_probs = log_probs_sum - jnp.log(args.n_client)      # log of sum, so need this minus to get the average

        print("[Time] ", datetime.now(), "log_probs are computed ")
        # These operations can not be jitted
        # mask = jnp.where(log_probs > -1, True, False)
        # samples = samples[mask]
        # log_probs = log_probs[mask]
        # n_samples = samples.shape[0]

        print("[Time] ", datetime.now(), "Mask out the low prob data points")
        # n_samples_limit = args.sup_batch_size * 500
        print("Initial samples", n_samples)
        # if n_samples >= n_samples_limit:
        #     samples = samples[:n_samples_limit,:]
        #     log_probs = log_probs[:n_samples_limit]
        #     n_samples = n_samples_limit
        #     print("usable samples", n_samples)




        #
        # # ====================  TODO : tidy below code  ====================
        # ------------------- student learns from these data and PDFs -------------------
        # first see the initial test loss of student


        test_loss = evaluate_sup(params_server, samples, log_probs, variables)
        print("[Student] Initial test_supervised_loss ", test_loss)



        test_nll_old = evaluate_unsup(params_server, test_data, variables)
        val_nll_old = evaluate_unsup(params_server, val_data, variables)

        print("[Time] ", datetime.now(), "Start to train on these data and PDFs")
        # Then student trains on these data
        train_epoch_sup_student = 1
        batch_size = args.sup_batch_size
        # It seems that the learning rate should be very small
        learning_rate = args.sup_learning_rate
        steps_per_epoch_sup_student = int(n_samples // batch_size)
        total_steps = steps_per_epoch_sup_student * train_epoch_sup_student
        # It seems that another state is needed for student

        # 【Manually build the state_student】
        # 【Note】 The tx (optimizer in optax) should be reused, in order to avoid the fast_train_flow_sup being recompiled each time
        # Refer: https://github.com/google/flax/discussions/3100
        state_student = train_state.TrainState.create(apply_fn=model.apply, params=params_server, tx=tx_sup)

        train_samples = samples.reshape((int(n_samples / batch_size), batch_size, n_dim))
        train_log_probs = log_probs.reshape((int(n_samples / batch_size), batch_size))
        print("[Time] ", datetime.now(), "Start to call fast_train_flow_sup")

        # ====================  【Supervised training】  ====================
        state_student = fast_train_flow_sup(state_student, variables, train_samples, train_log_probs, args.sup_itr_num, num_sup_epochs)

        print("[Time] ", datetime.now(), "Finished supervised training!")

        # ------------------- Print cache size to check if the function is recompiled -------------------
        # print("fast_train_flow_sup cache size: ", fast_train_flow_sup._cache_size())
        # # print(fast_train_flow_sup._cache())
        # print("evaluate_sup cache size:        ", evaluate_sup._cache_size())
        # print("compute_expect_log_prob size:        ", compute_expect_log_prob._cache_size())
        # print("train_step cache size: ", train_step_sup._cache_size())

        test_loss = evaluate_sup(state_student.params, samples, log_probs, variables)
        print("[Time] ", datetime.now(), "[New Student] After supervised training, test_loss of student is ", test_loss)



        # ------------------- Use NLL to test student model -------------------
        test_nll = evaluate_unsup(state_student.params, test_data, variables)
        print("[Time] ", datetime.now(), "[New Student] Test NLL  ", test_nll)


        print("[Time] ", datetime.now(), "[Old Student] Test NLL  ", test_nll_old)


        total_sup_cnt += 1
        if test_loss <= test_nll_old:
            sup_better_test += 1


        val_nll = evaluate_unsup(state_student.params, val_data, variables)
        print("[Time] ", datetime.now(), "[New Student] Val NLL  ", val_nll)


        print("[Time] ", datetime.now(), "[Old Student] Val NLL  ", val_nll_old)

        if val_nll <= val_nll_old:
            sup_better_val += 1
        print("test sup better proportion: {:.2f}".format(100.0 * sup_better_test / total_sup_cnt))
        print("val  sup better proportion: {:.2f}".format(100.0 * sup_better_val / total_sup_cnt))

        # test_nll = evaluate_unsup(params_client_list[-1], test_data, variables)
        # print(datetime.now(), "[One Teacher] Test NLL  ", test_nll)
        # 【Update server params】
        if val_nll <= val_nll_old:
            params_server_new = state_student.params

        state = ServerState(params_server_new, state.opt_state, state.client_states)


        if args.save_ckpt_in_training:
            """ Store the ckpt of save the clients' models during training """
            workdir = os.path.join(PROJECT_PATH, f"output/{args.dataset_name}/")
            if args.IF_DPSGD == False:
                subdir = f"{args.model_name}-alpha-{args.loc_alpha}-nc-{args.n_client}-epoch-{args.num_epochs}-batch-{args.batch_size}-lr-{args.learning_rate}-nl-{args.n_layers}-hidden-{args.n_hiddens[0]}-DP-{args.IF_DPSGD}"
            else:
                subdir = f"{args.model_name}-alpha-{args.loc_alpha}-nc-{args.n_client}-epoch-{args.num_epochs}-batch-{args.batch_size}-lr-{args.learning_rate}-nl-{args.n_layers}-hidden-{args.n_hiddens[0]}-DP-{args.IF_DPSGD}-clip-{args.l2_norm_clip}-noise-{noise_multiplier}-epsion-{EPSILON:.2f}"
            print("During traing, save client's models to\n", workdir + subdir)
            for ith_c, client_id in enumerate(client_diagnostics.keys()):
                client_delta_params = client_diagnostics[client_id]["delta_params"]
                to_save_state = {"params": client_delta_params}
                to_save_state = process_model_dict_by_model_type(to_save_state,
                                                                 args.model_name,
                                                                 args.n_layers,
                                                                 variables)
                # first create the workdir + subdir
                if not os.path.exists(workdir + subdir):
                    os.makedirs(workdir + subdir)
                pickle.dump(to_save_state, open(workdir + subdir + f"/client-{ith_c}-epoch-{itr}-params.pkl", "wb"))

                # print(client_id)
                # print(client_state)
                # print(client_state.params)
                # print(client_state.


        # state, client_diagnostics = algorithm.apply(state, clients)

        if itr % args.check_interval == 0:
            st_time = time.time()

            # nll = evaluate(state.params, test_data, variables)
            # nll = evaluate(state.params, val_data, variables)
            nll = evaluate_unsup(state.params, val_data, variables)
            if nll > 100 or jnp.isnan(nll):
                print("nll is ", nll)
                return 100000
            value = nll
            loss_values = loss_values.at[int(itr // args.check_interval)].set(value)



            # need set
            # if recent 5 loss values are increasing, then stop
            # 【Below is early stop for `power`】
            # 2024-02-21 Note： This early stop is only usable for power dataset
            # if itr > 15:
            #     if best_loss > 0:
            #         print("Early stop!")
            #         return 100000


            if value < best_loss:
                best_state = state
                best_loss = value

            # print(f"[{itr}]Test Loss (NLL): ", nll)
            print("")
            print(f"[{itr}]Val Loss (NLL): ", nll)
            print(f"[{itr}]Best Val Loss (NLL): ", best_loss)
            print("-- Evaluation Time used: ", time.time() - st_time)
            print("All loss values are ", loss_values)



    workdir = os.path.join(PROJECT_PATH, f"output/{args.dataset_name}/")
    if args.IF_DPSGD == False:
        subdir = f"sup-{day}-{args.model_name}-alpha-{args.loc_alpha}-nc-{args.n_client}-epoch-{args.num_epochs}-batch-{args.batch_size}-lr-{args.learning_rate}-nl-{args.n_layers}-hidden-{args.n_hiddens[0]}-DP-{args.IF_DPSGD}"
    else:
        subdir = f"sup-{day}-{args.model_name}-alpha-{args.loc_alpha}-nc-{args.n_client}-epoch-{args.num_epochs}-batch-{args.batch_size}-lr-{args.learning_rate}-nl-{args.n_layers}-hidden-{args.n_hiddens[0]}-DP-{args.IF_DPSGD}-clip-{args.l2_norm_clip}-noise-{noise_multiplier}-epsion-{EPSILON:.2f}"

    try:
        # best_state.client_states = None
        # save_checkpoint(best_state, workdir + subdir)


        to_save_state = {"params": best_state.params}

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




# search_space = {
#     'l2_norm_clip': [0.5],
#     # 'learning_rate': [0.001 * i for i in range(1, 10, 2)] + [0.01 * i for i in range(1, 10, 2)] + [0.1, 0.2, 0.3, 0.4, 0.5]
#     'learning_rate': [0.08],
#     'sup_learning_rate' : [1e-8],
#     'sup_batch_size' : [512],
#     # 'sup_learning_rate' : [1e-7],
# }


#  use val set
# search_space = {
#         'l2_norm_clip': [0.1 * i for i in range( 2, 16, 2)],           # 8
#         'learning_rate': [0.01 * i for i in range(2, 14, 2)] ,         # 6
#         'sup_learning_rate' : [1e-8, 5e-9],                                       # 1
#         # 'sup_batch_size': [128],  # 1
#         # 'sup_itr_num': [1440]  # 4
#     'sup_batch_size': [512],  # 1
#     'sup_itr_num': [360]  # 4
# }


# tmp
# search_space = {
#         'l2_norm_clip': [1],           # 8
#         'learning_rate': [0.04] ,         # 6
#         'sup_learning_rate' : [1e-8],                                       # 1
#         # 'sup_batch_size': [128],  # 1
#         # 'sup_itr_num': [1440]  # 4
#     'sup_batch_size': [128],  # 1
#     'sup_itr_num': [500]  # 4
# }




# search 1
# search_space = {
#     'l2_norm_clip': [0.1 * i for i in range( 2, 16, 2)],           # 8
#     'learning_rate': [0.01 * i for i in range(2, 14, 2)] ,
#     'sup_learning_rate' : [1e-8, 5e-9],                                       # 1
#     'sup_batch_size' :[512],                                                          # 1
#     'sup_itr_num' : [500]                    # 3
# }


# # search 2
# search_space = {
#     'l2_norm_clip': [0.1 * i for i in range( 2, 16, 2)],           # 8
#     'learning_rate': [0.01 * i for i in range(2, 14, 2)] ,
#     'sup_learning_rate' : [1e-8, 5e-9],                                       # 1
#     'sup_batch_size' :[128],                                                          # 1
#     'sup_itr_num' : [500]                    # 3
# }


# search 3
# search_space = {
#     'l2_norm_clip': [0.1 * i for i in range( 2, 16, 2)],           # 8
#     'learning_rate': [0.01 * i for i in range(2, 14, 2)] ,
#     'sup_learning_rate' : [1e-8, 5e-9],                                       # 1
#     'sup_batch_size' :[512],                                                          # 1
#     'sup_itr_num' : [2000]                    # 3
# }


# search_space = {
#     'l2_norm_clip': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1, 2, 3, 4, 5],
#     # 'learning_rate': [0.001 * i for i in range(1, 10, 2)] + [0.01 * i for i in range(1, 10, 2)] + [0.1, 0.2, 0.3, 0.4, 0.5]
#     'learning_rate': [0.01 * i for i in range(1, 10)] + [0.1, 0.2, 0.3, 0.4, 0.5],
#     'sup_learning_rate' : [1e-8],
# }





search_space = {
    'l2_norm_clip': [1],           # 8
    'learning_rate': [0.08] ,
    'sup_learning_rate' : [1e-6],                                       # 1
    'sup_batch_size' :[512],                                                          # 1
    'sup_itr_num' : [1000],                    # 3
    'num_sup_epochs' : [10]
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



