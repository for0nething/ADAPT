
import flax.training.train_state
import jax
import jax.numpy as jnp  # JAX NumPy
import jax.random as random  # JAX random
from tqdm import trange
# from flax.training import checkpoints
import optax  # Optimizers
from flax.training import train_state  # Useful dataclass to keep train state
import numpy as np
import my_tree_util
import copy
import jax.lax as lax
Array = jnp.ndarray
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple


def make_training_loop_unsupervised(model, IF_DPSGD=False):
    """
        The `unsupervised` training loop for NF, i.e., using maximum likelihood estimation
    """
    def train_step(batch, state, variables):
        """ Train for a single step. """
        def loss(params, batch):
            log_det = model.apply(
                {"params": params, "variables": variables}, batch, method=model.log_prob
            )
            return -jnp.mean(log_det)
        grad_fn = jax.value_and_grad(loss)

        if IF_DPSGD == True:

            this_batch_size = batch.shape[0]
            # Insert dummy dimension in axis 1 to use jax.vmap over the batch
            batch = jax.tree_util.tree_map(lambda x: x[:, None], batch)
            # Use jax.vmap across the batch to extract per-example gradients
            grad_fn = jax.vmap(grad_fn, in_axes=(None,  0))

        value, grad = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grad)
        value = jnp.average(value)
        return value, state
    train_step = jax.jit(train_step)

    def train_epoch(rng, state, variables, train_ds, batch_size):
        """Train for a single epoch."""
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size
        if steps_per_epoch > 0:
            perms = jax.random.permutation(rng, train_ds_size)
            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch = train_ds[perm, ...]
                value, state = train_step(batch, state, variables)
        else:
            value, state = train_step(train_ds, state, variables)
        return value, state
    def evaluate_batch(params, batch, variables):
        """ Compute the loss of the model on a batch of data. """
        log_det = model.apply(
            {"params": params, "variables": variables}, batch, method=model.log_prob
        )
        log_likelihood = -jnp.mean(log_det)
        return log_likelihood
    evaluate_batch = jax.jit(evaluate_batch)
    def evaluate(params, data, variables):
        """ Compute the average loss of the model on a dataset. """
        test_batch_size = 1601 * 16
        num_test_steps = int(np.ceil(data.shape[0] / test_batch_size))
        ll_values = jnp.zeros(num_test_steps)
        for i in range(num_test_steps):
            batch_data = data[i * test_batch_size:(i + 1) * test_batch_size, :]
            ll_values = ll_values.at[i].set(evaluate_batch(params, batch_data, variables))
        return jnp.mean(ll_values)
    evaluate = jax.jit(evaluate)
    def train_flow(rng, state, variables, data, num_epochs, batch_size, val_data=None, pruneParams=None):
        """ Train a model for a specified number of epochs. """
        if val_data is None:
            val_data = data
        loss_values = jnp.zeros(num_epochs)
        # print("see gradient meanings")
        # print(state.params.keys())
        gradient_norms = jnp.zeros((num_epochs, len(state.params.keys())))
        pbar = trange(num_epochs, desc="Training NF", miniters=int(num_epochs / 10))
        best_state = state
        best_loss = 1e9
        state_list = []

        for epoch in pbar:
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            value, state = train_epoch(input_rng, state, variables, data, batch_size)
            state_list.append(state)

            eval_value = evaluate(state.params, val_data, variables)
            # print("Eval loss: %.3f" % eval_value)
            value = eval_value
            loss_values = loss_values.at[epoch].set(value)
            print("value is ", value, "loss_value is ", loss_values[epoch])
            if value < best_loss:
                best_state = state
                best_loss = value
            if value > 100 or jnp.isnan(value):
                loss_values= loss_values.at[0].set(100000)
                break

            print("All loss values are ", loss_values)
            pbar.set_description(f"Training NF, current loss: {value:.3f},  best loss: {best_loss:.3f}")

        return rng, best_state, loss_values, gradient_norms, state_list, variables

    return train_flow, train_epoch, train_step, evaluate



def make_training_loop_supervised(model, IF_DPSGD=False):
    """
        The `supervised` training loop for NF, i.e., using data and dataPDF to train the model
    """

    # @jax.jit
    def train_step(batch, batchPDF, state, variables):
        # 【MSE】
        def loss(params, batch, batchPDF):
            log_prob = model.apply(
                {"params": params, "variables": variables}, batch, method=model.log_prob
            )
            batchPDF = jnp.exp(batchPDF)
            log_prob = jnp.exp(log_prob)
            # return jnp.mean(jnp.square(log_prob - batchPDF))    # original loss



            # =====================  The following is a loss with Orthogonal Regulation  ================
            leaves = jax.tree_util.tree_leaves(params)

            first_mask = variables["conditioner_0"]["conditioner"]["layers_0"]["mask"]
            second_mask = variables["conditioner_0"]["conditioner"]["layers_1"]["layers_0"]["mask"]

            orthogonal_regularization = 0
            reg_coef = 1e-4
            # reg_coef = 0
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
                    tmp = jnp.multiply(tmp, (ones - eyes))
                    orthogonal_regularization = orthogonal_regularization + jnp.sum(jnp.square(tmp))
            # jax.debug.print("original   loss: {x}",x=jnp.mean(jnp.square(log_prob - batchPDF)))
            # jax.debug.print("orthogonal loss: {x}",x=reg_coef * orthogonal_regularization)
            return jnp.mean(jnp.square(log_prob - batchPDF)) + reg_coef * orthogonal_regularization
        
        
        # KL divergence
        # def loss(params, batch, batchPDF):
        #     log_det = model.apply(
        #         {"params": params, "variables": variables}, batch, method=model.log_prob
        #     )
        #     smooth = 1e-12
        #     # smooth = 0
        #     # # ------------------- operates on predicted pdf -------------------
        #     # # log_prob -> prob
        #     # predict_probs = jnp.exp(log_det) + smooth
        #     # # normalize predict_probs
        #     # predict_probs = predict_probs / jnp.sum(predict_probs)
        #     #
        #     #
        #     # # ------------------- operates on target pdf -------------------
        #     # # log_prob -> prob
        #     batchPDF = jnp.exp(batchPDF) + smooth
        #     # # normalize batchPDF
        #     # batchPDF = batchPDF / jnp.sum(batchPDF)
        #
        #     return optax.kl_divergence(log_det, batchPDF)


        # 【normalize KL divergence】
        # def loss(params, batch, batchPDF):
        #     log_det = model.apply(
        #         {"params": params, "variables": variables}, batch, method=model.log_prob
        #     )
        #     # smooth = 1e-12
        #     smooth = 1e-10
        #     # # ------------------- operates on predicted pdf -------------------
        #     # # log_prob -> prob
        #     predict_probs = jnp.exp(log_det) + smooth
        #     # # normalize predict_probs
        #     predict_probs = predict_probs / jnp.sum(predict_probs)
        #     #
        #     #
        #     # # ------------------- operates on target pdf -------------------
        #     # # log_prob -> prob
        #     batchPDF = jnp.exp(batchPDF) + smooth
        #     # # normalize batchPDF
        #     batchPDF = batchPDF / jnp.sum(batchPDF)
        #
        #     return optax.kl_divergence(jnp.log(predict_probs), batchPDF)
        #
        #
        grad_fn = jax.grad(loss)

        grad = grad_fn(state.params, batch, batchPDF)
        state = state.apply_gradients(grads=grad)

        return state
    # because the param 2, i.e. state has the same size as the returned state, so use donate_argnums here
    # But note that the original state can not be called after this func
    train_step = jax.jit(train_step, donate_argnums=(2))


    def train_epoch(rng, state, variables, train_ds, batch_size, batchPDFs):
        """Train for a single epoch."""
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size
        if steps_per_epoch > 0:
            perms = jax.random.permutation(rng, train_ds_size)
            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch = train_ds[perm, ...]
                batchPDF = batchPDFs[perm]
                state = train_step(batch, batchPDF, state, variables)
        else:
            state = train_step(train_ds, batchPDFs, state, variables)
        return state

    # 【MSE】
    # def evaluate_batch(params, batch, batchPDF, variables):
    #     log_det = model.apply(
    #         {"params": params, "variables": variables}, batch, method=model.log_prob
    #     )
    #     return jnp.mean(jnp.square(log_det - batchPDF))


    # 【normalize KL divergence】
    def evaluate_batch(params, batch, batchPDF, variables):
        log_det = model.apply(
            {"params": params, "variables": variables}, batch, method=model.log_prob
        )

        # KL divergence
        smooth = 1e-12
        # # ------------------- operates on predicted pdf -------------------
        # # log_prob -> prob
        predict_probs = jnp.exp(log_det) + smooth
        # # normalize predict_probs
        predict_probs = predict_probs / jnp.sum(predict_probs)
        #
        #
        # # ------------------- operates on target pdf -------------------
        # # log_prob -> prob
        batchPDF = jnp.exp(batchPDF) + smooth
        # # normalize batchPDF
        batchPDF = batchPDF / jnp.sum(batchPDF)
        return optax.kl_divergence(jnp.log(predict_probs), batchPDF)


    evaluate_batch = jax.jit(evaluate_batch)

    def evaluate(params, data, dataPDFs, variables):
        test_batch_size = 1601 * 16
        num_test_steps = int(np.ceil(data.shape[0] / test_batch_size))
        ll_values = jnp.zeros(num_test_steps)
        for i in range(num_test_steps):
            batch_data = data[i * test_batch_size:(i + 1) * test_batch_size, :]
            batchPDF = dataPDFs[i * test_batch_size:(i + 1) * test_batch_size]
            ll_values = ll_values.at[i].set(evaluate_batch(params, batch_data, batchPDF, variables))
        # print("all ll_values")
        # jax.debug.print("{x}", x=ll_values)
        return jnp.mean(ll_values)
    evaluate = jax.jit(evaluate)


    def compute_expect_log_prob(samples, client_params, variables, log_probs_array):

        for i, params_teacher in enumerate(client_params):
            log_probs = model.apply({"params": params_teacher, "variables": variables}, samples,
                                    method=model.log_prob)
            log_probs_array = log_probs_array.at[i, :].set(log_probs)
        log_probs = jax.scipy.special.logsumexp(log_probs_array, axis=0)
        return log_probs
    compute_expect_log_prob = jax.jit(compute_expect_log_prob)

    def get_samples(base_samples, params, variables):


        samples = model.apply(
            {"params": params, "variables": variables}, base_samples, method=model.inverse
        )
        return samples
    get_samples = jax.jit(get_samples)

    def train_flow(rng, state, variables, data, num_epochs, batch_size, batchPDFs, val_data=None, pruneParams=None):
        if val_data is None:
            val_data = data
        loss_values = jnp.zeros(num_epochs)
        # print("see gradient meanings")
        # print(state.params.keys())
        gradient_norms = jnp.zeros((num_epochs, len(state.params.keys())))  # 20 layers  TODO: Make this more general
        pbar = trange(num_epochs, desc="Training NF", miniters=int(num_epochs / 10))
        best_state = state
        best_loss = 1e9
        state_list = []

        # 【new】
        train_ds_size = len(data)
        steps_per_epoch = train_ds_size // batch_size
        for epoch in pbar:
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            state = train_epoch(input_rng, state, variables, data, batch_size, batchPDFs=batchPDFs)
            # print('Train loss: %.3f' % value)
            state_list.append(state)

            """ Here assumes that val_data is the same as train data"""
            eval_value = evaluate(state.params, val_data, batchPDFs, variables)
            # print("Eval loss: %.3f" % eval_value)
            value = eval_value
            loss_values = loss_values.at[epoch].set(value)
            print("value is ", value, "loss_value is ", loss_values[epoch])

            if value < best_loss:
                best_state = state
                best_loss = value
            if value > 1000000 or jnp.isnan(value):
                loss_values = loss_values.at[0].set(1000000)
                break
            print("All loss values are ", loss_values)
            pbar.set_description(f"Training NF, current loss: {value:.3f},  best loss: {best_loss:.3f}")
        return rng, best_state, loss_values, gradient_norms, state_list, variables



    def fast_train_flow(state : flax.training.train_state.TrainState, variables: Dict, train_data: Array, train_dataPDFs: Array, step_num : int, num_sup_epochs : int)-> optax.OptState:

        def body_fun(i : int, carry :optax.OptState)->optax.OptState:

            batch, batchPDF = train_data[i], train_dataPDFs[i]
            updated_state = train_step(batch, batchPDF, carry, variables)
            return updated_state


        # num_sup_epochs = 10
        final_state = state
        for i in range(num_sup_epochs):
            final_state = lax.fori_loop(0, step_num, body_fun, final_state)
        return final_state

    # fast_train_flow = jax.jit(fast_train_flow, donate_argnums=0)
    fast_train_flow = jax.jit(fast_train_flow, static_argnums=(4,5))

    return train_flow, train_epoch, train_step, evaluate, compute_expect_log_prob, get_samples, fast_train_flow


def create_train_state(model, n_dim, rng, learning_rate, params, momentum=0.9, IF_DPSGD=False,
                       l2_norm_clip=3, noise_multiplier=0.7, seed=1234):


    if IF_DPSGD == False:
        # tx = optax.adam(learning_rate, momentum)
        tx = optax.adam(learning_rate)
    else:
        #  【DP-SGD optimizer】
        tx = optax.dpsgd(learning_rate=learning_rate,
                         l2_norm_clip=l2_norm_clip,
                         noise_multiplier=noise_multiplier,
                         momentum=momentum,
                         seed=seed)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def learning_rate_schedule(init_lr, steps_per_epoch, epochs):

    def schedule(count):
        epoch = count // steps_per_epoch  # Assuming you have a variable to keep track of the current training step
        return init_lr * jnp.cos(epoch / epochs * jnp.pi / 2)
        # return init_lr / (2 ** epoch)  # Reduce the learning rate by a factor of 2 each epoch
    return schedule


def create_schedule_train_state(model, n_dim, rng, learning_rate,
                                steps_per_epoch,
                                epochs,
                                momentum=0.9,
                                IF_DPSGD=False,
                                l2_norm_clip=3,
                                noise_multiplier=0.7,
                                seed=1234,
                                total_steps=10000,
                                grad_norm_clip_value=5.0,
                                nesterov=False,
                                params=None
                                ):
    if params is None:
        params = model.init(rng, jnp.ones((1, n_dim)))["params"]

    # Cosine decay of the learning rate.
    scheduler = optax.cosine_decay_schedule(learning_rate, decay_steps=total_steps, alpha=0)


    # scheduler = learning_rate_schedule(learning_rate, steps_per_epoch=steps_per_epoch, epochs=epochs)


    if IF_DPSGD == False:
        # Combining gradient transforms using `optax.chain`.

        gradient_transform = optax.chain(


            optax.clip_by_global_norm(grad_norm_clip_value),





            # If use momentum version SGD
            (optax.trace(decay=momentum, nesterov=nesterov)
             if momentum is not None else optax.identity()),


            # If use adam
            # optax.scale_by_adam(),
            optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1.0)
        )


        frozen_layer_lim = 0

        def if_frozen(node_str):
            # if node_str has format conditioner_i and i < frozen_layer_lim return True
            # if node_str has format scalar_i and i < frozen_layer_lim return True
            # otherwise return False
            if "conditioner_" in node_str:
                id = int(node_str.split("_")[1])
                if id < frozen_layer_lim:
                    return True
            if "scalar_" in node_str:
                id = int(node_str.split("_")[1])
                if id < frozen_layer_lim:
                    return True
            return False

        def compute_mask_labels(kp, value):
            # print("KP:    ", kp)
            for node in kp:
                if if_frozen(node.key):
                    return "zero"
            return "g"

        mask_labels = my_tree_util.tree_map_with_path(compute_mask_labels, params)
        # print(mask_labels)

        gradient_transform = optax.multi_transform({"g": gradient_transform, "zero": optax.set_to_zero()},
                                                   mask_labels)
    else:
        # Combining gradient transforms using `optax.chain`.
        gradient_transform = optax.chain(
            # optax.clip_by_global_norm(grad_norm_clip_value),


            optax.differentially_private_aggregate(l2_norm_clip=l2_norm_clip,
                                                   noise_multiplier=noise_multiplier,
                                                   seed=seed),  # Use the updates from adam.

            # my_DPSGD.differentially_private_aggregate(l2_norm_clip=l2_norm_clip,
            #                                        noise_multiplier=noise_multiplier,
            #                                        seed=seed),  # Use the updates from adam.




            ###########  If use sgd with momentum  ##########
            (optax.trace(decay=momentum, nesterov=nesterov)
             if momentum is not None else optax.identity()),


            ###########  If use adam  ##########
            # adam  seems to be impossible, it always performs very bad for DP training
            # optax.scale_by_adam(),

            optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1.0)
        )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=gradient_transform)



# def restore_checkpoint(state, workdir):
#   return checkpoints.restore_checkpoint(workdir, state)
# 
# 
# def save_checkpoint(state, workdir):
#   if jax.process_index() == 0:
#     # get train state from the first replica
#     # state = jax.device_get(jax.tree_map(lambda x: x[0], state))
#     # step = int(state.step)
#     step = int(0)
#     checkpoints.save_checkpoint(workdir, state, step, keep=3)




def process_model_dict_by_model_type(to_save_state, model_name, n_layers, variables):
    """
    Parameters
    ----------
    to_save_state:  a dict {"params": best_state.params}    containing the params needed to process according to the model_name
    model_name:     the name of the model, e.g. RQSpline, MaskRQSpline
    n_layers:       the number of layers of the model
    variables:      the variables of the model

    Returns
    -------
    The updated param dict that is processed accordign to model_name
    e.g. MaskRQspline will multiply the mask and the model params
    """
    to_save_params = copy.deepcopy(to_save_state["params"])
    to_save_params = to_save_params.unfreeze()

    if model_name == "MaskRQSpline":



        # ==============================       Below are newly added when layers_0 and layers_4 in conditioner also use mask       =================================
        # 【new for when layers_0 also use mask】
        mask_00 = variables["conditioner_0"]["conditioner"]["layers_0"]["mask"]
        # 【new for when layers_4 also use mask】
        # mask_04 = variables["conditioner_0"]["conditioner"]["layers_4"]["mask"]



        mask_0 = variables["conditioner_0"]["conditioner"]["layers_1"]["layers_0"]["mask"]
        mask_1 = variables["conditioner_0"]["conditioner"]["layers_1"]["layers_1"]["mask"]

        test_mask_0 = variables["conditioner_0"]["conditioner"]["layers_2"]["layers_0"]["mask"]
        test_mask_1 = variables["conditioner_0"]["conditioner"]["layers_2"]["layers_1"]["mask"]

        assert jnp.all(mask_0 == test_mask_0)
        assert jnp.all(mask_1 == test_mask_1)

        for i in range(10):
            l1 = "conditioner_{}".format(i)
            l2 = "conditioner"
            l3_1 = "layers_1"
            l3_2 = "layers_2"
            l4_0 = "layers_0"
            l4_1 = "layers_1"

            # ==============================       Below are newly added when layers_0 and layers_4 in conditioner also use mask       =================================
            # 【new for when layers_0 also use mask】
            l_0 = "layers_0"
            to_save_params[l1][l2][l_0]["kernel"] = mask_00 * to_save_params[l1][l2][l_0]["kernel"]

            # 【new for when layers_4 also use mask】
            # l_4 = "layers_4"
            # to_save_params[l1][l2][l_4]["kernel"] = mask_04 * to_save_params[l1][l2][l_4]["kernel"]






            # to_save_params[l1][l2][l3_1][l4_0]["kernel"] *= mask_0
            to_save_params[l1][l2][l3_1][l4_0]["kernel"] = jnp.where(mask_0,
                                                                     to_save_params[l1][l2][l3_1][l4_0]["kernel"],
                                                                     0)
            to_save_params[l1][l2][l3_1][l4_1]["kernel"] = jnp.where(mask_1,
                                                                     to_save_params[l1][l2][l3_1][l4_1]["kernel"],
                                                                     0)

            to_save_params[l1][l2][l3_2][l4_0]["kernel"] = jnp.where(mask_0,
                                                                     to_save_params[l1][l2][l3_2][l4_0]["kernel"],
                                                                     0)
            to_save_params[l1][l2][l3_2][l4_1]["kernel"] = jnp.where(mask_1,
                                                                     to_save_params[l1][l2][l3_2][l4_1]["kernel"],
                                                                     0)



    elif model_name == "PosAddRQSpline":
        #
        pre_masks = variables["masks"]
        # to_save_params = state.params.unfreeze()

        for i in range(n_layers):
            l1 = "conditioner_{}".format(i)
            l2 = "conditioner"
            l3_1 = "layers_1"
            l3_2 = "layers_2"
            l4_0 = "layers_0"
            l4_1 = "layers_1"

            to_save_params[l1][l2][l3_1][l4_0]["bias"] += pre_masks[0, :].reshape(-1)
            to_save_params[l1][l2][l3_1][l4_1]["bias"] += pre_masks[1, :].reshape(-1)
            to_save_params[l1][l2][l3_2][l4_0]["bias"] += pre_masks[0, :].reshape(-1)
            to_save_params[l1][l2][l3_2][l4_1]["bias"] += pre_masks[1, :].reshape(-1)
            # to_save_params[l1][l2][l3_1][l4_0]["kernel"] *= mask_0
            # to_save_params[l1][l2][l3_1][l4_0]["kernel"] = jnp.where(pre_masks,
            #                                                          to_save_params[l1][l2][l3_1][l4_0]["kernel"],
            #                                                          0)

    elif model_name == "PosMulRQSpline":
        pre_masks = variables["masks"]
        # to_save_params = state.params.unfreeze()

        for i in range(n_layers):
            l1 = "conditioner_{}".format(i)
            l2 = "conditioner"
            l3_1 = "layers_1"
            l3_2 = "layers_2"
            l4_0 = "layers_0"
            l4_1 = "layers_1"

            to_save_params[l1][l2][l3_1][l4_0]["bias"] *= pre_masks[0, :].reshape(-1)
            to_save_params[l1][l2][l3_1][l4_1]["bias"] *= pre_masks[1, :].reshape(-1)
            to_save_params[l1][l2][l3_2][l4_0]["bias"] *= pre_masks[0, :].reshape(-1)
            to_save_params[l1][l2][l3_2][l4_1]["bias"] *= pre_masks[1, :].reshape(-1)

            to_save_params[l1][l2][l3_1][l4_0]["kernel"] *= pre_masks[0, :]
            to_save_params[l1][l2][l3_1][l4_1]["kernel"] *= pre_masks[1, :]
            to_save_params[l1][l2][l3_2][l4_0]["kernel"] *= pre_masks[0, :]
            to_save_params[l1][l2][l3_2][l4_1]["kernel"] *= pre_masks[1, :]
    #
    to_save_params = jax.tree_map(lambda x: x._value, to_save_params)
    to_save_state = {"params": to_save_params}
    return to_save_state