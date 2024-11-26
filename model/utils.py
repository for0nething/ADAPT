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


def make_training_loop(model, IF_DPSGD=False):
    """
    Create a function that trains an NF model.

    Args:
        model: a neural network model with a `loss` method.

    Returns:
        train_flow (Callable): wrapper function that trains the model.

    """
    def train_step(batch, state, variables):
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

        # jax.debug.print("In step see trace {x}", x=state.opt_state[1].trace['conditioner_0']['conditioner']['layers_0']['kernel'][0, :2])

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

    def get_step_gradient(batch, state, variables):
        def loss(params, batch):
            log_det = model.apply(
                {"params": params, "variables": variables}, batch, method=model.log_prob
            )
            return -jnp.mean(log_det)

        grad_fn = jax.value_and_grad(loss)
        value, grad = grad_fn(state.params, batch)
        return grad
    get_step_gradient = jax.jit(get_step_gradient)

    from optax._src import numerics


    def evaluate_batch(params, batch, variables):
        # log_det = model.apply(
        #     {"params": params, "variables": variables}, batch, method=model.log_prob
        # )
        log_det = model.apply(
            {"params": params, "variables": variables}, batch, method=model.log_prob
        )
        log_likelihood = -jnp.mean(log_det)
        return log_likelihood

    evaluate_batch = jax.jit(evaluate_batch)

    def evaluate(params, data, variables):
        test_batch_size = 1601 * 16
        num_test_steps = int(np.ceil(data.shape[0] / test_batch_size))
        ll_values = jnp.zeros(num_test_steps)
        for i in range(num_test_steps):
            batch_data = data[i * test_batch_size:(i + 1) * test_batch_size, :]
            # batch = {'data': jnp.asarray(batch_data)}
            # log_det = model.apply(
            #     {"params": params, "variables": variables}, batch_data, method=model.log_prob
            # )
            # nll = -jnp.mean(log_det)
            ll_values = ll_values.at[i].set(evaluate_batch(params, batch_data, variables))
        return jnp.mean(ll_values)

    def train_flow(rng, state, variables, data, num_epochs, batch_size, val_data=None, pruneParams=None):
        if val_data is None:
            val_data = data
        loss_values = jnp.zeros(num_epochs)
        print("see gradient meanings")
        print(state.params.keys())
        gradient_norms = jnp.zeros((num_epochs, len(state.params.keys())))    # 20 layers  TODO: Make this more general
        pbar = trange(num_epochs, desc="Training NF", miniters=int(num_epochs / 10))
        best_state = state
        best_loss = 1e9
        state_list = []

        for epoch in pbar:
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)

            # Run an optimization step over a training batch
            value, state = train_epoch(input_rng, state, variables, data, batch_size)
            # print('Train loss: %.3f' % value)
            state_list.append(state)

            eval_value = evaluate(state.params, val_data, variables)
            # print("Eval loss: %.3f" % eval_value)
            value = eval_value
            loss_values = loss_values.at[epoch].set(value)

            print("value is ", value, "loss_value is ", loss_values[epoch])
            # if loss_values[epoch] < best_loss:
            if value < best_loss:
                best_state = state
                best_loss = value

            if value > 100 or jnp.isnan(value):
                loss_values= loss_values.at[0].set(100000)
                break

            print("All loss values are ", loss_values)

            pbar.set_description(f"Training NF, current loss: {value:.3f},  best loss: {best_loss:.3f}")
            # if num_epochs > 10:
            #     if epoch % int(num_epochs / 10) == 0:
            #         pbar.set_description(f"Training NF, current loss: {value:.3f}")
            # else:
            #     if epoch == num_epochs:
            #         pbar.set_description(f"Training NF, current loss: {value:.3f}")


        return rng, best_state, loss_values, gradient_norms, state_list, variables

    return train_flow, train_epoch, train_step, evaluate


def create_train_state(model, n_dim, rng, learning_rate, momentum, IF_DPSGD=False,
                       l2_norm_clip=3, noise_multiplier=0.7, seed=1234):
    params = model.init(rng, jnp.ones((1, n_dim)))["params"]


    if IF_DPSGD == False:
        tx = optax.adam(learning_rate, momentum)

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
                                nesterov=False
                                ):
    params = model.init(rng, jnp.ones((1, n_dim)))["params"]

    # Cosine decay of the learning rate.
    scheduler = optax.cosine_decay_schedule(learning_rate, decay_steps=total_steps, alpha=0)


    # scheduler = learning_rate_schedule(learning_rate, steps_per_epoch=steps_per_epoch, epochs=epochs)


    if IF_DPSGD == False:
        # Combining gradient transforms using `optax.chain`.

        gradient_transform = optax.chain(


            optax.clip_by_global_norm(grad_norm_clip_value),



            # optax.inject_hyperparams(optax.clip_by_global_norm)(
            # max_norm=optax.linear_schedule(init_value=10, end_value=2,transition_steps=100),
            # ),

            # (optax.trace(decay=momentum, nesterov=nesterov)
            #  if momentum is not None else optax.identity()),

            # optax.scale_by_adam(b1=momentum),  # Use the updates from adam.

            ########################   adam     ############################
            optax.scale_by_adam(),  # Use the updates from adam.


            optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1.0)
        )

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


            ############## 【 momentum】     ############################
            (optax.trace(decay=momentum, nesterov=nesterov)
             if momentum is not None else optax.identity()),
            optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1.0)
        )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=gradient_transform)


def sample_nf(model, param, rng_key, n_sample, variables):
    rng_key, subkey = random.split(rng_key)
    samples = model.apply(
        {"params": param, "variables": variables}, subkey, n_sample, method=model.sample
    )
    # samples = jnp.flip(samples[0],axis=1)
    return rng_key, samples


# def restore_checkpoint(state, workdir):
#   return checkpoints.restore_checkpoint(workdir, state)
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
        # [Note] If it wasn't for MaskRQSpline, it wouldn't work now, so we need to comment out the code below
        # to_save_params = state.params.unfreeze()


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