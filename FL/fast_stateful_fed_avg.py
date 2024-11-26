# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example of stateful clients in federated averaging.

This is an example project demonstrating how to implment stateful clients in
FedJAX. We introduce a counter on each client, which tracks the total number of
steps of model training on each client. For example, if client A has been
sampled m times at round n, and each round ran with q iterations, then the
counter for client A would be q*m at round n.

This is an extension of the federated averaging implementation at
`fedjax.algorithms.fed_avg` , so we use `fedjax.for_each_client` here. We do
this to provide an easy to extend example that uses `fedjax.for_each_client`.

NOTE: Implementing this wihout `fedjax.for_each_client` should be very
straightforward and will be more flexible (e.g. this implementation with
`fedjax.for_each_client` will not work if the client states are not all PyTrees
with the same structure).
In this example, client states are stored in memory as part
of algorithm state. This could prove resource intensive if the client state size
or the number of clients is too large.
"""
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import fedjax
import jax
import jax.numpy as jnp
import chex
Grads = fedjax.Params


def create_train_for_each_client(grad_fn, client_optimizer):
  """Builds client_init, client_step, client_final for for_each_client."""

  def client_init(server_params, client_input):
    opt_state = client_optimizer.init(server_params)

    """ Initialize the state left from the previous round"""
    # not-DP  adam       clip -> adam -> schedule -> -1
    # modified_opt_state = (opt_state[0], opt_state[1]._replace(count=client_input['state'].adam_count,
    #                                                           mu=client_input['state'].adam_mu,
    #                                                           nu=client_input['state'].adam_nu)
    #                                                           , opt_state[2]._replace(count=client_input['state'].num_steps), opt_state[3])

    # DP   DP_aggregate -> trace -> schedule -> -1
    modified_opt_state = (opt_state[0]._replace(rng_key=client_input['state'].DP_rng),
    # modified_opt_state = (opt_state[0],
                          opt_state[1]._replace(trace=client_input['state'].trace),
                          opt_state[2]._replace(count=client_input['state'].num_steps),
                          opt_state[3])

    # 【DP】DP_aggregate -> adam -> schedule -> -1
    # modified_opt_state = (opt_state[0]._replace(rng_key=client_input['state'].DP_rng),
    #                       opt_state[1]._replace(count=client_input['state'].adam_count,
    #                                             mu=client_input['state'].adam_mu,
    #                                             nu=client_input['state'].adam_nu),
    #                       opt_state[2]._replace(count=client_input['state'].num_steps),
    #                       opt_state[3])


    # 【Initialized using the state from the previous round】
    opt_state = modified_opt_state


    # jax.debug.print("UPD scale by schedule count {x}", x=opt_state[-2].count)

    client_step_state = {
        'params': server_params,
        'opt_state': opt_state,
        'rng': client_input['rng'],
        'state': client_input['state'],     # Using the state initialization from the previous round doesn't matter as long as num_steps is right
    }

    return client_step_state

  def client_step(client_step_state, batch):
    rng, use_rng = jax.random.split(client_step_state['rng'])

    # todo: This seems to be only correct for DP scenario, maybe incorrect for non-DP
    batch["data"] = jax.tree_util.tree_map(lambda x: x[:, None], batch["data"])

    grads = grad_fn(client_step_state['params'], batch["data"], use_rng)

    opt_state, params = client_optimizer.apply(grads,
                                               client_step_state['opt_state'],
                                               client_step_state['params'])

    # The state can be set arbitrarily
    next_client_step_state = {
        'params': params,
        'opt_state': opt_state,
        'rng': rng,
        # Add to count of total number of steps of training for the client.
        'state': ClientState(
            # num_steps=client_step_state['state'].num_steps + 1,
            num_steps=-1,

            # trace=opt_state[1].trace
            trace=None,

            # adam_count=client_step_state['state'].adam_count + 1,
            adam_count=-1,
            adam_mu=None,
            adam_nu=None,

            DP_rng = None

            # this_opt_state = opt_state
        ),
    }
    return next_client_step_state

  def client_final(server_params, client_step_state):
    delta_params = jax.tree_util.tree_map(
        lambda a, b: a - b, server_params, client_step_state['params']
    )
    opt_state = client_step_state["opt_state"]


    # Try to use opt_state to update the Settings, because the count in opt_state is the latest
    # Many of the variables in client_step_state['state'] are not maintained in client_step
    # Update trace only when final
    # client_state = ClientState(client_step_state['state'].num_steps, trace=opt_state[-3].trace)

    # opt_state[1] should be the adam state
    # client_state = ClientState(client_step_state['state'].num_steps, trace=None,

    # 【not-DP】  有adam       clip -> adam -> schedule -> -1
    # client_state = ClientState(num_steps=opt_state[2].count,
    #                            trace=None,
    #                            adam_count=opt_state[1].count,
    #                            adam_mu=opt_state[1].mu,
    #                            adam_nu=opt_state[1].nu)

    # 【DP】   DP_aggregate -> trace -> schedule -> -1
    client_state = ClientState(num_steps=opt_state[2].count,
                               trace=opt_state[1].trace,
                               adam_count=-1,
                               adam_mu=None,
                               adam_nu=None,
                               DP_rng = opt_state[0].rng_key
                               )

    # 【DP】   DP_aggregate -> adam -> schedule -> -1
    # client_state = ClientState(num_steps=opt_state[2].count,
    #                            trace=None,
    #                            adam_count=opt_state[1].count,
    #                            adam_mu=opt_state[1].mu,
    #                            adam_nu=opt_state[1].nu,
    #                            DP_rng = opt_state[0].rng_key
    #                            )

    client_output = {
        'delta_params': delta_params,
        'state': client_state,
    }

    return client_output

  return fedjax.for_each_client(client_init, client_step, client_final)


@fedjax.dataclass
class ClientState:
  """State of client maintained over rounds.

  Attributes:
    num_steps: A counter tracking the total number of training steps per client.
  """
  # 【for scheduler】
  num_steps: int

  # 【for trace】
  # this_opt_state: tuple
  trace : chex.ArrayTree

  # 【for adam】
  adam_count : int
  # adam_count : chex.Array
  adam_mu : chex.ArrayTree = None
  adam_nu : chex.ArrayTree = None

  # 【for DP】
  DP_rng : chex.Array = None




@fedjax.dataclass
class ServerState:
  """State of server passed between rounds.

  Attributes:
    params: A pytree representing the server model parameters.
    opt_state: A pytree representing the server optimizer state.
    client_states: A dict of pytrees representing client states. In this case,
      we just keep track of total number of training steps using a dictionary
      with the key 'num_steps'.
  """
  params: fedjax.Params
  opt_state: fedjax.OptState
  client_states: Dict[fedjax.ClientId, ClientState]



# import jaxlib
# import flax
# def create_zero_tree(x):
#     if isinstance(x, jnp.ndarray):
#         return jnp.zeros_like(x)
#     elif isinstance(x, jaxlib.xla_extension.DeviceArray):
#         return jnp.zeros_like(x)
#     elif isinstance(x, (tuple, list)):
#         return type(x)(create_zero_tree(subtree) for subtree in x)
#     elif isinstance(x, dict):
#         return {key: create_zero_tree(subtree) for key, subtree in x.items()}
#     elif isinstance(x, flax.core.frozen_dict.FrozenDict):
#         return flax.core.frozen_dict.FrozenDict({key: create_zero_tree(subtree) for key, subtree in x.items()})
#     else:
#         raise ValueError("Unsupported pytree node type")

def federated_averaging(
    grad_fn: Callable[
        [fedjax.Params, fedjax.BatchExample, fedjax.PRNGKey], Grads
    ],
    client_optimizer: fedjax.Optimizer,
    server_optimizer: fedjax.Optimizer,
    client_batch_hparams: fedjax.ShuffleRepeatBatchHParams,
) -> fedjax.FederatedAlgorithm:
  """Builds stateful federated averaging that maintains client state.

  Args:
    grad_fn: A function from (params, batch_example, rng) to gradients.
      This can be created with `fedjax.model_grad`.
    client_optimizer: Optimizer for local client training.
    server_optimizer: Optimizer for server update.
    client_batch_hparams: Hyperparameters for batching client dataset for train.

  Returns:
    FederatedAlgorithm
  """
  train_for_each_client = create_train_for_each_client(grad_fn,
                                                       client_optimizer)

  def init(params: fedjax.Params) -> ServerState:
    opt_state = server_optimizer.init(params)
    client_states = {}
    return ServerState(params, opt_state, client_states)

  def apply(
      server_state: ServerState,
      clients: Sequence[
          Tuple[fedjax.ClientId, fedjax.ClientDataset, fedjax.PRNGKey]
      ],
  ) -> Tuple[ServerState, Mapping[fedjax.ClientId, Any]]:
    client_num_examples = {cid: len(cds) for cid, cds, _ in clients}
    batch_clients = []

    # zero_params = create_zero_tree(server_state.params)
    # initial_trace = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t), server_state.params)
    initial_trace = fedjax.tree_util.tree_zeros_like(server_state.params)
    params_server_past = server_state.params
    for cid, cds, crng in clients:
      batch_cds = cds.shuffle_repeat_batch(client_batch_hparams)
      if cid not in server_state.client_states:
        # Initialize client state total training steps counter.
        # server_state.client_states[cid] = ClientState(num_steps=0)
        # server_state.client_states[cid] = ClientState(num_steps=0, trace=initial_trace)

        # [Need to initialize each client's ClientState]
        # Note in particular that the rng for DP is now set here
        # todo: whether to initialize with crng
        server_state.client_states[cid] = ClientState(num_steps=0, trace=initial_trace,
                                                      adam_count=0, adam_mu=initial_trace, adam_nu=initial_trace,
                                                      # DP_rng=jax.random.PRNGKey(1234))
                                                      DP_rng=crng)
        # server_state.client_states[cid] = ClientState(num_steps=0, this_opt_state=None)
      client_input = {'rng': crng, 'state': server_state.client_states[cid]}
      batch_clients.append((cid, batch_cds, client_input))

    client_diagnostics = {}
    # Running weighted mean of client updates. We do this iteratively to avoid
    # loading all the client outputs into memory since they can be prohibitively
    # large depending on the model parameters size.
    delta_params_sum = fedjax.tree_util.tree_zeros_like(server_state.params)
    num_examples_sum = 0
    for client_id, client_output in train_for_each_client(
        server_state.params, batch_clients):
      delta_params = client_output['delta_params']
      # Update client state after the round of training.
      server_state.client_states[client_id] = client_output['state']
      num_examples = client_num_examples[client_id]
      delta_params_sum = fedjax.tree_util.tree_add(
          delta_params_sum,
          fedjax.tree_util.tree_weight(delta_params, num_examples),
      )
      num_examples_sum += num_examples
      # We record the l2 norm of client updates as an example, but it is not
      # required for the algorithm.
      client_diagnostics[client_id] = {
          'delta_l2_norm': fedjax.tree_util.tree_l2_norm(delta_params),
          'delta_params' : delta_params,
          'model_params': fedjax.tree_util.tree_add(params_server_past, delta_params),
      }
    mean_delta_params = fedjax.tree_util.tree_inverse_weight(
        delta_params_sum, num_examples_sum
    )
    server_state = server_update(server_state, mean_delta_params)

    # jax.debug.print("server_params {x}", x=server_state.params['conditioner_0']['conditioner']['layers_0']['kernel'][0,:2])

    return server_state, client_diagnostics

  def server_update(server_state, mean_delta_params):
    opt_state, params = server_optimizer.apply(mean_delta_params,
                                               server_state.opt_state,
                                               server_state.params)
    return ServerState(params, opt_state, server_state.client_states)

  return fedjax.FederatedAlgorithm(init, apply)
