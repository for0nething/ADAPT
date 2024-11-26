""" Model Construction Function """
from parameterSetting import *
from nflows import transforms
import torch
from nflows import utils
import nflows.nn as nn_
import torch.nn.functional as F

if LOAD == "JAX":
    def create_linear_transform(features, i):
    #     masks = np.ones(features)
        return transforms.CompositeTransform([
            transforms.Permutation(torch.IntTensor(permutation_list[i])),
    #         transforms.RandomPermutation(features=features),
    #         transforms.AffineTransform()
            transforms.PointwiseAffineTransform()
    #         transforms.AffineCouplingTransform(masks=masks,)
    #         transforms.SVDLinear(features, num_householder=10, identity_init=True)
        ])

    def create_base_transform(i, features):
        # tmp_mask = utils.create_alternating_binary_mask(features, even=(i % 2 == 0))
        return transforms.coupling.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),     
            transform_net_create_fn=lambda in_features, out_features: nn_.nets.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                context_features=None,
                num_blocks=num_transform_blocks,
                activation=F.relu,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm
            ),
            num_bins=num_bins,
    #         tails='linear',
            tails=None,
            # tails='linear',
            tail_bound=tail_bound,
            # apply_unconditional_transform=True,
            apply_unconditional_transform=False,
            min_bin_width=1e-4,
            min_bin_height=1e-4,
            min_derivative=1e-4
        )


    # torch.masked_select()
    def create_transform(features):
        transform = transforms.CompositeTransform([
            transforms.CompositeTransform([
                create_linear_transform(features, i),
                create_base_transform(i, features)
            ]) for i in range(num_flow_steps)
        ]
        #                                           + [
        #     create_linear_transform(features, num_flow_steps)
        # ]
        )
        return transform


def assign_transform_params(model_dict, transform):
    # Use the parameters to initialize the nflows model
    #  - permutation：set when construction
    #  - scalar
    #  - RQSpline
    #  - base distribution : no need to change
    # for i in range(num_flow_steps -1, -1, -1):
    for i in range(num_flow_steps):
        print("!!", i)
        # 【Scalar affine】 shifts & scales
        scalar_i = "scalar_{}".format(i)
        shifts = torch.Tensor(model_dict[scalar_i]["shifts"])
        scales = torch.Tensor(model_dict[scalar_i]["scales"])
        # print("shifts.shape")
        # print(shifts.shape)
        # print("scales.shape")
        # print(scales.shape)

        transform._transforms[i]._transforms[0]._transforms[1]._shift = torch.nn.Parameter(shifts)
        transform._transforms[i]._transforms[0]._transforms[1]._scale = torch.nn.Parameter(scales)

        # 【RQ-spline parameters】
        conditioner_i = "conditioner_{}".format(i)
        transform_net = transform._transforms[i]._transforms[1].transform_net  # read-only, can not modify

        # 【initial layer】:  6*256
        # initial_layer = transform_net.initial_layer
        initial_layer_weight = torch.Tensor(model_dict[conditioner_i]["layers_0"]["kernel"]).T
        initial_layer_bias = torch.Tensor(model_dict[conditioner_i]["layers_0"]["bias"])
        # print("see initial_layer shape ", model_dict[conditioner_i]["layers_0"]["kernel"].shape)
        # print("see initial_layer_weight.shape: ", initial_layer_weight.shape)
        # print("correct shape ", transform._transforms[i]._transforms[1].transform_net.initial_layer.weight.shape)
        # print("correct shape ", transform._transforms[i]._transforms[1].transform_net.initial_layer.bias.shape)

        transform._transforms[i]._transforms[1].transform_net.initial_layer.weight = torch.nn.Parameter(
            initial_layer_weight)
        transform._transforms[i]._transforms[1].transform_net.initial_layer.bias = torch.nn.Parameter(
            initial_layer_bias)

        # 【hidden layers】: 2x  256*256
        for block_id, net in enumerate(transform_net.blocks):
            linear_layers = net.linear_layers
            for linear_id, linear_layer in enumerate(linear_layers):
                hidden_weight = torch.Tensor(
                    model_dict[conditioner_i]["layers_{}".format(block_id + 1)]["layers_{}".format(linear_id)][
                        "kernel"]).T
                hidden_bias = torch.Tensor(
                    model_dict[conditioner_i]["layers_{}".format(block_id + 1)]["layers_{}".format(linear_id)]["bias"])
                transform._transforms[i]._transforms[1].transform_net.blocks[block_id].linear_layers[
                    linear_id].weight = torch.nn.Parameter(hidden_weight)
                transform._transforms[i]._transforms[1].transform_net.blocks[block_id].linear_layers[
                    linear_id].bias = torch.nn.Parameter(hidden_bias)

        # 【final layer】: 256*75
        final_layer_weight = torch.Tensor(model_dict[conditioner_i]["layers_4"]["kernel"]).T
        final_layer_bias = torch.Tensor(model_dict[conditioner_i]["layers_4"]["bias"])
        # print("correct final_layer shape ", transform._transforms[i]._transforms[1].transform_net.final_layer.weight.shape)
        transform._transforms[i]._transforms[1].transform_net.final_layer.weight = torch.nn.Parameter(
            final_layer_weight)
        transform._transforms[i]._transforms[1].transform_net.final_layer.bias = torch.nn.Parameter(final_layer_bias)
    return transform