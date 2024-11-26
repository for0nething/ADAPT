import numpy as np
import torch
from torch.nn import functional as F

from nflows.transforms.base import InputOutsideDomain
from nflows.utils import torchutils

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
    enable_identity_init=False,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        # unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        # constant = np.log(np.exp(1 - min_derivative) - 1)
        # unnormalized_derivatives[..., 0] = constant
        # unnormalized_derivatives[..., -1] = constant
        #
        # outputs[outside_interval_mask] = inputs[outside_interval_mask]
        # logabsdet[outside_interval_mask] = 0


        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1),
                                         value=constant)

        outputs += outside_interval_mask * inputs
        distrax_kwargs = {}
        # logabsdet += outside_interval_mask*0
    else:
        # new, currently only supports forward. For `inverse`, needs further modify the code according to Distrax
        below_interval_mask = (inputs < -tail_bound)
        above_interval_mask = (inputs > tail_bound)
        distrax_kwargs = {"below_interval_mask": below_interval_mask,
                          "above_interval_mask": above_interval_mask,
                          "outside_interval_linear": True,
                          "original_inputs": inputs}
        # constant = np.log(np.exp(1 - min_derivative) - 1)
        # unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1),
        #                                  value=constant)

        # outputs += outside_interval_mask * inputs
        # raise RuntimeError("{} tails are not implemented.".format(tails))


    # if torch.any(inside_interval_mask):
    #     (
    #         outputs[inside_interval_mask],
    #         logabsdet[inside_interval_mask],
    #     ) = rational_quadratic_spline(
    #         inputs=inputs[inside_interval_mask],
    #         unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
    #         unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
    #         unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
    #         inverse=inverse,
    #         left=-tail_bound,
    #         right=tail_bound,
    #         bottom=-tail_bound,
    #         top=tail_bound,
    #         min_bin_width=min_bin_width,
    #         min_bin_height=min_bin_height,
    #         min_derivative=min_derivative,
    #         enable_identity_init=enable_identity_init,
    #     )
    (
        inside_outputs,
        inside_logabsdet,
    ) = rational_quadratic_spline(
        # Clamp inputs to the domain to prevent out of domain errors
        inputs=inputs.clamp(min=-tail_bound, max=tail_bound),
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **distrax_kwargs
    )
    # print("see inside_interval_mask , ", inside_interval_mask.shape)
    # print(inside_interval_mask)
    # print("see outputs , ", outputs.shape)
    # print(outputs)
    # print("see logabsdet ", logabsdet.shape)
    # print(logabsdet)

    # outputs = inside_outputs
    # logabsdet = logabsdet
    if tails == "linear":
        outputs += inside_interval_mask * inside_outputs
        logabsdet += inside_interval_mask * inside_logabsdet
    else:
        outputs = inside_outputs
        logabsdet = inside_logabsdet

    # print("#" * 50)
    # print("see inside_interval_mask , ", inside_interval_mask.shape)
    # print(inside_interval_mask)
    # print("see outputs , ",outputs.shape)
    # print(outputs)
    # print("see logabsdet ", logabsdet.shape)
    # print(logabsdet)

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
    enable_identity_init=False,
    outside_interval_linear=False,
    below_interval_mask=None,
    above_interval_mask=None,
    original_inputs=None
):
    # print("min(inputs) ", torch.min(inputs))
    # print("max(inputs) ", torch.max(inputs))
    # print("limits: ", left, right)
    if torch.min(inputs) < left or torch.max(inputs) > right:

        raise InputOutsideDomain()

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)

    # 【Original version width】
    # widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    # # cumwidths = torch.cumsum(widths, dim=-1)
    # # cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    # # cumwidths = (right - left) * cumwidths + left
    # # cumwidths[..., 0] = left
    #
    # widths *= right - left



    # 【distrax version width】
    widths = widths * ((right - left) - min_bin_width * num_bins)  + min_bin_width



    cumwidths = F.pad(widths, pad=(1, 0), mode="constant", value=left)
    cumwidths = torch.cumsum(cumwidths, dim=-1)
    # Make right-most knot at the right boundary
    cumwidths[..., -1] = right
    # widths = cumwidths[..., 1:] - cumwidths[..., :-1]




    # 【Original version derivative】
    # if enable_identity_init: #flow is the identity if initialized with parameters equal to zero
    #     beta = np.log(2) / (1 - min_derivative)
    # else: #backward compatibility
    #     beta = 1
    # derivatives = min_derivative + F.softplus(unnormalized_derivatives, beta=beta)


    # 【distrax's version of derivative】
    min_derivative = torch.Tensor([min_derivative])
    offset = torch.log(torch.exp(1 - min_derivative) - 1)
    derivatives =  F.softplus(unnormalized_derivatives + offset) + min_derivative





    heights = F.softmax(unnormalized_heights, dim=-1)



    # 【Original version height】
    # heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    # # cumheights = torch.cumsum(heights, dim=-1)
    # # cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    # # cumheights = (top - bottom) * cumheights + bottom
    # # cumheights[..., 0] = bottom
    #
    # heights *= top - bottom

    # 【distrax version of height】
    heights = heights * ((top - bottom) - min_bin_height * num_bins) + min_bin_height



    cumheights = F.pad(heights, pad=(1, 0), mode="constant", value=bottom)
    cumheights = torch.cumsum(cumheights, dim=-1)
    # Make top-most knot at the top boundary
    cumheights[..., -1] = top
    # heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        # The original code in nflows
        # bin_idx = torchutils.searchsorted(cumheights, inputs)[..., None]

        # After modification, use torch.searchsorted
        cumheights[..., -1] += 1e-6
        bin_idx = torch.searchsorted(cumheights, inputs[..., None], side="right") - 1


    else:
        # The nflows version,  any pytorch version seems to work just fine because it's searchsorted by hand
        # bin_idx = torchutils.searchsorted(cumwidths, inputs)[..., None]

        # searchsorted is faster but not supported in earlier versions of pytorch due to api changes
        cumwidths[..., -1] += 1e-6
        bin_idx = torch.searchsorted(cumwidths, inputs[..., None], side="right") - 1


    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    # print("【see inputs】")
    # print(inputs)
    # print()
    #
    # print(f"!!see 【bin_widths】   shape:[{widths.shape}]" )
    # print(widths)
    # print()
    #
    # print(f"see 【bin_heights】  shape:[{heights.shape}]")
    # print(heights)
    # print()
    #
    # print(f"see 【input_widths】: shape [{input_bin_widths.shape}]")
    # print(input_bin_widths)
    # print()
    #
    # print(f"see 【input_heights】: shape [{input_heights.shape}]")
    # print(input_heights)
    # print()
    #
    #
    # # derivatives[:,:,0] = 1
    # # derivatives[:,:,-1]=1
    # print(f"see 【derivatitives】: shape [{derivatives.shape}]")
    # print(derivatives)
    # print()


    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # root = (- b + torch.sqrt(discriminant)) / (2 * a)
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        if outside_interval_linear == True:
            # new
            # If x is outside the spline range, we default to a linear transformation.
            #
            #
            # print((inputs - cumwidths[...,0]).shape)
            # print(((inputs - cumwidths[...,0]) * derivatives[..., 0]).shape)
            # print(cumheights[0].shape)
            # print(below_interval_mask.shape)

            # print("cumwidths   ", cumwidths.shape)
            # print(cumwidths)

            # print("cumheights   ", cumheights.shape)
            # print(cumheights)

            # print("bounds ")
            # print(cumwidths[...,0])
            # print(cumheights[...,0])
            # print(derivatives[...,0])

            # print("original outputs ")
            # print(outputs)
            # print("below_interval_mask")
            # print(below_interval_mask)

            outputs[below_interval_mask] = 0
            outputs[above_interval_mask] = 0
            logabsdet[below_interval_mask] = 0
            logabsdet[above_interval_mask] = 0

            outputs += below_interval_mask * ((original_inputs - cumwidths[..., 0]) * derivatives[...,0] + cumheights[...,0])
            outputs += above_interval_mask * ((original_inputs - cumwidths[..., -1]) * derivatives[...,-1] + cumheights[...,-1])

            logabsdet += below_interval_mask * torch.log(derivatives[...,0])
            logabsdet += above_interval_mask * torch.log(derivatives[...,-1])

            # print("outputs is ")
            # print(outputs)

        return outputs, logabsdet
