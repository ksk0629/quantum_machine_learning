import random

import numpy as np
import qiskit_algorithms
import torch


def fix_seed(seed: int):
    """Fix the random seeds to have reproducibility.

    :param int seed: random seed
    """
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    qiskit_algorithms.utils.algorithm_globals.random_seed = 12345


def get_parameter_dict(
    parameter_names: list[str], parameters: list[float]
) -> dict[str, float]:
    """Get the dictionary whose keys are names of the given parameters and the values are parameter values.

    :param list[str] parameter_names: names of parameters
    :param list[float] parameters: patameter values
    :return dict[str, float]: parameter dictionary
    """
    parameter_dict = {
        parameter_name: parameter
        for parameter_name, parameter in zip(parameter_names, parameters)
    }
    return parameter_dict


def calc_2d_output_shape(
    height: int,
    width: int,
    kernel_size: tuple[int, int],
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> tuple[int, int]:
    """Calculate an output shape of convolutional or pooling layers.
    Referred to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.

    :param int in_height: input height
    :param int in_width: input width
    :param tuple[int, int] kernel_size: kernel size
    :param int stride: stride, defaults to 1
    :param int padding: padding, defaults to 0
    :param int dilation: dilation, defaults to 1
    :return tuple[int, int]: output shape
    """
    output_height = np.floor(
        (height + 2 * padding - dilation * (kernel_size[0] - 1) - 1) / stride + 1
    )
    output_width = np.floor(
        (width + 2 * padding - dilation * (kernel_size[1] - 1) - 1) / stride + 1
    )
    return (output_height, output_width)
