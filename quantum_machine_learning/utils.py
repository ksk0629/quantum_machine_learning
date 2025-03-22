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


def calculate_accuracy(predicted_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Calculate accuracy.

    :param np.ndarray predicted_labels: predicted labels
    :param np.ndarray true_labels: true labels
    :raises ValueError: if predicted_labels and true_labels have the different lengths
    :return float: accuracy
    """
    if len(predicted_labels) != len(true_labels):
        msg = f"Given predicted_labels and true_labels must be the same lengths, but {len(predicted_labels)} and {len(true_labels)}."
        raise ValueError(msg)

    num_correct = (predicted_labels == true_labels).sum()
    return num_correct / len(predicted_labels)


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


def encode_through_arcsin(data: np.ndarray) -> np.ndarray:
    """Encode the given data through arcsin,
    which is given in the paper https://arxiv.org/pdf/2103.11307.

    :param np.ndarray data: data
    :return np.ndarray: encoded data
    """
    return 2 * np.arcsin(np.sqrt(data))


def count_ones(result: dict[str, int]) -> int:
    """Count the number of ones in the most likely outcome of the circuit result.

    :param dict[str, int] result: qiskit circuit result
    :return int: number of ones
    """
    key_1 = "1"
    # Sort the resuly by the frequency.
    sorted_result = dict(sorted(result.items(), key=lambda item: -item[1]))
    # Get the most likely result.
    most_likely_result = list(sorted_result.keys())[0]
    # Count the number of ones.
    num_ones = most_likely_result.count(key_1)
    return num_ones


def encode_according_to_threshold(
    data: np.ndarray, threshold: float, low_value: float, high_value: float
) -> np.ndarray:
    """Encode according to the given low_value and high_value using the given threshold.
    Each datapoint being larger than threshold is encoded into the given high_value,
    otherwise, the given high_value.

    :param np.ndarray data: data to be encoded
    :param float threshold: threshold
    :param float low_value: low value to be substituted
    :param float high_value: high value to be substituted
    :return np.ndarray: encoded data
    """
    low_indices = np.where(data <= threshold)
    high_indices = np.where(data > threshold)
    data[low_indices] = low_value
    data[high_indices] = high_value

    return data


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
