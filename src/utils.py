import random
import warnings

import numpy as np
import qiskit_algorithms
from sklearn.preprocessing import MinMaxScaler
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


def calculate_fidelity_from_swap_test(result: dict[str, int]) -> float:
    """Calculate the quantum fidelity from the result of cswap test.

    :param dict[str, int] result: result of cswap test
    :raises ValueError: if the given result contains nothing or more than two keys
    :raises ValueError: if the given result is consisted with other thatn 0 and/or 1
    :raises ValueError: if the calculated quantum fidelity is negative
    :return float: quantum fidelity
    """
    key_0 = "0"
    key_1 = "1"

    if len(result) > 2 or len(result) == 0:
        msg = f"The argument result must be the dict whose keys are {key_0} and {key_1} as the result of CSWAP test, but {result}."
        raise ValueError(msg)

    keys = set(result.keys())
    if set(key_0) != keys and set(key_1) != keys and set([key_0, key_1]) != keys:
        msg = f"The keys of a given result must consist with {key_0} and/or {key_1}, but {keys}."
        raise ValueError(msg)

    num_zeros = result[key_0] if key_0 in result else 0
    if num_zeros == 0:
        return 0
    num_ones = result[key_1] if key_1 in result else 0
    probability_zero = num_zeros / (num_zeros + num_ones)
    fidelity = 2 * probability_zero - 1
    if fidelity < 0:
        fidelity = 0

    return fidelity


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


def normalise_data(data: np.ndarray) -> np.ndarray:
    """Normalise each data, which correponds to each row of the given data.

    :param np.ndarray data: data
    :return np.ndarray: normalised data
    """
    return data / np.linalg.norm(data, axis=1, keepdims=1)


def scale_data(data: np.ndarray) -> np.ndarray:
    """Scale each datam which corresponds to each row.

    :param np.ndarray data: data
    :return np.ndarray: scaled data
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)


def encode_through_arcsin(data: np.ndarray) -> np.ndarray:
    """Encode the given data through arcsin,
    which is given in the paper https://arxiv.org/pdf/2103.11307.

    :param np.ndarray data: data
    :return np.ndarray: encoded data
    """
    return 2 * np.arcsin(np.sqrt(data))


def pad_data(data: np.ndarray, pad_value: float = 0) -> np.ndarray:
    """Pad data with the given filling value.

    :param np.ndarray data: data
    :param float pad_value: value to pad given data, defaults to 0
    :return np.ndarray: padded data
    """
    if data.shape[1] % 2 != 0:
        new_shape = [0, 0]
        new_shape[0] = data.shape[0]
        new_shape[1] = data.shape[1] + 1
        padded_data = np.zeros(new_shape) + pad_value
        padded_data[:, :-1] = data
    return padded_data
