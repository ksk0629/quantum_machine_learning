import os
import random

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


def get_basic_info_path(model_dir_path: str, file_prefix: str | None = None) -> str:
    """Get basic_info.pkl path.

    :param str model_dir_path: path to directory
    :param str | None file_prefix: file prefix, defaults to None
    :return str: path to basic information file
    """
    filename = (
        f"{file_prefix}basic_info.pkl" if file_prefix is not None else "basic_info.pkl"
    )
    return os.path.join(model_dir_path, filename)


def get_circuit_path(model_dir_path: str, file_prefix: str | None = None) -> str:
    """Get circuit.qpy path.

    :param str model_dir_path: path to directory
    :param str | None file_prefix: file prefix, defaults to None
    :return str: path to circuit file path
    """
    filename = f"{file_prefix}circuit.qpy" if file_prefix is not None else "circuit.qpy"
    return os.path.join(model_dir_path, filename)


def get_trainable_parameters_path(
    model_dir_path: str, file_prefix: str | None = None
) -> str:
    """Get trainable_parameters.pkl path.

    :param str model_dir_path: path to directory
    :param str | None file_prefix: file prefix, defaults to None
    :return str: path to trainable_parameters file path
    """
    filename = (
        f"{file_prefix}trainable_parameters.pkl"
        if file_prefix is not None
        else "trainable_parameters.pkl"
    )
    return os.path.join(model_dir_path, filename)


def get_data_parameters_path(
    model_dir_path: str, file_prefix: str | None = None
) -> str:
    """Get data_parameters.pkl path.

    :param str model_dir_path: path to directory
    :param str | None file_prefix: file prefix, defaults to None
    :return str: path to data_parameters file path
    """
    filename = (
        f"{file_prefix}data_parameters.pkl"
        if file_prefix is not None
        else "data_parameters.pkl"
    )
    return os.path.join(model_dir_path, filename)


def get_trained_parameters_path(
    model_dir_path: str, file_prefix: str | None = None
) -> str:
    """Get trained_parameters.pkl path.

    :param str model_dir_path: path to directory
    :param str | None file_prefix: file prefix, defaults to None
    :return str: path to trained_parameters file path
    """
    filename = (
        f"{file_prefix}trained_parameters.pkl"
        if file_prefix is not None
        else "trained_parameters.pkl"
    )
    return os.path.join(model_dir_path, filename)


def get_sliding_window_single_channel_data(
    data: np.ndarray, window_size: tuple[int, int]
) -> np.ndarray:
    """Get the sliding window data.

    :param np.ndarray data: two dimensional data
    :param tuple[int, int] window_size: window size
    :return np.ndarray: data whose each entry is sliding window
    """
    return np.lib.stride_tricks.sliding_window_view(data, window_size)


def get_sliding_window_multi_channel_data(
    data: np.ndarray, window_size: tuple[int, int]
) -> np.ndarray:
    """Get the sliding window data.

    :param np.ndarray data: three dimensional data
    :param tuple[int, int] window_size: window size
    :return np.ndarray: data whose each entry is sliding window
    """
    sliding_window_data = []
    for single_channel in data:
        sliding_window_data.append(
            get_sliding_window_single_channel_data(
                data=single_channel, window_size=window_size
            )
        )
    return np.array(sliding_window_data)


def get_sliding_window_batch_data(
    batch_data: np.ndarray, window_size: tuple[int, int]
) -> np.ndarray:
    """Get the sliding window data.

    :param np.ndarray batch_data: batch data whose shape is [batch, channels, height, width]
    :param tuple[int, int] window_size: window size
    :return np.ndarray: batch data whose each entry is sliding window
    """
    processed_batch_data = []
    for data in batch_data:
        processed_batch_data.append(
            get_sliding_window_multi_channel_data(data=data, window_size=window_size)
        )
    return np.array(processed_batch_data)


def encode_according_to_threshold(
    data: np.ndarray, threshold: float, low_value: float, high_value: float
) -> np.ndarray:
    """Encode according to the given low_value and high_value using the given threshold.
    Each datapoint being smaller than threshold is encoded into the given low_value,
    otherwise, the given high_value.

    :param np.ndarray data: data to be encoded
    :param float threshold: threshold
    :param float low_value: low value to be substituted
    :param float high_value: high value to be substituted
    :return np.ndarray: encoded data
    """
    low_indices = np.where(data < threshold)
    high_indices = np.where(data >= threshold)
    data[low_indices] = low_value
    data[high_indices] = high_value

    return data
