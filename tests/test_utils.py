import os
import random

import numpy as np
import pytest
import qiskit_algorithms
import torch

import src.utils as utils


class TestUtils:

    @classmethod
    def setup_class(cls):
        cls.seed = 91
        cls.model_dir_path = "./../test"

    def test_fix_seed_with_self_args(self):
        """Normal test;
        Run fix_seed and generate random integers through each module and do the same thing.

        Check if the generated integers are the same.
        """
        low = 0
        high = 100000

        utils.fix_seed(self.seed)
        x_random = random.randint(low, high)
        x_qiskit = qiskit_algorithms.utils.algorithm_globals.random.integers(low, high)
        x_np = np.random.randint(low, high)
        x_torch = torch.randint(low=low, high=high, size=(1,))

        utils.fix_seed(self.seed)
        assert x_random == random.randint(low, high)
        assert x_qiskit == qiskit_algorithms.utils.algorithm_globals.random.integers(
            low, high
        )
        assert x_np == np.random.randint(low, high)
        assert x_torch == torch.randint(low=low, high=high, size=(1,))

    @pytest.mark.parametrize(
        "result",
        [
            {"0": 1, "1": 1, "2": 1},
            {},
            {"0": 1, "-1": 1},
            {"1": 1, "3": 1},
            {"-2": 1},
        ],
    )
    def test_calculate_fidelity_from_swap_test_with_invalid_arg(self, result):
        """Abnormal test;
        Run calculate_fidelity_from_swap_test with an invalid argument.

        Check if ValueError happens.
        """
        with pytest.raises(ValueError):
            utils.calculate_fidelity_from_swap_test(result)

    @pytest.mark.parametrize(
        "result",
        [
            {"0": 1},
            {"0": 1, "1": 1},
            {"1": 1},
        ],
    )
    def test_calculate_fidelity_from_swap_test_with_valid_arg(self, result):
        """Normal test;
        Run calculate_fidelity_from_swap_test with a valid argument.

        Check if
        - the return value is between 0 and 1
        """
        fidelity = utils.calculate_fidelity_from_swap_test(result)
        assert 0 <= fidelity <= 1

    @pytest.mark.parametrize(
        "predicted_labels_and_true_labels", [[[1], [2, 3]], [[1, 2], [3]]]
    )
    def test_calculate_accuracy_with_invalid_args(
        self, predicted_labels_and_true_labels
    ):
        """Abnormal test;
        Run calculate_accuracy with an invalid arguments.

        Check if ValueError happens.
        """
        (predicted_labels, true_labels) = predicted_labels_and_true_labels
        with pytest.raises(ValueError):
            utils.calculate_accuracy(
                predicted_labels=predicted_labels, true_labels=true_labels
            )

    @pytest.mark.parametrize(
        "predicted_labels_and_true_labels_and_accuracy",
        [[[1], [1], 1], [[1, 2], [1, 3], 0.5], [[1, 2], [2, 1], 0]],
    )
    def test_calculate_accuracy_with_valid_args(
        self, predicted_labels_and_true_labels_and_accuracy
    ):
        """Abnormal test;
        Run calculate_accuracy with a valid arguments.

        Check if the return value, which is an accuracy, is correct.
        """
        (predicted_labels, true_labels, true_accuracy) = (
            predicted_labels_and_true_labels_and_accuracy
        )
        predicted_labels = np.array(predicted_labels)
        true_labels = np.array(true_labels)
        accuracy = utils.calculate_accuracy(
            predicted_labels=predicted_labels, true_labels=true_labels
        )
        assert accuracy == true_accuracy

    @pytest.mark.parametrize(
        "parameter_dict", [{"x": 1.2, "1": 2}, {"y": 3.1}, {"layers!": 901}]
    )
    def test_get_parameter_dict(self, parameter_dict):
        parameter_names = list(parameter_dict.keys())
        parameters = list(parameter_dict.values())
        result = utils.get_parameter_dict(
            parameter_names=parameter_names, parameters=parameters
        )

        assert result == parameter_dict

    @pytest.mark.parametrize("data", [[[2, 3], [1, 1]], [[1, 1]]])
    def test_normalise_data(self, data):
        """Normal test;
        Run normalise_data.

        Check if the vectors are normalised.
        """
        result = utils.normalise_data(data)

        normalised_data = []
        for _d in data:
            normalised_data.append(_d / np.linalg.norm(_d))
        normalised_data = np.array(normalised_data)

        assert np.allclose(result, normalised_data)

    @pytest.mark.parametrize("data", [[[0.1, 0.9], [1, 1]], [[1, 1]]])
    def test_encode_through_arcsin(self, data):
        """Normal test;
        Run encode_through_arcsin.

        Check if the data is encoded as it should be.
        """
        result = utils.encode_through_arcsin(data)
        encoded_data = 2 * np.arcsin(np.sqrt(data))
        assert np.allclose(result, encoded_data)

    def test_get_basic_info_path(self):
        """Normal test;
        Run get_basic_info_path function.

        Check if the return value is self.model_dir_path/basic_info.pkl.
        """
        assert utils.get_basic_info_path(self.model_dir_path) == os.path.join(
            self.model_dir_path, "basic_info.pkl"
        )

    def test_get_circuit_path(self):
        """Normal test;
        Run get_circuit_path function.

        Check if the return value is self.model_dir_path/circuit.qpy.
        """
        assert utils.get_circuit_path(self.model_dir_path) == os.path.join(
            self.model_dir_path, "circuit.qpy"
        )

    def test_get_trainable_parameters_path(self):
        """Normal test;
        Run get_trainable_parameters_path function.

        Check if the return value is self.model_dir_path/trainable_parameters.pkl.
        """
        assert utils.get_trainable_parameters_path(self.model_dir_path) == os.path.join(
            self.model_dir_path, "trainable_parameters.pkl"
        )

    def test_get_data_parameters_path(self):
        """Normal test;
        Run get_data_parameters_path function.

        Check if the return value is self.model_dir_path/data_parameters.pkl.
        """
        assert utils.get_data_parameters_path(self.model_dir_path) == os.path.join(
            self.model_dir_path, "data_parameters.pkl"
        )

    def test_get_trained_parameters_path(self):
        """Normal test;
        Run get_trained_parameters_path function.

        Check if the return value is self.model_dir_path/trained_parameters.pkl.
        """
        assert utils.get_trained_parameters_path(self.model_dir_path) == os.path.join(
            self.model_dir_path, "trained_parameters.pkl"
        )

    def test_get_classical_torch_model_path(self):
        """Normal test;
        run get_classical_torch_model_path function.

        Check if the return value is self.model_dir_path/classical_model.pth.
        """
        assert utils.get_classical_torch_model_path(
            self.model_dir_path
        ) == os.path.join(self.model_dir_path, "classical_model.pth")

    @pytest.mark.parametrize("window_size", [(2, 2), (3, 3)])
    def test_get_sliding_window_batch_data(self, window_size):
        """Normal test;
        run get_sliding_window_batch_data.

        Check if each sliding window is as expected.
        """
        batch_data = np.arange(5 * 2 * 16).reshape((5, 2, 4, 4))
        sliding_window_batch_data = utils.get_sliding_window_batch_data(
            batch_data=batch_data, window_size=window_size
        )

        for sliding_window_data, data in zip(sliding_window_batch_data, batch_data):

            for sliding_window_single_channel, single_channel in zip(
                sliding_window_data, data
            ):
                for row_index in range(len(single_channel)):
                    for column_index in range(len(single_channel[row_index])):
                        try:
                            window = sliding_window_single_channel[row_index][
                                column_index
                            ]
                            row_start = row_index
                            row_end = row_start + window_size[0]
                            column_start = column_index
                            column_end = column_start + window_size[1]
                            cropped = single_channel[
                                row_start:row_end, column_start:column_end
                            ]
                            print(cropped)
                        except IndexError:
                            continue
                        assert np.allclose(window, cropped)

    def test_encode_according_to_threshold(self):
        """Normal test;
        run encode_according_to_threshold.

        Check if
        - the data is encoded as it should be.
        - the data shape and the encoded data shape is the same.
        """
        length = 2 * 3 * 4 * 4
        data = np.arange(length).reshape((2, 3, 4, 4))
        threshold = length // 2
        encoded_data = utils.encode_according_to_threshold(
            data=data, threshold=threshold, low_value=0, high_value=1
        )
        assert encoded_data.sum() == threshold
        assert data.shape == encoded_data.shape
