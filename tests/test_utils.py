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
            {"0": 1, "1": -3},
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
