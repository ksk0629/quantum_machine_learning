import numpy as np
import pytest

from quantum_machine_learning.utils.calculation_utils import CalculationUtils


class TestCalculationUtils:

    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.utils
    def test_calculate_cross_entropy_zero(self):
        """Normal test;
        run calculate_cross_entropy with arguments of which the cross entropy is 0.

        Check if the return value is 0.
        """
        probabilities_list = [
            {"A": 1, "B": 0, "C": 0},
            {"A": 0, "B": 1, "C": 0},
            {"A": 0, "B": 0, "C": 1},
        ]
        true_labels = ["A", "B", "C"]
        cross_entropy = CalculationUtils.calculate_cross_entropy(
            probabilities_list=probabilities_list,
            true_labels=true_labels,
        )
        assert cross_entropy == 0

    @pytest.mark.utils
    def test_calculate_accuracy(self):
        """Normal test;
        run calculate_accuracy with arguments of which the accuracies are 0, 1/2 and 1.

        Check if the return value is correct.
        """
        predicted_labels = ["A", "B", "C"]
        true_labels = ["B", "C", "A"]
        shold_be_zero = CalculationUtils.calculate_accuracy(
            predicted_labels=predicted_labels, true_labels=true_labels
        )
        assert shold_be_zero == 0

        predicted_labels = ["A", "B", "C", "D"]
        true_labels = ["B", "C", "C", "D"]
        shold_be_half = CalculationUtils.calculate_accuracy(
            predicted_labels=predicted_labels, true_labels=true_labels
        )
        assert shold_be_half == 0.5

        predicted_labels = ["A", "B", "C", "D"]
        true_labels = ["A", "B", "C", "D"]
        shold_be_one = CalculationUtils.calculate_accuracy(
            predicted_labels=predicted_labels, true_labels=true_labels
        )
        assert shold_be_one == 1

    @pytest.mark.utils
    def test_safe_log_e(self):
        """Normal test;
        run sage_log_e with e^(-1) and 0.

        Check if the return values are -1 and log_e(1e-10).
        """
        value = np.exp(-1)
        should_be_minus_one = CalculationUtils.safe_log_e(value=value)
        assert should_be_minus_one == -1

        value = 0
        result = CalculationUtils.safe_log_e(value=value)
        assert result == np.log(1e-16)

    @pytest.mark.utils
    def test_calc_2d_output_shape(self):
        """Normal test;
        run calc_2d_output_shape.

        Check if the returned shape is appropriate.
        """
        height = 50
        width = 25
        kernel_size = (5, 5)
        (out_height, out_width) = CalculationUtils.calc_2d_output_shape(
            height=height, width=width, kernel_size=kernel_size
        )

        assert out_height == 46
        assert out_width == 21
