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
        pass

    @pytest.mark.utils
    def test_calculate_accuracy(self):
        """Normal test;
        run calculate_accuracy with arguments of which the accuracies are 0, 1/2 and 1.

        Check if the return value is correct.
        """
        pass

    @pytest.mark.utils
    def test_safe_log_e_with_minus_1(self):
        """Normal test;
        run sage_log_e with 1e-1.

        Check if the return value is -1.
        """
        pass

    @pytest.mark.utils
    def test_safe_log_e_with_zero(self):
        """Normal test;
        run safe_log_e with 0.

        Check if the return value is -16.
        """
        pass

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
