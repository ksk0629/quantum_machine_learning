import pytest

from quantum_machine_learning.utils.data_utils import DataUtils


class TestDataUtils:

    @classmethod
    def setup_class(cls):
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
        (out_height, out_width) = DataUtils.calc_2d_output_shape(
            height=height, width=width, kernel_size=kernel_size
        )

        assert out_height == 46
        assert out_width == 21
