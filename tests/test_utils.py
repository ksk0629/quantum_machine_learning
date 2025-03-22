import random

import numpy as np
import pytest
import qiskit_algorithms
import torch

import quantum_machine_learning.utils as utils


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
        "parameter_dict", [{"x": 1.2, "1": 2}, {"y": 3.1}, {"layers!": 901}]
    )
    def test_get_parameter_dict(self, parameter_dict):
        parameter_names = list(parameter_dict.keys())
        parameters = list(parameter_dict.values())
        result = utils.get_parameter_dict(
            parameter_names=parameter_names, parameters=parameters
        )

        assert result == parameter_dict

    def test_calc_2d_output_shape(self):
        """Normal test;
        run calc_2d_output_shape.

        Check if the returned shape is appropriate.
        """
        height = 50
        width = 25
        kernel_size = (5, 5)
        (out_height, out_width) = utils.calc_2d_output_shape(
            height=height, width=width, kernel_size=kernel_size
        )

        assert out_height == 46
        assert out_width == 21
