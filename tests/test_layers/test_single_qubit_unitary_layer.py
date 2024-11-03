import pytest

from src.layers.single_qubit_unitary_layer import SingleQubitUnitaryLayer


class TestSingleQubitUnitaryLayer:
    @classmethod
    def setup_class(cls):
        cls.param_prefix = "test_single_qubit_unitary"
        cls.q_conv_nn = SingleQubitUnitaryLayer(param_prefix=cls.param_prefix)
