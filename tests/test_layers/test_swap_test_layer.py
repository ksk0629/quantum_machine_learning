import pytest

from quantum_machine_learning.layers.swap_test_layer import SwapTestLayer


class TestSwapTestLayer:
    @classmethod
    def setup_class(cls):
        cls.control_qubit = 0
        cls.qubit_pairs = [(1, 4), (2, 5), (3, 6)]
        cls.swap_test_layer = SwapTestLayer(
            control_qubit=cls.control_qubit,
            qubit_pairs=cls.qubit_pairs,
        )

    def test_init(self):
        """Normal test
        Check if the class that was prepared in setup_class has
        - the same control_qubit as self.control_qubit.
        - the same qubit_pairs as self.qubit_pairs.
        """
        assert self.swap_test_layer.control_qubit == self.control_qubit
        assert self.swap_test_layer.qubit_pairs == self.qubit_pairs
