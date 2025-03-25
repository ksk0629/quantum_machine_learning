import pytest

from quantum_machine_learning.layers.swap_test_layer import SwapTestLayer


class TestSwapTestLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    def test(self):
        """Normal test;
        Create an instance of SwapTestLayer.

        Check if
        - its control_qubit is the same as the given control_qubit.
        - its qubit_pairs is the same as the given qubit_pairs.
        - its control_qubits is the same as the new control_qubit after substituting a new control_qubit.
        - its qubit_pairs is the same as the new qubit_pairs after substituting a new qubit_pairs.
        """
        num_state_qubits = 5
        control_qubit = 0
        qubit_pairs = [(1, 3), (2, 4)]
        layer = SwapTestLayer(
            num_state_qubits, control_qubit=control_qubit, qubit_pairs=qubit_pairs
        )
        assert layer.control_qubit == control_qubit
        assert layer.qubit_pairs == qubit_pairs

        layer._build()  # For the coverage

        new_control_qubit = 1
        new_qubit_pairs = [(2, 3)]
        layer.control_qubit = new_control_qubit
        layer.qubit_pairs = new_qubit_pairs
        assert layer.control_qubit == new_control_qubit
        assert layer.qubit_pairs == new_qubit_pairs
