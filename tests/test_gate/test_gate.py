import numpy as np
import pytest
import qiskit

from quantum_machine_learning.gate.s_swap_gate import SSwapGate


class TestGate:

    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.gate
    def test_sswap_matrix_representation(self):
        """Normal test;
        Check if
        - the type of the instance of SSWapGate is qiskit.circuit.Gate.
        - the matrix representation of SSwapGate is correct.
        """
        circuit = qiskit.QuantumCircuit(2)
        s_swap_gate = SSwapGate()
        assert isinstance(s_swap_gate, qiskit.circuit.Gate)
        circuit.append(s_swap_gate, [0, 1])

        unitary = qiskit.quantum_info.Operator(circuit)
        correct_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 0.5 * (1 + 1j), 0.5 * (1 - 1j), 0],
                [0, 0.5 * (1 - 1j), 0.5 * (1 + 1j), 0],
                [0, 0, 0, 1],
            ]
        )
        assert np.allclose(unitary.data, correct_matrix)
