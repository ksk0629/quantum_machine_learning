import numpy as np
import pytest
import qiskit

from src.s_swap_gate import SSwapGate


class TestSSwapGate:

    @classmethod
    def setup_class(cls):
        cls.sswap = SSwapGate()
        cls.correct_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 0.5 * (1 + 1j), 0.5 * (1 - 1j), 0],
                [0, 0.5 * (1 - 1j), 0.5 * (1 + 1j), 0],
                [0, 0, 0, 1],
            ]
        )

    def test_init(self):
        """Normal test;
        Check if the SSwapGate created in setup_class is the instance of qiskit.circuit.Gate.
        """
        assert isinstance(self.sswap, qiskit.circuit.Gate)

    def test_matrix_representation(self):
        """Normal test;
        Check if the matrix representation of SSwapGate is correct.
        """
        circuit = qiskit.QuantumCircuit(2)
        circuit.append(self.sswap, [0, 1])

        unitary = qiskit.quantum_info.Operator(circuit)
        assert np.allclose(unitary.data, self.correct_matrix)
