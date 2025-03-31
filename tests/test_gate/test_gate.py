import numpy as np
import pytest
import qiskit

from quantum_machine_learning.gate.s_swap_gate import SSwapGate


class TestGate:

    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.gate
    def test_init_with_defaults(self):
        """Normal test;
        create an instance of the square root swap gate.

        Check if
        1. its name is "√SWAP".
        2. the number of qubits to be applied is two.
        """
        s_swap_gate = SSwapGate()
        # 1. its name is "√SWAP".
        assert s_swap_gate.name == "√SWAP"
        # 2. the number of qubits to be applied is two.
        assert s_swap_gate.num_qubits == 2

    @pytest.mark.gate
    def test_sswap_matrix_representation(self):
        """Normal test;
        create an instance of the square root swap gate.

        Check if
        1. the matrix representation of SSwapGate is correct.
        """
        circuit = qiskit.QuantumCircuit(2)
        s_swap_gate = SSwapGate()
        # 1. the matrix representation of SSwapGate is correct.
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
