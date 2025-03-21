import itertools

import pytest
import qiskit

from quantum_machine_learning.layers.random_layer import RandomLayer


class TestRandomLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5, 10])
    def test_init(self, num_qubits):
        """Normal test
        Create RandomLayer instance with several num_qubits arguments.

        Check if
        - random_layer.num_qubits is the same as the num_qubits argument.
        - the return value od random_layer() is instance of qiskit.QuantumCircuit.
        - the number of gates applied to the returned circuit is appropriate.
        - each gate applied to the returned circuit is either one or two qubit gate.
        """
        random_layer = RandomLayer(
            num_qubits=num_qubits,
        )
        assert random_layer.num_qubits == num_qubits

        circuit = random_layer()
        assert isinstance(circuit, qiskit.QuantumCircuit)

        decomposed_data = circuit.decompose().data
        num_gates = len(decomposed_data)
        maximum_num_two_qubit_gates = len(
            list(itertools.combinations(range(num_qubits), 2))
        )
        maximum_num_one_qubit_gate = 2 * num_qubits
        assert num_gates <= maximum_num_one_qubit_gate + maximum_num_two_qubit_gates

        for gate in decomposed_data:
            assert gate.operation.num_qubits in (1, 2)
