import pytest

from src.encoders.x_encoder import XEncoder


class TestXEncoder:
    @classmethod
    def setup_class(cls):
        cls.num_qubits = 4
        cls.x_encoder = XEncoder(
            num_qubits=cls.num_qubits,
        )

    def test_init(self):
        """Normal test;
        Check if the x_encoder created in setup_class has
        - num_qubits as same as self.num_qubits.
        - num_params as same as self.num_qubits.
        """
        assert self.x_encoder.num_qubits == self.num_qubits
        assert self.x_encoder.num_params == self.num_qubits

    @pytest.mark.parametrize("num_qubits", [2, 4, 5, 10])
    def test_get_circuit(self, num_qubits):
        """Normal test;
        Run get_circuit with various num_qubits.

        Check if
        - the number of qubits of the circuit is the same as num_qubits.
        - the number of paramters of the circuit is the same of num_qubits.
        - each qubit has only one gate, rx.
        """
        x_encoder = XEncoder(num_qubits=num_qubits)
        circuit = x_encoder().decompose()

        # Check the number of qubits.
        assert circuit.num_qubits == num_qubits
        # Check the number of parameters.
        assert len(circuit.parameters) == num_qubits

        for index, instruction in enumerate(circuit.data):
            # Check if the current gate is rx.
            assert instruction.name == "rx"

            # Check if the position of the gate is correct.
            qubit_applied = instruction.qubits[0]._index
            assert qubit_applied == index
