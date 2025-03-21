import pytest

from quantum_machine_learning.encoders.yz_encoder import YZEncoder


class TestYZEncoder:
    @classmethod
    def setup_class(cls):
        cls.num_qubits = 4
        cls.yz_encoder = YZEncoder(
            num_qubits=cls.num_qubits,
        )

    def test_init(self):
        """Normal test;
        Check if the yz_encoder created in setup_class has
        - num_qubits as same as self.num_qubits.
        - num_params as double as self.num_qubits.
        """
        assert self.yz_encoder.num_qubits == self.num_qubits
        assert self.yz_encoder.num_params == self.num_qubits * 2

    @pytest.mark.parametrize("num_qubits", [2, 4, 5, 10])
    def test_get_circuit(self, num_qubits):
        """Normal test;
        Run get_circuit with various num_qubits.

        Check if
        - the number of qubits of the circuit is the same as num_qubits.
        - the number of paramters of the circuit is the double of num_qubits.
        - each qubit has only two gates, ry and rz in this order.
        """
        yz_encoder = YZEncoder(num_qubits=num_qubits)
        circuit = yz_encoder().decompose()

        # Check the number of qubits.
        assert circuit.num_qubits == num_qubits
        # Check the number of parameters.
        assert len(circuit.parameters) == num_qubits * 2

        qubit_index = 0
        gates = []
        for index_inst, instruction in enumerate(circuit.data):
            # Check if the gate is either ry or rz.
            should_ry = True if index_inst % 2 == 0 else False
            if should_ry:
                assert instruction.name == "ry"
            else:
                assert instruction.name == "rz"
            gates.append(instruction.name)

            # Check if the current qubit is correct.
            qubit_applied = instruction.qubits[0]._index
            assert qubit_applied == qubit_index

            # Check if the gates are in the correct order.
            assert len(gates) <= 2
            if len(gates) == 2:
                assert gates == ["ry", "rz"]
                qubit_index += 1
                gates = []
