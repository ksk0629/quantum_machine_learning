import pytest
import qiskit

from src.layers.single_qubit_unitary_layer import SingleQubitUnitaryLayer


class TestSingleQubitUnitaryLayer:
    @classmethod
    def setup_class(cls):
        cls.param_prefix = "test_single_qubit_unitary"
        cls.single_qubit_unitary_layer = SingleQubitUnitaryLayer(
            param_prefix=cls.param_prefix
        )

    def test_init(self):
        """Normal test
        Check if the class that was prepared in setup_class has param_prefix as same as self.param_prefix.
        """
        assert self.single_qubit_unitary_layer.param_prefix == self.param_prefix

    @pytest.mark.parametrize("num_qubits", [2, 5, 8])
    @pytest.mark.parametrize("applied_qubits", [[0, 1], [1], [0, 2, 4], [1, 3, 5, 7]])
    def test_get_circuit(self, num_qubits, applied_qubits):
        """Normal and abnormal test;
        Run get_circuit with verious num_qubits and applied_qubits defined above.

        Case 1: applied_qubits has a larger number than num_qubits
            Check if qiskit.circuit.exceptions.CircuitError happens.
        Case 2: All elements in applied_qubits is equal to or less than num_qubits.
            Check if
            - the number of parameters equals to double of the number of elements of applied_qubits.
            - the number of gates equals to double of the number of elements of applied_qubits.
            - each gate is either ry or ry.
            - the number of ry gates eqauls to the number of elements of applied_qubits.
        """
        maximal_applied_qubit = max(applied_qubits)

        if maximal_applied_qubit > num_qubits - 1:
            with pytest.raises(qiskit.circuit.exceptions.CircuitError):
                self.single_qubit_unitary_layer.get_circuit(
                    num_qubits=num_qubits, applied_qubits=applied_qubits
                )
        else:
            circuit = self.single_qubit_unitary_layer.get_circuit(
                num_qubits=num_qubits, applied_qubits=applied_qubits
            )
            # Check the number of the parameters.
            correct_num_parameters = 2 * len(applied_qubits)
            assert len(circuit.parameters) == correct_num_parameters
            # Check the number of the gates.
            gate_data = circuit.decompose().data
            correct_num_gates = 2 * len(applied_qubits)
            assert len(gate_data) == correct_num_gates
            # Check the kinds of the gates.
            gate_names = set([gate.name for gate in gate_data])
            assert gate_names == set(["ry", "rz"])
            # Check the number of ry gates.
            correct_num_ry_gates = len(applied_qubits)
            num_ry_gates = len([gate for gate in gate_data if gate.name == "ry"])
            assert num_ry_gates == correct_num_ry_gates
