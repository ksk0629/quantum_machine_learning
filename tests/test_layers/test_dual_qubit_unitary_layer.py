import pytest
import qiskit

from quantum_machine_learning.layers.dual_qubit_unitary_layer import DualQubitUnitaryLayer


class TestDualQubitUnitaryLayer:
    @classmethod
    def setup_class(cls):
        cls.num_qubits = 4
        cls.applied_qubit_pairs = [(0, 1), (0, 3)]
        cls.param_prefix = "test_dual_qubit_unitary"
        cls.dual_qubit_unitary_layer = DualQubitUnitaryLayer(
            num_qubits=cls.num_qubits,
            applied_qubit_pairs=cls.applied_qubit_pairs,
            param_prefix=cls.param_prefix,
        )

    def test_init(self):
        """Normal test
        Check if the class that was prepared in setup_class has
        - num_qubits as same as self.num_qubits.
        - applied_qubit_pairs as same as self.applied_qubit_pairs.
        - param_prefix as same as self.param_prefix.
        - num_params as double as the number of sell.applied_qubit_pairs.
        """
        assert self.dual_qubit_unitary_layer.num_qubits == self.num_qubits
        assert (
            self.dual_qubit_unitary_layer.applied_qubit_pairs
            == self.applied_qubit_pairs
        )
        assert self.dual_qubit_unitary_layer.param_prefix == self.param_prefix
        assert (
            self.dual_qubit_unitary_layer.num_params
            == len(self.applied_qubit_pairs) * 2
        )

    @pytest.mark.parametrize("num_qubits", [2, 5, 8])
    @pytest.mark.parametrize(
        "applied_qubit_pairs",
        [[(0, 1)], [(0, 2), (0, 1), (0, 4)], [(0, 1), (0, 2), (0, 3), (0, 7)]],
    )
    def test_get_circuit(self, num_qubits, applied_qubit_pairs):
        """Normal and abnormal test;
        Run get_circuit with verious num_qubits and applied_qubit_pairs defined above.

        Case 1: applied_qubit_pairs has a pair having a larger number than num_qubits
            Check if qiskit.circuit.exceptions.CircuitError happens.
        Case 2: All pairs in applied_qubit_pairs have two numbers being equal to or less than num_qubits.
            Check if
            - the number of parameters equals to double of the number of elements of applied_qubit_pairs.
            - the number of gates equals to double of the number of elements of applied_qubit_pairs.
            - each gate is either ryy or rzz.
            - the number of ryy gates eqauls to the number of elements of applied_qubit_pairs.
        """
        applied_qubits = []
        for applied_qubit_pair in applied_qubit_pairs:
            applied_qubits.append(applied_qubit_pair[0])
            applied_qubits.append(applied_qubit_pair[1])
        maximal_applied_qubit = max(applied_qubits)

        dual_qubit_unitary_layer = DualQubitUnitaryLayer(
            num_qubits=num_qubits,
            applied_qubit_pairs=applied_qubit_pairs,
            param_prefix=self.param_prefix,
        )

        if maximal_applied_qubit > num_qubits - 1:
            with pytest.raises(qiskit.circuit.exceptions.CircuitError):
                dual_qubit_unitary_layer.get_circuit()
        else:
            circuit = dual_qubit_unitary_layer.get_circuit()
            # Check the number of the parameters.
            correct_num_parameters = 2 * len(applied_qubit_pairs)
            assert len(circuit.parameters) == correct_num_parameters
            # Check the number of the gates.
            gate_data = circuit.decompose().data
            correct_num_gates = 2 * len(applied_qubit_pairs)
            assert len(gate_data) == correct_num_gates
            # Check the kinds of the gates.
            gate_names = set([gate.name for gate in gate_data])
            assert gate_names == set(["ryy", "rzz"])
            # Check the number of ryy gates.
            correct_num_ryy_gates = len(applied_qubit_pairs)
            num_ry_gates = len([gate for gate in gate_data if gate.name == "ryy"])
            assert num_ry_gates == correct_num_ryy_gates
