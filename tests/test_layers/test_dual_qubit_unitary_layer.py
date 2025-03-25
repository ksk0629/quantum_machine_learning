import math

import pytest
import qiskit

from quantum_machine_learning.layers.dual_qubit_unitary_layer import (
    DualQubitUnitaryLayer,
)


class TestDualQubitUnitaryLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3])
    def test_normal(self, num_state_qubits):
        """Normal test;
        Create an instance of DualQubitUnitaryLayer class.

        Check if
        - the type of its yy_parameters is qiskit.circuit.ParameterVector.
        - the length of its yy_parameters is the same as the number of combinations of choosing two qubits in num_state_qubits qubits.
        - the type of its zz_parameters is qiskit.circuit.ParameterVector.
        - the length of its zz_parameters is the same as the number of combinations of choosing two qubits in num_state_qubits qubits.
        - its num_parameters is the double as the number of combinations of choosing two qubits in num_state_qubits qubits.
        - the type of its parameters is list.
        - the first element of its parameters is the same as its yy_parameters.
        - the second element of its parameters is the same as its zz_parameters.
        - the type of itself is qiskit.QuantumCircuit.
        - the above things are still correct with a new num_state_qubits
          after substituting the new num_state_qubits.
        """
        layer = DualQubitUnitaryLayer(num_state_qubits=num_state_qubits)
        parameter_length = math.comb(num_state_qubits, 2)
        assert isinstance(layer.yy_parameters, qiskit.circuit.ParameterVector)
        assert len(layer.yy_parameters) == parameter_length
        assert isinstance(layer.zz_parameters, qiskit.circuit.ParameterVector)
        assert len(layer.zz_parameters) == parameter_length
        assert layer.num_parameters == 2 * parameter_length
        assert isinstance(layer.parameters, list)
        assert layer.parameters[0] == layer.yy_parameters
        assert layer.parameters[1] == layer.zz_parameters
        assert isinstance(layer, qiskit.QuantumCircuit)

        layer._build()  # For the coverage

        new_num_state_qubits = num_state_qubits + 1
        new_parameter_length = math.comb(new_num_state_qubits, 2)
        layer.num_state_qubits = new_num_state_qubits
        assert isinstance(layer.yy_parameters, qiskit.circuit.ParameterVector)
        assert len(layer.yy_parameters) == new_parameter_length
        assert isinstance(layer.zz_parameters, qiskit.circuit.ParameterVector)
        assert len(layer.zz_parameters) == new_parameter_length
        assert layer.num_parameters == 2 * new_parameter_length
        assert isinstance(layer.parameters, list)
        assert layer.parameters[0] == layer.yy_parameters
        assert layer.parameters[1] == layer.zz_parameters
        assert isinstance(layer, qiskit.QuantumCircuit)

    @pytest.mark.layer
    def test_with_one_num_state_qubits(self):
        """Normal test;
        Create an instance of DualQubitUnitaryLayer class with 1 state qubit.

        Check if
        - AttributeError happens when to_gate() runs.
        - AttributeError happens when to_gate() runs after substituting a new odd number of state qubits.
        """
        num_state_qubits = 1
        layer = DualQubitUnitaryLayer(num_state_qubits=num_state_qubits)
        with pytest.raises(AttributeError):
            layer.to_gate()

        num_state_qubits = 2
        layer = DualQubitUnitaryLayer(num_state_qubits=num_state_qubits)
        layer.num_state_qubits = 1
        with pytest.raises(AttributeError):
            layer.to_gate()

    @pytest.mark.layer
    def test_with_qubit_applied_pairs(self):
        """Normal test;
        Create an instance of DualQubitUnitaryLayer class with qubit_applied_pairs.

        Check if
        - the length of its decomposed data is the same as the length of the given qubit_applied_pairs.
        - the length of its yy_parameters is the same as the length of the given qubit_applied_pairs.
        - the length of its zz_parameters is the same as the length of the given qubit_applied_pairs.
        - each qubits in its decomposed data is the same as the each element of the given qubit_applied_pairs.
        - the above things are still correct after substituting a new qubit_applied_pairs.
        """
        num_state_qubits = 5
        qubit_applied_pairs = [(0, 2), (0, 4)]
        layer = DualQubitUnitaryLayer(
            num_state_qubits=num_state_qubits, qubit_applied_pairs=qubit_applied_pairs
        )
        assert len(layer.yy_parameters) == len(qubit_applied_pairs)
        assert len(layer.zz_parameters) == len(qubit_applied_pairs)
        for pair_index in range(len(qubit_applied_pairs)):
            index = pair_index * 2
            data = layer.decompose().data[index]
            assert data.qubits[0]._index == qubit_applied_pairs[pair_index][0]
            assert data.qubits[1]._index == qubit_applied_pairs[pair_index][1]

            data = layer.decompose().data[index + 1]
            assert data.qubits[0]._index == qubit_applied_pairs[pair_index][0]
            assert data.qubits[1]._index == qubit_applied_pairs[pair_index][1]

        new_qubit_applied_pairs = [(1, 2)]
        layer.qubit_applied_pairs = new_qubit_applied_pairs
        assert len(layer.yy_parameters) == len(new_qubit_applied_pairs)
        assert len(layer.zz_parameters) == len(new_qubit_applied_pairs)
        for pair_index in range(len(new_qubit_applied_pairs)):
            index = pair_index * 2
            data = layer.decompose().data[index]
            assert data.qubits[0]._index == new_qubit_applied_pairs[pair_index][0]
            assert data.qubits[1]._index == new_qubit_applied_pairs[pair_index][1]

            data = layer.decompose().data[index + 1]
            assert data.qubits[0]._index == new_qubit_applied_pairs[pair_index][0]
            assert data.qubits[1]._index == new_qubit_applied_pairs[pair_index][1]
