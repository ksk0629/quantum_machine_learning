import pytest
import qiskit

from quantum_machine_learning.layers.single_qubit_unitary_layer import (
    SingleQubitUnitaryLayer,
)


class TestSingleQubitUnitaryLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3])
    def test_normal(self, num_state_qubits):
        """Normal test;
        Create an instance of SingleQubitUnitaryLayer class.

        Check if
        - the type of its y_parameters is qiskit.circuit.ParameterVector.
        - the length of its y_parameters is the same as the given num_state_qubits.
        - the type of its z_parameters is qiskit.circuit.ParameterVector.
        - the length of its z_parameters is the same as the given num_state_qubits.
        - its num_parameters is the double as the given num_state_qubits.
        - the type of its parameters is list.
        - the first element of its parameters is the same as its y_parameters.
        - the second element of its parameters is the same as its z_parameters.
        - the type of itself is qiskit.QuantumCircuit.
        - the above things are still correct with a new num_state_qubits
          after substituting the new num_state_qubits.
        """
        layer = SingleQubitUnitaryLayer(num_state_qubits=num_state_qubits)
        assert isinstance(layer.y_parameters, qiskit.circuit.ParameterVector)
        assert len(layer.y_parameters) == num_state_qubits
        assert isinstance(layer.z_parameters, qiskit.circuit.ParameterVector)
        assert len(layer.z_parameters) == num_state_qubits
        assert layer.num_parameters == 2 * num_state_qubits
        assert isinstance(layer.parameters, list)
        assert layer.parameters[0] == layer.y_parameters
        assert layer.parameters[1] == layer.z_parameters
        assert isinstance(layer, qiskit.QuantumCircuit)

        layer._build()  # For the coverage

        new_num_state_qubits = num_state_qubits + 1
        layer.num_state_qubits = new_num_state_qubits
        assert isinstance(layer.y_parameters, qiskit.circuit.ParameterVector)
        assert len(layer.y_parameters) == new_num_state_qubits
        assert isinstance(layer.z_parameters, qiskit.circuit.ParameterVector)
        assert len(layer.z_parameters) == new_num_state_qubits
        assert layer.num_parameters == 2 * new_num_state_qubits
        assert isinstance(layer.parameters, list)
        assert layer.parameters[0] == layer.y_parameters
        assert layer.parameters[1] == layer.z_parameters
        assert isinstance(layer, qiskit.QuantumCircuit)

    @pytest.mark.layer
    def test_with_qubits_applied(self):
        """Normal test;
        Create an instance of SingleQubitUnitaryLayer class with qubits_applied.

        Check if
        - the length of its decomposed data is the same as the length of the given qubits_applied.
        - the length of its y_parameters is the same as the length of the given qubits_applied.
        - the length of its z_parameters is the same as the length of the given qubits_applied.
        - each qubits in its decomposed data is the same as the each element of the given qubits_applied.
        - the above things are still correct after substitutin a new qubits_applied.
        """
        num_state_qubits = 5
        qubits_applied = [0, 3, 4]
        layer = SingleQubitUnitaryLayer(
            num_state_qubits=num_state_qubits, qubits_applied=qubits_applied
        )
        assert len(layer.y_parameters) == len(qubits_applied)
        assert len(layer.z_parameters) == len(qubits_applied)
        for qubit_index in range(len(qubits_applied)):
            data_index = qubit_index * 2
            data = layer.decompose().data[data_index]
            assert data.qubits[0]._index == qubits_applied[qubit_index]

            data = layer.decompose().data[data_index + 1]
            assert data.qubits[0]._index == qubits_applied[qubit_index]

        new_qubit_applied = [1]
        layer.qubits_applied = new_qubit_applied
        assert len(layer.y_parameters) == len(new_qubit_applied)
        assert len(layer.z_parameters) == len(new_qubit_applied)
        for qubit_index in range(len(new_qubit_applied)):
            data_index = qubit_index * 2
            data = layer.decompose().data[data_index]
            assert data.qubits[0]._index == new_qubit_applied[qubit_index]

            data = layer.decompose().data[data_index + 1]
            assert data.qubits[0]._index == new_qubit_applied[qubit_index]

    @pytest.mark.layer
    def test_with_parameter_prefix(self):
        """Normal test;
        create an instance of SingleQubitUnitaryLayer with specified parameter_prefix.

        Check if
        - its parameter_prefix is the same as the given parameter_prefix.
        - its parameter_prefix is the same as the new given parameter_prefix
          after substituting a new parameter_prefix.
        """
        num_state_qubits = 2
        parameter_prefix = "prefix!"
        layer = SingleQubitUnitaryLayer(
            num_state_qubits=num_state_qubits, parameter_prefix=parameter_prefix
        )
        assert layer.parameter_prefix == parameter_prefix

        new_parameter_prefix = "new parameter_prefix!"
        layer.parameter_prefix = new_parameter_prefix
        assert layer.parameter_prefix == new_parameter_prefix
