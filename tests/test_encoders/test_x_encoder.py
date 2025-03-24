import pytest
import qiskit

from quantum_machine_learning.encoders.x_encoder import XEncoder


class TestXEncoder:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.encoder
    @pytest.mark.parametrize("data_dimension", [2, 3])
    def test_init(self, data_dimension):
        """Normal test;
        Create an XEcnoder instance and chenge the data dimension.

        Check if
        - its num_encoding_qubits is the same as the given data_dimension.
        - its num_parameters is the same as the given data_dimension.
        - the type of its parameters is list.
        - the length of its parameters is 1.
        - the type of the first element of its parameters is qiskit.circuit.ParameterVector.
        - the length of the first element of its parameters is the same as the given data_dimension.
        - the above things are preserved with the new_data_dimension after substituting new_data_dimension.
        """
        x_encoder = XEncoder(data_dimension=data_dimension)
        assert x_encoder.num_encoding_qubits == data_dimension
        assert x_encoder.num_parameters == data_dimension
        assert isinstance(x_encoder.parameters, list)
        assert len(x_encoder.parameters) == 1
        assert isinstance(x_encoder.parameters[0], qiskit.circuit.ParameterVector)
        assert len(x_encoder.parameters[0]) == data_dimension

        new_data_dimension = 10
        x_encoder.data_dimension = new_data_dimension
        assert x_encoder.num_encoding_qubits == new_data_dimension
        assert x_encoder.num_parameters == new_data_dimension
        assert isinstance(x_encoder.parameters, list)
        assert len(x_encoder.parameters) == 1
        assert isinstance(x_encoder.parameters[0], qiskit.circuit.ParameterVector)
        assert len(x_encoder.parameters[0]) == new_data_dimension
