import pytest
import qiskit

from quantum_machine_learning.encoders.yz_encoder import YZEncoder


class TestYZEncoder:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.encoder
    @pytest.mark.parametrize("data_dimension", [2, 3])
    def test_init(self, data_dimension):
        """Normal test;
        Create an YZEcnoder instance and chenge the data dimension.

        Check if
        - its num_encoding_qubits is
            - the half of the given data_dimension if it is even.
            - the half of the 1 + the given data_dimension if it is odd.
        - its num_parameters is
            - the same as the given data_dimension if it is even.
            - the same as 1 + the given data_dimension if it is odd.
        - the type of its parameters is list.
        - the length of its parameters is 2.
        - the type of the first element of its parameters is qiskit.circuit.ParameterVector.
        - the type of the second element of its parameters is qiskit.circuit.ParameterVector.
        - the length of the first element of its parameters is
            - the half of the given data_dimension if it is even.
            - the half of the 1 + the given data_dimension if it is odd.
        - the above things are preserved with the new_data_dimension after substituting new_data_dimension.
        """
        yz_encoder = YZEncoder(data_dimension=data_dimension)
        assert isinstance(yz_encoder.parameters, list)
        assert len(yz_encoder.parameters) == 2
        assert isinstance(yz_encoder.parameters[0], qiskit.circuit.ParameterVector)
        assert isinstance(yz_encoder.parameters[1], qiskit.circuit.ParameterVector)
        if data_dimension % 2 == 0:  # Even
            assert yz_encoder.num_parameters == data_dimension
            assert yz_encoder.num_encoding_qubits == data_dimension // 2
            assert len(yz_encoder.parameters[0]) == data_dimension // 2
        else:  # Odd
            assert yz_encoder.num_parameters == data_dimension + 1
            assert yz_encoder.num_encoding_qubits == (data_dimension + 1) // 2
            assert len(yz_encoder.parameters[0]) == (data_dimension + 1) // 2

        new_data_dimension = data_dimension + 1
        yz_encoder.data_dimension = new_data_dimension
        assert isinstance(yz_encoder.parameters, list)
        assert len(yz_encoder.parameters) == 2
        assert isinstance(yz_encoder.parameters[0], qiskit.circuit.ParameterVector)
        assert isinstance(yz_encoder.parameters[1], qiskit.circuit.ParameterVector)
        if new_data_dimension % 2 == 0:  # Even
            assert yz_encoder.num_parameters == new_data_dimension
            assert yz_encoder.num_encoding_qubits == new_data_dimension // 2
            assert len(yz_encoder.parameters[0]) == new_data_dimension // 2
        else:  # Odd
            assert yz_encoder.num_parameters == new_data_dimension + 1
            assert yz_encoder.num_encoding_qubits == (new_data_dimension + 1) // 2
            assert len(yz_encoder.parameters[0]) == (new_data_dimension + 1) // 2
