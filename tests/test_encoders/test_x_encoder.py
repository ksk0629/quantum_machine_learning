import pytest
import qiskit

from quantum_machine_learning.encoders.x_encoder import XEncoder


class TestXEncoder:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.encoder
    @pytest.mark.parametrize("data_dimension", [1, 2, 5, 6])
    def test_init_with_defaults(self, data_dimension):
        """Normal test;
        create an instance of XEncoder with defaults.

        Check if
        1. its data_dimension is the same as the given data_dimension.
        2. its name is "XEncoder"
        3. its num_encoding_qubits is the same as the given data_dimension.
        """
        x_encoder = XEncoder(data_dimension=data_dimension)
        # 1. its data_dimension is the same as the given data_dimension.
        assert x_encoder.data_dimension == data_dimension
        # 2. its name is "XEncoder"
        assert x_encoder.name == "XEncoder"
        # 3. its num_encoding_qubits is the same as the given data_dimension.
        assert x_encoder.num_encoding_qubits == data_dimension

    @pytest.mark.encoder
    @pytest.mark.parametrize("data_dimension", [1, 2, 5, 6])
    def test_build(self, data_dimension):
        """Normal test;
        run _build method.

        Check if
        1. its num_qubits is the same as the given data_dimension.
        2. its num_parameters is the same as the given data_dimension.
        3. itself, after performing decompose() method, contains only RX gates such that
            3.1 the number of the gates is the same as data_dimension.
            3.2 they are parametrised.
            3.3 each gate is applied to each qubit.
        4. the above things correctly hold after setting a new data_dimension.
        """
        x_encoder = XEncoder(data_dimension=data_dimension)
        applied_qubits = set()  # for 3.3
        # 1. its num_qubits is the same as the given data_dimension.
        assert x_encoder.num_qubits == data_dimension
        # 2. its num_parameters is the same as the given data_dimension.
        assert x_encoder.num_parameters == data_dimension
        # 3. itself, after performing decompose() method, contains only RX gates such that
        decomposed_x_encoder = x_encoder.decompose()
        gates = decomposed_x_encoder.data
        assert all(gate.operation.name == "rx" for gate in gates)
        #     3.1 the number of the gates is the same as data_dimension.
        assert len(gates) == data_dimension
        for gate in gates:
            #     3.2 they are parametrised.
            assert len(gate.operation.params) == 1  # == 1: since it's a RX gate

            applied_qubits.add(gate.qubits[0]._index)  # [0]: since it's a RX gate
        #     3.3 each gate is applied to each qubit.
        assert applied_qubits == set(range(data_dimension))

        # 4. the above things correctly hold after setting a new data_dimension.
        new_data_dimension = data_dimension + 1
        x_encoder.data_dimension = new_data_dimension
        applied_qubits = set()  # for 4-3.3
        # 4-1. its num_qubits is the same as the given data_dimension.
        assert x_encoder.num_qubits == new_data_dimension
        # 4-2. its num_parameters is the same as the given data_dimension.
        assert x_encoder.num_parameters == new_data_dimension
        # 4-3. itself, after performing decompose() method, contains only RX gates such that
        decomposed_x_encoder = x_encoder.decompose()
        gates = decomposed_x_encoder.data
        assert all(gate.operation.name == "rx" for gate in gates)
        #     4-3.1 the number of the gates is the same as data_dimension.
        assert len(gates) == new_data_dimension
        for gate in gates:
            #     4-3.2 they are parametrised.
            assert len(gate.operation.params) == 1  # == 1: since it's a RX gate

            applied_qubits.add(gate.qubits[0]._index)  # [0]: since it's a RX gate
        #     4-3.3 each gate is applied to each qubit.
        assert applied_qubits == set(range(new_data_dimension))
