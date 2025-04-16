import pytest

from quantum_machine_learning.layers.circuits.feature_maps.x_angle import XAngle


class TestXAngle:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.encoder
    @pytest.mark.parametrize("data_dimension", [1, 2, 5, 6])
    def test_init_with_defaults(self, data_dimension):
        """Normal test;
        create an instance of XAngle with defaults.

        Check if
        1. its data_dimension is the same as the given data_dimension.
        2. its name is "XAngle"
        3. its num_state_qubits is the same as the given data_dimension.
        """
        feature_map = XAngle(data_dimension=data_dimension)
        # 1. its data_dimension is the same as the given data_dimension.
        assert feature_map.data_dimension == data_dimension
        # 2. its name is "XAngle"
        assert feature_map.name == "XAngle"
        # 3. its num_state_qubits is the same as the given data_dimension.
        assert feature_map.num_state_qubits == data_dimension

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
        feature_map = XAngle(data_dimension=data_dimension)
        applied_qubits = set()  # for 3.3
        # 1. its num_qubits is the same as the given data_dimension.
        assert feature_map.num_qubits == data_dimension
        # 2. its num_parameters is the same as the given data_dimension.
        assert feature_map.num_parameters == data_dimension
        # 3. itself, after performing decompose() method, contains only RX gates such that
        decomposed_x_encoder = feature_map.decompose()
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
        feature_map.data_dimension = new_data_dimension
        applied_qubits = set()  # for 4-3.3
        # 4-1. its num_qubits is the same as the given data_dimension.
        assert feature_map.num_qubits == new_data_dimension
        # 4-2. its num_parameters is the same as the given data_dimension.
        assert feature_map.num_parameters == new_data_dimension
        # 4-3. itself, after performing decompose() method, contains only RX gates such that
        decomposed_x_encoder = feature_map.decompose()
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
