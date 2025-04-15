import pytest

from quantum_machine_learning.encoders.yz_encoder import YZEncoder


class TestYZEncoder:
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
        2. its name is "YZEncoder"
        3. its num_encoding_qubits is data_dimension / 2 if it's even, otherwise (data_dimension + 1) / 2.
        """
        yz_encoder = YZEncoder(data_dimension=data_dimension)
        # 1. its data_dimension is the same as the given data_dimension.
        assert yz_encoder.data_dimension == data_dimension
        # 2. its name is "YZEncoder"
        assert yz_encoder.name == "YZEncoder"
        # 3. its num_encoding_qubits is data_dimension / 2 if it's even, otherwise (data_dimension + 1) / 2.
        if data_dimension % 2 == 0:  # Even
            assert yz_encoder.num_encoding_qubits == int(data_dimension / 2)
        else:  # Odd
            assert yz_encoder.num_encoding_qubits == int((data_dimension + 1) / 2)

    @pytest.mark.encoder
    @pytest.mark.parametrize("data_dimension", [1, 2, 5, 6])
    def test_build(self, data_dimension):
        """Normal test;
        run _build method.

        Check if
        1. its num_qubits is the same as its num_encoding_qubits.
        2. its num_parameters is 2 * num_encoding_qubits.
        3. itself, after performing decompose() method, contains only RY and RZ gates such that
            3.1 the number of the gates is 2 * num_encoding_qubits.
            3.2 they are parametrised.
            3.3 one of each gate is applied to each qubit.
        4. the above things correctly hold after setting a new data_dimension.
        """
        yz_encoder = YZEncoder(data_dimension=data_dimension)
        applied_qubits_ry = set()  # for 3.3
        applied_qubits_rz = set()  # for 3.3
        # 1. its num_qubits is the same as its num_encoding_qubits.
        assert yz_encoder.num_qubits == yz_encoder.num_encoding_qubits
        # 2. its num_parameters is 2 * num_encoding_qubits.
        assert yz_encoder.num_parameters == 2 * yz_encoder.num_encoding_qubits
        # 3. itself, after performing decompose() method, contains only RY and RZ gates such that
        decomposed_yz_encoder = yz_encoder.decompose()
        gates = decomposed_yz_encoder.data
        assert all(gate.operation.name in ["ry", "rz"] for gate in gates)
        #     3.1 the number of the gates is 2 * num_encoding_qubits.
        assert len(gates) == 2 * yz_encoder.num_encoding_qubits
        for gate in gates:
            #     3.2 they are parametrised.
            assert (
                len(gate.operation.params) == 1
            )  # == 1: since it's either an RY or an RZ gate

            if gate.operation.name == "ry":
                applied_qubits_ry.add(
                    gate.qubits[0]._index
                )  # [0]: since it's either an RY or an RZ gate
            else:  # gate must be either an RY or an RZ, thus this is RZ.
                applied_qubits_rz.add(
                    gate.qubits[0]._index
                )  # [0]: since it's either an RY or an RZ gate
        #     3.3 one of each gate is applied to each qubit.
        assert applied_qubits_ry == set(range(yz_encoder.num_encoding_qubits))
        assert applied_qubits_rz == set(range(yz_encoder.num_encoding_qubits))

        # 4. the above things correctly hold after setting a new data_dimension.
        new_data_dimension = data_dimension + 1
        yz_encoder.data_dimension = new_data_dimension
        applied_qubits_ry = set()  # for 4-3.3
        applied_qubits_rz = set()  # for 4-3.3
        # 4-1. its num_qubits is the same as its num_encoding_qubits.
        assert yz_encoder.num_qubits == yz_encoder.num_encoding_qubits
        # 4-2. its num_parameters is 2 * num_encoding_qubits.
        assert yz_encoder.num_parameters == 2 * yz_encoder.num_encoding_qubits
        # 4-3. itself, after performing decompose() method, contains only RY and RZ gates such that
        decomposed_yz_encoder = yz_encoder.decompose()
        gates = decomposed_yz_encoder.data
        assert all(gate.operation.name in ["ry", "rz"] for gate in gates)
        #     4-3.1 the number of the gates is 2 * num_encoding_qubits.
        assert len(gates) == 2 * yz_encoder.num_encoding_qubits
        for gate in gates:
            #     4-3.2 they are parametrised.
            assert (
                len(gate.operation.params) == 1
            )  # == 1: since it's either an RY or an RZ gate

            if gate.operation.name == "ry":
                applied_qubits_ry.add(
                    gate.qubits[0]._index
                )  # [0]: since it's either an RY or an RZ gate
            else:  # gate must be either an RY or an RZ, thus this is RZ.
                applied_qubits_rz.add(
                    gate.qubits[0]._index
                )  # [0]: since it's either an RY or an RZ gate
        #     4-3.3 one of each gate is applied to each qubit.
        assert applied_qubits_ry == set(range(yz_encoder.num_encoding_qubits))
        assert applied_qubits_rz == set(range(yz_encoder.num_encoding_qubits))
