import itertools
import random
import string

import pytest
import qiskit

from quantum_machine_learning.layers.entanglement_unitary_layer import (
    EntanglementUnitaryLayer,
)


class TestEntanglementUnitaryLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3, 6, 7])
    def test_init_with_defaults(self, num_state_qubits):
        """Normal test;
        create an instance of EntanglementUnitaryLayer with the default values.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its qubit_applied_pairs is all the combinations of two qubits.
        3. its parameter_prefix is the empty string.
        4. its name is "EntanglementUnitary".
        5. the parameter_prefix is the same as the new parameter_prefix after setting it.
        """
        layer = EntanglementUnitaryLayer(num_state_qubits=num_state_qubits)
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        assert layer.num_state_qubits == num_state_qubits
        # 2. its qubit_applied_pairs is all the combinations of two qubits.
        qubits = list(range(layer.num_state_qubits))
        all_combinations_of_two_qubits = list(itertools.combinations(qubits, 2))
        assert len(layer.qubit_applied_pairs) == len(all_combinations_of_two_qubits)
        assert set(layer.qubit_applied_pairs) == set(all_combinations_of_two_qubits)
        # 3. its parameter_prefix is the empty string.
        assert layer.parameter_prefix == ""
        # 4. its name is "EntanglementUnitary".
        assert layer.name == "EntanglementUnitary"
        # 5. the parameter_prefix is the same as the new parameter_prefix after setting it.
        parameter_prefix = "HEY"
        layer.parameter_prefix = parameter_prefix
        assert layer.parameter_prefix == parameter_prefix

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3, 6, 7])
    def test_init_with_qubit_applied_pairs(self, num_state_qubits):
        """Normal test;
        create an instance of EntanglementUnitaryLayer by giving qubit_applied_pairs.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its qubit_applied_pairs is the same as the given qubit_applied_paris.
        3. its parameter_prefix is the empty string.
        4. its name is "EntanglementUnitary".
        """
        random.seed(901)

        qubits = list(range(num_state_qubits))
        all_combinations_of_two_qubits = list(itertools.combinations(qubits, 2))

        num_trials = (
            100
            if len(all_combinations_of_two_qubits) > 100
            else len(all_combinations_of_two_qubits)
        )
        for _ in range(num_trials):
            k = random.randint(1, num_state_qubits - 1)
            qubit_applied_pairs = random.choices(all_combinations_of_two_qubits, k=k)
            layer = EntanglementUnitaryLayer(
                num_state_qubits=num_state_qubits,
                qubit_applied_pairs=qubit_applied_pairs,
            )
            # 1. its num_state_qubits is the same as the given num_state_qubits.
            assert layer.num_state_qubits == num_state_qubits
            # 2. its qubit_applied_pairs is the same as the given qubit_applied_paris.
            assert layer.qubit_applied_pairs == qubit_applied_pairs
            # 3. its parameter_prefix is the empty string.
            assert layer.parameter_prefix == ""
            # 4. its name is "EntanglementUnitary".
            assert layer.name == "EntanglementUnitary"

    @pytest.mark.layer
    def test_init_with_name(self):
        """Normal test;
        create an instance of EntanglementUnitaryLayer by giving name.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its qubit_applied_pairs is all the combinations of two qubits.
        3. its parameter_prefix is the empty string.
        4. its name is the same as the given name.
        """
        random.seed(901)  # For reproducibility

        chars = string.ascii_letters + string.digits
        num_state_qubits = 2

        num_trials = 100
        for _ in range(num_trials):
            name = "".join(random.choice(chars) for _ in range(64))
            layer = EntanglementUnitaryLayer(
                num_state_qubits=num_state_qubits, name=name
            )
            # 1. its num_state_qubits is the same as the given num_state_qubits.
            assert layer.num_state_qubits == num_state_qubits
            # 2. its qubit_applied_pairs is all the combinations of two qubits.
            qubits = list(range(layer.num_state_qubits))
            all_combinations_of_two_qubits = list(itertools.combinations(qubits, 2))
            assert len(layer.qubit_applied_pairs) == len(all_combinations_of_two_qubits)
            assert set(layer.qubit_applied_pairs) == set(all_combinations_of_two_qubits)
            # 3. its parameter_prefix is the empty string.
            assert layer.parameter_prefix == ""
            # 4. its name is the same as the given name.
            assert layer.name == name

    @pytest.mark.layer
    def test_qubit_applied_pairs(self):
        """Normal test;
        call the setter and getter of qubit_applied_pairs.

        Check if
        1. its qubit_applied_pairs is the same as the given qubit_applied_pairs.
        2. the length of its parameters is double in the size of its qubit_applied_pairs.
        3. its qubit_applied_pairs is all the combinations of two qubits after setting None in qubit_applied_paris.
        4. the length of its parameters is double in the size of its qubit_applied_pairs.
        5. its qubit_applied_pairs is the empty list after setting 1 in num_state_qubits.
        6. its qubit_applied_pairs is the same as the new qubit_applied_pairs after setting a new qubit_applied_pairs.
        7. the length of its parameters is double in the size of its qubit_applied_pairs.
        """
        num_state_qubits = 7
        qubit_applied_pairs = [(0, 1), (0, 6), (1, 3)]
        layer = EntanglementUnitaryLayer(
            num_state_qubits=num_state_qubits, qubit_applied_pairs=qubit_applied_pairs
        )
        # 1. its qubit_applied_pairs is the same as the given qubit_applied_pairs.
        assert layer.qubit_applied_pairs == qubit_applied_pairs
        # 2. the length of its parameters is double in the size of its qubit_applied_pairs.
        assert len(layer.parameters) == 2 * len(layer.qubit_applied_pairs)
        # 3. its qubit_applied_pairs is all the combinations of two qubits after setting None in qubit_applied_paris.
        layer.qubit_applied_pairs = None
        qubits = list(range(layer.num_state_qubits))
        all_combinations_of_two_qubits = list(itertools.combinations(qubits, 2))
        assert len(layer.qubit_applied_pairs) == len(all_combinations_of_two_qubits)
        assert set(layer.qubit_applied_pairs) == set(all_combinations_of_two_qubits)
        # 4. the length of its parameters is double in the size of its qubit_applied_pairs.
        assert len(layer.parameters) == 2 * len(layer.qubit_applied_pairs)
        # 5. its qubit_applied_pairs is the empty list after setting 1 in num_state_qubits.
        layer.num_state_qubits = 1
        assert layer.qubit_applied_pairs == []
        # 6. its qubit_applied_pairs is the same as the new qubit_applied_pairs after setting a new qubit_applied_pairs.
        layer.num_state_qubits = 7
        new_qubit_applied_pairs = [(0, 1), (1, 2), (3, 4), (5, 6)]
        layer.qubit_applied_pairs = new_qubit_applied_pairs
        assert layer.qubit_applied_pairs == new_qubit_applied_pairs
        # 7. the length of its parameters is double in the size of its qubit_applied_pairs.
        assert len(layer.parameters) == 2 * len(layer.qubit_applied_pairs)

    @pytest.mark.layer
    def test_valid_check_configuration(self):
        """Normal test;
        run _build() to see if _check_configuration works with more than one num_state_qubits.

        Check if
        1. no error arises.
        """
        num_state_qubits = 2
        layer = EntanglementUnitaryLayer(num_state_qubits=num_state_qubits)
        # 1. no error arises.
        layer._build()

    @pytest.mark.layer
    def test_invalid_check_configuration(self):
        """Abormal test;
        run _build() to see if _check_configuration works with one num_state_qubits.

        Check if
        1. AttributeError arises.
        """
        num_state_qubits = 1
        layer = EntanglementUnitaryLayer(num_state_qubits=num_state_qubits)
        # 1. AttributeError arises.
        with pytest.raises(AttributeError):
            layer._build()

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3, 6, 7])
    def test_build(self, num_state_qubits):
        """Normal test;
        run _build().

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its qubit_applied_pairs is all the combinations of two qubits.
        3. itself, after performing decompose() method, contains only CRY and CRZ gates such that
            3.1 the number of the gates is twice the size of its qubit_applied_pairs.
            3.2 they are parametrised.
            3.3 each gate is applied to two qubits.
        4. its num_state_qubits is the same as the new num_state_qubits after setting it.
        5. its qubit_applied_pairs is all the combinations of two qubits.
        6. itself, after performing decompose() method, contains only CRY and CRZ gates such that
            6.1 the number of the gates is twice the size of its qubit_applied_pairs
            6.2 they are parametrised.
            6.3 each gate is applied to two qubits.
        7. its qubit_applied_pairs is the same as the new qubit_applied_pairs after setting it.
        8. itself, after performing decompose() method, contains only CRY and CRZ gates such that
            8.1 the number of the gates is twice the size of its qubit_applied_pairs
            8.2 they are parametrised.
            8.3 each gate is applied to two qubits.
        """
        layer = EntanglementUnitaryLayer(num_state_qubits=num_state_qubits)
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        assert layer.num_state_qubits == num_state_qubits
        # 2. its qubit_applied_pairs is all the combinations of two qubits.
        qubits = list(range(layer.num_state_qubits))
        all_combinations_of_two_qubits = list(itertools.combinations(qubits, 2))
        assert len(layer.qubit_applied_pairs) == len(all_combinations_of_two_qubits)
        assert set(layer.qubit_applied_pairs) == set(all_combinations_of_two_qubits)
        # 3. itself, after performing decompose() method, contains only CY and CZ gates such that
        decomposed_x_layer = layer.decompose()
        gates = decomposed_x_layer.data
        assert all(
            gate.operation.name == "cry" or gate.operation.name == "crz"
            for gate in gates
        )
        #     3.1 the number of the gates is twice the size of its qubit_applied_pairs.
        assert len(gates) == 2 * len(layer.qubit_applied_pairs)

        for gate in gates:
            #     3.2 they are parametrised.
            assert (
                len(gate.operation.params) == 1
            )  # == 1: since it's either an CRY or an CRZ gate
            #     3.3 each gate is applied to two qubits.
            assert gate.operation.num_qubits == 2
        # 4. its num_state_qubits is the same as the new num_state_qubits after setting it.
        new_num_state_qubits = num_state_qubits + 1
        layer.num_state_qubits = new_num_state_qubits
        assert layer.num_state_qubits == new_num_state_qubits
        # 5. its qubit_applied_pairs is all the combinations of two qubits.
        qubits = list(range(layer.num_state_qubits))
        all_combinations_of_two_qubits = list(itertools.combinations(qubits, 2))
        assert len(layer.qubit_applied_pairs) == len(all_combinations_of_two_qubits)
        assert set(layer.qubit_applied_pairs) == set(all_combinations_of_two_qubits)
        # 6. itself, after performing decompose() method, contains only CRY and CRZ gates such that
        decomposed_x_layer = layer.decompose()
        gates = decomposed_x_layer.data
        assert all(
            gate.operation.name == "cry" or gate.operation.name == "crz"
            for gate in gates
        )
        #     6.1 the number of the gates is twice the size of its qubit_applied_pairs
        assert len(gates) == 2 * len(layer.qubit_applied_pairs)

        for gate in gates:
            #     6.2 they are parametrised.
            assert (
                len(gate.operation.params) == 1
            )  # == 1: since it's either an CRY or an CRZ gate
            #     6.3 each gate is applied to two qubits.
            assert gate.operation.num_qubits == 2
        # 7. its qubit_applied_pairs is the same as the new qubit_applied_pairs after setting it.
        new_qubit_applied_pairs = [(0, 1)]
        layer.qubit_applied_pairs = new_qubit_applied_pairs
        assert layer.qubit_applied_pairs == new_qubit_applied_pairs
        # 8. itself, after performing decompose() method, contains only CRY and CRZ gates such that
        decomposed_x_layer = layer.decompose()
        gates = decomposed_x_layer.data
        assert all(
            gate.operation.name == "cry" or gate.operation.name == "crz"
            for gate in gates
        )
        #     17.1 the number of the gates is twice the size of its qubit_applied_pairs
        assert len(gates) == 2 * len(layer.qubit_applied_pairs)

        for gate in gates:
            #     17.2 they are parametrised.
            assert (
                len(gate.operation.params) == 1
            )  # == 1: since it's either an CRY or an CRZ gate
            #     17.3 each gate is applied to two qubits.
            assert gate.operation.num_qubits == 2
