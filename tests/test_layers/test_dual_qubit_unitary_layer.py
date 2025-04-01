import random
import string
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
    @pytest.mark.parametrize("num_state_qubits", [2, 3, 6, 7])
    def test_init_with_defaults(self, num_state_qubits):
        """Normal test;
        create an instance of DualQubitUnitaryLayer with the default values.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its qubit_applied_pairs is all the combinations of two qubits.
        3. its parameter_prefix is the empty string.
        4. its name is "DualQubitUnitaryLayer".
        5. the length of its _parameters is 2.
        6. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        7. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        """
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        # 2. its qubit_applied_pairs is all the combinations of two qubits.
        # 3. its parameter_prefix is the empty string.
        # 4. its name is "DualQubitUnitaryLayer".
        # 5. the length of its _parameters is 2.
        # 6. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        # 7. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3, 6, 7])
    def test_init_with_qubit_applied_pairs(self, num_state_qubits):
        """Normal test;
        create an instance of DualQubitUnitaryLayer by giving qubit_applied_pairs.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its qubit_applied_pairs is the same as the given qubit_applied_paris.
        3. its parameter_prefix is the empty string.
        4. its name is "DualQubitUnitaryLayer".
        5. the length of its _parameters is 2.
        6. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        7. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        """
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        # 2. its qubit_applied_pairs is the same as the given qubit_applied_paris.
        # 3. its parameter_prefix is the empty string.
        # 4. its name is "DualQubitUnitaryLayer".
        # 5. the length of its _parameters is 2.
        # 6. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        # 7. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.

    @pytest.mark.layer
    def test_init_with_name(self, num_state_qubits):
        """Normal test;
        create an instance of DualQubitUnitaryLayer by giving name.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its qubit_applied_pairs is all the combinations of two qubits.
        3. its parameter_prefix is the empty string.
        4. its name is the same as the given name.
        5. the length of its _parameters is 2.
        6. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        7. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        """
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        # 2. its qubit_applied_pairs is all the combinations of two qubits.
        # 3. its parameter_prefix is the empty string.
        # 4. its name is the same as the given name.
        # 5. the length of its _parameters is 2.
        # 6. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        # 7. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.

    @pytest.mark.layer
    def test_qubit_applied_pairs(self):
        """Normal test;
        call the setter and getter of qubit_applied_pairs.

        Check if
        1. its qubit_applied_pairs is the same as the given qubit_applied_pairs.
        2. the length of its _parameters is 2.
        3. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        4. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        5. its qubit_applied_pairs is all the combinations of two qubits after setting None in qubit_applied_paris.
        6. the length of its _parameters is 2.
        7. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        8. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        9. its qubit_applied_pairs is the empty list after setting 1 in num_state_qubits.
        10. the length of its _parameters is 2.
        11. both elements of its _parameters are None.
        12. its qubit_applied_pairs is the same as the new qubit_applied_pairs after setting a new qubit_applied_pairs.
        13. the length of its _parameters is 2.
        14. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        15. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        """
        # 1. its qubit_applied_pairs is the same as the given qubit_applied_pairs.
        # 2. the length of its _parameters is 2.
        # 3. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        # 4. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        # 5. its qubit_applied_pairs is all the combinations of two qubits after setting None in qubit_applied_paris.
        # 6. the length of its _parameters is 2.
        # 7. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        # 8. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        # 9. its qubit_applied_pairs is the empty list after setting 1 in num_state_qubits.
        # 10. the length of its _parameters is 2.
        # 11. both elements of its _parameters are None.
        # 12. its qubit_applied_pairs is the same as the new qubit_applied_pairs after setting a new qubit_applied_pairs.
        # 13. the length of its _parameters is 2.
        # 14. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        # 15. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.

    @pytest.mark.layer
    def test_valid_check_configuration(self):
        """Normal test;
        run _build() to see if _check_configuration works with more than one num_state_qubits.

        Check if
        1. no error arises.
        """
        # 1. no error arises.

    @pytest.mark.layer
    def test_valid_check_configuration(self):
        """Normal test;
        run _build() to see if _check_configuration works with one num_state_qubits.

        Check if
        1. AttributeError arises.
        """
        # 1. AttributeError arises.

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3, 6, 7])
    def test_build(self, num_state_qubits):
        """Normal test;
        run _build().

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its qubit_applied_pairs is all the combinations of two qubits.
        3. the length of its _parameters is 2.
        4. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        5. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        6. itself, after performing decompose() method, contains only RYY and RZZ gates such that
            6.1 the number of the gates is the same as qubit_applied_pairs.
            3.2 they are parametrised.
            3.3 each gate is applied to two qubits.
        7. its num_state_qubits is the same as the new num_state_qubits after setting it.
        8. its qubit_applied_pairs is all the combinations of two qubits.
        9. the length of its _parameters is 2.
        10. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        11. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        12. itself, after performing decompose() method, contains only RYY and RZZ gates such that
            12.1 the number of the gates is the same as qubit_applied_pairs.
            12.2 they are parametrised.
            12.3 each gate is applied to two qubits.
        13. its qubit_applied_pairs is the same as the new qubit_applied_pairs after setting it.
        14. the length of its _parameters is 2.
        15. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        16. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        17. itself, after performing decompose() method, contains only RYY and RZZ gates such that
            17.1 the number of the gates is the same as qubit_applied_pairs.
            17.2 they are parametrised.
            17.3 each gate is applied to two qubits.
        """
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        # 2. its qubit_applied_pairs is all the combinations of two qubits.
        # 3. the length of its _parameters is 2.
        # 4. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        # 5. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        # 6. itself, after performing decompose() method, contains only RYY and RZZ gates such that
        #     6.1 the number of the gates is the same as qubit_applied_pairs.
        #     3.2 they are parametrised.
        #     3.3 each gate is applied to two qubits.
        # 7. its num_state_qubits is the same as the new num_state_qubits after setting it.
        # 8. its qubit_applied_pairs is all the combinations of two qubits.
        # 9. the length of its _parameters is 2.
        # 10. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        # 11. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        # 12. itself, after performing decompose() method, contains only RYY and RZZ gates such that
        #     12.1 the number of the gates is the same as qubit_applied_pairs.
        #     12.2 they are parametrised.
        #     12.3 each gate is applied to two qubits.
        # 13. its qubit_applied_pairs is the same as the new qubit_applied_pairs after setting it.
        # 14. the length of its _parameters is 2.
        # 15. the types of both elements of its _parameters are qiskit.circuit.ParameterVector.
        # 16. the lengths of both elements of its _parameters are the same as one of its qubit_applied_pairs.
        # 17. itself, after performing decompose() method, contains only RYY and RZZ gates such that
        #     17.1 the number of the gates is the same as qubit_applied_pairs.
        #     17.2 they are parametrised.
        #     17.3 each gate is applied to two qubits.
