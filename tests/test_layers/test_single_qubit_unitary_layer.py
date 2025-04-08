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
    @pytest.mark.parametrize("num_state_qubits", [1, 2, 5, 6])
    def test_init_with_defaults(self, num_state_qubits):
        """Normal test:
        Create an instance of SingleQubitUnitaryLayer.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its name is "SingleQubitUnitary".
        3. its qubits_applied is [0, 1, ..., num_state_qubits].
        """

    @pytest.mark.layer
    def test_qubits_applied(self):
        """Normal test:
        Call the setter and getter of qubits_applied.

        Check if
        1. its qubit_applied is the same as the given qubit_applied.
        2. its _is_built is False.
        3. its _is_built is True after calling _build().
        4. the type of its parameters is list.
        5. the lengths of each element of its parameters are the length of its qubit_applied.
        6. its qubit_applied is [0, 1, ..., num_state_qubits] after setting None.
        7. its _is_built is False.
        8. its _is_built is True after calling _build().
        9. the type of its parameters is list.
        10. the lengths of each element of its parameters are the length of its qubit_applied.
        11. its qubit_applied is a new given qubit_applied after setting a new one.
        12. its _is_built is False.
        13. the type of its parameters is list.
        14. the lengths of each element of its parameters are the length of its qubit_applied.
        """
