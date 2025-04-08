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
        4. the parameter_prefix is the same as the new parameter_prefix after setting it.
        """
        layer = SingleQubitUnitaryLayer(num_state_qubits=num_state_qubits)
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        assert layer.num_state_qubits == num_state_qubits
        # 2. its name is "SingleQubitUnitary".
        assert layer.name == "SingleQubitUnitary"
        # 3. its qubits_applied is [0, 1, ..., num_state_qubits].
        assert layer.qubits_applied == list(range(num_state_qubits))
        # 4. the parameter_prefix is the same as the new parameter_prefix after setting it.
        parameter_prefix = "HEY"
        layer.parameter_prefix = parameter_prefix
        assert layer.parameter_prefix == parameter_prefix

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
        13. its _is_built is True after calling _build().
        14. the type of its parameters is list.
        15. the lengths of each element of its parameters are the length of its qubit_applied.
        """
        num_state_qubits = 4
        qubits_applied = [0, 2]
        layer = SingleQubitUnitaryLayer(
            num_state_qubits=num_state_qubits, qubits_applied=qubits_applied
        )
        # 1. its qubit_applied is the same as the given qubit_applied.
        assert layer.qubits_applied == qubits_applied
        # 2. its _is_built is False.
        assert not layer._is_built
        # 3. its _is_built is True after calling _build().
        layer._build()
        assert layer._is_built
        # 4. the type of its parameters is list.
        assert isinstance(layer.parameters, list)
        # 5. the lengths of each element of its parameters are the length of its qubit_applied.
        assert len(layer.parameters[0]) == len(qubits_applied)
        assert len(layer.parameters[1]) == len(qubits_applied)
        # 6. its qubit_applied is [0, 1, ..., num_state_qubits] after setting None.
        layer.qubits_applied = None
        assert layer.qubits_applied == list(range(layer.num_state_qubits))
        # 7. its _is_built is False.
        assert not layer._is_built
        # 8. its _is_built is True after calling _build().
        layer._build()
        assert layer._is_built
        # 9. the type of its parameters is list.
        assert isinstance(layer.parameters, list)
        # 10. the lengths of each element of its parameters are the length of its qubit_applied.
        assert len(layer.parameters[0]) == len(layer.qubits_applied)
        assert len(layer.parameters[1]) == len(layer.qubits_applied)
        # 11. its qubit_applied is a new given qubit_applied after setting a new one.
        new_qubits_applied = [1]
        layer.qubits_applied = new_qubits_applied
        assert layer.qubits_applied == new_qubits_applied
        # 12. its _is_built is False.
        assert not layer._is_built
        # 13. its _is_built is True after calling _build().
        layer._build()
        assert layer._is_built
        # 14. the type of its parameters is list.
        assert isinstance(layer.parameters, list)
        # 15. the lengths of each element of its parameters are the length of its qubit_applied.
        assert len(layer.parameters[0]) == len(layer.qubits_applied)
        assert len(layer.parameters[1]) == len(layer.qubits_applied)
