import itertools

import pytest
import qiskit

from quantum_machine_learning.layers.random_layer import (
    GateInfo,
    SelectOptions,
    RandomLayer,
)


class TestRandomLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    def test_available_gates(self):
        """Normal test;
        Create an instance of RandomLayer with/without avialble_gates.

        Check if
        - the type of its available_gates is list without giving the available_gates argument.
        - the types of all the elements is GateInfo without giving the avialable_gates argument.
        - its available_gates is the same as the given available_gates.
        - the type of its available_gates is list after substituting None to available_gates of an instance created with some avilable_gates.
        - the types of all the elements is GateInfo after substituting None to available_gates of an instance created with some avilable_gates.
        """
        num_state_qubits = 5
        layer = RandomLayer(num_state_qubits=num_state_qubits)
        assert isinstance(layer.available_gates, list)
        for available_gate in layer.available_gates:
            assert isinstance(available_gate, GateInfo)

        available_gates = [GateInfo(qiskit.circuit.library.XGate, 1, 0)]
        layer = RandomLayer(
            num_state_qubits=num_state_qubits, available_gates=available_gates
        )
        assert layer.available_gates == available_gates

        layer.available_gates = None
        assert isinstance(layer.available_gates, list)
        for available_gate in layer.available_gates:
            assert isinstance(available_gate, GateInfo)

    @pytest.mark.layer
    def test_threshold(self):
        """Normal test;
        Create an instance of RandomLayer with/without threshold.

        Check if
        - its threshold is 0.5 without giving a threshold argument.
        - its threshold is the same as the given threshold.
        """
        num_state_qubits = 5
        layer = RandomLayer(num_state_qubits=num_state_qubits)
        assert layer.threshold == 0.5

        threshold = 0.1
        layer = RandomLayer(num_state_qubits=num_state_qubits, threshold=threshold)
        assert layer.threshold == threshold

    @pytest.mark.layer
    def test_select_options(self):
        """Normal test;
        Create an instance of RandomLayer with/without select_options.

        Check if
        - the type of its select_options is SelectOptions.
        - its select_options is SelectOptions({1: 2 * num_state_qubits**2, 2: 1}, {1: 0, 2: 1}) withtout giving a select_options argument.
        - its select_options is the same as the given select_options.
        """
        num_state_qubits = 5
        layer = RandomLayer(num_state_qubits=num_state_qubits)
        select_options = SelectOptions({1: 2 * num_state_qubits**2, 2: 1}, {1: 0, 2: 1})
        assert layer.select_options == select_options

        select_options = SelectOptions({1: 1, 2: 2}, {1: 1, 2: 2})
        layer = RandomLayer(
            num_state_qubits=num_state_qubits, select_options=select_options
        )
        assert layer.select_options == select_options

    @pytest.mark.layer
    def test_invalid_available_gates(self):
        """Abnormal test;
        Attempt to set a list of non GateInfo class.

        Check if ValueError happens.
        """
        num_state_qubits = 5
        available_gates = [
            [GateInfo(qiskit.circuit.library.XGate, 1, 0)],
            1,  # Invalid
            [GateInfo(qiskit.circuit.library.XGate, 1, 0)],
        ]
        with pytest.raises(ValueError):
            RandomLayer(
                num_state_qubits=num_state_qubits, available_gates=available_gates
            )
