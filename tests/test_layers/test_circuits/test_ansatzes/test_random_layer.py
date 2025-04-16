import itertools
import random
import string

import pytest
import qiskit

from quantum_machine_learning.layers.circuits.ansatzes.random_layer import (
    GateInfo,
    SelectOptions,
    RandomLayer,
)


class TestRandomLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [1, 2, 5, 6])
    def test_init_with_defaults(self, num_state_qubits):
        """Normal test:
        Create an instance of RandomLayer with defaults arguments.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its name is "RandomLayer".
        3. its seed is 901.
        4. its available_gates is the same as RandomLayer.DEFAULT_AVAILABEL_GATES.
        5. its threshold is the same as RandomLayer.DEFAULT_THRESHOLD.
        6. its select_options is the same as RandomLayer.DEFAULT_SELECT_OPTIONS(num_state_qubits).
        """
        layer = RandomLayer(num_state_qubits=num_state_qubits)
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        assert layer.num_state_qubits == num_state_qubits
        # 2. its name is "RandomLayer".
        assert layer.name == "RandomLayer"
        # 3. its seed is 901.
        assert layer.seed == 901
        # 4. its available_gates is the same as RandomLayer.DEFAULT_AVAILABEL_GATES.
        assert layer.available_gates == RandomLayer.DEFAULT_AVAILABEL_GATES
        # 5. its threshold is the same as RandomLayer.DEFAULT_THRESHOLD.
        assert layer.threshold == RandomLayer.DEFAULT_THRESHOLD
        # 6. its select_options is the same as RandomLayer.DEFAULT_SELECT_OPTIONS(num_state_qubits).
        assert layer.select_options == RandomLayer.DEFAULT_SELECT_OPTIONS(
            num_state_qubits
        )

    @pytest.mark.layer
    def test_init_with_name(self):
        """Normal test:
        Create an instance of RandomLayer with name.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its name is the same as the given name.
        3. its seed is 901.
        4. its available_gates is the same as RandomLayer.DEFAULT_AVAILABEL_GATES.
        5. its threshold is the same as RandomLayer.DEFAULT_THRESHOLD.
        6. its select_options is the same as RandomLayer.DEFAULT_SELECT_OPTIONS(num_state_qubits).
        """
        random.seed(901)  # For reproducibility

        chars = string.ascii_letters + string.digits
        num_state_qubits = 2

        num_trials = 100
        for _ in range(num_trials):
            name = "".join(random.choice(chars) for _ in range(64))
            layer = RandomLayer(num_state_qubits=num_state_qubits, name=name)
            # 1. its num_state_qubits is the same as the given num_state_qubits.
            assert layer.num_state_qubits == num_state_qubits
            # 2. its name is the same as the given name.
            assert layer.name == name
            # 3. its seed is 901.
            assert layer.seed == 901
            # 4. its available_gates is the same as RandomLayer.DEFAULT_AVAILABEL_GATES.
            assert layer.available_gates == RandomLayer.DEFAULT_AVAILABEL_GATES
            # 5. its threshold is the same as RandomLayer.DEFAULT_THRESHOLD.
            assert layer.threshold == RandomLayer.DEFAULT_THRESHOLD
            # 6. its select_options is the same as RandomLayer.DEFAULT_SELECT_OPTIONS(num_state_qubits).
            assert layer.select_options == RandomLayer.DEFAULT_SELECT_OPTIONS(
                num_state_qubits
            )

    @pytest.mark.layer
    def test_available_gates(self):
        """Normal test:
        Call the setter and getter of available_gates.

        Check if
        1. its available_gates is the same as the given available_gates.
        2. its _is_built is True after calling _build().
        3. its availabel_gates is RandomLayer.DEFAULT_AVAILABEL_GATES after setting None in available_gates.
        4. its _is_built is False.
        5. its _is_built is True after calling _build().
        6. its available_gates is a new available_gates after settings a new one.
        7. its _is_built is False.
        """
        available_gates = [GateInfo(qiskit.circuit.library.CRXGate, 2, 1)]
        layer = RandomLayer(num_state_qubits=2, available_gates=available_gates)
        # 1. its available_gates is the same as the given available_gates.
        assert layer.available_gates == available_gates
        # 2. its _is_built is True after calling _build().
        layer._build()
        assert layer._is_built
        # 3. its availabel_gates is RandomLayer.DEFAULT_AVAILABEL_GATES after setting None in available_gates.
        layer.available_gates = None
        assert layer.available_gates == RandomLayer.DEFAULT_AVAILABEL_GATES
        # 4. its _is_built is False.
        assert not layer._is_built
        # 5. its _is_built is True after calling _build().
        layer._build()
        assert layer._is_built
        # 6. its available_gates is a new available_gates after settings a new one.
        new_available_gates = [
            GateInfo(qiskit.circuit.library.CRYGate, 2, 1),
            GateInfo(qiskit.circuit.library.CRZGate, 2, 1),
        ]
        layer.available_gates = new_available_gates
        assert layer.available_gates == new_available_gates
        # 7. its _is_built is False.
        assert not layer._is_built

    @pytest.mark.layer
    def test_threshold(self):
        """Normal test:
        Call the setter and getter of threshold.

        Check if
        1. its threshold is the same as the given threshold.
        2. its _is_built is True after calling _build().
        3. its threshold is RandomLayer.DEFAULT_THRESHOLD after setting None.
        4. its _is_built is False.
        5. its _is_built is True after calling _build().
        6. its threshold is the same as the new given threshold after setting a new one.
        7. its _is_built is False.
        """
        threshold = 1
        layer = RandomLayer(num_state_qubits=2, threshold=threshold)
        # 1. its threshold is the same as the given threshold.
        assert layer.threshold == threshold
        # 2. its _is_built is True after calling _build().
        layer._build()
        assert layer._is_built
        # 3. its threshold is RandomLayer.DEFAULT_THRESHOLD after setting None.
        layer.threshold = None
        assert layer.threshold == RandomLayer.DEFAULT_THRESHOLD
        # 4. its _is_built is False.
        assert not layer._is_built
        # 5. its _is_built is True after calling _build().
        layer._build()
        assert layer._is_built
        # 6. its threshold is the same as the new given threshold after setting a new one.
        new_threshold = 0.1
        layer.threshold = new_threshold
        assert layer.threshold == new_threshold
        # 7. its _is_built is False.
        assert not layer._is_built

    @pytest.mark.layer
    def test_seed(self):
        """Normal test:
        Call the setter and getter of seed.

        Check if
        1. its seed is the same as the given seed.
        2. its _is_built is True after calling _build().
        3. its seed is None after setting None.
        4. its _is_built is False.
        5. its _is_built is True after calling _build().
        6. its seed is the same as the new given seed after setting a new one.
        7. its _is_built is False.
        """
        seed = 57
        layer = RandomLayer(num_state_qubits=2, seed=seed)
        # 1. its seed is the same as the given seed.
        assert layer.seed == seed
        # 2. its _is_built is True after calling _build().
        layer._build()
        assert layer._is_built
        # 3. its seed is None after setting None.
        layer.seed = None
        assert layer.seed is None
        # 4. its _is_built is False.
        assert not layer._is_built
        # 5. its _is_built is True after calling _build().
        layer._build()
        assert layer._is_built
        # 6. its seed is the same as the new given seed after setting a new one.
        new_seed = 91
        layer.seed = new_seed
        assert layer.seed == new_seed
        # 7. its _is_built is False.
        assert not layer._is_built

    @pytest.mark.layer
    def test_select_options(self):
        """Normal test:
        Call the setter and getter of select_options.

        Check if
        1. its select_options is the same as the given select_options.
        2. its _is_built is True after calling _build().
        3. its select_options is RandomLayer.DEFUALT_SELECT_OPTIONS(num_state_qubits) after setting None.
        4. its _is_built is False.
        5. its _is_built is True after calling _build().
        6. its select_options is the same as the new given select_options after setting a new one.
        7. its _is_built is False.
        """
        select_options = SelectOptions(
            {1: 1, 2: 2, 3: 3, 4: 4},  # a maximum number of gates
            {1: 1, 2: 1, 3: 1, 4: 1},  # a minimum number of gates
        )
        layer = RandomLayer(num_state_qubits=2, select_options=select_options)
        # 1. its select_options is the same as the given select_options.
        assert layer.select_options == select_options
        # 2. its _is_built is True after calling _build().
        layer._build()
        assert layer._is_built
        # 3. its select_options is RandomLayer.DEFUALT_SELECT_OPTIONS(num_state_qubits) after setting None.
        layer.select_options = None
        assert layer.select_options == RandomLayer.DEFAULT_SELECT_OPTIONS(
            layer.num_state_qubits
        )
        # 4. its _is_built is False.
        assert not layer._is_built
        # 5. its _is_built is True after calling _build().
        layer._build()
        assert layer._is_built
        # 6. its select_options is the same as the new given select_options after setting a new one.
        new_select_options = SelectOptions(
            {1: 1},  # a maximum number of gates
            {1: 0},  # a minimum number of gates
        )
        layer.select_options = new_select_options
        assert layer.select_options == new_select_options
        # 7. its _is_built is False.
        assert not layer._is_built

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3, 6, 7])
    def test_connection_probabilities(self, num_state_qubits):
        """Normal test:
        Call the getter of connection_probabilities.

        Check if
        1. its connection_probabilities is an empty dict before calling _build()
        2. the keys of its connection_probabilities is all combinations of each qubit after calling _build().
        3. the values of its connection_probabilities is in between 0 and 1.
        """
        layer = RandomLayer(num_state_qubits=num_state_qubits)
        # 1. its connection_probabilities is an empty dict before calling _build()
        assert layer.connection_probabilities == dict()

        layer._build()
        keys = set(list(layer.connection_probabilities.keys()))
        all_qubits = list(range(layer.num_state_qubits))
        combinations = []
        for n in range(1, layer.num_state_qubits + 1):
            combinations.extend(list(itertools.combinations(all_qubits, n)))
        # 2. the keys of its connection_probabilities is all combinations of each qubit after calling _build().
        assert keys == set(combinations)
        # 3. the values of its connection_probabilities is in between 0 and 1.
        assert all(0 <= value <= 1 for value in layer.connection_probabilities.values())

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3, 6, 7])
    def test_selected_gates(self, num_state_qubits):
        """Normal test:
        Call the getter of selected_gates.

        Check if
        1. its selected_gates is an empty list before calling _build()
        2. gate of each element of its selected_gates is in its availabel_gates after calling _build().
        3. every element of qubits of each element of its selected_gates is valid.
        4. gate of each element of its selected_gates is in the new available_gates after setting a new one and calling _build().
        5. every element of qubits of each element of its selected_gates is valid.
        """
        layer = RandomLayer(num_state_qubits=num_state_qubits)
        # 1. its selected_gates is an empty list before calling _build()
        assert layer.selected_gates == []

        layer._build()
        all_gate_classes = [
            available_gate.gate for available_gate in layer.available_gates
        ]
        for selected_gate in layer.selected_gates:
            # 2. gate of each element of its selected_gates is in its availabel_gates after calling _build().
            assert selected_gate.gate.base_class in all_gate_classes

            for qubit in selected_gate.qubits:
                # 3. every element of qubits of each element of its selected_gates is valid.
                assert qubit <= layer.num_state_qubits - 1

        new_available_gates = [GateInfo(qiskit.circuit.library.CRXGate, 2, 1)]
        layer.available_gates = new_available_gates
        layer._build()
        all_gate_classes = [
            available_gate.gate for available_gate in layer.available_gates
        ]
        for selected_gate in layer.selected_gates:
            # 4. gate of each element of its selected_gates is in the new available_gates after setting a new one and calling _build().
            assert selected_gate.gate.base_class in all_gate_classes

            for qubit in selected_gate.qubits:
                # 5. every element of qubits of each element of its selected_gates is valid.
                assert qubit <= layer.num_state_qubits - 1

    @pytest.mark.layer
    def test_invalid_check_configuration(self):
        """Abnormal test:
        Run _build() to see if _check_configuration works with one num_state_qubits.

        Check if
        1. AttributeError arises when setting invalid available gates.
        2. AttributeError arises when setting invalid select options.
        """
        available_gates = [GateInfo(qiskit.circuit.library.CCXGate, 3, 0)]
        select_options = SelectOptions(
            {1: 1},  # max
            {1: 2},  # min
        )
        layer = RandomLayer(
            num_state_qubits=2,
            available_gates=available_gates,
            select_options=select_options,
        )
        # 1. AttributeError arises when setting invalid available gates.
        with pytest.raises(AttributeError):
            layer._build()

        layer.available_gates = None
        # 2. AttributeError arises when setting invalid select options.
        with pytest.raises(AttributeError):
            layer._build()
