import dataclasses
import itertools
import random
from typing import Callable, Final

import numpy as np
import qiskit
import qiskit.circuit

from quantum_machine_learning.layers.base_layer import BaseLayer
from quantum_machine_learning.gate.s_swap_gate import SSwapGate
from quantum_machine_learning.utils.utils import Utils


@dataclasses.dataclass
class GateInfo:
    """Gate infomration data class"""

    gate: qiskit.circuit.Gate
    num_qubits: int
    num_parameters: int


@dataclasses.dataclass
class SelectedGate:
    """Selected gate data class"""

    gate: qiskit.circuit.Gate
    qubits: tuple[int, ...]


@dataclasses.dataclass
class SelectOptions:
    max_num_gates: dict[
        int, int
    ]  # The key is the number of target qubits and the value is the maximum number of gates.
    min_num_gates: dict[
        int, int
    ]  # The key is the number of target qubits and the value is the minimum number of gates.


class RandomLayer(BaseLayer):
    """RandomLayer class. This is introduced as the quanvolutional filter
    in https://arxiv.org/pdf/1904.04767.
    """

    DEFAULT_AVAILABEL_GATES: Final[list[qiskit.circuit.Gate]] = [
        GateInfo(qiskit.circuit.library.RXGate, 1, 1),
        GateInfo(qiskit.circuit.library.RYGate, 1, 1),
        GateInfo(qiskit.circuit.library.RZGate, 1, 1),
        GateInfo(qiskit.circuit.library.PhaseGate, 1, 1),
        GateInfo(qiskit.circuit.library.UGate, 1, 3),
        GateInfo(qiskit.circuit.library.TGate, 1, 0),
        GateInfo(qiskit.circuit.library.HGate, 1, 0),
        GateInfo(qiskit.circuit.library.CUGate, 2, 4),
        GateInfo(SSwapGate, 2, 0),
        GateInfo(qiskit.circuit.library.CXGate, 2, 0),
        GateInfo(qiskit.circuit.library.SwapGate, 2, 0),
    ]
    DEFAULT_THRESHOLD: Final[float] = 0.5
    DEFAULT_SELECT_OPTIONS: Final[Callable[[int], SelectOptions]] = (
        lambda num_state_qubits: SelectOptions(
            {1: 2 * num_state_qubits**2, 2: 1},  # a maximum number of gates
            {1: 0, 2: 1},  # a minimum number of gates
        )
    )

    def __init__(
        self,
        num_state_qubits: int,
        available_gates: list[GateInfo] | None = None,
        threshold: float | None = None,
        select_options: SelectOptions | None = None,
        seed: int = 901,
        name: str | None = "RandomLayer",
    ):
        """Initialise this layer.

        :param int num_state_qubits: the number of state qubits
        :param list[GateInfo] | None available_gates: a set of gates, defaults to None
        :param float | None threshold: a threshold of connection probability, defaults to None
        :param SelectOptions | None select_options: a select options for gates, None
        :param int | None seed: a random seed, defaults to 901
        :param str | None name: the name of this encoder, defaults to "RandomLayer"
        """
        self._selected_gates: list[SelectedGate] | None = None
        self._connection_probabilities: dict[list[int], float] | None = None
        self._available_gates: list[GateInfo] | None = None
        self._threshold: float | None = None
        self._select_options: SelectOptions | None = None
        self._seed = None

        super().__init__(num_state_qubits=num_state_qubits, name=name)

        self.available_gates = available_gates
        self.threshold = threshold
        self.select_options = select_options
        self.seed = seed

    @property
    def available_gates(self) -> list[GateInfo]:
        """Return available gates.
        If it is None, then return the standard availalbe gates defined here.

        :return list[GateInfo]: the available gates
        """
        if self._available_gates is not None:
            return self._available_gates
        else:
            return RandomLayer.DEFAULT_AVAILABEL_GATES

    @available_gates.setter
    def available_gates(self, available_gates: list[GateInfo] | None) -> None:
        """Set the new available gates if valid and reset the register.

        :param list[GateInfo] available_gates: a new available gates
        """
        self._available_gates = available_gates
        self._reset_register()

    @property
    def threshold(self) -> float:
        """Return the threshold of connection probability.

        :return float: the threshold
        """
        if self._threshold is None:
            return RandomLayer.DEFAULT_THRESHOLD
        else:
            return self._threshold

    @threshold.setter
    def threshold(self, threshold: float | None) -> None:
        """Set the new threshold and reset the register.

        :param float | None threshold: a new threshold
        """
        self._threshold = threshold
        self._reset_register()

    @property
    def seed(self) -> int | None:
        """Return the random seed.

        :return int | None: the random seed
        """
        return self._seed

    @seed.setter
    def seed(self, seed: int | None) -> None:
        self._seed = seed
        self._reset_register()

    @property
    def select_options(self) -> SelectOptions:
        """Return the select options for gates of each number of qubits.
        If it is None, then return its standard.

        :return SelectOptions: the select options for gates
        """
        if self._select_options is not None:
            return self._select_options
        else:

            return RandomLayer.DEFAULT_SELECT_OPTIONS(self.num_state_qubits)

    @select_options.setter
    def select_options(self, select_options: SelectOptions | None) -> None:
        """Set the new select options for gates of each number of qubits and reset the register.

        :param SelectOptions select_options: a new select options of gates of each number of qubits
        """
        self._select_options = select_options
        self._reset_register()

    @property
    def connection_probabilities(self) -> dict[list[int], float]:
        """Return the connection probabilities between qubits.

        :return dict[list[int], float]: the connection probabilities
        """
        if self._connection_probabilities is None:
            return dict()
        else:
            return self._connection_probabilities

    def _set_connection_probabilieis(self):
        """Randomly assign connection probabilities between each qubit.

        :return dict[tuple[int, int], float]: the connection probabilities whose key is a pair of qubits and value is the connection probability
        """
        # Initialise the connection probabilities.
        connection_probabilities = dict()
        all_qubits = list(range(self.num_state_qubits))  # type: ignore
        for n in range(1, self.num_state_qubits + 1):
            for conbination in itertools.combinations(all_qubits, n):
                # Set the connection probability.
                connection_probabilities[tuple(conbination)] = np.random.rand()

        self._connection_probabilities = connection_probabilities

    @property
    def selected_gates(self) -> list[SelectedGate]:
        """Return the selected gates.

        :return list[SelectedGate]: the selected gate
        """
        if self._selected_gates is None:
            return []
        else:
            return self._selected_gates

    def _set_gates_based_on_probabilities(self, target_num_qubits: int) -> None:
        """Select gates based on probabilities based on the connection probabilities.

        :param int target_num_qubits: the target number of qubits
        :param float threshold: threshould whether or not a gate is applied to the qubits, defaults to 0.5
        """
        # Get candidate gates whose number of target qubits is the same as the given target_num_qubits.
        target_available_gates = [
            avilable_gate
            for avilable_gate in self.available_gates
            if avilable_gate.num_qubits == target_num_qubits
        ]  # type: ignore
        # End if there is no target_available_gates.
        if len(target_available_gates) == 0:
            return

        for (
            target_qubits,
            connection_probability,
        ) in self.connection_probabilities.items():
            if len(target_qubits) != 1 and connection_probability <= self.threshold:
                # Skip the pair.
                continue
            if len(target_qubits) != target_num_qubits:
                # Skip the pair.
                continue

            # Randomly select a gate.
            selected_gate_info = random.choice(target_available_gates)

            # Set the parameter if needed.
            if selected_gate_info.num_parameters > 0:
                params = np.random.rand(selected_gate_info.num_parameters) * 2 * np.pi
                selected_gate = selected_gate_info.gate(*params)
            else:
                selected_gate = selected_gate_info.gate()

            # Shuffle the qubits.
            shuffled_target_qubits = [
                *target_qubits
            ]  # Need to cast to list as the key is tuple.
            random.shuffle(shuffled_target_qubits)

            # Add the gate into the selected gates.
            self._selected_gates.append(
                SelectedGate(selected_gate, shuffled_target_qubits)
            )

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :raises AttributeError: if there is a gate requiring a larget number qubits than its num_state_qubits
        :raises AttributeError: if its select_options is invalid.
        :return bool: if the configuration is valid
        """

        valid = super()._check_configuration(raise_on_failure=raise_on_failure)
        if not all(
            1 <= available_gate.num_qubits <= self.num_state_qubits
            for available_gate in self.available_gates
        ):
            valid = False
            if raise_on_failure:
                error_msg = f"available_gates must be a set of gates being able to apply to this circuit having {self.num_state_qubits} qubits. However, some gates require more qubits."
                raise AttributeError(error_msg)

        num_qubits_for_max = set(list(self.select_options.max_num_gates.keys()))
        num_qubits_for_min = set(list(self.select_options.min_num_gates.keys()))
        num_qubits_list = list(num_qubits_for_max & num_qubits_for_min)  # intersection
        for num_qubits in num_qubits_list:
            max_num = self.select_options.max_num_gates[num_qubits]
            min_num = self.select_options.min_num_gates[num_qubits]
            if max_num < min_num:
                valid = False
                if raise_on_failure:
                    error_msg = f"For {num_qubits} qubits, the maximum number of gates vs the minimum number of gates = {max_num} vs {min_num}, which is invalid."
                    raise AttributeError(error_msg)

        return valid

    def _reset_register(self) -> None:
        """Reset the register."""
        qreg = qiskit.QuantumRegister(self.num_state_qubits)
        self.qregs = [qreg]

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Fix the seed.
        if self.seed is not None:
            Utils.fix_seed(self.seed)

        # Set connection probabilities between qubits.
        self._set_connection_probabilieis()

        # Get gates randomly in available gates.
        self._selected_gates = []
        for n in range(1, self.num_state_qubits + 1):  # type: ignore
            # Get a maximum number of n-qubit gates.
            if n in self.select_options.max_num_gates:
                max_num = self.select_options.max_num_gates[n]
            else:
                max_num = 1

            # Get a minimum number of n-qubit gates.
            if n in self.select_options.min_num_gates:
                min_num = self.select_options.min_num_gates[n]
            else:
                min_num = 1

            # Determine the number of n-qubit gates.
            if max_num == min_num:
                num_gates = max_num
            else:
                num_gates = np.random.randint(low=min_num, high=max_num)

            for _ in range(num_gates):
                self._set_gates_based_on_probabilities(target_num_qubits=n)

        # Shuffle the order of the gates.
        random.shuffle(self._selected_gates)

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs, name=self.name)

        # Apply the gates to the circuit.
        for selected_gate in self._selected_gates:  # type: ignore
            circuit.append(selected_gate.gate, selected_gate.qubits)

        self.append(circuit.to_gate(), self.qubits)
