import random

import numpy as np
import qiskit
import qiskit.circuit

from src.layers.base_layer import BaseLayer
from src.s_swap_gate import SSwapGate


class RandomLayer(BaseLayer):
    """RandomLayer class. This is introduced as the quanvolutional filter
    in https://arxiv.org/pdf/1904.04767.
    """

    def __init__(
        self,
        num_qubits: int,
        available_gates: list[tuple[qiskit.circuit.Gate, int, int]] | None = None,
    ):
        """Initialise this layer.

        :param int num_qubits: _description_
        :param list[tuple[qiskit.circuit.Gate, int]] | None available_gates: set of tuple of available gates, number of qubits, number of parameters, that may be used in this layer, defaults to None
        """
        self.num_qubits = num_qubits

        if available_gates is not None:
            self.available_gates = available_gates
        else:
            self.available_gates = [
                (qiskit.circuit.library.RXGate, 1, 1),
                (qiskit.circuit.library.RYGate, 1, 1),
                (qiskit.circuit.library.RZGate, 1, 1),
                (qiskit.circuit.library.PhaseGate, 1, 1),
                (qiskit.circuit.library.UGate, 1, 3),
                (qiskit.circuit.library.TGate, 1, 0),
                (qiskit.circuit.library.HGate, 1, 0),
                (qiskit.circuit.library.CUGate, 2, 4),
                (SSwapGate, 2, 0),
                (qiskit.circuit.library.CXGate, 2, 0),
                (qiskit.circuit.library.SwapGate, 2, 0),
            ]

    def get_circuit(self) -> qiskit.QuantumCircuit:
        """Get the random layer.

        :return qiskit.QuantumCircuit: random layer
        """
        # Get connection probabilities between each qubit.
        connection_probabilities = self.__get_connection_probabilieis()

        # Get the gates randomly.
        selected_gates = []
        selected_gates += self.__select_gates_based_on_probabilities(
            target_num_qubits=2, connection_probabilities=connection_probabilities
        )
        num_gates = np.random.randint(low=0, high=2 * self.num_qubits)
        selected_gates += self.__select_gates(target_num_qubits=1, num_gates=num_gates)

        # Shuffle the order of the gates.
        random.shuffle(selected_gates)

        # Apply the gates to the circuit.
        circuit = qiskit.QuantumCircuit(self.num_qubits, name="Random Layer")
        for gate, qubits in selected_gates:
            circuit.append(gate, qubits)

        circuit_inst = circuit.to_instruction()
        circuit = qiskit.QuantumCircuit(self.num_qubits)
        circuit.append(circuit_inst, list(range(self.num_qubits)))

        return circuit

    def __get_connection_probabilieis(self):
        """Randomly assign connection probabilities between each qubit."""
        connection_probabilities = dict()
        for index in range(self.num_qubits):
            for next_index in range(index + 1, self.num_qubits):
                # Set the connection probability.
                connection_probabilities[(index, next_index)] = np.random.rand()
        return connection_probabilities

    def __select_gates_based_on_probabilities(
        self,
        target_num_qubits: int,
        connection_probabilities: dict[list[int], float],
        threshold: float = 0.5,
    ) -> list[tuple[qiskit.circuit.Instruction, list[int]]]:
        """Select gates based on probabilities.

        :param int target_num_qubits: target number of qubits
        :param dict[list[int], float] connection_probabilities: connection probabilities between qubits, the first element is the list of qubits and the other is the corresponding probability
        :param float threshold: threshould whether or not a gate is applied to the qubits, defaults to 0.5
        :return list[tuple[qiskit.circuit.Instruction, list[int]]]: selected gates, the first element is the selected gate and the other is the qubits to which the gate is applied
        """
        target_available_gates = [
            (gate, num_qubits, num_params)
            for (gate, num_qubits, num_params) in self.available_gates
            if num_qubits == target_num_qubits
        ]

        selected_gates = []
        for target_qubits, connection_probability in connection_probabilities.items():
            if connection_probability <= threshold:
                # Skip the pair.
                continue

            # Select a gate.
            selected_gate = self.__select_gate(availabel_gates=target_available_gates)

            # Shuffle the qubits.
            shuffled_target_qubits = [*target_qubits]
            random.shuffle(
                shuffled_target_qubits
            )  # key is tuple. Need to cast to list.

            selected_gates.append((selected_gate, shuffled_target_qubits))

        return selected_gates

    def __select_gates(
        self, target_num_qubits: int, num_gates: int
    ) -> list[tuple[qiskit.circuit.Instruction, list[int]]]:
        target_available_gates = [
            (gate, num_qubits, num_params)
            for (gate, num_qubits, num_params) in self.available_gates
            if num_qubits == target_num_qubits
        ]
        available_qubit_sequence = range(self.num_qubits)

        selected_gates = []
        for _ in range(num_gates):
            # Select a gate.
            selected_gate = self.__select_gate(availabel_gates=target_available_gates)

            # Get the target qubits.
            target_qubits = np.random.choice(
                available_qubit_sequence, size=target_num_qubits, replace=False
            ).tolist()

            # Keep the selected gate.
            selected_gates.append((selected_gate, target_qubits))
        return selected_gates

    def __select_gate(
        self,
        availabel_gates: list[tuple[qiskit.circuit.Gate, int, int]],
        max: float = 2 * np.pi,
    ) -> qiskit.circuit.Instruction:
        # Select a two-qubit gate.
        (selected_gate_class, _, num_params) = random.choice(availabel_gates)

        # Set random parameters to the gate.
        if num_params > 0:
            params = np.random.rand(num_params) * max
            selected_gate = selected_gate_class(*params)
        else:
            selected_gate = selected_gate_class()

        return selected_gate
