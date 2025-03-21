import qiskit

from quantum_machine_learning.layers.base_learnable_layer import BaseLearnableLayer


class EntanglementUnitaryLayer(BaseLearnableLayer):
    """EntanglementUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(
        self,
        num_qubits: int,
        applied_qubit_pairs: list[tuple[int, int]],
        param_prefix: str,
    ):
        """Initialise this layer.

        :param int num_qubits: number of qubits
        :param list[int] applied_qubit_pairs: list of qubit pairs, (control qubit, target qubit), to which entanglement unitary is applied
        :param str param_prefix: parameter prefix
        """
        self.num_qubits = num_qubits
        self.applied_qubit_pairs = applied_qubit_pairs
        super().__init__(param_prefix=param_prefix)

    @property
    def num_params(self) -> int:
        """Get the number of paramters.

        :return int: number of parameters
        """
        return len(self.applied_qubit_pairs) * 2

    def __get_pattern(
        self, params: qiskit.circuit.ParameterVector
    ) -> qiskit.QuantumCircuit:
        """Return the entanglement unitary layer pattern.

        :param qiskit.circuit.ParameterVector params: parameter vector
        :return qiskit.QuantumCircuit: entanglement unitary layer pattern
        """
        pattern = qiskit.QuantumCircuit(2)
        pattern.cry(params[0], 0, 1)
        pattern.crz(params[1], 0, 1)

        return pattern

    def get_circuit(
        self,
    ) -> qiskit.QuantumCircuit:
        """Get the entanglement unitary layer circuit.

        :return qiskit.QuantumCircuit: entanglement unitary layer circuit
        """
        # Get parameters.
        params = qiskit.circuit.ParameterVector(
            self.param_prefix, length=self.num_params
        )

        # Make a quantum circuit having the entanglement unitary at the specified qubit pairs.
        circuit = qiskit.QuantumCircuit(
            self.num_qubits, name="Entanglement Unitary Layer"
        )
        for index, applied_qubit_pair in enumerate(self.applied_qubit_pairs):
            param_start_index = index * 2
            circuit.compose(
                self.__get_pattern(
                    params=params[param_start_index : 2 + param_start_index],
                ),
                applied_qubit_pair,
                inplace=True,
            )

        circuit_inst = circuit.to_instruction()
        circuit = qiskit.QuantumCircuit(self.num_qubits)
        circuit.append(circuit_inst, list(range(self.num_qubits)))

        return circuit
