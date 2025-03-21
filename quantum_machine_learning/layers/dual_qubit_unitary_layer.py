import qiskit

from quantum_machine_learning.layers.base_learnable_layer import BaseLearnableLayer


class DualQubitUnitaryLayer(BaseLearnableLayer):
    """DualQubitUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(
        self,
        num_qubits: int,
        applied_qubit_pairs: list[tuple[int, int]],
        param_prefix: str,
    ):
        """Initialise this layer.

        :param int num_qubits: number of qubits
        :param list[int] applied_qubit_pairs: list of qubit pairs to which dual qubit unitary is applied
        :param str param_prefix: parameter prefix
        """
        self.num_qubits = num_qubits
        self.applied_qubit_pairs = applied_qubit_pairs
        super().__init__(param_prefix=param_prefix)

    @property
    def num_params(self) -> int:
        return len(self.applied_qubit_pairs) * 2

    def __get_pattern(
        self, params: qiskit.circuit.ParameterVector
    ) -> qiskit.QuantumCircuit:
        """Return the dual qubit unitary layer pattern.

        :param qiskit.circuit.ParameterVector params: parameter vector
        :return qiskit.QuantumCircuit: dual qubit unitary layer pattern
        """
        pattern = qiskit.QuantumCircuit(2)
        pattern.ryy(params[0], 0, 1)
        pattern.rzz(params[1], 0, 1)

        return pattern

    def get_circuit(
        self,
    ) -> qiskit.QuantumCircuit:
        """Get the dual qubit unitary layer circuit.

        :return qiskit.QuantumCircuit: dual qubit unitary layer circuit
        """
        # Get parameters.
        params = qiskit.circuit.ParameterVector(
            self.param_prefix, length=self.num_params
        )

        # Make a quantum circuit having the dual qubit unitary at the specified qubit pairs.
        circuit = qiskit.QuantumCircuit(
            self.num_qubits, name="Dual Qubit Unitary Layer"
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
