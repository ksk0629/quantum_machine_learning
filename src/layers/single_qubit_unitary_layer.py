import qiskit

from src.layers.base_learnable_layer import BaseLearnableLayer


class SingleQubitUnitaryLayer(BaseLearnableLayer):
    """SingleQubitUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(self, num_qubits: int, applied_qubits: list[int], param_prefix: str):
        """initialise the layer.

        :param int num_qubits: number of qubits
        :param list[int] applied_qubits: list of qubits to which single qubit unitary is applied
        :param str param_prefix: parameter prefix
        """
        self.num_qubits = num_qubits
        self.applied_qubits = applied_qubits
        super().__init__(param_prefix=param_prefix)

    @property
    def num_params(self) -> int:
        """Get the number of parameters.

        :return int: number of parameters
        """
        return len(self.applied_qubits) * 2

    def __get_pattern(
        self, params: qiskit.circuit.ParameterVector
    ) -> qiskit.QuantumCircuit:
        """Return the single qubit unitary layer pattern.

        :param qiskit.circuit.ParameterVector params: parameter vector
        :return qiskit.QuantumCircuit: single qubit unitary layer pattern
        """
        pattern = qiskit.QuantumCircuit(1)
        pattern.ry(params[0], 0)
        pattern.rz(params[1], 0)

        return pattern

    def get_circuit(
        self,
    ) -> qiskit.QuantumCircuit:
        """Get the single qubit unitary layer circuit.

        :return qiskit.QuantumCircuit: single qubit unitary layer circuit
        """
        # Get parameters.
        params = qiskit.circuit.ParameterVector(
            self.param_prefix, length=self.num_params
        )

        # Make a quantum circuit having single qubit unitary at the specified qubits.
        circuit = qiskit.QuantumCircuit(
            self.num_qubits, name="Single Qubit Unitary Layer"
        )
        for index, applied_qubit in enumerate(self.applied_qubits):
            param_start_index = index * 2
            circuit.compose(
                self.__get_pattern(
                    params=params[param_start_index : 2 + param_start_index],
                ),
                applied_qubit,
                inplace=True,
            )

        circuit_inst = circuit.to_instruction()
        circuit = qiskit.QuantumCircuit(self.num_qubits)
        circuit.append(circuit_inst, list(range(self.num_qubits)))

        return circuit
