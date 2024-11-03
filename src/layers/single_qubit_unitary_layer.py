import qiskit

from src.layers.base_learnable_layer import BaseLearnableLayer


class SingleQubitUnitaryLayer(BaseLearnableLayer):
    """SingleQubitUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(self, param_prefix: str):
        super().__init__(param_prefix=param_prefix)

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
        self, num_qubits: int, applied_qubits: list[int]
    ) -> qiskit.QuantumCircuit:
        """Get the single qubit unitary layer circuit.

        :param int num_qubits: number of qubits
        :param list[int] applied_qubits: list of qubits to which single qubit unitary is applied
        :return qiskit.QuantumCircuit: single qubit unitary layer circuit
        """
        # Get parameters.
        num_params = len(applied_qubits) * 2
        params = qiskit.circuit.ParameterVector(self.param_prefix, length=num_params)

        # Make a quantum circuit having single qubit unitary at the specified qubits.
        circuit = qiskit.QuantumCircuit(num_qubits, name="Single Qubit Unitary Layer")
        for index, applied_qubit in enumerate(applied_qubits):
            param_start_index = index * 2
            circuit.compose(
                self.__get_pattern(
                    params=params[param_start_index : 2 + param_start_index],
                ),
                applied_qubit,
                inplace=True,
            )

        circuit_inst = circuit.to_instruction()
        circuit = qiskit.QuantumCircuit(num_qubits)
        circuit.append(circuit_inst, list(range(num_qubits)))

        return circuit
