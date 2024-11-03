import qiskit

from src.layers.base_learnable_layer import BaseLearnableLayer


class DualQubitUnitaryLayer(BaseLearnableLayer):
    """DualQubitUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(self, param_prefix: str):
        super().__init__(param_prefix=param_prefix)

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
        self, num_qubits: int, applied_qubit_pairs: list[tuple[int, int]]
    ) -> qiskit.QuantumCircuit:
        """Get the dual qubit unitary layer circuit.

        :param int num_qubits: number of qubits
        :param list[int] applied_qubit_pairs: list of qubit pairs to which dual qubit unitary is applied
        :return qiskit.QuantumCircuit: dual qubit unitary layer circuit
        """
        # Get parameters.
        num_params = len(applied_qubit_pairs) * 2
        params = qiskit.circuit.ParameterVector(self.param_prefix, length=num_params)

        # Make a quantum circuit having the dual qubit unitary at the specified qubit pairs.
        circuit = qiskit.QuantumCircuit(num_qubits, name="Dual Qubit Unitary Layer")
        for index, applied_qubit_pair in enumerate(applied_qubit_pairs):
            param_start_index = index * 2
            circuit.compose(
                self.__get_pattern(
                    params=params[param_start_index : 2 + param_start_index],
                ),
                applied_qubit_pair,
                inplace=True,
            )

        circuit_inst = circuit.to_instruction()
        circuit = qiskit.QuantumCircuit(num_qubits)
        circuit.append(circuit_inst, list(range(num_qubits)))

        return circuit

    @classmethod
    def get(
        cls,
        param_prefix: str,
        num_qubits: int,
        applied_qubit_pairs: list[tuple[int, int]],
    ) -> qiskit.QuantumCircuit:
        """Call get_circuit as a class method.

        :param str param_prefix: prefix of parameter name
        :param int num_qubits: number of qubits
        :param list[int] applied_qubit_pairs: list of qubit pairs to which dual qubit unitary is applied
        :return qiskit.QuantumCircuit: dual qubit unitary layer circuit
        """
        dual_qubit_unitary_layer = cls(param_prefix=param_prefix)
        circuit = dual_qubit_unitary_layer.get_circuit(
            num_qubits=num_qubits, applied_qubit_pairs=applied_qubit_pairs
        )
        return circuit
