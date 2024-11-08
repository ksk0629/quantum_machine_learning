import qiskit

from src.encoders.base_encoder import BaseEncoder


class YZEncoder(BaseEncoder):
    """YZEncoder class, which is used in https://arxiv.org/pdf/2103.11307."""

    def __init__(self, num_qubits: int):
        """Initialise this encoder.

        :param int num_qubits: number of qubits.
        """
        self.num_qubits = num_qubits

    @property
    def num_params(self) -> int:
        return self.num_qubits * 2

    def get_circuit(self) -> qiskit.QuantumCircuit:
        # Get parameters.
        params = qiskit.circuit.ParameterVector("x", length=self.num_params)

        circuit = qiskit.QuantumCircuit(self.num_qubits)
        for index_qubit in range(self.num_qubits):
            index_first_param = index_qubit * 2
            circuit.ry(params[index_first_param], index_qubit)

            index_second_param = index_first_param + 1
            circuit.rz(params[index_second_param], index_qubit)

        circuit_inst = circuit.to_instruction()
        circuit = qiskit.QuantumCircuit(self.num_qubits)
        circuit.append(circuit_inst, list(range(self.num_qubits)))

        return circuit
