import qiskit

from src.encoders.base_encoder import BaseEncoder


class XEncoder(BaseEncoder):
    """XEncoder class"""

    def __init__(self, num_qubits: int):
        """Initialise this encoder.

        :param int num_qubits: number of qubits.
        """
        self.num_qubits = num_qubits

    @property
    def num_params(self) -> int:
        return self.num_qubits

    def get_circuit(self) -> qiskit.QuantumCircuit:
        # Get parameters.
        params = qiskit.circuit.ParameterVector("x", length=self.num_params)

        circuit = qiskit.QuantumCircuit(self.num_qubits, name="XEncoder")
        for index in range(self.num_qubits):
            circuit.rx(params[index], index)

        circuit_inst = circuit.to_instruction()
        circuit = qiskit.QuantumCircuit(self.num_qubits)
        circuit.append(circuit_inst, list(range(self.num_qubits)))

        return circuit
