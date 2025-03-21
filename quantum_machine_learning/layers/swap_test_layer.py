import qiskit

from quantum_machine_learning.layers.base_layer import BaseLayer


class SwapTestLayer(BaseLayer):
    """SwapTestLayer class"""

    def __init__(self, control_qubit: int, qubit_pairs: list[tuple[int, int]]):
        """Initialise this layer.

        :param int control_qubit: control qubit
        :param list[tuple[int, int]] qubit_pairs: qubit pairs to perform swap test
        """
        self.control_qubit = control_qubit
        self.qubit_pairs = qubit_pairs

    def get_circuit(self) -> qiskit.QuantumCircuit:
        """Get the swap test layer as a quantum circuit.

        :return qiskit.QuantumCircuit: swap test layer circuit
        """
        num_qubits = 1 + len(self.qubit_pairs) * 2
        circuit = qiskit.QuantumCircuit(num_qubits, name="Swap Test Layer")
        circuit.h(self.control_qubit)
        for qubit_1, qubit_2 in self.qubit_pairs:
            circuit.cswap(self.control_qubit, qubit_1, qubit_2)
        circuit.h(self.control_qubit)

        return circuit
