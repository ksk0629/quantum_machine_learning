import numpy as np
import qiskit


class SSwapGate(qiskit.circuit.Gate):
    """Square root swap gate class."""

    def __init__(self, label: str | None = None, *, duration=None, unit="dt"):
        """Create the square root of swap gate."""
        super().__init__("√SWAP", 2, [], label=label, duration=duration, unit=unit)

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""

        q = qiskit.QuantumRegister(2, "q")
        qc = qiskit.QuantumCircuit(q, name=self.name)
        rules = [
            (qiskit.circuit.library.CXGate(), [q[0], q[1]], []),
            (qiskit.circuit.library.CSXGate(), [q[1], q[0]], []),
            (qiskit.circuit.library.CXGate(), [q[0], q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc
