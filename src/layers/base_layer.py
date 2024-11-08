from abc import ABC, abstractmethod

import qiskit


class BaseLayer(ABC):
    """BaseLayer abstract class, which is all quantum layers inherit this."""

    def __init__(self):
        pass

    @abstractmethod
    def get_circuit(self) -> qiskit.QuantumCircuit:
        pass

    def __call__(self) -> qiskit.QuantumCircuit:
        return self.get_circuit()

    @classmethod
    def get(cls, *args) -> qiskit.QuantumCircuit:
        layer = cls(*args)
        circuit = layer.get_circuit()
        return circuit
