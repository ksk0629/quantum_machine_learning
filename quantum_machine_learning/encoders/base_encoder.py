from abc import ABC, abstractmethod

import qiskit


class BaseEncoder(ABC):
    """BaseEncoder class of which all encoders inherit this."""

    def __init__(self):
        pass

    @abstractmethod
    def get_circuit(self) -> qiskit.QuantumCircuit:
        pass

    def __call__(self) -> qiskit.QuantumCircuit:
        return self.get_circuit()

    @classmethod
    def get(cls, *args) -> qiskit.QuantumCircuit:
        encoder = cls(*args)
        circuit = encoder.get_circuit()
        return circuit
