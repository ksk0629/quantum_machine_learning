from abc import ABC, abstractmethod

import qiskit


class BaseLayer(ABC):
    """BaseLayer abstract class, which is all quantum layers inherit this."""

    def __init__(self):
        pass

    @abstractmethod
    def get_circuit(self) -> qiskit.QuantumCircuit:
        pass
