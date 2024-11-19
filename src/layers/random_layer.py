import qiskit

from src.layers.base_layer import BaseLayer


class RandomLayer(BaseLayer):
    """RandomLayer class. This is introduced as the quanvolutional filter
    in https://arxiv.org/pdf/1904.04767.
    """

    def __init__(self):
        pass

    def get_circuit(self) -> qiskit.QuantumCircuit:
        pass
