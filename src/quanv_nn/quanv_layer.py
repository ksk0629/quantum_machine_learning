import numpy as np
import qiskit
from qiskit import qpy, primitives

from src.encoders.x_encoder import XEncoder
from src.layers.random_layer import RandomLayer


class QuanvLayer:
    """Quanvolutional layer class."""

    def __init__(self, kernel_size: tuple[int, int], num_filters: int):
        """Initialise the quanvolutional layer.

        :param tuple[int, int] kernel_size: kernel size
        :param int num_filters: number of filiters
        """
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        num_qubits = self.kernel_size[0] * self.kernel_size[1]
        circuit = qiskit.QuantumCircuit(num_qubits, name="QuanvFilter")
        circuit.compose(
            XEncoder(num_qubits=num_qubits)(), range(num_qubits), inplace=True
        )
        circuit.barrier()

        self.filters = [
            circuit.compose(
                RandomLayer(num_qubits=num_qubits)(), range(num_qubits), inplace=False
            )
            for _ in range(self.num_filters)
        ]

    def __call__(
        self,
        data: np.ndarray,
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 8096,
    ):
        pass
