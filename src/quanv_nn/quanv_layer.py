import os
import pickle

import numpy as np
import qiskit
from qiskit import qpy, primitives

from src.encoders.x_encoder import XEncoder
from src.layers.random_layer import RandomLayer
import src.utils


class QuanvLayer:
    """Quanvolutional layer class."""

    def __init__(self, kernel_size: tuple[int, int], num_filters: int):
        """Initialise the quanvolutional layer.

        :param tuple[int, int] kernel_size: kernel size
        :param int num_filters: number of filiters
        """
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        circuit = qiskit.QuantumCircuit(self.num_qubits, name="QuanvFilter")
        circuit.compose(
            XEncoder(num_qubits=self.num_qubits)(), range(self.num_qubits), inplace=True
        )
        circuit.barrier()

        self.filters = []
        for _ in range(self.num_filters):
            _c = circuit.compose(
                RandomLayer(num_qubits=self.num_qubits)(),
                range(self.num_qubits),
                inplace=False,
            )
            _c.measure_all()
            self.filters.append(_c)

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits of each filter.

        :return int: number of qubits
        """
        return self.kernel_size[0] * self.kernel_size[1]

    def __call__(
        self,
        batch_data: np.ndarray,
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 8096,
    ) -> np.ndarray:
        """Call self.process.

        :param np.ndarray batch_data: batch data whose shape is [batch, data (whose length is the same as the number of qubits of each filter), depth(=channel)]
        :param qiskit.primitives.BaseSamplerV1  |  qiskit.primitives.BaseSamplerV2 sampler: sampler primitives, defaults to qiskit.primitives.StatevectorSampler
        :param int shots: number of shots
        :return np.ndarray: processed batch data
        """
        return self.process(batch_data=batch_data, sampler=sampler, shots=shots)

    def process(
        self,
        batch_data: np.ndarray,
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 8096,
    ) -> np.ndarray:
        """Process the given batch data through all filters.

        :param np.ndarray batch_data: batch data whose shape is [batch, depth(=channel), data (whose length is the same as the number of qubits of each filter)]
        :param qiskit.primitives.BaseSamplerV1  |  qiskit.primitives.BaseSamplerV2 sampler: sampler primitives, defaults to qiskit.primitives.StatevectorSampler
        :param int shots: number of shots
        :return np.ndarray: processed batch data
        :raises ValueError: if length of shape of batch_data is not 3
        :raises ValueError: if second element of shape of each data in batch_data
        """
        if len(batch_data.shape) != 3:
            msg = f"The given batch_data shape must be three [batch, depth, data], but {batch_data.shape}."
            raise ValueError(msg)

        is_data_shape_valid = all(
            [len(channel) != self.num_qubits for data in batch_data for channel in data]
        )
        if is_data_shape_valid:
            msg = f"The given batch_data must contain data having the shape as same as num.qubits {self.num_qubits}."
            raise ValueError(msg)

        process_one_data_np = np.vectorize(
            # signature: (m, n) means the length of the shape is two.
            #            () means a scalar
            #            ->(p, q) means the length of the output shape is two
            self.__process_one_data,
            signature="(m, n),(),()->(p, q)",
        )

        return process_one_data_np(batch_data, sampler, shots)

    def __process_one_data(
        self,
        data: np.ndarray,
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 8096,
    ) -> np.ndarray:
        """Process one data, whose shape is [depth(=channel), data (whose length is the same as the number of qubits of each filter)]

        :param np.ndarray data: one data
        :param qiskit.primitives.BaseSamplerV1  |  qiskit.primitives.BaseSamplerV2 sampler: sampler primitives, defaults to qiskit.primitives.StatevectorSampler
        :param int shots: number of shots
        :return np.ndarray: processed data whose shape is [depth(=channel) * filters, processed data (= 1 for now)]
        """
        # Create the combination of the circuit and parameters to run the circuits.
        pubs = []
        for filter in self.filters:
            for channel in data:
                parameters = src.utils.get_parameter_dict(
                    parameter_names=filter.parameters, parameters=channel
                )
                pubs.append((filter, parameters))

        # Run the sampler.
        job = sampler.run(pubs, shots=shots)
        # Count the number of ones from each result.
        results = job.result()
        results = [result.data.meas.get_counts() for result in results]
        processed_data_dimension = 1
        processed_data = np.empty(
            (len(self.filters) * len(data), processed_data_dimension)
        )
        processed_data[:, 0] = list(map(src.utils.count_ones, results))

        return processed_data
