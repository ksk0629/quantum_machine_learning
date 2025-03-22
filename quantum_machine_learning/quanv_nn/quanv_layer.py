import os
import pickle

import numpy as np
import qiskit
from qiskit import qpy, primitives

from quantum_machine_learning.encoders.x_encoder import XEncoder
from quantum_machine_learning.layers.random_layer import RandomLayer
import quantum_machine_learning.utils
from quantum_machine_learning.path_getter.quanv_nn_path_getter import QuanvNNPathGetter


class QuanvLayer:
    """Quanvolutional layer class."""

    def __init__(
        self, kernel_size: tuple[int, int], num_filters: int, is_loaded: bool = False
    ):
        """Initialise the quanvolutional layer.

        :param tuple[int, int] kernel_size: kernel size
        :param int num_filters: number of filiters
        :param bool is_loaded: if loaded mode
        """
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        circuit = qiskit.QuantumCircuit(self.num_qubits, name="QuanvFilter")
        circuit.compose(
            XEncoder(num_qubits=self.num_qubits)(), range(self.num_qubits), inplace=True
        )
        circuit.barrier()

        self.filters = []
        if not is_loaded:
            for _ in range(self.num_filters):
                _c = circuit.compose(
                    RandomLayer(num_qubits=self.num_qubits)(),
                    range(self.num_qubits),
                    inplace=False,
                )
                _c.measure_all()
                self.filters.append(_c)
            self.lookup_tables = []

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits of each filter.

        :return int: number of qubits
        """
        return self.kernel_size[0] * self.kernel_size[1]

    def get_lookup_tables_path(
        self, model_dir_path: str, file_prefix: str | None = None
    ) -> str:
        """Get lookup_tables.pkl path.

        :param str model_dir_path: path to directory
        :param str | None file_prefix: file prefix, defaults to None
        :return str: path to look-up tables file path
        """
        filename = (
            f"{file_prefix}lookup_tables.pkl"
            if file_prefix is not None
            else "lookup_tables.pkl"
        )
        return os.path.join(model_dir_path, filename)

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

        :param np.ndarray batch_data: batch data whose shape is (batch, data (whose length is the same as the number of qubits of each filter))
        :param qiskit.primitives.BaseSamplerV1  |  qiskit.primitives.BaseSamplerV2 sampler: sampler primitives, defaults to qiskit.primitives.StatevectorSampler
        :param int shots: number of shots
        :return np.ndarray: processed batch data, first index implies the filter and the other implies the data in the given barch data
        :raises ValueError: if length of shape of batch_data is not 2
        :raises ValueError: if each data's shape is not the same as self.num_qubits
        """
        if len(batch_data.shape) != 2:
            msg = f"The given batch_data shape must be three (batch, data), but {batch_data.shape}."
            raise ValueError(msg)

        is_data_shape_valid = all([len(data) != self.num_qubits for data in batch_data])
        if is_data_shape_valid:
            msg = f"The given batch_data must contain data having the shape as same as num.qubits {self.num_qubits}."
            raise ValueError(msg)

        process_one_data_np = np.vectorize(
            # signature: (m, n) means the length of the shape is two.
            #            () means a scalar
            #            ->(p, q) means the length of the output shape is two
            self.__process_one_data,
            signature="(l),(),()->(m, n)",
        )
        processed_batch_data = process_one_data_np(batch_data, sampler, shots)
        return np.hstack(processed_batch_data)

    def __process_one_data(
        self,
        data: np.ndarray,
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 8096,
    ) -> np.ndarray:
        """Process one data, which is one-dimensional array.

        :param np.ndarray data: one data
        :param qiskit.primitives.BaseSamplerV1  |  qiskit.primitives.BaseSamplerV2 sampler: sampler primitives, defaults to qiskit.primitives.StatevectorSampler
        :param int shots: number of shots
        :return np.ndarray: processed data through each filter, first index implies filter
        """
        # Get the outputs from self.lookup_tables.
        outputs_from_lookup_tables = self.__get_outputs_from_lookup_tables(data=data)
        if outputs_from_lookup_tables is not None:
            return outputs_from_lookup_tables

        # Create the combination of the circuit and parameters to run the circuits.
        pubs = []
        for filter in self.filters:
            parameters = quantum_machine_learning.utils.get_parameter_dict(
                parameter_names=filter.parameters, parameters=data
            )
            pubs.append((filter, parameters))

        # Run the sampler.
        job = sampler.run(pubs, shots=shots)
        # Count the number of ones from each result.
        results = job.result()
        results = [result.data.meas.get_counts() for result in results]
        processed_data = np.empty((len(self.filters), 1))
        processed_data[:, 0] = list(
            map(quantum_machine_learning.utils.count_ones, results)
        )

        return processed_data

    def build_lookup_tables(
        self,
        patterns: np.ndarray,
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 8096,
    ):
        """Build each look-up table to the given patterns.

        :param np.ndarray patterns: input patterns to make look-up tables
        :param qiskit.primitives.BaseSamplerV1  |  qiskit.primitives.BaseSamplerV2 sampler: sampler primitives, defaults to qiskit.primitives.StatevectorSampler
        :param int shots: number of shots
        """
        # Initialise the tables.
        self.lookup_tables = []

        # Process all patterns.
        output_patterns = self.process(
            batch_data=patterns, sampler=sampler, shots=shots
        )

        # Store the outputs.
        for filter_index in range(self.num_filters):
            target_output_patterns = output_patterns[filter_index]
            lookup_table = dict()

            for i_pattern, o_pattern in zip(patterns, target_output_patterns):
                lookup_table[tuple(i_pattern.tolist())] = float(o_pattern)
            self.lookup_tables.append(lookup_table)

    def __get_outputs_from_lookup_tables(self, data: np.ndarray) -> np.ndarray | None:
        """Get outputs from self.lookup_tables.

        :param np.ndarray data: data used as key
        :return np.ndarray | None: output from self.lookup_tables
        """
        # Initialise the output data.
        processed_data = np.empty((len(self.filters), 1))
        # Convert the data into the form of the key of each look-up table.
        key = tuple(data.tolist())
        for filter_index in range(self.num_filters):
            try:
                if key in self.lookup_tables[filter_index]:
                    processed_data[filter_index, 0] = self.lookup_tables[filter_index][
                        key
                    ]
                else:
                    return None
            except IndexError:
                # If thie error happens, possibly lookup_tables have not created.
                return None

        return processed_data

    def save(self, model_dir_path: str):
        """Save the filters to the directory specified by the given model_dir_path.

        :param str model_dir_path: path to the output directory.
        """
        # Create the directory specified by the argument output_dir_path.
        os.makedirs(model_dir_path)

        # Create PathGetter.
        path_getter = QuanvNNPathGetter(dir_path=model_dir_path)

        # Save the basic information of this QuanvLayer.
        basic_info = {
            "kernel_size": self.kernel_size,
            "num_filters": self.num_filters,
        }
        with open(path_getter.basic_info, "wb") as pkl_file:
            pickle.dump(basic_info, pkl_file)

        # Save the circuit.
        with open(path_getter.circuit, "wb") as qpy_file:
            qpy.dump(self.filters, qpy_file)

        # Save the look-up tables.
        lookup_tables_path = self.get_lookup_tables_path(model_dir_path)
        with open(lookup_tables_path, "wb") as pkl_file:
            pickle.dump(self.lookup_tables, pkl_file)

    @classmethod
    def load(cls, model_dir_path: str):
        """Load the filters from the directory specified by the given model_dir_path.

        :param str model_dir_path: path to the input directory.
        """
        # Create PathGetter.
        path_getter = QuanvNNPathGetter(dir_path=model_dir_path)

        # Load the basic information.
        with open(path_getter.basic_info, "rb") as pkl_file:
            basic_info = pickle.load(pkl_file)
        basic_info["is_loaded"] = True
        loaded_quanv_layer = cls(**basic_info)

        # Load the filters.
        with open(path_getter.circuit, "rb") as qpy_file:
            loaded_quanv_layer.filters = qpy.load(qpy_file)

        # Save the look-up tables.
        lookup_tables_path = loaded_quanv_layer.get_lookup_tables_path(model_dir_path)
        with open(lookup_tables_path, "rb") as pkl_file:
            loaded_quanv_layer.lookup_tables = pickle.load(pkl_file)

        return loaded_quanv_layer
