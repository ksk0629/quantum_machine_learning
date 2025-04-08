import os
import pickle

import numpy as np
import qiskit
from qiskit import qpy
import qiskit.providers

from quantum_machine_learning.encoders.x_encoder import XEncoder
from quantum_machine_learning.layers.random_layer import RandomLayer
from quantum_machine_learning.path_getter.quanv_nn_path_getter import QuanvNNPathGetter
from quantum_machine_learning.postprocessor.postprocessor import Postprocessor
from quantum_machine_learning.utils.circuit_utils import CircuitUtils
from quantum_machine_learning.preprocessor.preprocessor import Preprocessor


class MHQuanvLayer:
    """Quanvolutional layer class, suggested in https://arxiv.org/abs/1904.04767."""

    def __init__(self, kernel_size: tuple[int, int], num_filters: int, seed: int = 901):
        self._kernel_size = None
        self._num_filters = None
        self._seed = None
        self._filters = None

        self._build()

        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.seed = seed

    @property
    def kernel_size(self) -> tuple[int, int]:
        if self._kernel_size is None:
            return (0, 0)
        else:
            return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size: tuple[int, int] | None) -> None:
        self._kernel_size = kernel_size
        self._build()

    @property
    def num_filters(self) -> int:
        if self._num_filters is None:
            return 0
        else:
            return self._num_filters

    @num_filters.setter
    def num_filters(self, num_filters: int | None) -> None:
        self._num_filters = num_filters
        self._build()

    @property
    def seed(self) -> int | None:
        return self._seed

    @seed.setter
    def seed(self, seed: int | None) -> None:
        self._seed = seed
        self._build()

    @property
    def num_qubits(self) -> int:
        return self.kernel_size[0] * self.kernel_size[1]

    def _build(self) -> None:
        # Create the base circuit.
        base_circuit = qiskit.QuantumCircuit(self.num_qubits, name="QuanvFilter")
        base_circuit.compose(
            XEncoder(data_dimension=self.num_qubits),
            range(self.num_qubits),
            inplace=True,
        )

        # Create filters.
        self._filters = []
        self.lookup_tables = []
        for _ in range(self.num_filters):
            filter = base_circuit.compose(
                RandomLayer(num_state_qubits=self.num_qubits, seed=self.seed),
                range(self.num_qubits),
                inplace=False,
            )
            filter.measure_all()
            self._filters.append(filter)

    def __call__(
        self,
        data_2d: list[list[float]],
        backend: qiskit.providers.Backend,
        shots: int = 8192,
    ) -> np.ndarray:
        return self.process_data_2d(data_2d=data_2d, backend=backend, shots=shots)

    def process_data_2d(
        self,
        data_2d: list[list[float]],
        backend: qiskit.providers.Backend,
        shots: int = 8192,
    ) -> np.ndarray:
        data_2d_np = np.array(data_2d)
        if len(data_2d_np.shape) != 2:
            error_msg = f"The shape of the given data_2d must be three as its batched two-dimensional data, however it is {data_2d_np.shape}."
            raise ValueError(error_msg)

        # Prepare the data to be fed to this Quanvolutional Layer.
        windowed_data_2d = Preprocessor.window_batch_data(
            batch_data=data_2d_np, window_size=self.kernel_size
        )
        batch_size = (
            windowed_data_2d.shape[0] if len(windowed_data_2d.shape) == 4 else 1
        )
        num_channels = data_2d_np.shape[1]
        windowed_flattened_data_2d = windowed_data_2d.reshape(
            batch_size, num_channels, -1, self.num_qubits
        ).tolist()

        # Process the data.
        processed_batch_data = []
        new_height = data_2d_np.shape[2] - self.kernel_size[0] + 1 - 1 + 1
        new_width = data_2d_np.shape[3] - self.kernel_size[1] + 1 - 1 + 1
        for multi_channel_data_2d in windowed_flattened_data_2d:
            processed_multi_channel_data = []
            for _data_2d in multi_channel_data_2d:

                vectorised_process_datum_2d = np.vectorize(
                    # signature: (m, n) means the length of the shape is two.
                    #            () means a scalar
                    #            ->(p, q) means the length of the output shape is two
                    self._process_datum_2d,
                    signature="(l),(),()->(m, n)",
                )
                processed_data_2d = vectorised_process_datum_2d(
                    _data_2d, backend, shots
                )
                stacked_processed_data_2d = np.hstack(processed_data_2d)

                # Reshape the data as processed two-dimensional data.
                stacked_processed_data_2d = stacked_processed_data_2d.reshape(
                    self.quanv_layer.num_filters, new_height, new_width
                )
                processed_multi_channel_data.append(stacked_processed_data_2d)
            # Treat the output data in different channels and processed by different filters in the same way.
            # processed_multi_channel_data's shape
            # = (num_channels, num_filters, new_height, new_width)
            # -> (num_channels * num_filters, new_height, new_width)
            processed_multi_channel_data = np.vstack(processed_multi_channel_data)
            processed_batch_data.append(processed_multi_channel_data)

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

    def __process_one_data(
        self,
        data: np.ndarray,
        backend: qiskit.providers.BackendV2,
        shots: int = 8096,
    ) -> np.ndarray:
        """Process one data, which is one-dimensional array.

        :param np.ndarray data: one data
        :param qiskit.providers.BackendV2 backend: a backend
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
            parameters = CircuitUtils.get_parameter_dict(
                parameter_names=filter.parameters, parameters=data
            )
            pubs.append((filter, parameters))

        # Run the sampler.
        job = backend.run(pubs, shots=shots)
        # Count the number of ones from each result.
        results = job.result()
        results = [result.data.meas.get_counts() for result in results]
        processed_data = np.empty((len(self.filters), 1))
        processed_data[:, 0] = list(
            map(Postprocessor.count_one_bits_of_most_frequent_result, results)
        )

        return processed_data

    def build_lookup_tables(
        self,
        patterns: np.ndarray,
        backend: qiskit.providers.BackendV2,
        shots: int = 8096,
    ):
        """Build each look-up table to the given patterns.

        :param np.ndarray patterns: input patterns to make look-up tables
        :param qiskit.providers.BackendV2 backend: a backend
        :param int shots: number of shots
        """
        # Initialise the tables.
        self.lookup_tables = []

        # Process all patterns.
        output_patterns = self.process(
            batch_data=patterns, backend=backend, shots=shots
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
