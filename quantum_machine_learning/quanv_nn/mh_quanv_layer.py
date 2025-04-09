import dataclasses
import os
import pickle
from typing import Final
import yaml

import numpy as np
import qiskit
from qiskit import qpy

from quantum_machine_learning.encoders.x_encoder import XEncoder
from quantum_machine_learning.layers.random_layer import RandomLayer
from quantum_machine_learning.postprocessor.postprocessor import Postprocessor
from quantum_machine_learning.utils.calculation_utils import CalculationUtils
from quantum_machine_learning.preprocessor.preprocessor import Preprocessor


@dataclasses.dataclass
class MHQuanvLayerInfo:
    """MHQuanvLayer information data class.
    This class provides the enough information to save and load MHQuanvLayer class.
    """

    kernel_size: tuple[int, int]
    num_filters: int
    seed: int | None


class MHQuanvLayer:
    """Quanvolutional layer class, suggested in https://arxiv.org/abs/1904.04767."""

    # Define the file name to save and load.
    LAYER_FILENAME: Final[str] = "layer.yaml"
    # Define the encoding method to save and load.
    ENCODING: Final[str] = "utf-8"
    # Define the look-up table file name.
    LOOKUP_FILENAME: Final[str] = "look_up_table.pkl"
    # Define the circuit file name.
    FILTERS_FILENAME: Final[str] = "filters.qpy"

    def __init__(
        self,
        kernel_size: tuple[int, int],
        num_filters: int,
        seed: int | None = 901,
        lookup_mode: bool = True,
        build: bool = True,
    ):
        """Initialise this layer.

        :param tuple[int, int] kernel_size: a kernel size
        :param int num_filters: the number of filters
        :param int | None seed: a random seed, defaults to 901
        :param bool lookup_mode: if this layer is in look-up mode, defaults to True
        :param bool build: if this layer needs to be built when initialised, defaults to True
        """
        self._kernel_size: tuple[int, int] | None = None
        self._num_filters: int | None = None
        self._seed: int | None = None
        self._lookup_mode: bool | None = None
        self._filters: list[qiskit.QuantumCircuit] | None = None
        self._lookup_table: dict[tuple[int, ...], list[list[float]]] | None = None
        self._is_built: bool | None = False

        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.seed = seed
        self._lookup_mode = lookup_mode

        if build:
            self._build()  # Must be run after settings everything

    @property
    def kernel_size(self) -> tuple[int, int]:
        """Return the kernel size.

        :return tuple[int, int]: the kernel size
        """
        if self._kernel_size is None:
            return (0, 0)
        else:
            return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size: tuple[int, int] | None) -> None:
        """Set a new kernel size and re-build this layer.

        :param tuple[int, int] | None kernel_size: a new kernel size
        """
        self._kernel_size = kernel_size
        self._build()

    @property
    def num_filters(self) -> int:
        """Return the number of filters.

        :return int: the number of filters
        """
        if self._num_filters is None:
            return 0
        else:
            return self._num_filters

    @num_filters.setter
    def num_filters(self, num_filters: int | None) -> None:
        """Set the new number of filters and re-build this layer.

        :param int | None num_filters: the new number of filters
        """
        self._num_filters = num_filters
        self._build()

    @property
    def seed(self) -> int | None:
        """Return the random seed.

        :return int | None: the random seed
        """
        return self._seed

    @seed.setter
    def seed(self, seed: int | None) -> None:
        """Set a new random seed and re-build this layer.

        :param int | None seed: a new random seed
        """
        self._seed = seed
        self._build()

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits of each filter.

        :return int: the number of qubits.
        """
        return self.kernel_size[0] * self.kernel_size[1]

    def _build(self) -> None:
        """Build this layer."""
        # Create the base circuit.
        base_circuit = qiskit.QuantumCircuit(self.num_qubits, name="MHQuanvFilter")
        base_circuit.compose(
            XEncoder(data_dimension=self.num_qubits),
            range(self.num_qubits),
            inplace=True,
        )

        # Create each filter.
        self._filters = []
        self._lookup_table = dict()
        for _ in range(self.num_filters):
            filter = base_circuit.compose(
                RandomLayer(num_state_qubits=self.num_qubits, seed=self.seed),
                range(self.num_qubits),
                inplace=False,
            )
            filter.measure_all()
            self._filters.append(filter)

    def process_data_2d(
        self,
        data_2d: list[list[list[list[float]]]],
        backend: qiskit.providers.Backend,
        shots: int,
        optimisation_level: int,
    ) -> list[list[list[list[float]]]]:
        """Process two-dimensional data.

        :param list[list[list[list[float]]]] data_2d: multi-channel two-dimensional data ([batch, channel, height, width])
        :param qiskit.providers.Backend backend: a backend
        :param int shots: the number of shots
        :param int optimisation_level: a level of optimisation
        :raises ValueError: if the given data_2d doesn't have 4 axes
        :return list[list[list[list[float]]]]: processed data ([batch, channel, height, width])
        """
        # Check if the shape is correct.
        data_2d_np = np.array(data_2d)
        if len(data_2d_np.shape) != 4:
            error_msg = f"The shape of the given data_2d must be four as its batched multi-channel two-dimensional data, however it is {data_2d_np.shape}."
            raise ValueError(error_msg)

        # Prepare the data to be fed to this Quanvolutional Layer.
        windowed_data_2d = Preprocessor.window_batch_data(
            batch_data=data_2d_np, window_size=self.kernel_size
        )
        batch_size = data_2d_np.shape[0]
        num_channels = data_2d_np.shape[1]
        flattened_windowed_data = windowed_data_2d.reshape(
            batch_size, num_channels, -1, self.num_qubits
        ).tolist()

        # Process the data.
        processed_data_2d = []
        (new_height, new_width) = CalculationUtils.calc_2d_output_shape(
            height=data_2d_np.shape[2],
            width=data_2d_np.shape[3],
            kernel_size=self.kernel_size,
        )  # to re-shape processed data later
        for multi_channel_data in flattened_windowed_data:  # loop for batch
            processed_multi_channel_data = []

            for single_channel_data in multi_channel_data:  # loop for channel
                # Process the data.
                #  Note that single_channel_data contain multiple datum.
                #  Each datum was produced from a single channel data by windowing.
                processed_single_channel_data = self._process_single_channel_data(
                    single_channel_data=single_channel_data,
                    backend=backend,
                    shots=shots,
                    optimisation_level=optimisation_level,
                )  # [window, filter, processed_data]
                #
                stacked_processed_single_channel_data = np.hstack(
                    processed_single_channel_data
                )  # [filter, processed_data]

                # Reshape the flat data into a two-dimension one.
                stacked_processed_single_channel_data_2d = (
                    stacked_processed_single_channel_data.reshape(
                        self.num_filters, new_height, new_width
                    )
                ).tolist()

                processed_multi_channel_data.append(
                    stacked_processed_single_channel_data_2d
                )
            # Treat the output data in different channels and processed by different filters in the same way.
            #  (num_channels, num_filters, new_height, new_width)
            #  -> (num_channels * num_filters, new_height, new_width)
            processed_multi_channel_data = np.vstack(
                processed_multi_channel_data
            ).tolist()

            processed_data_2d.append(processed_multi_channel_data)

        return processed_data_2d

    def _process_single_channel_data(
        self,
        single_channel_data: list[list[float]],
        backend: qiskit.providers.BackendV2,
        shots: int,
        optimisation_level: int,
    ) -> list[list[list[float]]]:
        """Process single channel data.

        :param list[list[float]] single_channel_data: single channel data to be processed ([window, datum])
        :param qiskit.providers.BackendV2 backend: a backend
        :param int shots: the number of shots
        :param int optimisation_level: a level of optimisation
        :return list[list[float]]: processed single  channel data ([window, filter, processed_datum])
        """
        processed_single_channel_data = []
        for single_channel_datum in single_channel_data:
            # Process each (flattened) windowed datum.
            processed_single_channel_datum = self._process_single_channel_datum(
                datum=single_channel_datum,
                backend=backend,
                shots=shots,
                optimisation_level=optimisation_level,
            )  # [filter, processed_datum]

            processed_single_channel_data.append(processed_single_channel_datum)

        return processed_single_channel_data

    def _process_single_channel_datum(
        self,
        datum: list[float],
        backend: qiskit.providers.BackendV2,
        shots: int,
        optimisation_level: int,
    ) -> list[list[float]]:
        """Process a single channel datum.

        :param list[float] datum: a single channel datum
        :param qiskit.providers.BackendV2 backend: a backend
        :param int shots: the number of shots
        :param int optimisation_level: a level of optimisation
        :return list[list[float]]: a processed datum ([filter, processed_datum])
        """
        # Get the corresponding outputs from self.lookup_tables if exsited.
        if self._lookup_mode:
            processed_datum = self._look_up(datum=datum)
            if processed_datum is not None:
                return processed_datum

        # Create the combination of the circuit and parameters to run the circuits.
        pubs = []
        for filter in self._filters:
            # Transpile the filters.
            pass_manager = qiskit.transpiler.generate_preset_pass_manager(
                optimization_level=optimisation_level,
                backend=backend,
                seed_transpiler=self.seed,
            )
            transpiled_filter = pass_manager.run(filter)

            # Assign parameters on the filter.
            parameters = {
                parameter: d for parameter, d in zip(filter.parameters, datum)
            }
            primitive_unified_block = transpiled_filter.assign_parameters(parameters)

            pubs.append(primitive_unified_block)

        # Run the sampler.
        jobs = backend.run(pubs, shots=shots)
        # Count the number of ones from each result.
        results = jobs.result().results
        processed_datum = []
        for result in results:
            digit_result = {
                str(int(key, 16)): value for key, value in result.data.counts.items()
            }  # The keys are initially expressed in hex.
            processed_datum.append(
                [Postprocessor.count_one_bits_of_most_frequent_result(digit_result)]
            )

        # Store the processed data regardless of its lookup_mode.
        self._lookup_table[tuple(datum)] = processed_datum

        return processed_datum

    def _look_up(self, datum: list[float]) -> list[list[float]] | None:
        key = tuple(datum)
        if key in self._lookup_table:
            return self._lookup_table[key]
        else:
            return None

    def save(self, model_dir_path: str) -> None:
        """Save this layer.

        :param str model_dir_path: a path to the directory to be stored
        """
        # Create the directory.
        os.makedirs(model_dir_path, exist_ok=True)

        # Create the necessary data to save MHQuanvLayer.
        mh_quanv_layer_info = MHQuanvLayerInfo(
            kernel_size=self._kernel_size,
            num_filters=self._num_filters,
            seed=self._seed,
        )
        # Store the information as yaml.
        yaml_path = os.path.join(model_dir_path, MHQuanvLayer.LAYER_FILENAME)
        with open(yaml_path, "w", encoding=MHQuanvLayer.ENCODING) as yaml_file:
            yaml.dump(
                dataclasses.asdict(mh_quanv_layer_info),
                yaml_file,
                default_flow_style=False,
            )

        # Save the filters.
        filters_path = os.path.join(model_dir_path, MHQuanvLayer.FILTERS_FILENAME)
        with open(filters_path, "wb") as filters_file:
            qpy.dump(self._filters, filters_file)

        # Save the look-up table.
        lookup_table_path = os.path.join(model_dir_path, MHQuanvLayer.LOOKUP_FILENAME)
        with open(lookup_table_path, "wb") as lookup_table_file:
            pickle.dump(self._lookup_table, filters_file)

    @classmethod
    def load(cls, model_dir_path: str):
        """Load a MGQuanvLayer in the given directory.

        :param str model_dir_path: a path to the directory to load
        """
        # Create an instance from the information in the yaml file.
        yaml_path = os.path.join(model_dir_path, MHQuanvLayer.LAYER_FILENAME)
        with open(yaml_path, "r", encoding=MHQuanvLayer.ENCODING) as yaml_file:
            mh_quanv_layer_info = yaml.safe_load(yaml_file)
        mh_quanv_layer = cls(**mh_quanv_layer_info, build=False)

        # Load the filters.
        filters_path = os.path.join(model_dir_path, MHQuanvLayer.FILTERS_FILENAME)
        with open(filters_path, "rb") as filters_file:
            mh_quanv_layer._filters = qpy.load(filters_file)

        # Load the look-up table.
        lookup_table_path = os.path.join(model_dir_path, MHQuanvLayer.LOOKUP_FILENAME)
        with open(lookup_table_path, "rb") as lookup_table_file:
            mh_quanv_layer._lookup_table = pickle.load(filters_file)

        return mh_quanv_layer
