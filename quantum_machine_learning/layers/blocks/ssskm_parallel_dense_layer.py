from typing import Callable

import numpy as np
import qiskit
from qiskit.primitives import BackendEstimatorV2

from quantum_machine_learning.layers.circuits.bases.base_parametrised_layer import (
    BaseParametrisedLayer,
)
from quantum_machine_learning.layers.circuits.feature_maps.y_angle import YAngle
from quantum_machine_learning.layers.circuits.ansatzes.ssskm_dense_layer import (
    SSSKMDenseLayer,
)


class SSSKMParallelDenseLayer:
    """Quantum parallel dense layer class, suggested in https://iopscience.iop.org/article/10.1088/2632-2153/ad2aef"""

    def __init__(
        self,
        num_qubits: int,
        num_reputations: int,
        num_layers: int,
        parameter_prefix: str | None = None,
        encoder_class: BaseParametrisedLayer = YAngle,
        transformer: Callable[[list[float]], list[float]] = lambda data: (
            np.array(data) * np.pi
        ).tolist(),
        trainable_parameter_values: list[list[float]] | None = None,
        build: bool = True,
    ):
        """initialise the layer.

        :param int num_qubits: the number of state qubits
        :param int num_reputations: the number of reputations
        :param int num_layers: the number of SSSKMDenseLayer's
        :param str | None parameter_prefix: a prefix of the parameter names, defaults to None
        :param BaseParametrisedLayer encoder_class: an encoder class, defaults to YAngle
        :param Callable[[list[float]], list[float]] transformer: a transformer, defaults to lambda data: (np.array(data) * np.pi).tolist()
        :param list[list[float]] trainable_parameter_values: trainable parameter values to process data, defaults to None
        :param bool build: if each dense_layers must be built, defaults to True
        """
        self._num_qubits: int = num_qubits
        self._num_reputations: int = num_reputations
        self._num_layers: int = num_layers
        self._parameter_prefix: str | None = parameter_prefix
        self._encoder_class: BaseParametrisedLayer = encoder_class
        self._transformer: Callable[[list[float]], list[float]] = transformer
        self._trainable_parameter_values: list[list[float]] | None = (
            trainable_parameter_values
        )

        self._encoder = None
        self._dense_layers: list[SSSKMParallelDenseLayer] = []
        self._trainable_parameters: list[
            qiskit.circuit.parametertable.ParameterView
        ] = []

        if build:
            self._build()

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits contained in each dense layer.

        :return int: the number of qubits
        """
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set a new number of qubits and re-build the dense layers.

        :param int num_qubits: a new number of qubits
        """
        self._num_qubits = num_qubits
        self._build()

    @property
    def num_reputations(self) -> int:
        """Return the number of reputations of each dense layer.

        :return int: the number of reputations
        """
        return self._num_reputations

    @num_reputations.setter
    def num_reputations(self, num_reputations: int) -> None:
        """Set the new number of reputations and re-build the dense layers.

        :param int num_reputations: a new number of reputations
        """
        self._num_reputations = num_reputations
        self._build()

    @property
    def num_layers(self) -> int:
        """Return the number of dense layers.

        :return int: the number of dense layers
        """
        return self._num_layers

    @num_layers.setter
    def num_layers(self, num_layers: int) -> None:
        """Set a new number of layers and re-build the dense layers.

        :param int num_layers: a new number of layers
        """
        self._num_layers = num_layers
        self._build()

    @property
    def parameter_prefix(self) -> int:
        """Return the parameter prefix.

        :return int: the parameter prefix
        """
        if self._parameter_prefix is None:
            return ""
        else:
            return self._parameter_prefix

    @parameter_prefix.setter
    def parameter_prefix(self, parameter_prefix: str | None) -> None:
        """Set a new parameter prefix and re-build the dense layers.

        :param str | None parameter_prefix: a new parameter prefix
        """
        self._parameter_prefix = parameter_prefix
        self._build()

    @property
    def total_qubits(self) -> int:
        """Return the number of total qubits over all layers.

        :return int: the number of total qubits
        """
        return self.num_qubits * self.num_layers

    @property
    def encoding_parameters(self) -> qiskit.circuit.parametertable.ParameterView:
        """Return the parameters for encoding.

        :return qiskit.circuit.parametertable.ParameterView: the parameters for encoding
        """
        if self._encoder is None:
            return []
        else:
            return self._encoder.parameters

    def _build(self) -> None:
        """Build the circuit."""
        # Create the encoder.
        self._encoder = self._encoder_class(data_dimension=self.num_qubits)

        # Add dense layers.
        self._dense_layers = []
        self._trainable_parameters = []
        num_digits = len(str(self.num_layers))
        for index in range(self.num_layers):
            # Create a dense layer.
            zfilled_index = str(index).zfill(num_digits)
            parameter_prefix = f"dense{zfilled_index}"
            dense_layer = SSSKMDenseLayer(
                num_state_qubits=self.num_qubits,
                num_reputations=self.num_reputations,
                parameter_prefix=parameter_prefix,
            )
            # Store the parameters.
            self._trainable_parameters.append(dense_layer.parameters)
            # Add the encoder.
            dense_layer.compose(
                self._encoder, dense_layer.qubits, inplace=True, front=True
            )
            # Store the dense layer with the encoder.
            self._dense_layers.append(dense_layer)

    def run(
        self,
        data: list[float],
        backend_for_optimisation: qiskit.providers.BackendV2,
        estimator: BackendEstimatorV2,
        optimisation_level: int,
    ) -> list[float]:
        """Run this layer to process a given data.

        :param list[float] data: a data to be processed
        :param qiskit.providers.BackendV2 backend_for_optimisation: a backend to optimise the circuits
        :param qiskit.primitives.EstimatorV2 estimator: an estimator, which must be set options as you wish
        :param int optimisation_level: an optimisation level
        :raises ValueError: if the length of the data and its total_qubits are not the same
        :return list[float]: a processed data
        """
        if len(data) != self.total_qubits:
            error_msg = f"The length of the data to be processed, {len(data)}, and the number of total qubits, {self.total_qubits} must be the same, but not."
            raise ValueError(error_msg)

        # Prepare the preset pass manager.
        pm = qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager(
            backend=backend_for_optimisation, optimization_level=optimisation_level
        )
        # Process the data.
        processed_data = []
        for index in range(self.num_layers):
            # Get the corresponding layer, parameters and parameter values.
            dense_layer = self._dense_layers[index]
            tranable_parameters = self._trainable_parameters[index]
            trainable_parameter_values = self._trainable_parameter_values[index]

            # Get the corresponding data.
            data_start_index = index * self.num_qubits
            data_end_index = data_start_index + self.num_qubits
            partial_data = data[data_start_index:data_end_index]

            # Create the parameters map.
            parameters = {  # For encoding parameters
                parameter: parameter_value
                for parameter, parameter_value in zip(
                    self.encoding_parameters, self._transformer(partial_data)
                )
            } | {  # For trainable parameters
                parameter: parameter_value
                for parameter, parameter_value in zip(
                    tranable_parameters, trainable_parameter_values
                )
            }

            # Transpile the dense layer.
            assined_dense_layer = dense_layer.assign_parameters(parameters)
            transpiled_dense_layer = pm.run(assined_dense_layer)

            # Get the expectation values.
            for index in range(self.num_qubits):
                # Define the observable.
                pauli = ["I"] * self.num_qubits
                pauli[index] = "Y"
                pauli = ["".join(pauli)]
                coeff = [1.0]
                operator = qiskit.quantum_info.SparsePauliOp(pauli, coeff)
                observable = operator.apply_layout(transpiled_dense_layer.layout)

                # Get the expectation value.
                job = estimator.run([(transpiled_dense_layer, observable)])
                pub_result = job.result()[0]
                expectation_value = float(pub_result.data.evs)

                # Store the expectation value.
                processed_data.append(expectation_value)

        return processed_data
