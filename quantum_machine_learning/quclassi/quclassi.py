import dataclasses
import os
from typing import Callable, Final

import numpy as np
import qiskit
import qiskit.circuit
import yaml

from quantum_machine_learning.encoders.yz_encoder import YZEncoder
from quantum_machine_learning.layers.single_qubit_unitary_layer import (
    SingleQubitUnitaryLayer,
)
from quantum_machine_learning.layers.dual_qubit_unitary_layer import (
    DualQubitUnitaryLayer,
)
from quantum_machine_learning.layers.entanglement_unitary_layer import (
    EntanglementUnitaryLayer,
)
from quantum_machine_learning.postprocessor.postprocessor import Postprocessor


@dataclasses.dataclass
class QuClassiInfo:
    """QuClassi information data class.
    This class provides the enough information to save and load QuClassi class.
    """

    classical_data_size: int
    structure: str
    labels: list[str]
    initial_parameters: dict[str, list[float]] | None = None
    name: str | None = None


class QuClassi(qiskit.circuit.library.BlueprintCircuit):
    """QuClassi class."""

    # Define the constants used in this class.
    SINGLE_QUBIT_UNITARY: Final[str] = "s"
    DUAL_QUBIT_UNITARY: Final[str] = "d"
    ENTANGLEMENT_UNITARY: Final[str] = "e"
    ACCEPTABLE_STRUCTURE: Final[tuple[str, str, str]] = [
        SINGLE_QUBIT_UNITARY,
        DUAL_QUBIT_UNITARY,
        ENTANGLEMENT_UNITARY,
    ]
    # Define the classical register name.
    CREG_NAME: Final[str] = "creg"
    # Define the model file name to save and load.
    MODEL_FILENAME: Final[str] = "model.yaml"
    # Define the encoding method to save and load.
    ENCODING: Final[str] = "utf-8"

    def __init__(
        self,
        classical_data_size: int,
        structure: str,
        labels: list[str],
        transformer: Callable[[list[list[float]]], list[float]] | None = (
            lambda data: (2 * np.arcsin(np.sqrt(data))).tolist()
        ),
        initial_parameters: dict[str, list[float]] | None = None,
        name: str | None = "QuClassi",
    ):
        """Initialise QuClassi.

        :param int classical_data_size: the classical data size to be fed into QuClassi
        :param str structure: the strucutre of QuClassi
        :param list[str] labels: the labels to be classified
        :param Callable[[list[list[float]]], list[float]] | None transformer: a method to be applied to a data to be fed to the QuClassi, defaults to (lambda data: (2 * np.arcsin(np.sqrt(data))).tolist())
        :param dict[str, list[float]] initial_parameters: the initial parameters, defaults to None
        :param str | None name: the name of the cirucit, defaults to "QuClassi"
        """
        self._classical_data_size: int | None = None
        self._structure: str | None = None
        self._labels: list[str] | None = None
        self._parameter_values: dict[str, list[float]] | None = None
        self._ancilla_qreg: qiskit.QuantumCircuit | None = None
        self._data_qreg: qiskit.QuantumCircuit | None = None
        self._train_qreg: qiskit.QuantumCircuit | None = None
        self._ansatz: qiskit.QuantumCircuit | None = None
        self._feature_map: qiskit.QuantumCircuit | None = None
        self._transpiled: dict[int, qiskit.QuantumCircuit] | None = None
        self._transformer: Callable[[list[list[float]]], list[float]] = None

        super().__init__(name=name)  # For BlueprintCircuit

        self.classical_data_size = classical_data_size
        self.structure = structure
        self.labels = labels
        self.parameter_values = initial_parameters
        self._transformer = transformer

    @property
    def classical_data_size(self) -> int:
        """Return the classical data size.
        If it is None, return 0.

        :return int: the classical data size
        """
        if self._classical_data_size is None:
            return 0
        else:
            return self._classical_data_size

    @classical_data_size.setter
    def classical_data_size(self, classical_data_size: int) -> None:
        """Set the new classical data size and the reset the registers.

        :param int classical_data_size: a new classical data size
        """
        self._classical_data_size = classical_data_size
        self._reset_register()

    @property
    def using_classical_data_size(self) -> int:
        """Return the data size that this QuClassi actually uses.
        QuClassi assumes that the input classical data size is even,
        so this return value is actual classical_data_size + 1 if it's odd.

        :return int: data size that is evenised
        """
        if self.classical_data_size % 2 == 0:  # Even number
            return self.classical_data_size
        else:  # Odd number
            return self.classical_data_size + 1

    @property
    def num_data_qubits(self) -> int:
        """Get the number of data qubits of each circuit.

        :return int: the number of data qubits of each circuit.
        """
        return int(self.using_classical_data_size / 2)

    @property
    def num_train_qubits(self) -> int:
        """Get the number of train qubits of each circuit.

        :return int: the number of train qubits of each circuit.
        """
        return int(self.using_classical_data_size / 2)

    @property
    def structure(self) -> str:
        """Return the structure of this QuClassi.

        :return str: the structure
        """
        if self._structure is None:
            return ""
        else:
            return self._structure

    @structure.setter
    def structure(self, structure: str | None) -> None:
        """Set the new structure if it's valid, set the parameter values None and reset the registers.

        :param str structure | None: a new structure
        :raises ValueError: if a new structure is not constructed with acceptable letters
        """
        if structure is not None:
            valid = all(s in QuClassi.ACCEPTABLE_STRUCTURE for s in structure)
            if not valid:
                error_msg = f"""
                A given structure must be constructed with only {QuClassi.ACCEPTABLE_STRUCTURE}.
                However, it is '{structure}'.
                """
                raise ValueError(error_msg)

        self._structure = structure
        self._parameter_values = None

        self._reset_register()

    @property
    def labels(self) -> list[str]:
        """Return the labels.

        :return list[str]: the labels
        """
        if self._labels is None:
            return []
        else:
            return self._labels

    @labels.setter
    def labels(self, labels: list[str]) -> None:
        """Set the new labels and reset the parameters.

        :param list[str] labels: a new labels
        """
        self._labels = labels

    @property
    def parameter_values(self) -> dict[str, list[float]]:
        """Return the parameter values.

        :return dict[str, list[float]]: the parameter values
        """
        if self._parameter_values is None:
            return dict()
        else:
            return self._parameter_values

    @parameter_values.setter
    def parameter_values(self, parameter_values: dict[str, list[float]]) -> None:
        """Set the new parameter values in the same order as the labels.

        :param dict[str, list[float]] parameter_values: a new parameter values
        :raises AttributeError: if the labels hasn't been set yet
        :raises AttributeError: if the keys of the new parameter values are not the same as the labels.
        """
        if parameter_values is not None:
            if self.labels == []:
                raise AttributeError(
                    "In order to set parameter_values, the labels must be previously set."
                )

            if set(parameter_values.keys()) != set(self.labels):
                raise AttributeError(
                    f"""
                    All the keys of parameter_values must be the same as the labels: {self.labels}.
                    However, {parameter_values.keys()} is given.
                    """
                )

            # Sort by the labels.
            parameter_values = {label: parameter_values[label] for label in self.labels}

        self._parameter_values = parameter_values

    @property
    def trainable_parameters(
        self,
    ) -> qiskit.circuit.parametertable.ParameterView | None:
        """Return the trainable parameters.

        :return qiskit.circuit.ParameterExpression | None: trainable_parameters
        """
        if self._ansatz is None:
            return None
        else:
            return self._ansatz.parameters

    @property
    def data_parameters(self) -> qiskit.circuit.parametertable.ParameterView | None:
        """Return the data parameters.

        :return qiskit.circuit.ParameterExpression | None: data_parameters
        """
        if self._feature_map is None:
            return None
        else:
            return self._feature_map.parameters

    @property
    def with_measurement(self) -> qiskit.QuantumCircuit | None:
        """Return this QuClassi circuit with measurement.

        :return qiskit.QuantumCircuit | None: this quclassi with measurement
        """
        if self._ancilla_qreg is None:
            return None

        creg = qiskit.ClassicalRegister(1, name=QuClassi.CREG_NAME)
        circuit = self.copy()
        circuit.add_register(creg)
        circuit.measure(self._ancilla_qreg, creg)

        return circuit

    def _get_transpiled(
        self,
        optimisation_level: int,
        backend: qiskit.providers.Backend,
        seed: int = 901,
    ) -> qiskit.QuantumCircuit:
        """Get a transpiled QuClassi according to the given arguments.
        The transpiled circuit is stored to the member variable to return the same query quickly.

        :param int optimisation_level: an optimisation level
        :param qiskit.providers.Backend backend: a backend
        :param int seed: a random seed, defaults to 901
        :return qiskit.QuantumCircuit: the optimised QuClassi circuit
        """
        key = (backend, optimisation_level, seed)
        if self._transpiled is not None and key in self._transpiled:
            # If the transpiled circuit has been already set, return it.
            return self._transpiled[key]

        if self._transpiled is None:
            # If this is the first time transpiling the circuit, initialise self._transpiled.
            self._transpiled = dict()

        # Transpile the circuit and store it to the member variable.
        pass_manager = qiskit.transpiler.generate_preset_pass_manager(
            optimization_level=optimisation_level,
            backend=backend,
            seed_transpiler=seed,
        )
        self._transpiled[key] = pass_manager.run(self.with_measurement)

        return self._transpiled[key]

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :return bool: if the configuration is valid
        """
        valid = True
        if self._classical_data_size is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("classical_data_size must be set.")
        if self._labels is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("labels must be set.")

        return valid

    def _reset_register(self) -> None:
        """Reset the registers accodring to the numbers of data and train qubits."""
        # Reset the registers.
        self._ancilla_qreg = qiskit.AncillaRegister(1, name="ancilla")
        self._data_qreg = qiskit.QuantumRegister(self.num_data_qubits, name="data")
        self._train_qreg = qiskit.QuantumRegister(self.num_train_qubits, name="train")
        self.qregs = [self._ancilla_qreg, self._data_qreg, self._train_qreg]

        # Reset the circuits to be merged into this QuClassi.
        self._ansatz = qiskit.QuantumCircuit(self._train_qreg, name="train")
        self._feature_map = qiskit.QuantumCircuit(self._data_qreg, name="data")

        self._transpiled = dict()
        self._is_built = False

    def _build(self) -> None:
        """Build this circuit."""
        super()._build()

        # >>> Data qubits creation >>>
        self._feature_map.compose(
            YZEncoder(data_dimension=self.using_classical_data_size),
            range(self.num_data_qubits),
            inplace=True,
        )
        # <<< Data qubits creation <<<

        # >>> Train qubits creation >>>
        # Get the qubits to be applied the layers.
        qubits_applied = list(range(self.num_train_qubits))
        # Get the qubit pairs to be applied the layers.
        qubit_applied_pairs = [
            (qubit, qubit + 1) for qubit in range(self.num_data_qubits - 1)
        ]

        # Create the train qubits part.
        for index, letter in enumerate(self.structure):
            parameter_prefix = f"layer{index}"
            # Do not need to think about what if the letter might not match either of them
            # since QuClassi checks the structure when it's set.
            match letter:
                case QuClassi.SINGLE_QUBIT_UNITARY:
                    layer = SingleQubitUnitaryLayer(
                        num_state_qubits=self.num_train_qubits,
                        qubits_applied=qubits_applied,
                        parameter_prefix=parameter_prefix,
                    )
                case QuClassi.DUAL_QUBIT_UNITARY:
                    layer = DualQubitUnitaryLayer(
                        num_state_qubits=self.num_train_qubits,
                        qubit_applied_pairs=qubit_applied_pairs,
                        parameter_prefix=parameter_prefix,
                    )
                case QuClassi.ENTANGLEMENT_UNITARY:
                    layer = EntanglementUnitaryLayer(
                        num_state_qubits=self.num_train_qubits,
                        qubit_applied_pairs=qubit_applied_pairs,
                        parameter_prefix=parameter_prefix,
                    )

            # Add the layer to the ansatz.
            self._ansatz.compose(
                layer,
                range(self.num_train_qubits),
                inplace=True,
            )
        # <<< Train qubits creation <<<

        # >>> Whole circuit creation >>>
        circuit = qiskit.QuantumCircuit(*self.qregs, name=self.name)
        circuit.compose(self._ansatz.to_gate(), self._train_qreg, inplace=True)
        circuit.compose(self._feature_map.to_gate(), self._data_qreg, inplace=True)

        # Add the swap test.
        swap_test_layer = qiskit.QuantumCircuit(*self.qregs, name="SwapTest")
        swap_test_layer.h(self._ancilla_qreg)
        for train_qubit, data_qubit in zip(self._train_qreg, self._data_qreg):
            swap_test_layer.cswap(self._ancilla_qreg[0], train_qubit, data_qubit)
        swap_test_layer.h(self._ancilla_qreg)
        circuit.compose(swap_test_layer.to_gate(), self.qubits, inplace=True)
        # <<< Whole circuit creation <<<

        self.append(circuit.to_gate(), self.qubits)

    def _get_fidelities(
        self,
        datum: list[float],
        backend: qiskit.providers.Backend,
        shots: int = 8192,
        optimisation_level: int = 2,
        seed: int = 901,
    ) -> dict[str, float]:
        """Get fidelities between the given datum and the quantum states representing each label.

        :param list[float] datum: a datum to be classified
        :param qiskit.providers.Backend backend: a backend
        :param int shots: the number of shots, defaults to 8192
        :param int optimisation_level: the level of the optimisation, defaults to 2
        :param int seed: a random seed, defaults to 901
        :raises AttributeError: if the parameter values haven't been set
        :raises ValueError: if the shape of the given datum is not the same as the using classical data size
        :return dict[str, float]: the fidelities for each label
        """
        # Raise the error if the parameter values haven't been set.
        if self.parameter_values == {}:
            error_msg = "No parameter values are found. Set the parameter values first."
            raise AttributeError(error_msg)
        # Raise the error if the data shape does not meet the using_classical_data_size.
        if len(datum) != self.using_classical_data_size:
            error_msg = f"""
            The size of given datum must be {self.using_classical_data_size}.
            However, it is {len(datum)}.
            """
            raise ValueError(error_msg)

        # Build the circuit if not yet.
        if not self._is_built:
            self._build()

        # Transplie the circuits.
        transpiled_circuit = self._get_transpiled(
            optimisation_level=optimisation_level, backend=backend, seed=seed
        )

        # Load the data to the parameters.
        data_parameters = {
            data_parameter: d for data_parameter, d in zip(self.data_parameters, datum)
        }

        # Create primitive unified blocks.
        primitive_unified_blocks = []
        for parameter_values in self.parameter_values.values():
            # Set the parameter values.
            trained_parameters = {
                trainable_parameter: parameter_value
                for trainable_parameter, parameter_value in zip(
                    self.trainable_parameters, parameter_values
                )
            }

            # Add this primitive unified block to the dict.
            primitive_unified_block = transpiled_circuit.assign_parameters(
                {**trained_parameters, **data_parameters}
            )
            primitive_unified_blocks.append(primitive_unified_block)

        # Run the circuits.
        jobs = backend.run(primitive_unified_blocks, shots=shots)

        # Get the fidelities corresponding to each label.
        fidelities = {}
        results = jobs.result().results
        for result, label in zip(results, self.labels):
            digit_result = {
                str(int(key, 16)): value for key, value in result.data.counts.items()
            }  # The keys are initially expressed in hex.
            fidelities[label] = Postprocessor.calculate_fidelity_from_swap_test(
                digit_result
            )

        return fidelities

    def _get_probabilities(
        self,
        datum: list[float],
        backend: qiskit.providers.Backend,
        shots: int = 8192,
        optimisation_level: int = 2,
        seed: int | None = 901,
    ) -> dict[str, float]:
        """Get the probabilities, calculated through the softmax function,
        of belonging the given datum to each label.

        :param list[float] datum: a datum to be classified
        :param qiskit.providers.Backend backend: a backend
        :param int shots: the number of shots, defaults to 8192
        :param int optimisation_level: the level of the optimisation, defaults to 2
        :param int | None seed: a random seed, defaults to 901
        :return str: the predicted label
        """
        # Get the fidelities.
        fidelities = self._get_fidelities(
            datum=datum,
            backend=backend,
            shots=shots,
            optimisation_level=optimisation_level,
            seed=seed,
        )

        # Make the dictionary of the fidelities a vector.
        fidelity_vector = [fidelity for fidelity in fidelities.values()]

        # Calculate the probabilities using the softmax function.
        softmax = lambda x: np.exp(x) / sum(np.exp(x))
        probability_vector = softmax(fidelity_vector)

        # For developer: The summation of the entries of the probability vector must be one.
        total_probability = np.sum(probability_vector)
        development_error = f"FOR DEVELOPER: The summation of the entries of the probability vector must be 1, but {total_probability}."
        assert np.abs(total_probability - 1) < 1e-10, development_error

        # For developer: The dimension of the probability vector must be the same as the length of the labels.
        dim_probability_vector = len(probability_vector)
        num_labels = len(self.labels)
        development_error = f"FOR DEVELOPER: The dimension of the probability vector must be the same as the length of the label, but {dim_probability_vector} vs {num_labels}."
        assert dim_probability_vector == num_labels, development_error

        # Make the probability vector a dictionary.
        probabilities = {
            label: probability
            for label, probability in zip(self.labels, probability_vector)
        }

        return probabilities

    def _classify_datum(
        self,
        datum: list[float],
        backend: qiskit.providers.Backend,
        shots: int = 8192,
        optimisation_level: int = 2,
        seed: int = 901,
    ) -> str:
        """Classify one datum and return the label name.

        :param list[float] datum: a datum to be classified
        :param qiskit.providers.Backend backend: a backend
        :param int shots: the number of shots, defaults to 8192
        :param int optimisation_level: the level of the optimisation, defaults to 2
        :param int seed: a random seed, defaults to 901
        :return str: the predicted label
        """
        probabilities = self._get_probabilities(
            datum=datum,
            backend=backend,
            shots=shots,
            optimisation_level=optimisation_level,
            seed=seed,
        )

        # Find the most likely label.
        predicted_label = max(probabilities, key=probabilities.get)

        return predicted_label

    def classify(
        self,
        data: list[list[float]],
        backend: qiskit.providers.Backend,
        shots: int = 8192,
        optimisation_level: int = 2,
        seed: int = 901,
    ) -> list[str]:
        """Classify the given data.

        :param list[list[float]] data: data to be classcified
        :param qiskit.providers.Backend backend: a backend
        :param int shots: the number of shots, defaults to 8192
        :param int optimisation_level: the level of the optimisation, defaults to 2
        :param int seed: a random seed, defaults to 901
        :return list[str]: the classified labels
        """
        # Transform the data by self._transformer.
        if self._transformer is not None:
            data = self._transformer(data)

        # Append 0 to each datum if their dimension is not the same as the using classical data size.
        if len(data[0]) != self.using_classical_data_size:
            expanded_data = np.zeros((len(data), self.using_classical_data_size))
            expanded_data[:, :-1] = data
            data = expanded_data.tolist()

        # Classify each datum.
        predicted_labels = []
        for datum in data:
            predicted_labels.append(
                self._classify_datum(
                    datum=datum,
                    backend=backend,
                    shots=shots,
                    optimisation_level=optimisation_level,
                    seed=seed,
                )
            )

        return predicted_labels

    def save(self, model_dir_path: str):
        """Save this QuClassi.

        :param str model_dir_path: a path to the output directory.
        """
        # Raise the error if the parameter values haven't been set.
        if self.parameter_values == dict():
            error_msg = "No parameter values are found. Set the parameter values first."
            raise AttributeError(error_msg)

        # Create the directory specified by the argument output_dir_path.
        os.makedirs(model_dir_path, exist_ok=True)

        # Store the necessary data to load this QuClassi.
        #  Note that, there is no need to save the circuit itself
        #  because the circuit is uniquely determined from the variables.
        quclassi_info = QuClassiInfo(
            classical_data_size=self._classical_data_size,
            structure=self._structure,
            labels=self._labels,
            initial_parameters=self._parameter_values,
            name=self.name,
        )

        # Save the information as the yaml file.
        yaml_path = os.path.join(model_dir_path, QuClassi.MODEL_FILENAME)
        with open(yaml_path, "w", encoding=QuClassi.ENCODING) as yaml_file:
            yaml.dump(
                dataclasses.asdict(quclassi_info), yaml_file, default_flow_style=False
            )

    @classmethod
    def load(cls, model_dir_path: str):
        """Load a QuClassi stored in the given directory path.

        :param str model_dir_path: a path to the input directory.
        """
        # Read the information from the yaml file.
        yaml_path = os.path.join(model_dir_path, QuClassi.MODEL_FILENAME)
        with open(yaml_path, "r", encoding=QuClassi.ENCODING) as yaml_file:
            quclassi_info = yaml.safe_load(yaml_file)

        # Laod the model.
        quclassi = cls(**quclassi_info)

        return quclassi
