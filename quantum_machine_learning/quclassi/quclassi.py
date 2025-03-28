import dataclasses
import os
import pickle
from typing import Final

import numpy as np
import qiskit
from qiskit import qpy, primitives
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
from quantum_machine_learning.layers.swap_test_layer import SwapTestLayer
from quantum_machine_learning.path_getter.quclassi_path_getter import QuClassiPathGetter
from quantum_machine_learning.postprocessor.postprocessor import Postprocessor
from quantum_machine_learning.utils.circuit_utils import CircuitUtils


@dataclasses.dataclass
class QuClassiInfo:
    classical_data_size: int
    structure: str
    labels: list[str]
    initial_parameters: dict[str, list[float]] | None = None
    name: str | None = None


class QuClassi(qiskit.circuit.library.BlueprintCircuit):
    """QuClassi class."""

    # Define the constants used in this class.
    single_qubit_unitary: Final[str] = "s"
    dual_qubit_unitary: Final[str] = "d"
    entanglement_unitary: Final[str] = "e"
    acceptable_structure: Final[tuple[str, str, str]] = [
        single_qubit_unitary,
        dual_qubit_unitary,
        entanglement_unitary,
    ]
    # Define the classical register name.
    creg_name: Final[str] = "creg"
    # Define the model file name to save and load.
    model_filename: Final[str] = "model.yaml"
    # Define the encoding method to save and load.
    encoding: Final[str] = "utf-8"

    def __init__(
        self,
        classical_data_size: int,
        structure: str,
        labels: list[str],
        initial_parameters: dict[str, list[float]] | None = None,
        name: str | None = "QuClassi",
    ):
        """Initialise QuClassi.

        :param int classical_data_size: the classical data size to be fed into QuClassi
        :param str structure: the strucutre of QuClassi
        :param list[str] labels: the labels to be classified
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

        super().__init__(name=name)  # For BlueprintCircuit

        self.classical_data_size = classical_data_size
        self.structure = structure
        self.labels = labels
        self.parameter_values = initial_parameters

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
    def structure(self, structure: str) -> None:
        """Set the new structure if it's valid and reset the registers.

        :param str structure: a new structure
        :raises ValueError: if a new structure is not constructed with acceptable letters
        """
        valid = all(s in QuClassi.acceptable_structure for s in structure)
        if not valid:
            error_msg = f"""
            A given structure must be constructed with only {QuClassi.acceptable_structure}.
            However, it is '{structure}'.
            """
            raise ValueError(error_msg)

        self._structure = structure

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
        if self.labels is None:
            raise AttributeError(
                "In order to set parameter_values, the labels must be previously set."
            )

        if parameter_values is not None:
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
    def trainable_parameters(self) -> qiskit.circuit.ParameterExpression | None:
        """Return the trainable parameters.

        :return qiskit.circuit.ParameterExpression | None: trainable_parameters
        """
        if self._ansatz is None:
            return None
        else:
            return self._ansatz.parameters

    @property
    def data_parameters(self) -> qiskit.circuit.ParameterExpression | None:
        """Return the data parameters.

        :return qiskit.circuit.ParameterExpression | None: data_parameters
        """
        if self._feature_map is None:
            return None
        else:
            return self._feature_map.parameters

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

    def _build(self) -> None:
        """Build this circuit."""
        super()._build()

        # >>> Data qubits creation >>>
        feature_map = self._feature_map.compose(
            YZEncoder(data_dimension=self.using_classical_data_size),
            range(self.num_data_qubits),
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
        ansatz = self._ansatz.copy()
        for index, letter in enumerate(self.structure):
            parameter_prefix = f"layer{index}"
            # Do not need to think about what if the letter might not match either of them
            # since QuClassi checks the structure when it's set.
            match letter:
                case QuClassi.single_qubit_unitary:
                    layer = SingleQubitUnitaryLayer(
                        num_state_qubits=self.num_train_qubits,
                        qubits_applied=qubits_applied,
                        parameter_prefix=parameter_prefix,
                    )
                case QuClassi.dual_qubit_unitary:
                    layer = DualQubitUnitaryLayer(
                        num_state_qubits=self.num_train_qubits,
                        qubit_applied_pairs=qubit_applied_pairs,
                        parameter_prefix=parameter_prefix,
                    )
                case QuClassi.entanglement_unitary:
                    layer = EntanglementUnitaryLayer(
                        num_state_qubits=self.num_train_qubits,
                        qubit_applied_pairs=qubit_applied_pairs,
                        parameter_prefix=parameter_prefix,
                    )

            # Add the layer to the ansatz.
            ansatz.compose(
                layer,
                range(self.num_train_qubits),
                inplace=True,
            )
        # <<< Train qubits creation <<<

        # >>> Whole circuit creation >>>
        circuit = qiskit.QuantumCircuit(*self.qregs)
        circuit.compose(ansatz.to_gate(), self._train_qreg, inplace=True)
        circuit.compose(feature_map.to_gate(), self._data_qreg, inplace=True)

        # Add the swap test.
        swap_test_layer = qiskit.QuantumCircuit(*self.qregs, name="SwapTest")
        swap_test_layer.h(self._ancilla_qreg)
        for train_qubit, data_qubit in zip(self._train_qreg, self._data_qreg):
            swap_test_layer.cswap(self._ancilla_qreg[0], train_qubit, data_qubit)
        swap_test_layer.h(self._ancilla_qreg)
        circuit.compose(swap_test_layer.to_gate(), self.qubits, inplace=True)
        # <<< Whole circuit creation <<<

        self.append(circuit.to_gate(), self.qubits)

    def _classify_datum(
        self,
        datum: list[float],
        backend: qiskit.providers.Backend,
        shots: int = 9010,
    ) -> str:
        """Classify one datum and return the label name.

        :param list[float] datum: a datum to be classified
        :param qiskit.providers.Backend backend: a backend
        :param int shots: the number of shots, defaults to 9010
        :raises AttributeError: if the parameter values haven't been set
        :return str: the predicted label
        """
        # Raise the error if the parameter values haven't been set.
        if self.parameter_values is None:
            error_msg = "No parameter values are found. Set the parameter values first."
            raise AttributeError(error_msg)

        # Build the circuit if not yet.
        if not self._is_build:
            self._build()

        # Create a circuit with measurement.
        creg = qiskit.ClassicalRegister(1, name=QuClassi.creg_name)
        circuit = self.copy()
        circuit.add_register(creg)
        circuit.measure(self._ancilla_qreg, creg)

        # Load the data to the parameters.
        data_parameters = {
            data_parameter: d for data_parameter, d in zip(self.data_parameters, datum)
        }

        # Create primitive unified blocks.
        #  [0] will correspond to the first label
        #  ...
        #  [n] will correspond to the n-1 label ...
        primitive_unified_blocks = []  # [0] -> label
        for parameter_values in self.parameter_values.values():
            # Set the parameter values.
            trained_parameters = {
                trainable_parameter: parameter_value
                for trainable_parameter, parameter_value in zip(
                    self.trainable_parameters, parameter_values
                )
            }
            # Add this primitive unified block to the dict.
            primitive_unified_block = (
                circuit,
                {**trained_parameters, **data_parameters},
            )
            primitive_unified_blocks.append(primitive_unified_block)

        # Transplie the backend.
        pass_manager = qiskit.transpiler.generate_preset_pass_manager(
            optimization_level=3, backend=backend
        )
        # Run the circuits.
        jobs = pass_manager.run(primitive_unified_blocks, shots=shots)

        # Get the fidelities corresponding to each label.
        fidelities = {}
        results = jobs.result()
        for result, label in zip(results, self.labels):
            fidelities[label] = Postprocessor.calculate_fidelity_from_swap_test(
                result.data.creg.get_counts()  # creg is after QuClassi.creg_name
            )

        # Find the label whose value is the larest fidelity.
        predicted_label = max(fidelities, key=fidelities.get)

        return predicted_label

    def classify(
        self,
        data: list[list[float]],
        backend: qiskit.providers.Backend,
        shots: int = 9010,
    ) -> list[str]:
        """Classify the given data.

        :param list[list[float]] data: data to be classcified
        :param qiskit.providers.Backend backend: a backend
        :param int shots: the number of shots, defaults to 9010
        :return list[str]: the classified labels
        """
        # Classify each datum.
        predicted_labels = []
        for datum in data:
            predicted_labels.append(
                self._classify_datum(datum=datum, backend=backend, shots=shots)
            )

        return predicted_labels
