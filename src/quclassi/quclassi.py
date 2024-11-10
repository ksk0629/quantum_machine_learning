import os
import pickle

import numpy as np
import qiskit
from qiskit import qpy, primitives

from src.encoders.yz_encoder import YZEncoder
from src.layers.single_qubit_unitary_layer import SingleQubitUnitaryLayer
from src.layers.dual_qubit_unitary_layer import DualQubitUnitaryLayer
from src.layers.entanglement_unitary_layer import EntanglementUnitaryLayer


class QuClassi:
    """QuClassi class."""

    def __init__(self, classical_data_size: int, labels: list[str]):
        """Initialise QuClassi.

        :param int classical_data_size: data size of classical data, which not directly the same as number of qubits
        :param list[str] labels: list of labels
        """
        self.classical_data_size = classical_data_size
        self.labels = labels
        self.circuit = None
        self.trainable_parameters = None
        self.data_parameters = None
        self.trained_parameters = None

    @property
    def num_data_qubits(self) -> int:
        """Get the number of data qubits of each circuit.

        :return int: number of data qubits of each circuit.
        """
        return int(np.ceil(self.classical_data_size / 2))

    @property
    def num_train_qubits(self) -> int:
        """Get the number of train qubits of each circuit.

        :return int: number of train qubits of each circuit.
        """
        return int(np.ceil(self.classical_data_size / 2))

    @property
    def num_qubits(self) -> int:
        """Get the total number of qubits of each circuit.

        :return int: total number of qubits of each circuit
        """
        return self.num_data_qubits + self.num_train_qubits + 1

    @staticmethod
    def get_basic_info_path(model_dir_path: str):
        """Get basic_info.pkl path.

        :param str model_dir_path: path to directory
        """
        return os.path.join(model_dir_path, "basic_info.pkl")

    @staticmethod
    def get_circuit_path(model_dir_path: str):
        """Get circuit.qpy path.

        :param str model_dir_path: path to directory
        """
        return os.path.join(model_dir_path, "circuit.qpy")

    @staticmethod
    def get_trainable_parameters_path(model_dir_path: str):
        """Get trainable_parameters.pkl path.

        :param str model_dir_path: path to directory
        """
        return os.path.join(model_dir_path, "trainable_parameters.pkl")

    @staticmethod
    def get_data_parameters_path(model_dir_path: str):
        """Get data_parameters.pkl path.

        :param str model_dir_path: path to directory
        """
        return os.path.join(model_dir_path, "data_parameters.pkl")

    @staticmethod
    def get_trained_parameters_path(model_dir_path: str):
        """Get trained_parameters.pkl path.

        :param str model_dir_path: path to directory
        """
        return os.path.join(model_dir_path, "trained_parameters.pkl")

    def __call__(
        self,
        data: np.ndarray,
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 1024,
    ) -> str:
        """Call the classify function.

        :param np.ndarray data: input data, which is preprocessed if needed
        :param qiskit.primitives.BaseSamplerV1  |  qiskit.primitives.BaseSamplerV2 sampler: sampler primitives, defaults to qiskit.primitives.StatevectorSampler
        :param int shots: number of shots
        :return str: predicted label
        """
        return self.classify(data=data, sampler=sampler, shots=shots)

    def build(self, structure: str):
        """Build the circuit according to the given structure.

        :param str structure: structure of circuit
        :raises ValueError: if structure contains other than 's', 'd' or 'e'
        """
        # >>> Data qubits creation >>>
        feature_map = qiskit.QuantumCircuit(self.num_data_qubits, name="Data")
        feature_map.compose(
            YZEncoder(self.num_data_qubits)(),
            range(self.num_data_qubits),
            inplace=True,
        )
        # <<< Data qubits creation <<<

        # >>> Train qubits creation >>>
        # Get applied_qubits pattern.
        applied_qubits = list(range(self.num_train_qubits))
        # Get applied_qubit_pairs pattern.
        applied_qubit_pairs = [
            (qubit, qubit + 1) for qubit in range(self.num_data_qubits - 1)
        ]

        # Create the train qubits part.
        ansatz = qiskit.QuantumCircuit(self.num_train_qubits, name="Train")
        num_layers = 0
        for letter in structure:
            param_prefix = f"layer{num_layers}"
            match letter:
                case "s":
                    layer = SingleQubitUnitaryLayer(
                        num_qubits=self.num_train_qubits,
                        applied_qubits=applied_qubits,
                        param_prefix=param_prefix,
                    )
                case "d":
                    layer = DualQubitUnitaryLayer(
                        num_qubits=self.num_train_qubits,
                        applied_qubit_pairs=applied_qubit_pairs,
                        param_prefix=param_prefix,
                    )
                case "e":
                    layer = EntanglementUnitaryLayer(
                        num_qubits=self.num_train_qubits,
                        applied_qubit_pairs=applied_qubit_pairs,
                        param_prefix=param_prefix,
                    )
                case _:
                    # Raise ValueError if structure argument contains other than "s", "d" or "e",
                    # which stands for "SingleQubitUnitaryLayer", "DualQubitUnitaryLayer"
                    # and "EntanglementUnitaryLayer" respectively.
                    msg = f"structure argument must have only 's', 'd' or 'e', but {structure}"
                    raise ValueError(msg)
            # Add the layer to the ansatz.
            ansatz.compose(
                layer(),
                range(self.num_train_qubits),
                inplace=True,
            )
            num_layers += 1
        # <<< Train qubits creation <<<

        # >>> Whole circuit creation >>>
        circuit = qiskit.QuantumCircuit(self.num_qubits, 1, name="QuClassi")
        circuit.h(0)
        circuit.barrier()

        ansatz_qubits = range(1, self.num_train_qubits + 1)
        circuit.compose(ansatz, ansatz_qubits, inplace=True)
        self.trainable_parameters = ansatz.parameters
        circuit.barrier()

        feature_map_qubits = range(
            self.num_train_qubits + 1,
            self.num_data_qubits + self.num_train_qubits + 1,
        )
        circuit.compose(feature_map, feature_map_qubits, inplace=True)
        self.data_parameters = feature_map.parameters
        circuit.barrier()

        for ansatz_qubit, feature_map_qubit in zip(ansatz_qubits, feature_map_qubits):
            circuit.cswap(0, ansatz_qubit, feature_map_qubit)

        circuit.measure(0, 0)
        # <<< Whole circuit creation <<<

        self.circuit = circuit

    def get_fidelities(
        self,
        data: np.ndarray,
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 1024,
    ) -> dict[str, np.ndarray]:
        """Get fidelities between the input data and each representative state of self.labels.

        :param np.ndarray data: input data, which is preprocessed if needed
        :param qiskit.primitives.BaseSamplerV1  |  qiskit.primitives.BaseSamplerV2 sampler: sampler primitives, defaults to qiskit.primitives.StatevectorSampler
        :param int shots: number of shots
        :raises ValueError: if self.trained_parameters was not set.
        :return str: predicted label
        :return dict[str, np.ndarray]: fidelities
        """
        if self.trained_parameters is None:
            msg = "There is not trained_parameters set."
            raise ValueError(msg)

        # Set data as the data_parameters.
        data_parameters = {
            data_parameter: _d for data_parameter, _d in zip(self.data_parameters, data)
        }

        # Create the combination of the circuit and parameters to run the circuits.
        pubs = []
        for trained_parameters in self.trained_parameters:
            parameters = {
                trainable_parameter: trained_parameter
                for trainable_parameter, trained_parameter in zip(
                    self.trainable_parameters, trained_parameters
                )
            }
            parameters = {**parameters, **data_parameters}
            pubs.append((self.circuit, parameters))

        # Run the sampler.
        job = sampler.run(pubs, shots=shots)
        fidelities = {}
        results = job.result()
        for result, label in zip(results, self.labels):
            probability_zero = result.data.c.get_counts()["0"] / shots
            fidelity = 2 * probability_zero - 1
            if fidelity < 0:
                fidelity = 0
            fidelities[label] = fidelity

        return fidelities

    def classify(
        self,
        data: np.ndarray,
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 1024,
    ) -> str:
        """Classify the input data into one of self.labels.

        :param np.ndarray data: input data, which is preprocessed if needed
        :param qiskit.primitives.BaseSamplerV1  |  qiskit.primitives.BaseSamplerV2 sampler: sampler primitives, defaults to qiskit.primitives.StatevectorSampler
        :param int shots: number of shots
        :raises ValueError: if self.trained_parameters was not set.
        :return str: predicted label
        """
        if self.trained_parameters is None:
            msg = "There is not trained_parameters set."
            raise ValueError(msg)

        # Get the fidelities.
        fidelities = self.get_fidelities(data=data, sampler=sampler, shots=shots)
        # Find the label whose value is the maximal.
        label = max(fidelities, key=fidelities.get)

        return label

    def save(self, model_dir_path: str):
        """Save the circuit and parameters to the directory specified by the given model_dir_path.

        :param str model_dir_path: path to the output directory.
        """
        # Create the directory specified by the argument output_dir_path.
        os.makedirs(model_dir_path)

        # Save the basic information of this QuClassi.
        basic_info = {
            "classical_data_size": self.classical_data_size,
            "labels": self.labels,
        }
        basic_info_path = self.get_basic_info_path(model_dir_path)
        with open(basic_info_path, "wb") as pkl_file:
            pickle.dump(basic_info, pkl_file)

        # Save the circuit.
        circuit_path = self.get_circuit_path(model_dir_path)
        with open(circuit_path, "wb") as qpy_file:
            qpy.dump(self.circuit, qpy_file)

        # Save the parameters.
        trainable_parameters_path = self.get_trainable_parameters_path(model_dir_path)
        with open(trainable_parameters_path, "wb") as pkl_file:
            pickle.dump(self.trainable_parameters, pkl_file)
        data_parameters_path = self.get_data_parameters_path(model_dir_path)
        with open(data_parameters_path, "wb") as pkl_file:
            pickle.dump(self.data_parameters, pkl_file)
        trained_parameters_path = self.get_trained_parameters_path(model_dir_path)
        with open(trained_parameters_path, "wb") as pkl_file:
            pickle.dump(self.trained_parameters, pkl_file)

    @classmethod
    def load(cls, model_dir_path: str):
        """Load the circuit and parameters from the directory specified by the given model_dir_path.

        :param str model_dir_path: path to the input directory.
        """
        # Load the basic information.
        basic_info_path = cls.get_basic_info_path(model_dir_path)
        with open(basic_info_path, "rb") as pkl_file:
            basic_info = pickle.load(pkl_file)
        loaded_quclassi = cls(**basic_info)

        # Load the circuit.
        circuit_path = cls.get_circuit_path(model_dir_path)
        with open(circuit_path, "rb") as qpy_file:
            loaded_quclassi.circuit = qpy.load(qpy_file)[0]

        # Load the parameters.
        trainable_parameters_path = cls.get_trainable_parameters_path(model_dir_path)
        with open(trainable_parameters_path, "rb") as pkl_file:
            loaded_quclassi.trainable_parameters = pickle.load(pkl_file)
        data_parameters_path = cls.get_data_parameters_path(model_dir_path)
        with open(data_parameters_path, "rb") as pkl_file:
            loaded_quclassi.data_parameters = pickle.load(pkl_file)
        trained_parameters_path = cls.get_trained_parameters_path(model_dir_path)
        with open(trained_parameters_path, "rb") as pkl_file:
            loaded_quclassi.trained_parameters = pickle.load(pkl_file)

        return loaded_quclassi
