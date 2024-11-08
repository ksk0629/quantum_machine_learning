import numpy as np
import qiskit

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

    def __call__(self, data: np.ndarray) -> str:
        """Call the classify function.

        :param np.ndarray data: input data
        :return str: predicted label
        """
        return self.classify(data=data)

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
        circuit = qiskit.QuantumCircuit(self.num_qubits, name="QuClassi")
        circuit.h(0)
        circuit.barrier()

        ansatz_qubits = range(1, self.num_train_qubits + 1)
        circuit.compose(ansatz, ansatz_qubits, inplace=True)
        circuit.barrier()

        feature_map_qubits = range(
            self.num_train_qubits + 1,
            self.num_data_qubits + self.num_train_qubits + 1,
        )
        circuit.compose(feature_map, feature_map_qubits, inplace=True)
        circuit.barrier()

        for ansatz_qubit, feature_map_qubit in zip(ansatz_qubits, feature_map_qubits):
            circuit.cswap(0, ansatz_qubit, feature_map_qubit)
        # <<< Whole circuit creation <<<

        self.circuit = circuit

    def classify(self, data: np.ndarray) -> str:
        """Classify the input data into one of self.labels.

        :param np.ndarray data: inut data
        :return str: predicted label
        """
        pass

    def save(self, model_dir_path: str):
        """Save the circuit and parameters to the directory specified by the given model_dir_path.

        :param str model_dir_path: path to the output directory.
        """
        # Create the directory specified by the argument output_dir_path.

        # Save the circuit.

        # Save the parameters.
        pass

    @classmethod
    def load(cls, model_dir_path: str):
        """Load the circuit and parameters from the directory specified by the given model_dir_path.

        :param str model_dir_path: path to the input directory.
        """
        # Load the circuit.

        # Load the parameters.
        pass
