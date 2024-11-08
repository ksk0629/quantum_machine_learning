import numpy as np
import qiskit


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
        return np.ceil(self.classical_data_size / 2)

    @property
    def num_train_qubits(self) -> int:
        """Get the number of train qubits of each circuit.

        :return int: number of train qubits of each circuit.
        """
        return np.ceil(self.classical_data_size / 2)

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

    def build_circuits(self, structure: str):
        """Build the circuit according to the given structure.

        :param str structure: structure of circuit
        """
        # Check if the structure argument contains only "s", "d" or "e",
        # which stands for "SingleQubitUnitaryLayer", "DualQubitUnitaryLayer"
        # and "EntanglementUnitaryLayer" respectively.

        # Build a circuit according to the structure argument.
        pass

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
