import qiskit

from quantum_machine_learning_utils.encoders.base_encoder import BaseEncoder


class XEncoder(BaseEncoder):
    """XEncoder class"""

    def __init__(self, data_dimension: int, name: str = "XEncoder"):
        """Initialise this encoder.

        :param int data_dimension: the dimension of data
        :param str name: the name of the circuit
        """
        super().__init__(data_dimension=data_dimension, name=name)

    @property
    def num_encoding_qubits(self) -> int:
        """Return the number of qubits to be encoded.

        :return int: the number of encoding qubits
        """
        return self.data_dimension

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :return bool: if the configuration is valid
        """
        valid = super()._check_configuration(raise_on_failure=raise_on_failure)
        return valid

    def _reset_register(self) -> None:
        """Reset the register."""
        qreg = qiskit.QuantumRegister(self.num_encoding_qubits)
        self.qregs = [qreg]

    def _reset_parameters(self) -> None:
        """Reset the parameter vector."""
        self._parameters = [
            qiskit.circuit.ParameterVector("x", length=self.num_encoding_qubits)
        ]

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs, name=self.name)

        # Add the encoding part: the X-rotation.
        for index, parameter in enumerate(self.parameters[0]):  # type: ignore
            circuit.rx(parameter, index)

        self.append(circuit.to_gate(), self.qubits)
