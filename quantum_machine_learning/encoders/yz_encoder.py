from typing import Callable

import qiskit
import qiskit.circuit

from quantum_machine_learning.encoders.base_encoder import BaseEncoder


class YZEncoder(BaseEncoder):
    """YZEncoder class, which is used in https://arxiv.org/pdf/2103.11307."""

    def __init__(
        self,
        data_dimension: int,
        name: str = "yz_encoder",
        transformer: Callable[[list[float]], list[float]] | None = None,
    ):
        """Initialise this encoder.

        :param int data_dimension: the dimension of data
        :param str | None name: the name of this encoder, defaults to None
        :param Callable[[list[float]], list[float]] | None transformer: the data transformer, defaults to None
        """
        self._y_parameters = None
        self._z_parameters = None

        super().__init__(
            data_dimension=data_dimension, name=name, transformer=transformer
        )

    @property
    def num_encoding_qubits(self) -> int:
        """Return the number of qubits to be encoded.

        :return int: the number of encoding qubits
        """
        if self.data_dimension % 2 == 0:  # Even number
            return int(self.data_dimension // 2)
        else:  # Odd number
            return int((self.data_dimension + 1) // 2)

    @property
    def y_parameters(self) -> qiskit.circuit.ParameterVector:
        """Return the parameter vector for the Y-rotation of this circuit.

        :return qiskit.circuit.ParameterVecotr: the Y-rotation parameter vector
        """
        return self._y_parameters

    @property
    def z_parameters(self) -> qiskit.circuit.ParameterVector:
        """Return the parameter vector for the Z-rotation of this circuit.

        :return qiskit.circuit.ParameterVecotr: the Z-rotation parameter vector
        """
        return self._z_parameters

    def _check_configuration(self, raise_on_failure=True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :return bool: if the configuration is valid
        """
        valid = True

        return valid

    def _reset_register(self) -> None:
        """Reset the register."""
        qreg = qiskit.QuantumRegister(self.num_encoding_qubits)
        self.qregs = [qreg]

    def _reset_parameters(self) -> None:
        """Reset the parameter vector."""
        self._y_parameters = qiskit.circuit.ParameterVector(
            "y", length=self.num_encoding_qubits
        )
        self._z_parameters = qiskit.circuit.ParameterVector(
            "z", length=self.num_encoding_qubits
        )
        self._parameters = [self.y_parameters, self.z_parameters]

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs)

        # Add the encoding part: the rotation Y and Z.
        for index in range(self.num_encoding_qubits):
            circuit.ry(self.y_parameters[index], index)
            circuit.rz(self.z_parameters[index], index)

        self.append(circuit.to_gate(), self.qubits)
