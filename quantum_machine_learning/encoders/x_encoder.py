from typing import Callable

import qiskit

from quantum_machine_learning.encoders.base_encoder import BaseEncoder


class XEncoder(BaseEncoder):
    """XEncoder class"""

    def __init__(
        self,
        data_dimension: int,
        name: str = "x_encoder",
        transformer: Callable[[list[float]], list[float]] | None = None,
    ):
        """Initialise this encoder.

        :param int data_dimension: the dimension of data
        :param str name: the name of the circuit
        :param Callable[[list[float]], list[float]] | None transformer: the data transformer, defaults to None
        """
        super().__init__(
            data_dimension=data_dimension, name=name, transformer=transformer
        )

    @property
    def num_encoding_qubits(self) -> int | None:
        """Return the number of qubits to be encoded.

        :return int | None: the number of encoding qubits
        """
        return self.data_dimension

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
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
        self._parameters = [
            qiskit.circuit.ParameterVector("x", length=self.num_encoding_qubits)
        ]

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs)

        # Add the encoding part: the X-rotation.
        for index, parameter in enumerate(self.parameters[0]):  # type: ignore
            circuit.rx(parameter, index)

        self.append(circuit.to_gate(), self.qubits)
