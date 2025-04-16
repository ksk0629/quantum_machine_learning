import qiskit

from quantum_machine_learning.layers.circuits.base.base_parametrised_layer import (
    BaseParametrisedLayer,
)


class YAngle(BaseParametrisedLayer):
    """Y angle encoder class"""

    def __init__(self, data_dimension: int, name: str = "YAngle"):
        """Initialise this encoder.

        :param int data_dimension: the dimension of data
        :param str name: the name of the circuit
        """
        self._data_dimension: int = data_dimension

        super().__init__(num_state_qubits=self.data_dimension, name=name)

    @property
    def data_dimension(self) -> int:
        """Return the data dimension.

        :return int: the data dimension
        """
        return self._data_dimension

    @data_dimension.setter
    def data_dimension(self, data_dimension: int) -> None:
        """Set a new data dimension and num_state_qubits according to the new one.
        The setter of num_state_qubits defined in BaseLayer resets the register.

        :param int data_dimension: a new data dimension
        """
        self._data_dimension = data_dimension
        self.num_state_qubits = self._data_dimension

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :return bool: if the configuration is valid
        """
        valid = super()._check_configuration(raise_on_failure=raise_on_failure)
        return valid

    def _reset_register(self) -> None:
        """Reset the register."""
        qreg = qiskit.QuantumRegister(self.num_state_qubits)
        self.qregs = [qreg]

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs, name=self.name)

        # Add the encoding part: the X-rotation.
        num_digits = len(str(self.num_state_qubits))
        for qubit in range(self.num_state_qubits):
            parameter_index_str = str(qubit).zfill(num_digits)
            parameter = qiskit.circuit.Parameter(
                self._get_parameter_name(f"YAngle[{parameter_index_str}]")
            )
            circuit.ry(parameter, qubit)

        self.append(circuit.to_gate(), self.qubits)
