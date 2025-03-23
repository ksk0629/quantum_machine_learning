import qiskit

from quantum_machine_learning.encoders.base_encoder import BaseEncoder


class XEncoder(BaseEncoder):
    """XEncoder class"""

    def __init__(self, data_dimension: int, name: str = "x_encoder"):
        """Initialise this encoder.

        :param int data_dimension: the dimension of data
        :param str name: the name of the circuit
        """
        super().__init__(name=name)

        self._data_dimension = None
        self.__parameters = None
        self.data_dimension = data_dimension

    @property
    def data_dimension(self) -> int:
        """Return the dimension of data.

        :return int: the dimension of data
        """
        return self._data_dimension

    @data_dimension.setter
    def data_dimension(self, data_dimension: int):
        """Set the new dimension of data and reset the register and parameters.

        :param int data_dimension: the dimension of data
        """
        self._data_dimension = data_dimension
        self._reset_register()
        self._reset_parameters()

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters, which is the same as the number of qubits.

        :return int: the number of the parameters
        """
        return self.data_dimension

    @property
    def parameters(self) -> qiskit.circuit.ParameterVector:
        """Return the parameter vector of this circuit.

        :return qiskit.circuit.ParameterVecotr: the parameter vector
        """
        return self.__parameters

    def _check_configuration(self, raise_on_failure=True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :return bool: if the configuration is valid
        """
        valid = True

        return valid

    def _reset_register(self) -> None:
        """Reset the register."""
        qreg = qiskit.QuantumRegister(self.num_parameters)
        self.qregs = [qreg]

    def _reset_parameters(self) -> None:
        """Reset the parameter vector."""
        self.__parameters = qiskit.circuit.ParameterVector(
            "x", length=self.num_parameters
        )

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs)

        # Add the encoding part: the rotation X.
        for index, parameter in enumerate(self.__parameters):
            circuit.rx(parameter, index)

        self.append(circuit.to_gate(), self.qubits)
