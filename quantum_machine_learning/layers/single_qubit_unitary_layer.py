import qiskit

from quantum_machine_learning.layers.base_parametrised_layer import (
    BaseParametrisedLayer,
)


class SingleQubitUnitaryLayer(BaseParametrisedLayer):
    """SingleQubitUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(
        self,
        num_state_qubits: int,
        qubits_applied: list[int] | None = None,
        parameter_prefix: str | None = None,
        name: str | None = "SingleQubitUnitary",
    ):
        """initialise the layer.

        :param int num_state_qubits: the number of state qubits
        :param list[int] | None qubits_applied: qubits to be applied, defaults to None
        :param str | None parameter_prefix: a prefix of the parameter names, defaults to None
        :param str | None name: the name of this encoder, defaults to "SingleQubitUnitary"
        """
        self._qubits_applied = None
        self._y_parameters = None
        self._z_parameters = None

        super().__init__(
            num_state_qubits=num_state_qubits,
            parameter_prefix=parameter_prefix,
            name=name,
        )

        self.num_state_qubits = num_state_qubits
        self.qubits_applied = qubits_applied

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

    @property
    def qubits_applied(self) -> list[tuple[int, int]] | None:
        """Return the qubits to be applied.

        :return list[tuple[int, int]] | None: pairs of two-qubit to be applied
        """
        return self._qubits_applied

    @qubits_applied.setter
    def qubits_applied(self, qubits_applied: list[int] | None):
        """Set the qubits to be applied and reset the register and parameters.

        :param list[int] | None qubits_applied: a new qubits to be applied
        """
        self._qubits_applied = qubits_applied
        self._reset_parameters()
        self._reset_register()

    def _check_configuration(self, raise_on_failure=True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :return bool: if the configuration is valid
        """
        valid = True

        return valid

    def _reset_register(self) -> None:
        """Reset the register."""
        qreg = qiskit.QuantumRegister(self.num_state_qubits)
        self.qregs = [qreg]

    def _reset_parameters(self) -> None:
        """Reset the parameter vector."""
        # Make the parameter name according to the prefix.
        if self.parameter_prefix != "":
            prefix = f"{self.parameter_prefix}_"
        else:
            prefix = ""
        parameter_name = lambda name: f"{prefix}{name}"
        # Set the parameters.
        if self.qubits_applied is None:
            self._y_parameters = qiskit.circuit.ParameterVector(
                parameter_name("y"), length=self.num_state_qubits
            )
            self._z_parameters = qiskit.circuit.ParameterVector(
                parameter_name("z"), length=self.num_state_qubits
            )
        else:
            length = len(self.qubits_applied)
            self._y_parameters = qiskit.circuit.ParameterVector(
                parameter_name("y"), length=length
            )
            self._z_parameters = qiskit.circuit.ParameterVector(
                parameter_name("z"), length=length
            )
        self._parameters = [self.y_parameters, self.z_parameters]

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs)

        # Add the encoding part: the rotation Y and Z.
        if self.qubits_applied is None:
            for index in range(self.num_state_qubits):
                circuit.ry(self.y_parameters[index], index)
                circuit.rz(self.z_parameters[index], index)
        else:
            for index, qubit in enumerate(self.qubits_applied):
                circuit.ry(self.y_parameters[index], qubit)
                circuit.rz(self.z_parameters[index], qubit)

        self.append(circuit.to_gate(), self.qubits)
