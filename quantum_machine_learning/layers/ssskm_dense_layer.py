import qiskit

from quantum_machine_learning.layers.base_parametrised_layer import (
    BaseParametrisedLayer,
)


class SSSKMDenseLayer(BaseParametrisedLayer):
    """Quantum dense layer class, suggested in https://iopscience.iop.org/article/10.1088/2632-2153/ad2aef"""

    def __init__(
        self,
        num_state_qubits: int,
        parameter_prefix: str | None = None,
        name: str | None = "SSSKMDenseLayer",
    ):
        """initialise the layer.

        :param int num_state_qubits: the number of state qubits
        :param str | None parameter_prefix: a prefix of the parameter names, defaults to None
        :param str | None name: the name of this encoder, defaults to "SSSKMDenseLayer"
        """
        super().__init__(
            num_state_qubits=num_state_qubits,
            parameter_prefix=parameter_prefix,
            name=name,
        )

    def _check_configuration(self, raise_on_failure=True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :return bool: if the configuration is valid
        """
        valid = super()._check_configuration(raise_on_failure=raise_on_failure)
        return valid

    def _reset_register(self) -> None:
        """Reset the register."""
        pass
        # qreg = qiskit.QuantumRegister(self.num_state_qubits)
        # self.qregs = [qreg]

    def _reset_parameters(self) -> None:
        """Reset the parameter vector."""
        # Make the parameter name according to the prefix.
        pass
        # if self.parameter_prefix != "":
        #     prefix = f"{self.parameter_prefix}_"
        # else:
        #     prefix = ""
        # parameter_name = lambda name: f"{prefix}{name}"
        # # Set the parameters.
        # length = len(self.qubits_applied)
        # self._y_parameters = qiskit.circuit.ParameterVector(
        #     parameter_name("y"), length=length
        # )
        # self._z_parameters = qiskit.circuit.ParameterVector(
        #     parameter_name("z"), length=length
        # )
        # self._parameters = [self._y_parameters, self._z_parameters]

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()
