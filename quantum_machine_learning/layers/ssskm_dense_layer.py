import qiskit

from quantum_machine_learning.layers.base_parametrised_layer import (
    BaseParametrisedLayer,
)


class SSSKMDenseLayer(BaseParametrisedLayer):
    """Quantum dense layer class, suggested in https://iopscience.iop.org/article/10.1088/2632-2153/ad2aef"""

    def __init__(
        self,
        num_state_qubits: int,
        num_reputations: int,
        parameter_prefix: str | None = None,
        name: str | None = "SSSKMDenseLayer",
    ):
        """initialise the layer.

        :param int num_state_qubits: the number of state qubits
        :param int num_reputations: the number of reputations
        :param str | None parameter_prefix: a prefix of the parameter names, defaults to None
        :param str | None name: the name of this encoder, defaults to "SSSKMDenseLayer"
        """
        self._num_reputations: int | None = None

        super().__init__(
            num_state_qubits=num_state_qubits,
            parameter_prefix=parameter_prefix,
            name=name,
        )

        self.num_reputations = num_reputations

    @property
    def num_reputations(self) -> int:
        """Return the number of reputations.

        :return int: the number of reputations
        """
        if self._num_reputations is None:
            return 0
        else:
            return self._num_reputations

    @num_reputations.setter
    def num_reputations(self, num_reputations: int) -> None:
        """Set a new number of reputations and reset parameters and registers.

        :param int num_reputations: a new number of reputations
        """
        self._num_reputations = num_reputations
        self._reset_parameters()
        self._reset_register()

    def _check_configuration(self, raise_on_failure=True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :raises AttributeError: if the number of reputations is non-positive
        :raises AttributeError: if the number of state qubits is not greater than 1
        :return bool: if the configuration is valid
        """
        valid = super()._check_configuration(raise_on_failure=raise_on_failure)

        if self.num_reputations == 0:
            valid = False
            if raise_on_failure:
                error_msg = f"The number of reputations must be positive, but {self.num_reputations}."
                raise AttributeError(error_msg)
        if self.num_state_qubits == 1:
            valid = False
            if raise_on_failure:
                error_msg = f"The number of state qubits must be greater than 1, but {self.num_state_qubits}."
                raise AttributeError(error_msg)

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
        length = len(self.num_state_qubits) * (
            3 * self.num_reputations
        )  # [RZ*RY*RZ] * reputations
        parameters = qiskit.circuit.ParameterVector(parameter_name("w"), length=length)
        self._parameters = [parameters]

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs, name=self.name)

        # Add RZ, RY and RZ gates to each qubit.
        for qubit in range(self.num_state_qubits):

            for reputation_index in range(self.num_reputations):  # Loop for reputation
                for parameter_basic_index in range(3):  # RY * RZ * RY
                    parameter_index = parameter_basic_index + (3 * reputation_index)
                    parameter = self.parameters[0][parameter_index]

                    circuit.rz(parameter, qubit)
                    circuit.ry(parameter, qubit)
                    circuit.rz(parameter, qubit)

        # Add CNOT gates to entangle qubits.
        for qubit in range(self.num_state_qubits):
            next_qubit = (qubit + 1) % self.num_state_qubits
            circuit.cx(qubit, next_qubit)

        self.append(circuit.to_gate(), self.qubits)
