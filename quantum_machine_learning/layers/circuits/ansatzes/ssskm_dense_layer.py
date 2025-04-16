import qiskit

from quantum_machine_learning.layers.circuits.bases.base_parametrised_layer import (
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
    def num_reputations(self, num_reputations: int | None) -> None:
        """Set a new number of reputations and reset the registers.

        :param int | None num_reputations: a new number of reputations
        """
        self._num_reputations = num_reputations
        self._reset_register()

    def _check_configuration(self, raise_on_failure=True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :raises AttributeError: if the number of state qubits is not greater than 1
        :return bool: if the configuration is valid
        """
        valid = super()._check_configuration(raise_on_failure=raise_on_failure)

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

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs, name=self.name)

        # Add RZ, RY and RZ gates to each qubit.
        parameter_index = 0
        num_digits = len(str(3 * self.num_state_qubits * self.num_reputations))
        for qubit in range(self.num_state_qubits):
            for _ in range(self.num_reputations):  # Loop for reputation
                parameter_index_str = str(parameter_index).zfill(num_digits)
                parameter = qiskit.circuit.Parameter(
                    self._get_parameter_name(f"w[{parameter_index_str}]")
                )
                circuit.rz(parameter, qubit)
                parameter_index += 1

                parameter_index_str = str(parameter_index).zfill(num_digits)
                parameter = qiskit.circuit.Parameter(
                    self._get_parameter_name(f"w[{parameter_index_str}]")
                )
                circuit.ry(parameter, qubit)
                parameter_index += 1

                parameter_index_str = str(parameter_index).zfill(num_digits)
                parameter = qiskit.circuit.Parameter(
                    self._get_parameter_name(f"w[{parameter_index_str}]")
                )
                circuit.rz(parameter, qubit)
                parameter_index += 1

        # Add CNOT gates to entangle qubits.
        for qubit in range(self.num_state_qubits):
            next_qubit = (qubit + 1) % self.num_state_qubits
            circuit.cx(qubit, next_qubit)

        self.append(circuit.to_gate(), self.qubits)
