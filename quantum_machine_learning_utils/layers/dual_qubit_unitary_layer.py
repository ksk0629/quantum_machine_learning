import itertools

import qiskit

from quantum_machine_learning_utils.layers.base_parametrised_layer import (
    BaseParametrisedLayer,
)


class DualQubitUnitaryLayer(BaseParametrisedLayer):
    """DualQubitUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(
        self,
        num_state_qubits: int,
        qubit_applied_pairs: list[tuple[int, int]] | None = None,
        parameter_prefix: str | None = None,
        name: str | None = "DualQubitUnitary",
    ):
        """initialise the layer.

        :param int num_state_qubits: the number of state qubits
        :param list[tuple[int, int]] | None qubit_applied_pairs: pairs of two-qubit to be applied, defaults to None
        :param str | None parameter_prefix: a prefix of the parameter names, defaults to None
        :param str | None name: the name of this encoder, defaults to "DualQubitUnitary"
        """
        self._yy_parameters: qiskit.circuit.ParameterVector | None = None
        self._zz_parameters: qiskit.circuit.ParameterVector | None = None
        self._qubit_applied_pairs: list[tuple[int, int]] | None = None

        super().__init__(
            num_state_qubits=num_state_qubits,
            parameter_prefix=parameter_prefix,
            name=name,
        )

        self.num_state_qubits = num_state_qubits
        self.qubit_applied_pairs = qubit_applied_pairs

    @property
    def qubit_applied_pairs(self) -> list[tuple[int, int]]:
        """Return pairs of two qubits to be applied.

        :return list[tuple[int, int]]: pairs of two qubits to be applied
        """
        if self._qubit_applied_pairs is None:
            if self.num_state_qubits == 0 or self.num_state_qubits == 1:
                # If no qubits or only one qubit is there, a list of the pairs of two qubits should be empty.
                return []
            else:
                # If there are multiple qubits, return all the combinations.
                qubits = range(self.num_state_qubits)
                all_combinations = list(itertools.combinations(qubits, 2))
                return all_combinations
        else:
            return self._qubit_applied_pairs

    @qubit_applied_pairs.setter
    def qubit_applied_pairs(self, qubit_applied_pairs: list[tuple[int, int]]) -> None:
        """Set the pairs of two-qubit to be applied and reset the register and parameters.

        :param list[tuple[int, int]] qubit_applied_pairs: a new pairs of two-qubit to be applied
        """
        self._qubit_applied_pairs = qubit_applied_pairs
        self._reset_parameters()
        self._reset_register()

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

    def _reset_parameters(self) -> None:
        """Reset the parameter vector."""
        # Make the parameter name according to the prefix.
        if self.parameter_prefix != "":
            prefix = f"{self.parameter_prefix}_"
        else:
            prefix = ""
        parameter_name = lambda name: f"{prefix}{name}"
        # Set the parameters.
        if self.qubit_applied_pairs == []:
            self._yy_parameters = []
            self._zz_parameters = []
        else:
            self._yy_parameters = qiskit.circuit.ParameterVector(
                parameter_name("yy"), length=len(self.qubit_applied_pairs)
            )
            self._zz_parameters = qiskit.circuit.ParameterVector(
                parameter_name("zz"), length=len(self.qubit_applied_pairs)
            )

        self._parameters = [self._yy_parameters, self._zz_parameters]

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs, name=self.name)

        # Add the encoding part: the rotation Y and Z.
        for index, (qubit_1, qubit_2) in enumerate(self.qubit_applied_pairs):
            circuit.ryy(self._yy_parameters[index], qubit_1, qubit_2)
            circuit.rzz(self._zz_parameters[index], qubit_1, qubit_2)

        self.append(circuit.to_gate(), self.qubits)
