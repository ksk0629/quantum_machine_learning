import itertools
import math

import qiskit
import qiskit.circuit

from quantum_machine_learning.layers.base_parametrised_layer import (
    BaseParametrisedLayer,
)


class EntanglementUnitaryLayer(BaseParametrisedLayer):
    """EntanglementUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(
        self,
        num_state_qubits: int,
        qubit_applied_pairs: list[tuple[int, int]] | None = None,
        parameter_prefix: str | None = None,
        name: str | None = "EntanglementUnitary",
    ):
        """initialise the layer.

        :param int num_state_qubits: the number of state qubits
        :param list[tuple[int, int]] | None qubit_applied_pairs: pairs of two-qubit to be applied, defaults to None
        :param str parameter_prefix | None: a prefix of the parameter names, defaults to None
        :param str | None name: the name of this encoder, defaults to "EntanglementUnitary"
        """
        self._cy_parameters: qiskit.circuit.ParameterVector | None = None
        self._cz_parameters: qiskit.circuit.ParameterVector | None = None
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
        """Return pairs of two-qubit to be applied.

        :return list[tuple[int, int]]: pairs of two-qubit to be applied
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
    def qubit_applied_pairs(
        self, qubit_applied_pairs: list[tuple[int, int]] | None
    ) -> None:
        """Set the pairs of two-qubit to be applied and reset the register and parameters.

        :param list[tuple[int, int]] | None qubit_applied_pairs: a new pairs of two-qubit to be applied
        """
        self._qubit_applied_pairs = qubit_applied_pairs
        self._reset_parameters()
        self._reset_register()

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :return bool: if the configuration is valid
        """
        valid = True
        if self.num_state_qubits == 1:
            valid = False
            if raise_on_failure:
                error_msg = f"num_state_qubits must be larger than 1, but now {self.num_state_qubits}."
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
        if self.qubit_applied_pairs == []:
            self._cy_parameters = []
            self._cz_parameters = []
        else:
            self._cy_parameters = qiskit.circuit.ParameterVector(
                parameter_name("cy"), length=len(self.qubit_applied_pairs)
            )
            self._cz_parameters = qiskit.circuit.ParameterVector(
                parameter_name("cz"), length=len(self.qubit_applied_pairs)
            )

        self._parameters = [self._cy_parameters, self._cz_parameters]

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs, name=self.name)

        # Add the encoding part: the rotation controlled Y and Z rotations.
        if self.qubit_applied_pairs is None:
            index = 0
            for i in range(self.num_state_qubits):  # type: ignore
                for j in range(i + 1, self.num_state_qubits):  # type: ignore
                    circuit.cry(self._cy_parameters[index], i, j)  # type: ignore
                    circuit.crz(self._cz_parameters[index], i, j)  # type: ignore
                    index += 1
        else:
            for index, (qubit_1, qubit_2) in enumerate(self.qubit_applied_pairs):  # type: ignore
                circuit.cry(self._cy_parameters[index], qubit_1, qubit_2)  # type: ignore
                circuit.crz(self._cz_parameters[index], qubit_1, qubit_2)  # type: ignore

        self.append(circuit.to_gate(), self.qubits)
