from abc import ABC, abstractmethod

import qiskit

from quantum_machine_learning.layers.circuits.bases.base_layer import BaseLayer


class BaseParametrisedLayer(BaseLayer, ABC):
    """BaseParametrisedLayer abstract class."""

    def __init__(
        self,
        *regs: qiskit.QuantumRegister,
        num_state_qubits: int,
        parameter_prefix: str | None = None,
        name: str | None = None,
    ):
        """Initialise the BaseParametrisedLayer.

        :param int num_state_qubits: the number of state qubits
        :param str | None parameter_prefix: a prefix of the parameter names, defaults to None
        :param str | None name: the name of this encoder, defaults to None
        """
        self._parameter_prefix: str | None = None

        super().__init__(*regs, num_state_qubits=num_state_qubits, name=name)

        self.parameter_prefix = parameter_prefix

    @property
    def parameter_prefix(self) -> str:
        """Return the parameter prefix.
        If it is None, then return the empty string.

        :return str: a prefix of the parameter names
        """
        if self._parameter_prefix is None:
            return ""
        else:
            return self._parameter_prefix

    @parameter_prefix.setter
    def parameter_prefix(self, parameter_prefix: str) -> None:
        """Set a new prefix of the parameter names and reset the register.

        :param str | None parameter_prefix: a new prefix oh the parameter names
        """
        self._parameter_prefix = parameter_prefix
        self._reset_register()

    def _get_parameter_name(self, parameter_name: str) -> str:
        """Get the parameter name from its parametr_prefix and name.

        :param str parameter name: a parameter name
        :return str: the parameter name
        """
        if self.parameter_prefix != "":
            prefix = f"{self.parameter_prefix}_"
        else:
            prefix = ""

        return f"{prefix}{parameter_name}"

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :return bool: if the configuration is valid
        """
        valid = super()._check_configuration(
            raise_on_failure=raise_on_failure
        )  # From BaseLayer

        return valid
