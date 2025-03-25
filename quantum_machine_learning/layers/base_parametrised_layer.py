from abc import ABC, abstractmethod

import qiskit

from quantum_machine_learning.layers.base_layer import BaseLayer


class BaseParametrisedLayer(BaseLayer, ABC):
    """BaseParametrisedLayer abstract class."""

    def __init__(
        self,
        *regs,
        num_state_qubits: int,
        parameter_prefix: str | None = None,
        name: str | None = None
    ):
        """Initialise the BaseLayer.

        :param int num_state_qubits: the number of state qubits
        :param str parameter_prefix | None: a prefix of the parameter names, defaults to None
        :param str | None name: the name of this encoder, defaults to None
        """
        self._parameter_prefix = None

        super().__init__(*regs, num_state_qubits=num_state_qubits, name=name)
        self.parameter_prefix = parameter_prefix

    @property
    def num_state_qubits(self) -> int:
        """Return the number of the state qubits

        :return int: the number of the state qubits
        """
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: int) -> None:
        """Set the new number of the state qubits and reset the register and parameters.

        :param int num_state_qubits: the new number of of the state qubits
        """
        self._num_state_qubits = num_state_qubits
        self._reset_register()
        self._reset_parameters()

    @property
    def prameter_prefix(self) -> str:
        """Return the parameter prefix.
        If it is None, then return the empty string.

        :return str: a prefix of the parameter names
        """
        if self._parameter_prefix is None:
            return ""
        else:
            return self._parameter_prefix

    @prameter_prefix.setter
    def parameter_prefix(self, parameter_prefix: str) -> None:
        """Set a new prefix of the parameter names and reset the parameters and register.

        :param str parameter_prefix: a new prefix oh the parameter names
        """
        self._parameter_prefix = parameter_prefix
        self._reset_parameters()
        self._reset_register()

    @property
    def parameters(self) -> list[qiskit.circuit.ParameterVector]:
        """Return the parameter vector of this circuit.

        :return list[qiskit.circuit.ParameterVecotr]: the parameter vector
        """
        return self._parameters

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters.

        :return int: the number of parameters
        """
        if self._parameters is None:
            return 0
        else:
            num_parameters = sum(
                [len(parameter_vector) for parameter_vector in self._parameters]
            )
            return num_parameters

    @abstractmethod
    def _reset_parameters(self) -> None:
        """Reset the parameter vector, which is self._parameters"""
        pass
