from abc import ABC, abstractmethod
from typing import Callable

import qiskit


class BaseEncoder(qiskit.circuit.library.BlueprintCircuit, ABC):
    """BaseEncoder class of which all encoders inherit this."""

    def __init__(
        self,
        *regs: qiskit.QuantumCircuit,
        data_dimension: int,
        name: str | None = None,
    ):
        """Create a new encoder.

        :param int data_dimension: the dimension of data
        :param str | None name: the name of this encoder, defaults to None
        """
        super().__init__(*regs, name=name)

        self._parameters: list[qiskit.circuit.ParameterVector] | None = None
        self._data_dimension: int | None = None

        self.data_dimension = data_dimension

    @property
    def data_dimension(self) -> int:
        """Return the dimension of data.

        :return int: the dimension of data
        """
        if self._data_dimension is None:
            return 0
        else:
            return self._data_dimension

    @data_dimension.setter
    def data_dimension(self, data_dimension: int) -> None:
        """Set the new dimension of data and reset the register and parameters.

        :param int data_dimension: the dimension of data
        """
        self._data_dimension = data_dimension
        self._reset_register()
        self._reset_parameters()

    @property
    @abstractmethod
    def num_encoding_qubits(self) -> int | None:
        """Return the number of qubits to be encoded.

        :return int | None: the number of encoding qubits
        """
        pass

    @property
    def parameters(self) -> list[qiskit.circuit.ParameterVector] | None:
        """Return the parameter vector of this circuit.

        :return list[qiskit.circuit.ParameterVecotr] | None: the parameter vector
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

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :raises AttributeError: if its data_dimension is non-positive.
        :raises AttributeError: if its num_parameters is zero.
        :return bool: if the configuration is valid
        """
        valid = True

        if self.data_dimension <= 0:
            valid = False
            if raise_on_failure:
                error_msg = f"The data_dimension must be positive integer, but {self.data_dimension}."
                raise AttributeError(error_msg)

        if self.num_parameters == 0:
            valid = False
            if raise_on_failure:
                error_msg = f"The number of parameters must be positive integer, but {self.num_parameters}."
                raise AttributeError(error_msg)

        return valid

    @abstractmethod
    def _reset_register(self) -> None:
        """Reset the register."""
        pass

    @abstractmethod
    def _reset_parameters(self) -> None:
        """Reset the parameter vector, which is self._parameters"""
        pass
