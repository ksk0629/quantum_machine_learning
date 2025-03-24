from abc import ABC, abstractmethod
from typing import Callable

import qiskit


class BaseEncoder(qiskit.circuit.library.BlueprintCircuit, ABC):
    """BaseEncoder class of which all encoders inherit this."""

    def __init__(
        self,
        *regs,
        data_dimension: int,
        name: str | None = None,
        transformer: Callable[[list[float]], list[float]] | None = None
    ):
        """Create a new encoder.

        :param int data_dimension: the dimension of data
        :param str | None name: the name of this encoder, defaults to None
        :param Callable[[list[float]], list[float]] | None transformer: the data transformer, defaults to None
        """
        super().__init__(*regs, name=name)

        self._parameters = None
        self._transformer = transformer
        self._data_dimension = None
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
    def transformer(self) -> Callable[[list[float]], list[float]] | None:
        """Return the transformer.

        :return Callable[[list[float]], list[float]] | None: transformer
        """
        return self._transformer

    @transformer.setter
    def transformer(
        self, transformer: Callable[[list[float]], list[float]] | None
    ) -> None:
        """Set the transfromer.

        :param Callable[[list[float]], list[float]] | None transformer: transformer
        """
        self._transformer = transformer

    @property
    @abstractmethod
    def num_encoding_qubits(self) -> int:
        """Return the number of qubits to be encoded.

        :return int: the number of encoding qubits
        """
        pass

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

    def transform(self, data: list[float]) -> list[float]:
        """Transform the given data using self.__transformer.

        :param list[float] data: data to be transformed
        :return list[float]: transformed data
        """
        if self._transformer is None:
            return data
        else:
            return self._transformer(data)

    @abstractmethod
    def _reset_register(self) -> None:
        """Reset the register."""
        pass

    @abstractmethod
    def _reset_parameters(self) -> None:
        """Reset the parameter vector, which is self._parameters"""
        pass
