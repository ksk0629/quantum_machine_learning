from abc import ABC, abstractmethod

import qiskit


class BaseLayer(qiskit.circuit.library.BlueprintCircuit, ABC):
    """BaseLayer abstract class, which is all quantum layers inherit this."""

    def __init__(
        self,
        *regs: qiskit.QuantumRegister,
        num_state_qubits: int,
        name: str | None = None,
    ):
        """Initialise the BaseLayer.

        :param int num_state_qubits: the number of state qubits
        :param str | None name: the name of this encoder, defaults to None
        """
        self._num_state_qubits: int | None = None

        super().__init__(*regs, name=name)

        self.num_state_qubits = num_state_qubits

    @property
    def num_state_qubits(self) -> int:
        """Return the number of the state qubits

        :return int: the number of the state qubits
        """
        if self._num_state_qubits is None:
            return 0
        else:
            return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: int) -> None:
        """Set the new number of the state qubits and reset the register.

        :param int num_state_qubits: the new number of of the state qubits
        """
        self._num_state_qubits = num_state_qubits
        self._reset_register()

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :raises AttributeError: if its num_state_qubits is non-positive.
        :return bool: if the configuration is valid
        """
        valid = True

        if self.num_state_qubits <= 0:
            valid = False
            if raise_on_failure:
                error_msg = f"The num_state_qubits must be positive integer, but {self.num_state_qubits}."
                raise AttributeError(error_msg)

        return valid

    @abstractmethod
    def _reset_register(self) -> None:
        """Reset the register."""
        pass
