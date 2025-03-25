from abc import ABC, abstractmethod

import qiskit


class BaseLayer(qiskit.circuit.library.BlueprintCircuit, ABC):
    """BaseLayer abstract class, which is all quantum layers inherit this."""

    def __init__(self, *regs, num_state_qubits: int, name: str | None = None):
        """Initialise the BaseLayer.

        :param int num_state_qubits: the number of state qubits
        :param str | None name: the name of this encoder, defaults to None
        """
        self._num_state_qubits = None

        super().__init__(*regs, name=name)

        self.num_state_qubits = num_state_qubits

    @property
    def num_state_qubits(self) -> int:
        """Return the number of the state qubits

        :return int: the number of the state qubits
        """
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: int) -> None:
        """Set the new number of the state qubits and reset the register.

        :param int num_state_qubits: the new number of of the state qubits
        """
        self._num_state_qubits = num_state_qubits
        self._reset_register()

    @abstractmethod
    def _reset_register(self) -> None:
        """Reset the register."""
        pass
