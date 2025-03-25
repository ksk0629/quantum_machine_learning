import qiskit

from quantum_machine_learning.layers.base_layer import BaseLayer


class SwapTestLayer(BaseLayer):
    """SwapTestLayer class"""

    def __init__(
        self,
        num_state_qubits: int,
        control_qubit: int,
        qubit_pairs: list[tuple[int, int]],
        name: str | None = "CSWAPTestLayer",
    ):
        """Initialise this layer.

        :param int num_state_qubits: the number of state qubits
        :param int control_qubit: a control qubit
        :param list[tuple[int, int]] qubit_pairs: pairs of two qubits to be taken for the swap test
        :param str | None name: the name of this encoder, defaults to "CSWAPTestLayer"
        """
        self._num_state_qubits = None
        self._control_qubit = None
        self._qubit_pairs = None

        super().__init__(num_state_qubits=num_state_qubits, name=name)

        self.num_state_qubits = num_state_qubits
        self.control_qubit = control_qubit
        self.qubit_pairs = qubit_pairs

    @property
    def control_qubit(self) -> int:
        """Return the control qubit.

        :return int: control qubit
        """
        return self._control_qubit

    @control_qubit.setter
    def control_qubit(self, control_qubit: int):
        """Set the new control qubit and reset the register.

        :param int control_qubit: a new control qubit
        """
        self._control_qubit = control_qubit
        self._reset_register()

    @property
    def qubit_pairs(self) -> list[tuple[int, int]]:
        """Return the qubit pairs to be taken for the swap test.

        :return list[tuple[int, int]]: qubit pairs to be taken for the swap test
        """
        return self._qubit_pairs

    @qubit_pairs.setter
    def qubit_pairs(self, qubit_pairs: list[tuple[int, int]]):
        """Set the new qubit pairs to be taken for the swap test and reset the register.

        :param list[tuple[int, int]] qubit_pairs: a new qubit pairs to be taken for the swap test
        """
        self._qubit_pairs = qubit_pairs
        self._reset_register()

    def _check_configuration(self, raise_on_failure=True) -> bool:
        """Check if the current configuration is valid.

        :param bool raise_on_failure: if raise an error or not, defaults to True
        :return bool: if the configuration is valid
        """
        valid = True

        return valid

    def _reset_register(self) -> None:
        """Reset the register."""
        qreg = qiskit.QuantumRegister(self.num_state_qubits)
        self.qregs = [qreg]

    def _build(self) -> None:
        """Build the circuit."""
        super()._build()

        # Make the quantum circuit.
        circuit = qiskit.QuantumCircuit(*self.qregs)

        circuit.h(self.control_qubit)
        for qubit_1, qubit_2 in self.qubit_pairs:
            circuit.cswap(self.control_qubit, qubit_1, qubit_2)
        circuit.h(self.control_qubit)

        self.append(circuit.to_gate(), self.qubits)
