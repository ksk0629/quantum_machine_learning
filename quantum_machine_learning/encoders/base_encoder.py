from abc import ABC

import qiskit


class BaseEncoder(qiskit.circuit.library.BlueprintCircuit, ABC):
    """BaseEncoder class of which all encoders inherit this."""

    def __init__(self, *regs, name: str | None = None):
        """Create a new encoder.

        :param str | None name: name of this encoder, defaults to None
        """
        super().__init__(*regs, name=name)
