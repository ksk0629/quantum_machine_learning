import qiskit

from src.layers.base_learnable_layer import BaseLearnableLayer


class SingleQubitUnitaryLayer(BaseLearnableLayer):
    """SingleQubitUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(self, param_prefix: str):
        super().__init__(param_prefix=param_prefix)

    def __get_pattern(
        self, params: qiskit.circuit.ParameterVector
    ) -> qiskit.QuantumCircuit:
        """Return the single qubit unitary layer pattern.

        :param qiskit.circuit.ParameterVector params: parameter vector
        :return qiskit.QuantumCircuit: single qubit unitary layer pattern
        """
        pattern = qiskit.QuantumCircuit(1)
        pattern.ry(params[0], 0)
        pattern.rz(params[1], 0)

        return pattern
