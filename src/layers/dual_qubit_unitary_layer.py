import qiskit

from src.layers.base_learnable_layer import BaseLearnableLayer


class DualQubitUnitaryLayer(BaseLearnableLayer):
    """DualQubitUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(self, param_prefix: str):
        super().__init__(param_prefix=param_prefix)

    def __get_pattern(
        self, params: qiskit.circuit.ParameterVector
    ) -> qiskit.QuantumCircuit:
        """Return the dual qubit unitary layer pattern.

        :param qiskit.circuit.ParameterVector params: parameter vector
        :return qiskit.QuantumCircuit: dual qubit unitary layer pattern
        """
        pattern = qiskit.QuantumCircuit(2)
        pattern.rry(params[0], 0, 1)
        pattern.rrz(params[1], 0, 1)

        return pattern
