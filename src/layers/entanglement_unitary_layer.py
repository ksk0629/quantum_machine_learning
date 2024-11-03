import qiskit

from src.layers.base_learnable_layer import BaseLearnableLayer


class EntanglementUnitaryLayer(BaseLearnableLayer):
    """EntanglementUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(self, param_prefix: str):
        super().__init__(param_prefix=param_prefix)

    def __get_pattern(
        self, params: qiskit.circuit.ParameterVector
    ) -> qiskit.QuantumCircuit:
        """Return the entanglement unitary layer pattern.

        :param qiskit.circuit.ParameterVector params: parameter vector
        :return qiskit.QuantumCircuit: entanglement unitary layer pattern
        """
        pattern = qiskit.QuantumCircuit(2)
        pattern.cry(params[0], 0, 1)
        pattern.crz(params[1], 0, 1)

        return pattern
