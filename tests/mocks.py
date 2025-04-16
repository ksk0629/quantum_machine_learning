import qiskit


# === BaseLayer ===
from quantum_machine_learning.layers.circuits.bases.base_layer import BaseLayer


class BaseLayerNormalTester(BaseLayer):
    """This is a normal test class for BaseLayer.
    Thus, the docstrings will be omitted hereby.
    """

    def __init__(self, num_state_qubits, name=None):
        self._num_reset_registers = 0
        super().__init__(num_state_qubits=num_state_qubits, name=name)

    def _reset_register(self):
        self._num_reset_registers += 1

    def _check_configuration(self, raise_on_failure=True):
        valid = super()._check_configuration(raise_on_failure=raise_on_failure)

    def _build(self):
        super()._build()
        pass


class BaseLayerTesterWithoutResetRegister(BaseLayer):
    """This is an abnormal test class for BaseLayer without implementing _reset_register method.
    Thus, the docstrings will be omitted hereby.
    """

    def __init__(self, num_state_qubits, name=None):
        super().__init__(num_state_qubits=num_state_qubits, name=name)

    def _check_configuration(self, raise_on_failure=True):
        pass

    def _build(self):
        pass


# === BaseParametrisedLayer ===
from quantum_machine_learning.layers.circuits.bases.base_parametrised_layer import (
    BaseParametrisedLayer,
)


class BaseParametrisedLayerNormalTester(BaseParametrisedLayer):
    """This is a normal test class for BaseParametrisedLayer.
    Thus, the docstrings will be omitted hereby.
    """

    def __init__(self, num_state_qubits, parameter_prefix=None, name=None):
        self._num_reset_register = 0
        super().__init__(
            num_state_qubits=num_state_qubits,
            parameter_prefix=parameter_prefix,
            name=name,
        )

    def _reset_register(self):
        self._num_reset_register += 1

    def _check_configuration(self, raise_on_failure=True):
        super()._check_configuration(raise_on_failure=raise_on_failure)

    def _build(self):
        super()._build()
