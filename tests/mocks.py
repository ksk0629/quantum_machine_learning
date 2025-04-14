import qiskit


# === BaseEncoder ===
from quantum_machine_learning_utils.encoders.base_encoder import BaseEncoder


class BaseEncoderNormalTester(BaseEncoder):
    """Mock class of a child class of BaseEncoder for normal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(self, data_dimension, name=None):
        self._num_reset_register = 0
        self._num_reset_parameters = 0

        super().__init__(data_dimension=data_dimension, name=name)

    @property
    def num_encoding_qubits(self):
        return 1

    def _check_configuration(self, raise_on_failure=True):
        valid = super()._check_configuration()
        return valid

    def _build(self):
        super()._build()

    def _reset_register(self):
        self._num_reset_register += 1

    def _reset_parameters(self):
        self._num_reset_parameters += 1


class BaseEncoderTesterWithoutNumEncodingQubits(BaseEncoder):
    """Mock class of a child class of BaseEncoder for abnormal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(self, data_dimension, name=None):
        super().__init__(data_dimension=data_dimension, name=name)

    def _check_configuration(self, raise_on_failure=True):
        pass

    def _build(self):
        pass

    def _reset_register(self):
        pass

    def _reset_parameters(self):
        pass


class BaseEncoderTesterWithoutResetRegister(BaseEncoder):
    """Mock class of a child class of BaseEncoder for normal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(self, data_dimension: int, name=None):
        super().__init__(data_dimension=data_dimension, name=name)

    @property
    def num_encoding_qubits(self):
        pass

    def _check_configuration(self, raise_on_failure=True):
        pass

    def _build(self):
        pass

    def _reset_parameters(self):
        pass


class BaseEncoderTesterWithoutResetParameters(BaseEncoder):
    """Mock class of a child class of BaseEncoder for normal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(self, data_dimension, name=None):
        super().__init__(data_dimension=data_dimension, name=name)

    @property
    def num_encoding_qubits(self):
        pass

    def _check_configuration(self, raise_on_failure=True):
        pass

    def _build(self):
        pass

    def _reset_register(self):
        pass


# === BaseLayer ===
from quantum_machine_learning_utils.layers.base_layer import BaseLayer


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
from quantum_machine_learning_utils.layers.base_parametrised_layer import (
    BaseParametrisedLayer,
)


class BaseParametrisedLayerNormalTester(BaseParametrisedLayer):
    """This is a normal test class for BaseParametrisedLayer.
    Thus, the docstrings will be omitted hereby.
    """

    def __init__(self, num_state_qubits, parameter_prefix=None, name=None):
        self._num_reset_register = 0
        self._num_reset_parameters = 0
        super().__init__(
            num_state_qubits=num_state_qubits,
            parameter_prefix=parameter_prefix,
            name=name,
        )

    def _reset_register(self):
        self._num_reset_register += 1

    def _reset_parameters(self):
        self._num_reset_parameters += 1

    def _check_configuration(self, raise_on_failure=True):
        super()._check_configuration(raise_on_failure=raise_on_failure)

    def _build(self):
        super()._build()


class BaseParametrisedLayerTesterWithoutResetParameters(BaseParametrisedLayer):
    """This is an abnormal test class for BaseParametrisedLayer without implementing _reset_parameters method.
    Thus, the docstrings will be omitted hereby.
    """

    def __init__(self, num_state_qubits, parameter_prefix=None, name=None):
        super().__init__(
            num_state_qubits=num_state_qubits,
            parameter_prefix=parameter_prefix,
            name=name,
        )

    def _reset_register(self):
        pass

    def _check_configuration(self, raise_on_failure=True):
        pass

    def _build(self):
        pass
