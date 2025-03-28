import qiskit


# === BaseEncoder ===
from quantum_machine_learning.encoders.base_encoder import BaseEncoder


class BaseEncoderNormalTester(BaseEncoder):
    """Mock class of a child class of BaseEncoder for normal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(self, data_dimension, name=None, transformer=None):
        super().__init__(
            data_dimension=data_dimension, name=name, transformer=transformer
        )

    @property
    def num_encoding_qubits(self):
        return 1

    def _check_configuration(self, raise_on_failure=True):
        return True

    def _build(self):
        super()._build()
        circuit = qiskit.QuantumCircuit(1)
        self.append(circuit.to_gate(), self.qubits)

    def _reset_register(self):
        qreg = qiskit.QuantumRegister(1)
        self.qregs = [qreg]

    def _reset_parameters(self):
        self._parameters = [qiskit.circuit.ParameterVector("test_parameter", length=1)]


class BaseEncoderTesterWithoutNumEncodingQubits(BaseEncoder):
    """Mock class of a child class of BaseEncoder for abnormal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(self, data_dimension, name=None, transformer=None):
        super().__init__(
            data_dimension=data_dimension, name=name, transformer=transformer
        )

    def _check_configuration(self, raise_on_failure=True):
        return True

    def _build(self):
        super()._build()
        circuit = qiskit.QuantumCircuit(1)
        self.append(circuit.to_gate(), self.qubits)

    def _reset_register(self):
        qreg = qiskit.QuantumRegister(1)
        self.qregs = [qreg]

    def _reset_parameters(self):
        self._parameters = [qiskit.circuit.ParameterVector("test_parameter", length=1)]


class BaseEncoderTesterWithoutResetRegister(BaseEncoder):
    """Mock class of a child class of BaseEncoder for normal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(self, data_dimension: int, name=None, transformer=None):
        super().__init__(
            data_dimension=data_dimension, name=name, transformer=transformer
        )

    @property
    def num_encoding_qubits(self):
        return 1

    def _check_configuration(self, raise_on_failure=True):
        return True

    def _build(self):
        super()._build()
        circuit = qiskit.QuantumCircuit(1)
        self.append(circuit.to_gate(), self.qubits)

    def _reset_parameters(self):
        self._parameters = [qiskit.circuit.ParameterVector("test_parameter", length=1)]


class BaseEncoderTesterWithoutResetParameters(BaseEncoder):
    """Mock class of a child class of BaseEncoder for normal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(self, data_dimension, name=None, transformer=None):
        super().__init__(
            data_dimension=data_dimension, name=name, transformer=transformer
        )

    @property
    def num_encoding_qubits(self):
        return 1

    def _check_configuration(self, raise_on_failure=True):
        return True

    def _build(self):
        super()._build()
        circuit = qiskit.QuantumCircuit(1)
        self.append(circuit.to_gate(), self.qubits)

    def _reset_register(self):
        qreg = qiskit.QuantumRegister(1)
        self.qregs = [qreg]


# === BaseLayer ===
from quantum_machine_learning.layers.base_layer import BaseLayer


class BaseLayerNormalTester(BaseLayer):
    """This is a normal test class for BaseLayer.
    Thus, the docstrings will be omitted hereby.
    """

    def __init__(self, num_state_qubits, name=None):
        super().__init__(num_state_qubits=num_state_qubits, name=name)

    def _reset_register(self):
        qreg = qiskit.QuantumRegister(1)
        self.qregs = [qreg]

    def _check_configuration(self, raise_on_failure=True):
        return True

    def _build(self):
        super()._build()
        circuit = qiskit.QuantumCircuit(*self.qregs)
        self.append(circuit.to_gate(), self.qubits)


class BaseLayerTesterWithoutResetRegister(BaseLayer):
    """This is an abnormal test class for BaseLayer without implementing _reset_register method.
    Thus, the docstrings will be omitted hereby.
    """

    def __init__(self, num_state_qubits, name=None):
        super().__init__(num_state_qubits=num_state_qubits, name=name)

    def _check_configuration(self, raise_on_failure=True):
        return True

    def _build(self):
        super()._build()
        circuit = qiskit.QuantumCircuit(*self.qregs)
        self.append(circuit.to_gate(), self.qubits)


# === BaseParametrisedLayer ===
from quantum_machine_learning.layers.base_parametrised_layer import (
    BaseParametrisedLayer,
)


class BaseParametrisedLayerNormalTester(BaseParametrisedLayer):
    """This is a normal test class for BaseParametrisedLayer.
    Thus, the docstrings will be omitted hereby.
    """

    def __init__(self, num_state_qubits, parameter_prefix=None, name=None):
        super().__init__(
            num_state_qubits=num_state_qubits,
            parameter_prefix=parameter_prefix,
            name=name,
        )

    def _reset_register(self):
        qreg = qiskit.QuantumRegister(1)
        self.qregs = [qreg]

    def _reset_parameters(self):
        pass

    def _check_configuration(self, raise_on_failure=True):
        return True

    def _build(self):
        super()._build()
        circuit = qiskit.QuantumCircuit(*self.qregs)
        self.append(circuit.to_gate(), self.qubits)


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
        qreg = qiskit.QuantumRegister(1)
        self.qregs = [qreg]

    def _check_configuration(self, raise_on_failure=True):
        return True

    def _build(self):
        super()._build()
        circuit = qiskit.QuantumCircuit(*self.qregs)
        self.append(circuit.to_gate(), self.qubits)
