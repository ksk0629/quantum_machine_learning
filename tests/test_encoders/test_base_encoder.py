from typing import Callable

import pytest
import qiskit

from quantum_machine_learning.encoders.base_encoder import BaseEncoder


class BaseEncoderNormalTester(BaseEncoder):
    """Mock class of a child class of BaseEncoder for normal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(
        self,
        data_dimension: int,
        name: str | None = None,
        transformer: Callable[[list[float]], list[float]] | None = None,
    ):
        super().__init__(
            data_dimension=data_dimension, name=name, transformer=transformer
        )

    @property
    def num_encoding_qubits(self) -> int:
        return 1

    def _check_configuration(self, raise_on_failure=True) -> bool:
        return True

    def _build(self) -> None:
        super()._build()
        circuit = qiskit.QuantumCircuit(1)
        self.append(circuit.to_gate(), self.qubits)

    def _reset_register(self) -> None:
        qreg = qiskit.QuantumRegister(1)
        self.qregs = [qreg]

    def _reset_parameters(self) -> None:
        self._parameters = [qiskit.circuit.ParameterVector("test_parameter", length=1)]


class BaseEncoderTesterWithoutNumEncodingQubits(BaseEncoder):
    """Mock class of a child class of BaseEncoder for abnormal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(
        self,
        data_dimension: int,
        name: str | None = None,
        transformer: Callable[[list[float]], list[float]] | None = None,
    ):
        super().__init__(
            data_dimension=data_dimension, name=name, transformer=transformer
        )

    def _check_configuration(self, raise_on_failure=True) -> bool:
        return True

    def _build(self) -> None:
        super()._build()
        circuit = qiskit.QuantumCircuit(1)
        self.append(circuit.to_gate(), self.qubits)

    def _reset_register(self) -> None:
        qreg = qiskit.QuantumRegister(1)
        self.qregs = [qreg]

    def _reset_parameters(self) -> None:
        self._parameters = [qiskit.circuit.ParameterVector("test_parameter", length=1)]


class BaseEncoderTesterWithoutResetRegister(BaseEncoder):
    """Mock class of a child class of BaseEncoder for normal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(
        self,
        data_dimension: int,
        name: str | None = None,
        transformer: Callable[[list[float]], list[float]] | None = None,
    ):
        super().__init__(
            data_dimension=data_dimension, name=name, transformer=transformer
        )

    @property
    def num_encoding_qubits(self) -> int:
        return 1

    def _check_configuration(self, raise_on_failure=True) -> bool:
        return True

    def _build(self) -> None:
        super()._build()
        circuit = qiskit.QuantumCircuit(1)
        self.append(circuit.to_gate(), self.qubits)

    def _reset_parameters(self) -> None:
        self._parameters = [qiskit.circuit.ParameterVector("test_parameter", length=1)]


class BaseEncoderTesterWithoutResetParameters(BaseEncoder):
    """Mock class of a child class of BaseEncoder for normal test
    This is just for testing, so the docstrings will be omitted."""

    def __init__(
        self,
        data_dimension: int,
        name: str | None = None,
        transformer: Callable[[list[float]], list[float]] | None = None,
    ):
        super().__init__(
            data_dimension=data_dimension, name=name, transformer=transformer
        )

    @property
    def num_encoding_qubits(self) -> int:
        return 1

    def _check_configuration(self, raise_on_failure=True) -> bool:
        return True

    def _build(self) -> None:
        super()._build()
        circuit = qiskit.QuantumCircuit(1)
        self.append(circuit.to_gate(), self.qubits)

    def _reset_register(self) -> None:
        qreg = qiskit.QuantumRegister(1)
        self.qregs = [qreg]


class TestXEncoder:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.encoder
    def test_valid_child_class(self):
        """Normal test;
        Initialise the child class inheritating BaseEncoder for the test.

        Check if
        - its name is the given name.
        - its data_dimenstion is the given data_dimension.
        - the return value of its transformer is the return value of the given transformer.
        - the type of its parameters is list.
        - the type of teh first element of its parameters is qiskit.circuit.ParameterVector.
        - its num_parameters is 1.
        - the return value of its transformer after substituting a new transformer is
          the return value of the given new transformer.
        - the return value of its transformer after substituting new to transformer is the data.
        """
        # Create a BaseEncoderNormalTester instance.
        data_dimension = 2
        name = "tester"
        transformer = lambda x_list: [x * 2 for x in x_list]
        tester = BaseEncoderNormalTester(
            data_dimension=data_dimension, name=name, transformer=transformer
        )
        # Set data.
        data = [1, 2]
        transformed_data = transformer(data)

        assert tester.name == name
        assert tester.data_dimension == data_dimension
        assert tester.transform(data) == transformed_data
        assert isinstance(tester.parameters, list)
        assert isinstance(tester.parameters[0], qiskit.circuit.ParameterVector)
        assert tester.num_parameters == 1

        new_transformer = lambda x_list: [x * 3 for x in x_list]
        new_transformed_data = new_transformer(data)
        tester.transformer = new_transformer
        assert tester.transform(data) == new_transformed_data

        tester.transformer = None
        assert tester.transform(data) == data

    @pytest.mark.encoder
    def test_without_num_encoding_qubits(self):
        with pytest.raises(TypeError):
            BaseEncoderTesterWithoutNumEncodingQubits(data_dimension=1)

    @pytest.mark.encoder
    def test_without_reset_register(self):
        with pytest.raises(TypeError):
            BaseEncoderTesterWithoutResetRegister(data_dimension=1)

    @pytest.mark.encoder
    def test_without_reset_parameters(self):
        with pytest.raises(TypeError):
            BaseEncoderTesterWithoutResetParameters(data_dimension=1)
