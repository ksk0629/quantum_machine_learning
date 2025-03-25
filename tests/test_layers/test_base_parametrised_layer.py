import pytest
import qiskit

from quantum_machine_learning.layers.base_parametrised_layer import (
    BaseParametrisedLayer,
)


class BaseParametrisedLayerNormalTester(BaseParametrisedLayer):
    """This is a normal test class for BaseParametrisedLayer.
    Thus, the docstrings will be omitted hereby.
    """

    def __init__(
        self,
        num_state_qubits: int,
        parameter_prefix: str | None = None,
        name: str | None = None,
    ):
        super().__init__(
            num_state_qubits=num_state_qubits,
            parameter_prefix=parameter_prefix,
            name=name,
        )

    def _reset_register(self) -> None:
        qreg = qiskit.QuantumRegister(1)
        self.qregs = [qreg]

    def _reset_parameters(self) -> None:
        pass

    def _check_configuration(self, raise_on_failure=True) -> bool:
        return True

    def _build(self) -> None:
        super()._build()
        circuit = qiskit.QuantumCircuit(*self.qregs)
        self.append(circuit.to_gate(), self.qubits)


class BaseParametrisedLayerTesterWithoutResetParameters(BaseParametrisedLayer):
    """This is an abnormal test class for BaseParametrisedLayer without implementing _reset_parameters method.
    Thus, the docstrings will be omitted hereby.
    """

    def __init__(
        self,
        num_state_qubits: int,
        parameter_prefix: str | None = None,
        name: str | None = None,
    ):
        super().__init__(
            num_state_qubits=num_state_qubits,
            parameter_prefix=parameter_prefix,
            name=name,
        )

    def _reset_register(self) -> None:
        qreg = qiskit.QuantumRegister(1)
        self.qregs = [qreg]

    def _check_configuration(self, raise_on_failure=True) -> bool:
        return True

    def _build(self) -> None:
        super()._build()
        circuit = qiskit.QuantumCircuit(*self.qregs)
        self.append(circuit.to_gate(), self.qubits)


class TestBaseParametrisedLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3])
    def test_normal_inheritation_by_varying_num_state_qubits(self, num_state_qubits):
        """Normal test;
        Create a normal mock class of BaseParametrisedLayer using different num_state_qubits.

        Check if
        - its num_state_qubits is the same as the given num_state_qubits.
        - its parameter_prefix is the empty string.
        - the type of itself is qiskit.QuantumCircuit.
        - its num_state_qubits, after substituting a new num_state_qubits,
          is the same as the new given num_state_qubits.
        """
        tester = BaseParametrisedLayerNormalTester(num_state_qubits=num_state_qubits)
        assert tester.num_state_qubits == num_state_qubits
        assert tester.parameter_prefix == ""
        assert isinstance(tester, qiskit.QuantumCircuit)

        new_num_state_qubits = num_state_qubits + 1
        tester.num_state_qubits = new_num_state_qubits
        assert tester.num_state_qubits == new_num_state_qubits

    @pytest.mark.layer
    def test_normal_inheritation_with_given_parameter_prefix(self):
        """Normal test;
        Create a normal mock class of BaseParametrisedLayer that is given a parameter_prefix.

        Check if
        - its num_state_qubits is the same as the given num_state_qubits.
        - its parameter_prefix is the same as the given parameter_prefix.
        - the type of itself is qiskit.QuantumCircuit.
        - its parameter_prefix, after substituting a new num_parameter_prefix,
          is the same as the new given num_parameter_prefix.
        """
        num_state_qubits = 2
        parameter_prefix = "prefix!"
        tester = BaseParametrisedLayerNormalTester(
            num_state_qubits=num_state_qubits, parameter_prefix=parameter_prefix
        )
        assert tester.num_state_qubits == num_state_qubits
        assert tester.parameter_prefix == parameter_prefix
        assert isinstance(tester, qiskit.QuantumCircuit)

        new_parameter_prefix = "new!" + parameter_prefix
        tester.parameter_prefix = new_parameter_prefix
        assert tester.parameter_prefix == new_parameter_prefix
