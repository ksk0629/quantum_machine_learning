import pytest
import qiskit

from tests.mocks import (
    BaseParametrisedLayerNormalTester,
    BaseParametrisedLayerTesterWithoutResetParameters,
)


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
        - its num_parameters is 0 after substituting None to its _parameters.
        """
        tester = BaseParametrisedLayerNormalTester(num_state_qubits=num_state_qubits)
        assert tester.num_state_qubits == num_state_qubits
        assert tester.parameter_prefix == ""
        assert isinstance(tester, qiskit.QuantumCircuit)

        new_num_state_qubits = num_state_qubits + 1
        tester.num_state_qubits = new_num_state_qubits
        assert tester.num_state_qubits == new_num_state_qubits

        tester._parameters = None
        assert tester.num_parameters == 0

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

    @pytest.mark.layer
    def test_without_reset_parameters(self):
        """Abnormal test;
        Create an instance of a child class of BaseParametrisedLayer without implementing _reset_parameters.

        Check if TypeError happens.
        """
        with pytest.raises(TypeError):
            num_state_qubits = 2
            BaseParametrisedLayerTesterWithoutResetParameters(
                num_state_qubits=num_state_qubits
            )
