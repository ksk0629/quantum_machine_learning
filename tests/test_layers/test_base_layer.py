import pytest
import qiskit

from tests.mocks import BaseLayerNormalTester, BaseLayerTesterWithoutResetRegister


class TestBaseLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3])
    def test_normal_inheritation(self, num_state_qubits):
        """Normal test;
        Create a normal mock class of BaseLayer.

        Check if
        - its num_state_qubits is the same as the given num_state_qubits.
        - the type of itself is qiskit.QuantumCircuit.
        - its num_state_qubits, after substituting a new num_state_qubits,
          is the same as the new given num_state_qubits.
        """
        tester = BaseLayerNormalTester(num_state_qubits=num_state_qubits)
        assert tester.num_state_qubits == num_state_qubits
        assert isinstance(tester, qiskit.QuantumCircuit)

        new_num_state_qubits = num_state_qubits + 1
        tester.num_state_qubits = new_num_state_qubits
        assert tester.num_state_qubits == new_num_state_qubits

    @pytest.mark.layer
    def test_abnormal_inheritation(self):
        """Normal test;
        Create a abnormal mock class of BaseLayer.

        Check if TypeError happens.
        """
        with pytest.raises(TypeError):
            BaseLayerTesterWithoutResetRegister(num_state_qubits=2)
