import pytest

from quantum_machine_learning.layers.ssskm_dense_layer import SSSKMDenseLayer


class TestSSSKMDenseLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3])
    @pytest.mark.parametrize("num_reputations", [1, 2])
    def test_init_with_defaults(self, num_reputations):
        """Normal test:
        Create an instance of SSSKMDenseLayer.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its num_reputations is the same as the given num_reputations.
        3. its name is "SSSKMDenseLayer".
        """
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        # 2. its num_reputations is the same as the given num_reputations.
        # 3. its name is "SSSKMDenseLayer".

    @pytest.mark.layer
    def test_num_reputations(self):
        """Normal test:
        Call the setter and getter of num_reputations.

        Check if
        1. its num_reputations is the same as the given num_reputations.
        2. its _is_build is False.
        3. its _is_build is True after running _build().
        4. its parameters[0] has 3 * num_state_qubits * num_reputations elements.
        5. its num_reutations is 0 after setting None.
        6. its _is_build is False.
        7. its num_reputations is the same as the new one after setting a new one.
        8. its _is_build is False.
        9. its _is_build is True after running _build().
        10. its parameters[0] has 3 * num_state_qubits * num_reputations elements.
        """
        # 1. its num_reputations is the same as the given num_reputations.
        # 2. its _is_build is False.
        # 3. its _is_build is True after running _build().
        # 4. its parameters[0] has 3 * num_state_qubits * num_reputations elements.
        # 5. its num_reutations is 0 after setting None.
        # 6. its _is_build is False.
        # 7. its num_reputations is the same as the new one after setting a new one.
        # 8. its _is_build is False.
        # 9. its _is_build is True after running _build().
        # 10. its parameters[0] has 3 * num_state_qubits * num_reputations elements.

    @pytest.mark.layer
    def test_check_configuration_invlaid_num_state_qubits(self):
        """Abnormal test:
        Run _build() with num_state_qubits == 1.

        Check if
        1. AttributeError arises.
        """
        # 1. AttributeError arises.

    @pytest.mark.layer
    def test_check_configuration_invlaid_num_reputations(self):
        """Abnormal test:
        Run _build() with _num_reputations being None.

        Check if
        1. AttributeError arises.
        """
        # 1. AttributeError arises.
