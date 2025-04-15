import pytest

from quantum_machine_learning.layers.ssskm_dense_layer import SSSKMDenseLayer


class TestSSSKMDenseLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [2, 3])
    @pytest.mark.parametrize("num_reputations", [1, 2])
    def test_init_with_defaults(self, num_state_qubits, num_reputations):
        """Normal test:
        Create an instance of SSSKMDenseLayer.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its num_reputations is the same as the given num_reputations.
        3. its name is "SSSKMDenseLayer".
        """
        layer = SSSKMDenseLayer(
            num_state_qubits=num_state_qubits, num_reputations=num_reputations
        )
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        assert layer.num_state_qubits == num_state_qubits
        # 2. its num_reputations is the same as the given num_reputations.
        assert layer.num_reputations == num_reputations
        # 3. its name is "SSSKMDenseLayer".
        assert layer.name == "SSSKMDenseLayer"

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
        num_state_qubits = 4
        num_reputations = 3
        layer = SSSKMDenseLayer(
            num_state_qubits=num_state_qubits, num_reputations=num_reputations
        )
        # 1. its num_reputations is the same as the given num_reputations.
        assert layer.num_reputations == num_reputations
        # 2. its _is_build is False.
        assert not layer._is_built
        # 3. its _is_build is True after running _build().
        layer._build()
        assert layer._is_built
        # 4. its parameters[0] has 3 * num_state_qubits * num_reputations elements.
        correct_num_parameters = 3 * num_state_qubits * num_reputations
        assert len(layer.parameters[0]) == correct_num_parameters
        # 5. its num_reutations is 0 after setting None.
        layer.num_reputations = None
        assert layer.num_reputations == 0
        # 6. its _is_build is False.
        assert not layer._is_built
        # 7. its num_reputations is the same as the new one after setting a new one.
        new_num_reputations = 1
        layer.num_reputations = new_num_reputations
        assert layer.num_reputations == new_num_reputations
        # 8. its _is_build is False.
        assert not layer._is_built
        # 9. its _is_build is True after running _build().
        layer._build()
        assert layer._is_built
        # 10. its parameters[0] has 3 * num_state_qubits * num_reputations elements.
        new_correct_num_parameters = 3 * num_state_qubits * new_num_reputations
        assert len(layer.parameters[0]) == new_correct_num_parameters

    @pytest.mark.layer
    def test_check_configuration_invlaid_num_state_qubits(self):
        """Abnormal test:
        Run _build() with num_state_qubits == 1.

        Check if
        1. AttributeError arises.
        """
        layer = SSSKMDenseLayer(num_state_qubits=1, num_reputations=1)
        # 1. AttributeError arises.
        with pytest.raises(AttributeError):
            layer._build()

    @pytest.mark.layer
    def test_check_configuration_invlaid_num_reputations(self):
        """Abnormal test:
        Run _build() with _num_reputations being None.

        Check if
        1. AttributeError arises.
        """
        layer = SSSKMDenseLayer(num_state_qubits=2, num_reputations=None)
        # 1. AttributeError arises.
        with pytest.raises(AttributeError):
            layer._build()
