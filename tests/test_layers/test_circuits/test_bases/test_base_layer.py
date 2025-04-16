import random
import string

import pytest

from tests.mocks import BaseLayerNormalTester, BaseLayerTesterWithoutResetRegister


class TestBaseLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [1, 2, 5, 6])
    def test_init_with_defaults(self, num_state_qubits):
        """Normal test;
        create an instance of the normal mock class with default values.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its _num_reset_registers is 1.
           (This means that _reset_registers method ran when num_state_qubits was set.)
        """
        tester = BaseLayerNormalTester(num_state_qubits=num_state_qubits)
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        assert tester.num_state_qubits == num_state_qubits
        # 2. its _num_reset_registers is 1.
        assert tester._num_reset_registers == 1

    @pytest.mark.layer
    def test_init_with_name(self):
        """Normal test;
        create an instance of the normal mock class with giving name.

        Check if
        1. its name is the same as the given name.
        2. its num_state_qubits is the same as the given num_state_qubits.
        """
        random.seed(901)  # For reproducibility

        chars = string.ascii_letters + string.digits
        num_state_qubits = 2

        num_trials = 100
        for _ in range(num_trials):
            name = "".join(random.choice(chars) for _ in range(64))

            tester = BaseLayerNormalTester(num_state_qubits=num_state_qubits, name=name)
            # 1. its name is the same as the given name.
            assert tester.name == name
            # 2. its num_state_qubits is the same as the given num_state_qubits.
            assert tester.num_state_qubits == num_state_qubits

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [1, 2, 5, 6])
    def test_num_state_qubits(self, num_state_qubits):
        """Normal test;
        set and get num_state_qubits attribute.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its _num_reset_registers is 1.
           (This means that _reset_registers method ran when num_state_qubits was set.)
        3. its num_state_qubits is 0 after setting None in num_state_qubits.
        4. its _num_reset_registers is 2.
        5. its num_state_qubits is the new num_state_qubits after setting a new num_state_qubits.
        6. its _num_reset_registers is 3.
        """
        tester = BaseLayerNormalTester(num_state_qubits=num_state_qubits)
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        assert tester.num_state_qubits == num_state_qubits
        # 2. its _num_reset_registers is 1.
        assert tester._num_reset_registers == 1
        # 3. its num_state_qubits is 0 after setting None in num_state_qubits.
        tester.num_state_qubits = None
        assert tester.num_state_qubits == 0
        # 4. its _num_reset_registers is 2.
        assert tester._num_reset_registers == 2
        # 5. its num_state_qubits is the new num_state_qubits after setting a new num_state_qubits.
        new_num_state_qubits = num_state_qubits + 3
        tester.num_state_qubits = new_num_state_qubits
        assert tester.num_state_qubits == new_num_state_qubits
        # 6. its _num_reset_registers is 3.
        assert tester._num_reset_registers == 3

    @pytest.mark.layer
    def test_valid_check_configuration(self):
        """Normal test;
        run _build method to see if _check_configuration works when its _num_state_qubits is positive.

        Check if
        1. no error arises.
        """
        num_state_qubits = 2
        tester = BaseLayerNormalTester(num_state_qubits=num_state_qubits)
        # 1. no error arises.
        tester._build()

    @pytest.mark.layer
    def test_invalid_check_configuration(self):
        """Normal test;
        run _build method to see if _check_configuration works when its _num_state_qubits is None.

        Check if
        1. AttributeError arises.
        """
        num_state_qubits = None
        tester = BaseLayerNormalTester(num_state_qubits=num_state_qubits)
        with pytest.raises(AttributeError):
            # 1. AttributeError arises.
            tester._build()

    @pytest.mark.layer
    def test_without_reste_register(self):
        """Abnormal test;
        create an instance of abnormal mock class for BaseEncoder,
        without implementing _reset_register method.

        Check if TypeError happens.
        """
        with pytest.raises(TypeError):
            BaseLayerTesterWithoutResetRegister(num_state_qubits=2)
