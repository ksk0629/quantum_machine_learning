import random
import string

import pytest
import qiskit

from tests.mocks import BaseParametrisedLayerNormalTester


class TestBaseParametrisedLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [1, 2, 5, 6])
    def test_init_with_defaults(self, num_state_qubits):
        """Normal test;
        create an instance of a mock class for BaseParametrisedClass.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its parameter_prefix is "".
        3. its _num_reset_register is 2.
           (This means that _reset_register method ran when num_state_qubits and parameter_prefix were set.)
        """
        tester = BaseParametrisedLayerNormalTester(num_state_qubits=num_state_qubits)
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        assert tester.num_state_qubits == num_state_qubits
        # 2. its parameter_prefix is "".
        assert tester.parameter_prefix == ""
        # 3. its _num_reset_register is 2.
        assert tester._num_reset_register == 2

    @pytest.mark.layer
    def test_init_with_name(self):
        """Normal test;
        create an instance of the normal mock class with giving name.

        Check if
        1. its name is the same as the given name.
        2. its num_state_qubits is the same as the given num_state_qubits.
        3. its parameter_prefix is "".
        """
        random.seed(901)  # For reproducibility

        chars = string.ascii_letters + string.digits
        num_state_qubits = 2

        num_trials = 100
        for _ in range(num_trials):
            name = "".join(random.choice(chars) for _ in range(64))

            tester = BaseParametrisedLayerNormalTester(
                num_state_qubits=num_state_qubits, name=name
            )
            # 1. its name is the same as the given name.
            assert tester.name == name
            # 2. its num_state_qubits is the same as the given num_state_qubits.
            assert tester.num_state_qubits == num_state_qubits
            # 3. its parameter_prefix is "".
            assert tester.parameter_prefix == ""

    @pytest.mark.layer
    def test_init_with_parameter_prefix(self):
        """Normal test;
        create an instance of the normal mock class with giving name.

        Check if
        1. its parameter_prefix is the same as the given parameter_prefix.
        2. its num_state_qubits is the same as the given num_state_qubits.
        """
        random.seed(901)  # For reproducibility

        chars = string.ascii_letters + string.digits
        num_state_qubits = 2

        num_trials = 100
        for _ in range(num_trials):
            parameter_prefix = "".join(random.choice(chars) for _ in range(64))

            tester = BaseParametrisedLayerNormalTester(
                num_state_qubits=num_state_qubits, parameter_prefix=parameter_prefix
            )
            # 1. its parameter_prefix is the same as the given parameter_prefix.
            assert tester.parameter_prefix == parameter_prefix
            # 2. its num_state_qubits is the same as the given num_state_qubits.
            assert tester.num_state_qubits == num_state_qubits

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [1, 2, 5, 6])
    def test_num_state_qubits(self, num_state_qubits):
        """Normal test;
        call the setter of num_state_qubits.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its _num_reset_register is 2.
           (This means that _reset_register method ran when num_state_qubits and parameter_prefix were set.)
        3. its num_state_qubits is the same as the new num_state_qubits after setting a new value in num_state_qubits.
        4. its _num_reset_register is 3.
        """
        tester = BaseParametrisedLayerNormalTester(num_state_qubits=num_state_qubits)
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        assert tester.num_state_qubits == num_state_qubits
        # 2. its _num_reset_register is 2.
        assert tester._num_reset_register == 2
        # 3. its num_state_qubits is the same as the new num_state_qubits after setting a new value in num_state_qubits.
        new_num_state_qubits = num_state_qubits + 3
        tester.num_state_qubits = new_num_state_qubits
        assert tester.num_state_qubits == new_num_state_qubits
        # 4. its _num_reset_register is 3.
        assert tester._num_reset_register == 3

    @pytest.mark.layer
    @pytest.mark.parametrize("parameter_prefix", ["hey", " this", " is ", "test "])
    def test_parameter_prefix(self, parameter_prefix):
        """Normal test;
        call the setter and getter of parameter_prefix.

        Check if
        1. its parameter_prefix is the same as the given parameter_prefix.
        2. its _num_reset_register is 2.
           (This means that _reset_register method ran when num_state_qubits and parameter_prefix were set.)
        3. its parameter_prefix is "" after settting None in it.
        4. its _num_reset_register is 3.
        5. its parameter_prefix is the same as the new parameter_prefix after setting a new value in it.
        6. its _num_reset_register is 4.
        """
        tester = BaseParametrisedLayerNormalTester(
            num_state_qubits=2, parameter_prefix=parameter_prefix
        )
        # 1. its parameter_prefix is the same as the given parameter_prefix.
        assert tester.parameter_prefix == parameter_prefix
        # 2. its _num_reset_register is 2.
        assert tester._num_reset_register == 2
        # 3. its parameter_prefix is "" after settting None in it.
        tester.parameter_prefix = None
        assert tester.parameter_prefix == ""
        # 4. its _num_reset_register is 3.
        assert tester._num_reset_register == 3
        # 5. its parameter_prefix is the same as the new parameter_prefix after setting a new value in it.
        new_parameter_prefix = parameter_prefix + "OK!"
        tester.parameter_prefix = new_parameter_prefix
        assert tester.parameter_prefix == new_parameter_prefix
        # 6. its _num_reset_register is 4.
        assert tester._num_reset_register == 4

    @pytest.mark.layer
    def test_get_parameter_name(self):
        """Normal test;
        Run _get_parameter_name.

        Check if
        1. the return value is with prefix if a prefix was given.
        2. the return value is just the given parameter_name if a prefix wasn't given.
        """
        tester = BaseParametrisedLayerNormalTester(num_state_qubits=2)
        parameter_name = "name"
        # 1. the return value is with prefix if a prefix was given.
        assert (
            tester._get_parameter_name(parameter_name=parameter_name) == parameter_name
        )
        # 2. the return value is just the given parameter_name if a prefix wasn't given.
        prefix = "prefix"
        tester.parameter_prefix = prefix
        correct_name = f"{prefix}_{parameter_name}"
        assert tester._get_parameter_name(parameter_name=parameter_name) == correct_name
