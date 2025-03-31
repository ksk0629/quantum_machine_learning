import random
import string

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
    @pytest.mark.parametrize("num_state_qubits", [1, 2, 5, 6])
    def test_init_with_defaults(self, num_state_qubits):
        """Normal test;
        create an instance of a mock class for BaseParametrisedClass.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its parameter_prefix is "".
        3. its _parameters is None.
        4. its _num_reset_register is 2.
           (This means that _reset_register method ran when num_state_qubits and parameter_prefix were set.)
        5. its _num_reset_parameters is 2.
           (This means that _reset_parameters method ran when num_state_qubits and parameter_prefix were set.)
        """
        tester = BaseParametrisedLayerNormalTester(num_state_qubits=num_state_qubits)
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        assert tester.num_state_qubits == num_state_qubits
        # 2. its parameter_prefix is "".
        assert tester.parameter_prefix == ""
        # 3. its _parameters is None.
        assert tester._parameters is None
        # 4. its _num_reset_register is 2.
        assert tester._num_reset_register == 2
        # 5. its _num_reset_parameters is 2.
        assert tester._num_reset_parameters == 2

    @pytest.mark.layer
    def test_init_with_name(self):
        """Normal test;
        create an instance of the normal mock class with giving name.

        Check if
        1. its name is the same as the given name.
        2. its num_state_qubits is the same as the given num_state_qubits.
        3. its parameter_prefix is "".
        4. its parameters is None.
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
            # 4. its parameters is None.
            assert tester._parameters is None

    @pytest.mark.layer
    def test_init_with_parameter_prefix(self):
        """Normal test;
        create an instance of the normal mock class with giving name.

        Check if
        1. its parameter_prefix is the same as the given parameter_prefix.
        2. its num_state_qubits is the same as the given num_state_qubits.
        4. its parameters is None.
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
            # 4. its parameters is None.
            assert tester._parameters is None

    @pytest.mark.layer
    @pytest.mark.parametrize("num_state_qubits", [1, 2, 5, 6])
    def test_num_state_qubits(self, num_state_qubits):
        """Normal test;
        call the setter of num_state_qubits.

        Check if
        1. its num_state_qubits is the same as the given num_state_qubits.
        2. its _num_reset_register is 2.
           (This means that _reset_register method ran when num_state_qubits and parameter_prefix were set.)
        3. its _num_reset_parameters is 2.
           (This means that _reset_parameters method ran when num_state_qubits and parameter_prefix were set.)
        4. its num_state_qubits is the same as the new num_state_qubits after setting a new value in num_state_qubits.
        5. its _num_reset_register is 3.
        6. its _num_reset_parameters is 3.
        """
        tester = BaseParametrisedLayerNormalTester(num_state_qubits=num_state_qubits)
        # 1. its num_state_qubits is the same as the given num_state_qubits.
        assert tester.num_state_qubits == num_state_qubits
        # 2. its _num_reset_register is 2.
        assert tester._num_reset_register == 2
        # 3. its _num_reset_parameters is 2.
        assert tester._num_reset_parameters == 2
        # 4. its num_state_qubits is the same as the new num_state_qubits after setting a new value in num_state_qubits.
        new_num_state_qubits = num_state_qubits + 3
        tester.num_state_qubits = new_num_state_qubits
        assert tester.num_state_qubits == new_num_state_qubits
        # 5. its _num_reset_register is 3.
        assert tester._num_reset_register == 3
        # 6. its _num_reset_parameters is 3.
        assert tester._num_reset_parameters == 3

    @pytest.mark.layer
    @pytest.mark.parametrize("parameter_prefix", ["hey", " this", " is ", "test "])
    def test_parameter_prefix(self, parameter_prefix):
        """Normal test;
        call the setter and getter of parameter_prefix.

        Check if
        1. its parameter_prefix is the same as the given parameter_prefix.
        2. its _num_reset_register is 2.
           (This means that _reset_register method ran when num_state_qubits and parameter_prefix were set.)
        3. its _num_reset_parameters is 2.
           (This means that _reset_parameters method ran when num_state_qubits and parameter_prefix were set.)
        4. its parameter_prefix is "" after settting None in it.
        5. its _num_reset_register is 3.
        6. its _num_reset_parameters is 3.
        7. its parameter_prefix is the same as the new parameter_prefix after setting a new value in it.
        8. its _num_reset_register is 4.
        9. its _num_reset_parameters is 4.
        """
        tester = BaseParametrisedLayerNormalTester(
            num_state_qubits=2, parameter_prefix=parameter_prefix
        )
        # 1. its parameter_prefix is the same as the given parameter_prefix.
        assert tester.parameter_prefix == parameter_prefix
        # 2. its _num_reset_register is 2.
        assert tester._num_reset_register == 2
        # 3. its _num_reset_parameters is 2.
        assert tester._num_reset_parameters == 2
        # 4. its parameter_prefix is "" after settting None in it.
        tester.parameter_prefix = None
        assert tester.parameter_prefix == ""
        # 5. its _num_reset_register is 3.
        assert tester._num_reset_register == 3
        # 6. its _num_reset_parameters is 3.
        assert tester._num_reset_parameters == 3
        # 7. its parameter_prefix is the same as the new parameter_prefix after setting a new value in it.
        new_parameter_prefix = parameter_prefix + "OK!"
        tester.parameter_prefix = new_parameter_prefix
        assert tester.parameter_prefix == new_parameter_prefix
        # 8. its _num_reset_register is 4.
        assert tester._num_reset_register == 4
        # 9. its _num_reset_parameters is 4.
        assert tester._num_reset_parameters == 4

    @pytest.mark.layer
    def test_parameters(self):
        """Normal test;
        call parameters property.

        Check if
        1. its parameters is None.
        2. its parameters is the given list after setting a new list of qiskit.circuit.ParameterVecotr.
        """
        # 1. its parameters is None.
        # 2. its parameters is the given list after setting a new list of qiskit.circuit.ParameterVecotr.

    @pytest.mark.layer
    def test_num_parameters(self):
        """Normal test;
        call num_parameters property.

        Check if
        1. its num_parameters is 0.
        2. its num_parameters is a total number of elements in each list after setting a new _parameters.
        """
        # 1. its num_parameters is 0.
        # 2. its num_parameters is a total number of elements in each list after setting a new _parameters.

    @pytest.mark.layer
    def test_valid_check_configuration(self):
        """Normal test;
        run _build() method to see if _check_configuration works when parameters were set.

        Check if
        1. no error arises.
        """

    @pytest.mark.layer
    def test_invalid_check_configuration(self):
        """Abnormal test;
        run _build() method to see if _check_configuration works when there is no parameters set.

        Check if
        1. AttributeError arises.
        """
        # 1. AttributeError arises.

    @pytest.mark.layer
    def test_without_reset_parameters(self):
        """Abnormal test;
        Create an instance of a child class of BaseParametrisedLayer without implementing _reset_parameters.

        Check if
        1. TypeError arises.
        """
        with pytest.raises(TypeError):
            # 1. TypeError arises.
            num_state_qubits = 2
            BaseParametrisedLayerTesterWithoutResetParameters(
                num_state_qubits=num_state_qubits
            )
