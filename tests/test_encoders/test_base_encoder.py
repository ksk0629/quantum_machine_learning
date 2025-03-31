import random
import string

import pytest
import qiskit

from tests.mocks import (
    BaseEncoderNormalTester,
    BaseEncoderTesterWithoutNumEncodingQubits,
    BaseEncoderTesterWithoutResetRegister,
    BaseEncoderTesterWithoutResetParameters,
)


class TestXEncoder:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.encoder
    @pytest.mark.parametrize("data_dimension", [1, 2, 5, 6])
    def test_init_with_defaults(self, data_dimension):
        """Normal test;
        create an instance of the normal mock test class for BaseEncoder.

        Check if
        1. its data_dimension is the same as the given data_dimension.
        """
        tester = BaseEncoderNormalTester(data_dimension=data_dimension)
        # 1. its data_dimension is the same as the given data_dimension.
        assert tester.data_dimension == data_dimension

    @pytest.mark.encoder
    def test_init_with_name(self):
        """Normal test;
        create an instance of the normal mock test class for BaseEncoder.

        Check if
        1. its name is the same as the given name.
        2. its data_dimension is the same as the given data_dimension.
        """
        random.seed(10)  # For reproducibility

        chars = string.ascii_letters + string.digits
        name = "".join(random.choice(chars) for _ in range(64))

        data_dimension = 2
        tester = BaseEncoderNormalTester(name=name, data_dimension=data_dimension)
        # 1. its name is the same as the given name.
        assert tester.name == name
        # 2. its data_dimension is the same as the given data_dimension.
        assert tester.data_dimension == data_dimension

    @pytest.mark.encoder
    def test_data_dimension(self):
        """Normal test;
        call the data_dimension attribute.

        Check if
        1. the return value is the same as the given data_dimension.
        2. its _num_reset_register and _num_reset_parameters are both 1.
           (This means _reset_register and _reset_parameters worked
           when the instance was created.)
        3. the return value is 0 after setting None to its data_dimension.
        4. its _num_reset_register and _num_reset_parameters are both 2.
           (This means _reset_register and _reset_parameters worked
           when the new data_dimension, None, was set)
        """
        data_dimension = 2
        tester = BaseEncoderNormalTester(data_dimension=data_dimension)
        # 1. the return value is the same as the given data_dimension.
        assert tester.data_dimension == data_dimension
        # 2. its _num_reset_register and _num_reset_parameters are both 1.
        assert tester._num_reset_register == 1
        assert tester._num_reset_parameters == 1
        # 3. the return value is 0 after setting None to its data_dimension.
        tester.data_dimension = None
        assert tester.data_dimension == 0
        # 4. its _num_reset_register and _num_reset_parameters are both 2.
        assert tester._num_reset_register == 2
        assert tester._num_reset_parameters == 2

    @pytest.mark.encoder
    @pytest.mark.parametrize("seed", [91, 57, 901])
    def test_parameters(self, seed):
        """Normal test;
        Set a list of qiskit.circuit.ParameterVector in _parameters.

        Check if
        1. its parameters is the same as the given list.
        2. its num_parameters is the total number of each number of the ParameterVector in the list.
        3. its num_parameters is 0 after setting None in its _parameters.
        """
        random.seed(seed)  # For reproducibility

        # Determine the number of parameter vectors.
        min_num_parameter_vectors = 1
        max_num_parameter_vectors = 100
        num_parameter_vectors = random.randint(
            min_num_parameter_vectors, max_num_parameter_vectors
        )
        # Determine the lengths of each paraheter vectors.
        min_length = 1
        max_length = 10
        lengths = [
            random.randint(min_length, max_length) for _ in range(num_parameter_vectors)
        ]
        # Create the parameters to be set.
        parameters = [
            qiskit.circuit.ParameterVector(name=f"name", length=length)
            for length in lengths
        ]

        data_dimension = 2
        tester = BaseEncoderNormalTester(data_dimension=data_dimension)
        tester._parameters = parameters
        # 1. the return value is the given list.
        assert tester.parameters == parameters
        # 2. its num_parameters is the total number of each number of the ParameterVector in the list.
        assert tester.num_parameters == sum(lengths)
        # 3. its num_parameters is 0 after setting None in its _parameters.
        tester._parameters = None
        assert tester.num_parameters == 0

    @pytest.mark.encoder
    def test_abnormal_check_configuration(self):
        """Abnormal test;
        run _build() method to see if _check_configuration works.

        Check if
        1. AttributeError arises after None is set in its data_dimension.
        2. AttributeError arises after None is set in its _parameters.
        """
        data_dimension = 2
        # 1. AttributeError arises after None is set in its data_dimension.
        tester = BaseEncoderNormalTester(data_dimension=data_dimension)
        tester.data_dimension = None
        with pytest.raises(AttributeError):
            tester._build()
        # 2. AttributeError arises after None is set in its _parameters.
        tester = BaseEncoderNormalTester(data_dimension=data_dimension)
        tester._parameters = None
        with pytest.raises(AttributeError):
            tester._build()

    @pytest.mark.encoder
    def test_normal_check_configuration(self):
        """Abnormal test;
        run _build() method to see if _check_configuration works.

        Check if
        1. no error arises after setting some list of ParameterVectors in _parameters.
        """
        data_dimension = 2
        tester = BaseEncoderNormalTester(data_dimension=data_dimension)
        # 1. no error arises after setting some list of ParameterVectors in _parameters.
        parameters = [qiskit.circuit.ParameterVector(name=f"name", length=2)]
        tester._parameters = parameters
        tester._build()

    @pytest.mark.encoder
    def test_without_num_encoding_qubits(self):
        """Abnormal test;
        create an instance of abnormal mock class for BaseEncoder,
        without implementing num_encoding_qubits property.

        Check if
        1. TypeError arises.
        """
        # 1. TypeError arises.
        with pytest.raises(TypeError):
            BaseEncoderTesterWithoutNumEncodingQubits(data_dimension=2)

    @pytest.mark.encoder
    def test_without_reset_register(self):
        """Abnormal test;
        create an instance of abnormal mock class for BaseEncoder,
        without implementing _reset_register method.

        Check if
        1. TypeError arises.
        """
        # 1. TypeError arises.
        with pytest.raises(TypeError):
            BaseEncoderTesterWithoutResetRegister(data_dimension=2)

    @pytest.mark.encoder
    def test_without_reset_parameters(self):
        """Abnormal test;
        create an instance of abnormal mock class for BaseEncoder,
        without implementing _reset_register method.

        Check if
        1. TypeError arises.
        """
        # 1. TypeError arises.
        with pytest.raises(TypeError):
            BaseEncoderTesterWithoutResetParameters(data_dimension=2)
