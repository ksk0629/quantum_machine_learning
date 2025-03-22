import numpy as np
import pytest

from quantum_machine_learning.preprocessor.preprocessor import Preprocessor


class TestUtils:

    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.preprocessor
    @pytest.mark.parametrize(
        "data",
        [[[2, 3], [1, 1]], [[1, 1, 1]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]],
    )
    def test_normalise_data(self, data):
        """Normal test;
        Run normalise_data.

        Check if the vectors are normalised.
        """
        result = Preprocessor.normalise_data(data)

        normalised_vectors = []
        for datum in data:
            normalised_vectors.append(_v / np.linalg.norm(datum))
        normalised_vectors = np.array(normalised_vectors)

        assert np.allclose(result, normalised_vectors)

    @pytest.mark.preprocessor
    @pytest.mark.parametrize(
        "data_and_filling_value",
        [
            ([[2, 3], [1, 1]], 0),
            ([[1, 1, 1]], 1),
            ([[1, 1], [1, 1]], 0),
            ([[1, 1, 1], [1, 1, 1]], 0),
        ],
    )
    def test_evenise_data_dimension(self, data_and_filling_value):
        """Normal test;
        Run evenise_data_dimension.

        Check if the dimension of each datum is even.
        """
        (data, filling_value) = data_and_filling_value
        data = np.array(data)
        result = Preprocessor.evenise_data_dimension(
            data=data, filling_value=filling_value
        )

        if data.shape[1] % 2 == 0:
            # If the original dimension is even, then nothing must have happened.
            assert np.allclose(result, data)
        else:
            # If the original dimension is odd, then the first number of elements must be the same as the original data.
            assert np.allclose(result[:, :-1], data)
            # Also, the last elements must be the same as the filling value.
            fillings = np.zeros((result.shape[0],)) + filling_value
            assert np.allclose(result[:, -1], fillings)
