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
            normalised_vectors.append(datum / np.linalg.norm(datum))
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

    @pytest.mark.preprocessor
    def test_window_single_channel_data_with_2x2_window(self):
        """Normal test;
        Run window_single_channel_data with two-dimensional window.

        Check if the return value is windowed data.
        """
        data_2d = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )
        correct_windowed_data = np.array(
            [
                [[[1, 2], [5, 6]], [[2, 3], [6, 7]], [[3, 4], [7, 8]]],
                [[[5, 6], [9, 10]], [[6, 7], [10, 11]], [[7, 8], [11, 12]]],
                [[[9, 10], [13, 14]], [[10, 11], [14, 15]], [[11, 12], [15, 16]]],
            ]
        )
        windowed_data = Preprocessor.window_single_channel_data(
            data_2d=data_2d, window_size=(2, 2)
        )
        assert np.allclose(correct_windowed_data, windowed_data)

    @pytest.mark.preprocessor
    def test_window_single_channel_data_with_3x3_window(self):
        """Normal test;
        Run window_single_channel_data with two-dimensional window.

        Check if the return value is windowed data.
        """
        data_2d = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )
        correct_windowed_data = np.array(
            [
                [
                    [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
                    [[2, 3, 4], [6, 7, 8], [10, 11, 12]],
                ],
                [
                    [[5, 6, 7], [9, 10, 11], [13, 14, 15]],
                    [[6, 7, 8], [10, 11, 12], [14, 15, 16]],
                ],
            ]
        )
        windowed_data = Preprocessor.window_single_channel_data(
            data_2d=data_2d, window_size=(3, 3)
        )
        assert np.allclose(correct_windowed_data, windowed_data)
