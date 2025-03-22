import numpy as np
import pytest

from quantum_machine_learning.preprocessor.preprocessor import Preprocessor


class TestPreprocessor:

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

    @pytest.mark.preprocessor
    def test_window_multi_channel_data_with_2x2_window(self):
        """Normal test;
        Run window_multi_channel_data with two-dimensional window.

        Check if the return value is windowed data.
        """
        data_3d = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                [
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32],
                ],
                [
                    [33, 34, 35, 36],
                    [37, 38, 39, 40],
                    [41, 42, 43, 44],
                    [45, 46, 47, 48],
                ],
            ]
        )
        correct_windowed_data = np.array(
            [
                [
                    [[[1, 2], [5, 6]], [[2, 3], [6, 7]], [[3, 4], [7, 8]]],
                    [[[5, 6], [9, 10]], [[6, 7], [10, 11]], [[7, 8], [11, 12]]],
                    [[[9, 10], [13, 14]], [[10, 11], [14, 15]], [[11, 12], [15, 16]]],
                ],
                [
                    [[[17, 18], [21, 22]], [[18, 19], [22, 23]], [[19, 20], [23, 24]]],
                    [[[21, 22], [25, 26]], [[22, 23], [26, 27]], [[23, 24], [27, 28]]],
                    [[[25, 26], [29, 30]], [[26, 27], [30, 31]], [[27, 28], [31, 32]]],
                ],
                [
                    [[[33, 34], [37, 38]], [[34, 35], [38, 39]], [[35, 36], [39, 40]]],
                    [[[37, 38], [41, 42]], [[38, 39], [42, 43]], [[39, 40], [43, 44]]],
                    [[[41, 42], [45, 46]], [[42, 43], [46, 47]], [[43, 44], [47, 48]]],
                ],
            ]
        )
        windowed_data = Preprocessor.window_multi_channel_data(
            data_3d=data_3d, window_size=(2, 2)
        )
        assert np.allclose(correct_windowed_data, windowed_data)

    @pytest.mark.preprocessor
    def test_window_batch_data_with_single_channel_with_2x2_window(self):
        """Normal test;
        Run window_batch_data with single channel data and two-dimensional window.

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
        # Make batch single channel data and the truth.
        num_batch = 3
        num_channels = 1
        batch_single_channel_data = np.zeros(
            (num_batch, num_channels, data_2d.shape[0], data_2d.shape[1])
        )
        batch_correct_windowed_data = np.zeros(
            (
                num_batch,
                num_channels,
                correct_windowed_data.shape[0],
                correct_windowed_data.shape[1],
                correct_windowed_data.shape[2],
                correct_windowed_data.shape[3],
            )
        )
        for i in range(num_batch):
            offset = np.max(data_2d) * i
            batch_single_channel_data[i, 0, :, :] = data_2d + offset
            batch_correct_windowed_data[i, 0, :, :, :, :] = (
                correct_windowed_data + offset
            )

        windowed_batch_data = Preprocessor.window_batch_data(
            batch_data=batch_single_channel_data, window_size=(2, 2)
        )
        assert np.allclose(batch_correct_windowed_data, windowed_batch_data)

    @pytest.mark.preprocessor
    def test_window_batch_data_with_multi_channel_with_2x2_window(self):
        """Normal test;
        Run window_batch_data with multi-channel data and two-dimensional window.

        Check if the return value is windowed data.
        """
        data_3d = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                [
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32],
                ],
                [
                    [33, 34, 35, 36],
                    [37, 38, 39, 40],
                    [41, 42, 43, 44],
                    [45, 46, 47, 48],
                ],
            ]
        )
        correct_windowed_data = np.array(
            [
                [
                    [[[1, 2], [5, 6]], [[2, 3], [6, 7]], [[3, 4], [7, 8]]],
                    [[[5, 6], [9, 10]], [[6, 7], [10, 11]], [[7, 8], [11, 12]]],
                    [[[9, 10], [13, 14]], [[10, 11], [14, 15]], [[11, 12], [15, 16]]],
                ],
                [
                    [[[17, 18], [21, 22]], [[18, 19], [22, 23]], [[19, 20], [23, 24]]],
                    [[[21, 22], [25, 26]], [[22, 23], [26, 27]], [[23, 24], [27, 28]]],
                    [[[25, 26], [29, 30]], [[26, 27], [30, 31]], [[27, 28], [31, 32]]],
                ],
                [
                    [[[33, 34], [37, 38]], [[34, 35], [38, 39]], [[35, 36], [39, 40]]],
                    [[[37, 38], [41, 42]], [[38, 39], [42, 43]], [[39, 40], [43, 44]]],
                    [[[41, 42], [45, 46]], [[42, 43], [46, 47]], [[43, 44], [47, 48]]],
                ],
            ]
        )
        # Make batch single channel data and the truth.
        num_batch = 3
        batch_multi_channel_data = np.zeros(
            (num_batch, data_3d.shape[0], data_3d.shape[1], data_3d.shape[2])
        )
        batch_correct_windowed_data = np.zeros(
            (
                num_batch,
                correct_windowed_data.shape[0],  # = data_3d.shape[0]
                correct_windowed_data.shape[1],
                correct_windowed_data.shape[2],
                correct_windowed_data.shape[3],
                correct_windowed_data.shape[4],
            )
        )
        for i in range(num_batch):
            offset = np.max(data_3d) * i
            batch_multi_channel_data[i, :, :, :] = data_3d + offset
            batch_correct_windowed_data[i, :, :, :, :, :] = (
                correct_windowed_data + offset
            )

        windowed_batch_data = Preprocessor.window_batch_data(
            batch_data=batch_multi_channel_data, window_size=(2, 2)
        )
        print(windowed_batch_data)
        print("==========")
        print(batch_correct_windowed_data)
        assert np.allclose(batch_correct_windowed_data, windowed_batch_data)
