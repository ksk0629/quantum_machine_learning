import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    """Class for preprocessing data."""

    @staticmethod
    def normalise_data(data: np.ndarray) -> np.ndarray:
        """Normalise each datum, which correponds to each row of the given data.
        Say an input is [[1, 1], [2,2]].
        Then, the preprocessed one the will be [1/sqrt(2)[1, 1], 1/(2*sqrt(2))[2, 2]]

        :param np.ndarray data: data to be normalised, the shape must be (any, any)
        :return np.ndarray: normalised data
        """
        return data / np.linalg.norm(data, axis=1, keepdims=1)

    @staticmethod
    def scale_data(
        data: np.ndarray,
    ) -> np.ndarray:  # No test for now as this completely depends on scikit-learn.
        """Scale each datum, which corresponds to each row, into the range [0, 1].

        :param np.ndarray data: data to be scaled
        :return np.ndarray: scaled data
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(data)

    @staticmethod
    def evenise_data_dimension(
        data: np.ndarray, filling_value: float = 0
    ) -> np.ndarray:
        """Make each dimension of datum in which the given data, even by adding filling value.

        :param np.ndarray data: data of which dimension of each datum will be even
        :param float filling_value: value to fill given data, defaults to 0
        :return np.ndarray: data of which dimension of each datum is even
        """
        # Early return if the given data has the even number of dimension.
        if data.shape[1] % 2 == 0:
            return data

        # Make a new shape whose number of data is the same and the dimension of each datum is the original plus 1.
        new_shape = [0, 0]
        new_shape[0] = data.shape[0]
        new_shape[1] = data.shape[1] + 1
        # Fill the additional data with the given value.
        evenised_data = np.zeros(new_shape) + filling_value  # Fill all at once.
        evenised_data[:, :-1] = (
            data  # Insert original data in the first-original dimension.
        )

        return evenised_data

    @staticmethod
    def window_single_channel_data(
        data_2d: np.ndarray, window_size: tuple[int, int]
    ) -> np.ndarray:
        """Window single channel data, which means get a windowed single channel data.

        :param np.ndarray data_2d: two dimensional data
        :param tuple[int, int] window_size: window size
        :return np.ndarray: three-dimensional data whose each entry is sliding window
        """
        return np.lib.stride_tricks.sliding_window_view(data_2d, window_size)
