import numpy as np


class Preprocess:
    """Class for preprocessing data."""

    @staticmethod
    def normalise_vectors(vectors: np.ndarray) -> np.ndarray:
        """Normalise each vector, which correponds to each row of the given vectors.
        Say an input is [[1, 1], [2,2]].
        Then, the preprocessed one the will be [1/sqrt(2)[1, 1], 1/(2*sqrt(2))[2, 2]]

        :param np.ndarray vectors: vectors to be normalised, the shape must be (any, any)
        :return np.ndarray: normalised vectors
        """
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=1)
