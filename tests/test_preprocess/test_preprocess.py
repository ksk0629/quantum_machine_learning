import numpy as np
import pytest

from quantum_machine_learning.preprocess.preprocess import Preprocess


class TestUtils:

    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.preprocess
    @pytest.mark.parametrize(
        "vectors",
        [[[2, 3], [1, 1]], [[1, 1, 1]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]],
    )
    def test_normalise_vectors(self, vectors):
        """Normal test;
        Run normalise_vectors.

        Check if the vectors are normalised.
        """
        result = Preprocess.normalise_vectors(vectors)

        normalised_vectors = []
        for _v in vectors:
            normalised_vectors.append(_v / np.linalg.norm(_v))
        normalised_vectors = np.array(normalised_vectors)

        assert np.allclose(result, normalised_vectors)
