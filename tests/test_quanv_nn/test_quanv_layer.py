import numpy as np
import pytest

from src.quanv_nn.quanv_layer import QuanvLayer


class TestQuanvLayer:
    @classmethod
    def setup_class(cls):
        cls.kernel_size = (2, 2)
        cls.num_filters = 3
        cls.quanv_layer = QuanvLayer(
            kernel_size=cls.kernel_size, num_filters=cls.num_filters
        )

        cls.batch_data = np.array(
            [
                [[1, 1, 1, 1], [1, 1, 0, 1], [1, 0, 0, 1]],
                [[0, 0, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1]],
                [[0, 1, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]],
            ]
        )
        num_batch = len(cls.batch_data)
        num_channels = cls.batch_data[0].shape[0]
        cls.correct_output_shape = (num_batch, num_channels * cls.num_filters, 1)

    def test_init(self):
        """Normal test;
        Check if self.quanv_layer has
        - the same kernel_size as self.kernel_size.
        - the same num_filters as self.num_filters.
        """
        assert self.quanv_layer.kernel_size == self.kernel_size
        assert self.quanv_layer.num_filters == self.num_filters

    def test_process_with_valid(self):
        """Normal test;
        Run process and __call__.

        Check if
        - the return value of process has the correct shape.
        - the return values of process and __call__ are the same.
        """
        processed_data_1 = self.quanv_layer.process(batch_data=self.batch_data)

        assert processed_data_1.shape == self.correct_output_shape

        processed_data_2 = self.quanv_layer(self.batch_data)
        assert np.allclose(processed_data_1, processed_data_2)

    def test_process_with_invalid_batch_data(self):
        """Abnormal test;
        Run process with invalid batch_data.

        Check if ValueError happens.
        """
        invalid_batch_data = np.array([[1, 1, 1, 1], [1, 1, 0, 1], [1, 0, 0, 1]])
        with pytest.raises(ValueError):
            self.quanv_layer.process(batch_data=invalid_batch_data)

    def test_process_with_invalid_data(self):
        """Abnormal test;
        Run process with batch_data having invalid data.

        Check if ValueError happens.
        """
        invalid_batch_data = np.array(
            [
                [[1, 1, 1, 1, 5], [1, 1, 0, 1, 5], [1, 0, 0, 1, 5]],
                [[0, 0, 1, 1, 5], [1, 0, 0, 0, 5], [1, 1, 1, 1, 5]],
                [[0, 1, 1, 0, 5], [0, 0, 0, 0, 5], [1, 1, 0, 0, 5]],
            ]
        )
        with pytest.raises(ValueError):
            self.quanv_layer.process(batch_data=invalid_batch_data)
