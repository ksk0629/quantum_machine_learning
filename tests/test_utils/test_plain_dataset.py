import pytest
import torch

from quantum_machine_learning.utils.plain_dataset import PlainDataset


class TestDataUtils:

    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.utils
    def test(self):
        """Normal test;
        create an instance of PlainDataset.

        Check if
        - the length is the same as the given number of data.
        - each item of itself is the same of the given dataset.
        """
        num_data = 10
        data_dim = 12
        dataset_size = (num_data, data_dim)
        data = torch.rand((dataset_size))
        labels = torch.randint(low=0, high=1, size=(num_data,))
        plain_dataset = PlainDataset(x=data, y=labels)

        assert len(plain_dataset) == num_data

        for index in range(num_data):
            datum, label = plain_dataset[index]
            print(datum)
            print(data[index])
            assert torch.allclose(datum, data[index])
            assert torch.allclose(label, labels[index])
