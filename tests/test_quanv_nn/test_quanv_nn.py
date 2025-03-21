import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from quantum_machine_learning.quanv_nn.quanv_layer import QuanvLayer
from quantum_machine_learning.quanv_nn.quanv_nn import QuanvNN


# This class is defined for the fixed data size, which is defined in the test class.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(4, 16, 3)
        self.fc1 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        return x


class TestQuanvNN:
    @classmethod
    def setup_class(cls):
        cls.kernel_size = (2, 2)
        cls.num_filters = 2
        cls.model_dir_path = "./test/"
        cls.quanv_layer = QuanvLayer(
            kernel_size=cls.kernel_size, num_filters=cls.num_filters
        )
        cls.classical_model = Net()
        cls.quanv_nn = QuanvNN(
            quanv_layer=cls.quanv_layer, classical_model=cls.classical_model
        )

        num_batch = 2
        num_channels = 2
        height = 8  # This affects Net structure.
        width = 8  # This affects Net structure.
        data = np.arange(num_batch * num_channels * height * width).reshape(
            num_batch, num_channels, height, width
        )
        cls.data = torch.Tensor(data)

        cls.model_dir_path = "./test"

    def test_init(self):
        """Normal test;
        Check if self.quanv_nn has
        - the same quanv_layer as self.quanv_layer.
        - the same classical_model as self.classical_model.
        """
        assert self.quanv_nn.quanv_layer == self.quanv_layer
        assert self.quanv_nn.classical_model == self.classical_model

    def test_forward(self):
        """Normal test;
        Run forward.

        Check if the return value has the correct shape.
        """
        y = self.quanv_nn(self.data)
        assert len(y.shape) == 2
        assert y.shape[0] == 2
        assert y.shape[1] == 2

    def test_classify(self):
        """Normal test;
        Run classify.

        Check if the return value has the correct shape.
        """
        y = self.quanv_nn.classify(self.data)
        assert len(y.shape) == 1
        assert len(y) == 2

    def test_save(self):
        """Normal test;
        run self.quanv_nn.save.

        Check if
        - there is basic_info.pkl.
        - there is circuit.qpy.
        - there is lookup_tables.pkl.
        - there is classical_model.pth.
        """
        self.quanv_nn.save(model_dir_path=self.model_dir_path)

        basic_info_path = os.path.join(self.model_dir_path, "basic_info.pkl")
        assert os.path.isfile(basic_info_path)
        os.remove(basic_info_path)

        filters_path = os.path.join(self.model_dir_path, "circuit.qpy")
        assert os.path.isfile(filters_path)
        os.remove(filters_path)

        lookup_tables_path = os.path.join(self.model_dir_path, "lookup_tables.pkl")
        assert os.path.isfile(lookup_tables_path)
        os.remove(lookup_tables_path)

        classical_model_path = os.path.join(self.model_dir_path, "classical_model.pth")
        assert os.path.isfile(classical_model_path)
        os.remove(classical_model_path)

        os.rmdir(self.model_dir_path)
