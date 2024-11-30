import argparse
from functools import partial
import shutil
import yaml

import numpy as np
from qiskit import primitives
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import get_dataset
from src.quanv_nn.train import train
import src.utils as utils


class cnn(nn.Module):
    """Classical CNN class."""

    def __init__(self, in_dim: tuple[int, int, int], num_classes: int):
        super().__init__()
        # Set the first convolutional layer.
        conv_kernel_size = (5, 5)
        self.conv1 = nn.Conv2d(
            in_channels=in_dim[0], out_channels=50, kernel_size=conv_kernel_size
        )
        conv1_output_shape = utils.calc_2d_output_shape(
            height=in_dim[1], width=in_dim[2], kernel_size=conv_kernel_size
        )

        # Set the first pooling layer.
        pool_kernel_size = (2, 2)
        pool_stride = 2
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        pool1_output_shape = utils.calc_2d_output_shape(
            height=conv1_output_shape[0],
            width=conv1_output_shape[1],
            kernel_size=pool_kernel_size,
            stride=pool_stride,
        )

        # Set the second convolutional layer.
        num_conv2_filter = 64
        self.conv2 = nn.Conv2d(
            in_channels=50, out_channels=num_conv2_filter, kernel_size=conv_kernel_size
        )
        conv2_output_shape = utils.calc_2d_output_shape(
            height=pool1_output_shape[0],
            width=pool1_output_shape[1],
            kernel_size=conv_kernel_size,
        )

        # Set the second pooling layer.
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        pool2_output_shape = utils.calc_2d_output_shape(
            height=conv2_output_shape[0],
            width=conv2_output_shape[1],
            kernel_size=pool_kernel_size,
            stride=pool_stride,
        )

        # Set the first fully connected layer.
        fc1_input_size = int(
            num_conv2_filter * pool2_output_shape[0] * pool2_output_shape[1]
        )
        fc1_output_size = 1024
        self.fc1 = nn.Linear(in_features=fc1_input_size, out_features=fc1_output_size)

        # Set the dropout layer.
        self.dropout = nn.Dropout(p=0.4)

        # Set the second fully connected layer.
        self.fc2 = nn.Linear(in_features=fc1_output_size, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward data.

        :param torch.Tensor x: input data
        :return torch.Tensor: processed data
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Transform the output shape to input the fully connected layer.
        x = x.view(x.size()[0], -1)
        # print(len(x[0]))

        x = self.fc1(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return F.softmax(x, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and Evaluate QuanvNN with specified dataset."
    )
    parser.add_argument(
        "-c",
        "--config_yaml_path",
        required=False,
        type=str,
        default="./configs/local_config_quanv_nn.yaml",
    )
    args = parser.parse_args()

    config_yaml_path = args.config_yaml_path
    with open(config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)

    # Fix the seed.
    seed = config["general"]["seed"]
    utils.fix_seed(seed)

    # Get the dataset.
    dataset_options = config["dataset_options"]
    encode_method = partial(
        utils.encode_according_to_threshold, threshold=0, low_value=0, high_value=np.pi
    )
    dataset_options["encoding_method"] = encode_method
    dataset_name = dataset_options["name"]
    del dataset_options["name"]
    data, labels = get_dataset(dataset_name)
    dataset_options["seed"] = seed

    # Get options for TorchTrainer.
    trainer_options = config["trainer_options"]
    trainer_options["sampler"] = primitives.StatevectorSampler(seed=seed)

    # Get options for QuanvLayer.
    quanv_layer_options = config["quanv_layer_options"]

    # Crate a CNN.
    num_batch = quanv_layer_options["num_filters"] * data.shape[1]
    height = data.shape[2]
    width = data.shape[3]
    in_dim = (num_batch, height, width)
    num_classes = len(np.unique(labels))
    classical_model = cnn(in_dim=in_dim, num_classes=num_classes)

    # Get training_options.
    training_options = config["training_options"]

    train(
        data=data,
        labels=labels,
        quanv_layer_options=quanv_layer_options,
        classical_model=classical_model,
        dataset_options=dataset_options,
        epochs=trainer_options["epochs"],
        **training_options
    )

    shutil.copy2(config_yaml_path, training_options["model_dir_path"])
