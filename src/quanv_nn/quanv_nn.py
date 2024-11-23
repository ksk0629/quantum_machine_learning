import numpy as np
import torch
import torch.nn

from src.quanv_nn.quanv_layer import QuanvLayer
import src.utils as utils


class QuanvNN(torch.nn.Module):
    """Quanvolutional Neural Network class."""

    def __init__(self, quanv_layer: QuanvLayer, classical_model: torch.nn.Module):
        """Initialise the model.

        :param QuanvLayer quanv_layer: quanvolutional layer
        :param torch.nn.Module classical_model: classical model
        """
        super().__init__()
        self.quanv_layer = quanv_layer
        self.classical_model = classical_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the give data.

        :param torch.Tensor x: input data, whose shape must be (batch, channels, height, width)
        :return torch.Tensor: processed data
        """
        # Encode the data to one for quanv_layer.
        x_np = x.detach().cpu().numpy()
        encoded_x_np = utils.get_sliding_window_batch_data(
            x_np, self.quanv_layer.kernel_size
        )
        data_size = self.quanv_layer.kernel_size[0] * self.quanv_layer.kernel_size[1]
        batch_size = x_np.shape[0] if len(x_np.shape) == 4 else 1
        num_channels = x_np.shape[1]
        encoded_flattend_x_np = encoded_x_np.reshape(
            batch_size, num_channels, -1, data_size
        )
        processed_batch_data = []
        new_height = x_np.shape[2] - self.quanv_layer.kernel_size[0] + 1 - 1 + 1
        new_width = x_np.shape[3] - self.quanv_layer.kernel_size[1] + 1 - 1 + 1
        for multi_channel_data_np in encoded_flattend_x_np:
            processed_multi_channel_data = []
            for data_np in multi_channel_data_np:
                # Process the data through the QuanvLayer.
                processed_x_np = self.quanv_layer(
                    data_np
                )  # shape = (num_filters, dim_data)

                # Reshape the data as processed two-dimensional data.
                processed_x_np = processed_x_np.reshape(
                    self.quanv_layer.num_filters, new_height, new_width
                )
                processed_multi_channel_data.append(processed_x_np)
            # Treat the output data in different channels and processed by different filters in the same way.
            # processed_multi_channel_data's shape
            # = (num_channels, num_filters, new_height, new_width)
            # -> (num_channels * num_filters, new_height, new_width)
            processed_multi_channel_data = np.vstack(processed_multi_channel_data)
            processed_batch_data.append(processed_multi_channel_data)

        processed_batch_data = torch.Tensor(np.array(processed_batch_data))
        # processed_batch_data's shape
        # = (batch, num_channels * num_filters, new_height, new_width)

        output = self.classical_model(processed_batch_data)
        return output

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Classify the given data.

        :param torch.Tensor x: input data, whose shape must be (batch, channels, height, width)
        :return torch.Tensor: predicted label
        """
        probabilities = self.forward(x)
        return torch.argmax(probabilities, dim=1)
