import numpy as np


class DataUtils:
    """Utils, for data, class"""

    @staticmethod
    def calc_2d_output_shape(
        height: int,
        width: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> tuple[int, int]:
        """Calculate an output shape of convolutional or pooling layers.
        Referred to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.

        :param int in_height: input height
        :param int in_width: input width
        :param tuple[int, int] kernel_size: kernel size
        :param int stride: stride, defaults to 1
        :param int padding: padding, defaults to 0
        :param int dilation: dilation, defaults to 1
        :return tuple[int, int]: output shape
        """
        output_height = np.floor(
            (height + 2 * padding - dilation * (kernel_size[0] - 1) - 1) / stride + 1
        )
        output_width = np.floor(
            (width + 2 * padding - dilation * (kernel_size[1] - 1) - 1) / stride + 1
        )
        return (output_height, output_width)
