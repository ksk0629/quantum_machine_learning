import numpy as np


class CalculationUtils:
    """Utils, for calculation, class"""

    @staticmethod
    def calculate_cross_entropy(
        probabilities_list: list[dict[str, float]],
        true_labels: list[str],
    ) -> float:
        """Calculate the cross entropy.

        :param list[dict[str, float]] probabilities_list: a list of probabilities that are dictionaries whose keys are labels and values are their probabiliries
        :param list[str] true_labels: the true labels
        :raises ValueError: if the length of two of given arguments are not the same
        :return float: the cross entropy value
        """
        if len(probabilities_list) != len(true_labels):
            error_msg = f"The lengths of probabilities_list and true_labels must be same, but {len(probabilities_list)} vs {len(true_labels)}."
            raise ValueError(error_msg)

        cross_entropy = 0
        for true_label, probabilities in zip(true_labels, probabilities_list):
            predicted_label = max(probabilities, key=probabilities.get)
            if true_label == predicted_label:
                cross_entropy += -np.log(probabilities[predicted_label])
            else:
                cross_entropy += -np.log(1 - probabilities[predicted_label])

        cross_entropy /= len(true_labels)

        return cross_entropy

    @staticmethod
    def calculate_accuracy(
        predicted_labels: list[str], true_labels: list[str]
    ) -> float:
        """Calculate the accuracy.

        :param list[str] predicted_labels: predicted labels
        :param list[str] true_labels: the true labels
        :raises ValueError: if the lengths of two of the arguments are not the same
        :return float: the accuracy
        """
        if len(predicted_labels) != len(true_labels):
            error_mas = f"The lengths of the predicted and true labels must be the same, but {len(predicted_labels)} and {len(true_labels)}."
            raise ValueError(error_mas)

        num_correct = int((np.array(predicted_labels) == np.array(true_labels)).sum())
        return num_correct / len(predicted_labels)

    @staticmethod
    def safe_log_e(value: float) -> float:
        """Safely calculate the log_e.
        If the given value is 0, then the value is automatically set as 1e-16.

        :param float value: a value to be taken log_e
        :return float: the value of log_e
        """
        if value == 0:
            value = 1e-16
        return float(np.log(value))

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
        return (int(output_height), int(output_width))
