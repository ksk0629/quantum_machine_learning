import numpy as np

from src.quclassi.quclassi import QuClassi


class QuClassiTrainer:
    """QuClassiTrainer class that trains QuClassi with the algorithm introduced in the original paper:
    https://arxiv.org/pdf/2103.11307.
    """

    def __init__(
        self,
        quclassi: QuClassi,
        epochs: int = 25,
        learning_rate: float = 0.01,
        batch_size: int = 1,
        shuffle: bool = True,
        initial_weights: dict[str, np.ndarray] | None = None,
    ):
        """Initialise this trainer.

        :param QuClassi quclassi: quclassi to be trained
        :param int epochs: number of epochs, defaults to 25
        :param float learning_rate: learning rate, defaults to 0.01
        :param int batch_size: batch size, defaults to 1
        :param bool shuffle: whether dataset is shuffled or not, defaults to True
        :param dict[str, np.ndarray] | None initial_weights: initial weights, defaults to None
        :raises ValueError: if quclassi.labels and initial_weights.keys() do not match
        """
        if set(quclassi.labels) != set(initial_weights.keys()) or len(
            set(quclassi.labels)
        ) != len(initial_weights.keys()):
            msg = f"The labels the given quclassi has and the labels the given initial_weights has must be the same, but {quclassi.labels} and {initial_weights.keys()}"
            raise ValueError(msg)

        self.quclassi = quclassi
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.initial_weights = initial_weights

    def train(self, data: np.ndarray):
        pass

    def train_one_epoch(self, data: np.ndarray):
        pass
