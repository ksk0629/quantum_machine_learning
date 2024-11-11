import numpy as np
from qiskit import primitives
from tqdm.auto import tqdm

from src.quclassi.quclassi import QuClassi
import src.utils


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
        initial_paramters: np.ndarray | None = None,
    ):
        """Initialise this trainer.

        :param QuClassi quclassi: quclassi to be trained
        :param int epochs: number of epochs, defaults to 25
        :param float learning_rate: learning rate, defaults to 0.01
        :param int batch_size: batch size, defaults to 1
        :param bool shuffle: whether dataset is shuffled or not, defaults to True
        :param np.ndarray | None initial_paramters: initial parameters, defaults to None
        :raises ValueError: if the lengths of quclassi.labels and initial_paramters do not match
        """
        if initial_paramters is not None and len(set(quclassi.labels)) != len(
            initial_paramters
        ):
            msg = f"The labels the given quclassi has and the labels the given initial_weights has must be the same lengths, but {quclassi.labels} and {initial_paramters}"
            raise ValueError(msg)

        self.quclassi = quclassi
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        if initial_paramters is not None:
            self.current_parameters = initial_paramters
        else:
            self.current_parameters = (
                np.random.rand(
                    len(self.quclassi.trainable_parameters) * len(self.quclassi.labels)
                )
                * np.pi
            ).reshape((len(quclassi.labels), -1))

    def train(self, data: np.ndarray, labels: np.ndarray):
        """Train the quclassi.

        :param np.ndarray data: whole data
        :param np.ndarray labels: corresponding labels
        """
        # Separate the data.
        data_separated_label = dict()
        for label in self.quclassi.labels:
            target_indices = np.where(labels == label)
            data_separated_label[label] = data[target_indices]

        # Train self.quclassi.
        with tqdm(range(1, self.epochs + 1)) as tepoch:
            for epoch in tepoch:
                tepoch.set_description(f"Epoch {epoch} (train)")

                with tqdm(data_separated_label.items(), leave=False) as dataset:
                    for label, _d in dataset:
                        dataset.set_description(f"Label {label}")
                        self.train_one_epoch(data=_d, label=label, epoch=epoch)

        # Set the trained parameters to self.quclassi.
        self.quclassi.trained_parameters = self.current_parameters

    def train_one_epoch(
        self,
        data: np.ndarray,
        label: object,
        epoch: int,
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 1024,
    ):
        """Train the quclassi only one epoch for one class.

        :param np.ndarray data: data belonging to the given label
        :param object label: label to which the data belong
        :param int epoch: current epoch
        :param qiskit.primitives.BaseSamplerV1  |  qiskit.primitives.BaseSamplerV2 sampler: sampler primitives, defaults to qiskit.primitives.StatevectorSampler
        :param int shots: number of shots
        """
        # Shuffle the data if needed.
        if self.shuffle:
            np.random.shuffle(data)

        # Adjust the size of the data if needed.
        remainder = len(data) % self.batch_size
        if remainder != 0:
            data = np.concatenate((data, data[:remainder]))

        # Get the index of the target label, which corresponds to the target parameters.
        target_label_index = self.quclassi.labels.index(label)

        iterations = len(data) // self.batch_size
        with tqdm(range(iterations), leave=False) as titerations:
            for iteration in titerations:
                titerations.set_description(f"Iteration {iteration}")
                # Get target data for this iteration.
                start_index = iteration * self.batch_size
                end_index = iteration * self.batch_size + self.batch_size
                target_data = data[start_index:end_index]

                # Get the forward difference.
                forward_difference_parameters = self.current_parameters[
                    target_label_index
                ] + (np.pi / (2 * np.sqrt(epoch)))
                forward_difference_parameters = {
                    trainable_parameter: trained_parameter
                    for trainable_parameter, trained_parameter in zip(
                        self.quclassi.trainable_parameters,
                        forward_difference_parameters,
                    )
                }
                forward_difference_fidelities = self.get_fidelities(
                    data=target_data,
                    trained_parameters=forward_difference_parameters,
                    sampler=sampler,
                    shots=shots,
                )
                forward_difference_fidelity = -np.log(
                    np.average(forward_difference_fidelities)
                )

                # Get the backward differene.
                backward_difference_parameters = self.current_parameters[
                    target_label_index
                ] - (np.pi / (2 * np.sqrt(epoch)))
                backward_difference_parameters = {
                    trainable_parameter: trained_parameter
                    for trainable_parameter, trained_parameter in zip(
                        self.quclassi.trainable_parameters,
                        backward_difference_parameters,
                    )
                }
                backward_difference_fidelities = self.get_fidelities(
                    data=target_data,
                    trained_parameters=backward_difference_parameters,
                    sampler=sampler,
                    shots=shots,
                )
                backward_difference_fidelity = -np.log(
                    np.average(backward_difference_fidelities)
                )

                # Update the current parameters.
                diff = 0.5 * (
                    forward_difference_fidelity - backward_difference_fidelity
                )
                if diff <= 0:
                    diff = 10 ** (-10)
                self.current_parameters[target_label_index] -= diff * self.learning_rate

    def get_fidelities(
        self,
        data: np.ndarray,
        trained_parameters: dict[str, float],
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 1024,
    ) -> np.ndarray:
        """Get the sequence of fidelities.

        :param np.ndarray data: data to calculate fidelity
        :param dict[str, float] trained_parameters: parameters to calculate fidelity
        :param primitives.BaseSamplerV1  |  primitives.BaseSamplerV2 sampler: sampler, defaults to primitives.StatevectorSampler(seed=901)
        :param int shots: number of shots, defaults to 1024
        :return np.ndarray: sequence of fidelities
        """
        # Create the combination of the circuit and parameters to run the circuits.
        pubs = []
        for _td in data:
            data_parameters = {
                data_parameter: _d
                for data_parameter, _d in zip(self.quclassi.data_parameters, _td)
            }
            parameters = {**trained_parameters, **data_parameters}
            pubs.append((self.quclassi.circuit, parameters))

        # Run the sampler.
        job = sampler.run(pubs, shots=shots)
        # Calculate the sequence of the fidelities.
        fidelities = []
        results = job.result()
        for result in results:
            fidelities.append(
                src.utils.calculate_fidelity_from_swap_test(result.data.c.get_counts())
            )
        return fidelities
