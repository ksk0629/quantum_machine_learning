import os
import pickle

import numpy as np
import qiskit
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
        sampler: (
            primitives.BaseSamplerV1 | primitives.BaseSamplerV2
        ) = primitives.StatevectorSampler(seed=901),
        shots: int = 1024,
    ):
        """Initialise this trainer.

        :param QuClassi quclassi: quclassi to be trained
        :param int epochs: number of epochs, defaults to 25
        :param float learning_rate: learning rate, defaults to 0.01
        :param int batch_size: batch size, defaults to 1
        :param bool shuffle: whether dataset is shuffled or not, defaults to True
        :param np.ndarray | None initial_paramters: initial parameters, defaults to None
        :param qiskit.primitives.BaseSamplerV1  |  qiskit.primitives.BaseSamplerV2 sampler: sampler primitives, defaults to qiskit.primitives.StatevectorSampler
        :param int shots: number of shots
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
        self.parameters_history = []
        if initial_paramters is not None:
            self.current_parameters = initial_paramters
        else:
            self.current_parameters = (
                np.random.rand(
                    len(self.quclassi.trainable_parameters) * len(self.quclassi.labels)
                )
                * np.pi
            ).reshape((len(quclassi.labels), -1))
        self.parameters_history.append(self.current_parameters)
        self.sampler = sampler
        self.shots = shots

        # Initialise the histories.
        self.train_accuracies = []
        self.val_accuracies = []

    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        eval: bool = False,
    ):
        """Train the quclassi.

        :param np.ndarray train_data: whole train data
        :param np.ndarray train_labels: corresponding train labels
        :param np.ndarray val_data: whole validation data
        :param np.ndarray val_labels: corresponding validation labels
        :param bool eval: whether evaluation has to be done or not
        """
        # Separate the data.
        train_data_separated_label = dict()
        val_data_separated_label = dict()
        for label in self.quclassi.labels:
            target_train_indices = np.where(train_labels == label)
            train_data_separated_label[label] = train_data[target_train_indices]
            target_val_indices = np.where(val_labels == label)
            val_data_separated_label[label] = val_data[target_val_indices]

        # Train self.quclassi.
        with tqdm(range(1, self.epochs + 1)) as tepoch:
            for epoch in tepoch:
                tepoch.set_description(f"Epoch {epoch} (train)")

                with tqdm(self.quclassi.labels, leave=False) as tlabels:
                    for label in tlabels:
                        tlabels.set_description(f"Label {label}")
                        target_train_data = train_data_separated_label[label]
                        self.train_one_epoch(
                            train_data=target_train_data,
                            label=label,
                            epoch=epoch,
                        )
                    self.parameters_history.append(self.current_parameters)

                if eval:
                    self.quclassi.trained_parameters = self.current_parameters
                    # Get the accuracies.
                    predicted_train_labels = [
                        self.quclassi(data) for data in train_data
                    ]
                    self.train_accuracies.append(
                        src.utils.calculate_accuracy(
                            predicted_labels=predicted_train_labels,
                            true_labels=train_labels,
                        )
                    )
                    predicted_val_labels = [self.quclassi(data) for data in val_data]
                    self.val_accuracies.append(
                        src.utils.calculate_accuracy(
                            predicted_labels=predicted_val_labels,
                            true_labels=val_labels,
                        )
                    )

                    tepoch.set_postfix(
                        {
                            "Train Acc": self.train_accuracies[-1],
                            "Val Acc": self.val_accuracies[-1],
                        }
                    )

        if not eval:
            # Get the accuracies.
            self.quclassi.trained_parameters = self.current_parameters
            predicted_train_labels = [self.quclassi(data) for data in train_data]
            self.train_accuracies.append(
                src.utils.calculate_accuracy(
                    predicted_labels=predicted_train_labels,
                    true_labels=train_labels,
                )
            )
            predicted_val_labels = [self.quclassi(data) for data in val_data]
            self.val_accuracies.append(
                src.utils.calculate_accuracy(
                    predicted_labels=predicted_val_labels,
                    true_labels=val_labels,
                )
            )
        print(f"Train Accuracy: {self.train_accuracies[-1]}")
        print(f"Validation Accuracy: {self.val_accuracies[-1]}")

        # Set the trained parameters to self.quclassi.
        self.quclassi.trained_parameters = self.current_parameters

    def train_one_epoch(
        self,
        train_data: np.ndarray,
        label: object,
        epoch: int,
    ):
        """Train the quclassi only one epoch for one class.

        :param np.ndarray train_data: train data belonging to the given label
        :param object label: label to which the data belong
        :param int epoch: current epoch
        :param np.ndarray val_data: validation data belonging to the given label
        """
        # Shuffle the data if needed.
        if self.shuffle:
            np.random.shuffle(train_data)

        # Adjust the size of the data if needed.
        remainder = len(train_data) % self.batch_size
        if remainder != 0:
            data = np.concatenate((data, data[:remainder]))

        # Get the index of the target label, which corresponds to the target parameters.
        target_label_index = self.quclassi.labels.index(label)

        iterations = len(train_data) // self.batch_size
        with tqdm(range(iterations), leave=False) as titerations:
            for iteration in titerations:
                titerations.set_description(f"Iteration {iteration}")
                # Get target data for this iteration.
                start_index = iteration * self.batch_size
                end_index = iteration * self.batch_size + self.batch_size
                target_data = train_data[start_index:end_index]

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

    def run_sampler(
        self,
        data: np.ndarray,
        trained_parameters: dict[str, float],
    ) -> qiskit.providers.Job:
        """Run the given sampler.

        :param np.ndarray data: data to run the circuit.
        :param dict[str, float] trained_parameters: parameters to run the circuit
        :return qiskit.providers.Job: result of running sampler
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
        job = self.sampler.run(pubs, shots=self.shots)
        return job

    def get_fidelities(
        self,
        data: np.ndarray,
        trained_parameters: dict[str, float],
    ) -> np.ndarray:
        """Get the sequence of fidelities.

        :param np.ndarray data: data to calculate fidelity
        :param dict[str, float] trained_parameters: parameters to calculate fidelity
        :return np.ndarray: sequence of fidelities
        """
        job = self.run_sampler(
            data=data,
            trained_parameters=trained_parameters,
        )
        # Calculate the sequence of the fidelities.
        fidelities = []
        results = job.result()
        for result in results:
            fidelities.append(
                src.utils.calculate_fidelity_from_swap_test(result.data.c.get_counts())
            )
        return fidelities

    def save(self, model_dir_path: str):
        """Save the circuit and parameters to the directory specified by the given model_dir_path.

        :param str model_dir_path: path to the output directory.
        """
        # Save the model.
        self.quclassi.save(model_dir_path=model_dir_path)

        # Save the trained_parameters for each epoch.
        trained_parameter_path = self.quclassi.get_trained_parameters_path(
            model_dir_path=model_dir_path
        )

        name, extension = os.path.splitext(os.path.basename(trained_parameter_path))
        for index, parameters in enumerate(self.parameters_history):
            parameters_path = os.path.join(
                model_dir_path, f"{name}_{index:0>{len(str(self.epochs))}}{extension}"
            )
            with open(parameters_path, "wb") as pkl_file:
                pickle.dump(parameters, pkl_file)
