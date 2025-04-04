import copy
import os
from typing import Final

import numpy as np
import qiskit
from tqdm.auto import tqdm
import yaml

from quantum_machine_learning.quclassi.quclassi import QuClassi
from quantum_machine_learning.postprocessor.postprocessor import Postprocessor
from quantum_machine_learning.utils.utils import Utils
from quantum_machine_learning.utils.calculation_utils import CalculationUtils


class QuClassiTrainer:
    """QuClassiTrainer class that trains QuClassi with the algorithm introduced in the original paper:
    https://arxiv.org/pdf/2103.11307.
    """

    # Define the filename of parameters.
    PARAMETERS_PATH: Final[str] = "parameters.yaml"

    def __init__(
        self,
        quclassi: QuClassi,
        backend: qiskit.providers.Backend,
        epochs: int = 25,
        learning_rate: float = 0.01,
        batch_size: int = 1,
        shuffle: bool = True,
        shots: int = 8192,
        seed: int | None = 901,
        optimisation_level: int = 2,
    ):
        """Initialise this trainer.

        :param QuClassi quclassi: QuClassi to be trained
        :param qiskit.providers.Backend backend: a backend
        :param int epochs: the number of epochs, defaults to 25
        :param float learning_rate: a learning rate, defaults to 0.01
        :param int batch_size: a batch size, defaults to 1
        :param bool shuffle: if the data will be shuffled, defaults to True
        :param int shots: the number of shots, defaults to 8192
        :param int | None seed: a random seed, defaults to 901
        :param int optimisation_level: the level of the optimisation, defaults to 2
        """
        self.quclassi = quclassi
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.backend = backend
        self.shots = shots
        self.seed = seed
        self.optimisation_level = optimisation_level

        # Fix the np.random seed.
        Utils.fix_seed(seed=seed)

        # Initialise the histories.
        self.losses = None
        self.accuracies = None
        self.parameters = None

    def train(
        self,
        data: list[list[float]],
        labels: list[str],
        save_per_epoch: bool = False,
        model_dir_path: str | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Train the QuClassi.

        :param list[list[float]] data: data to train the QuClassi
        :param list[str] labels: labels to train the QuClassi
        :param bool save_per_epoch: if updated parameters should be saved per epoch, defaults to False
        :param str | None model_dir_path: a path to a directory to save parameters, defaults to None
        :raises ValueError: if the lengths of the data and labels are not the same
        :raises ValueError: if save_per_epoch is True and model_dir_path is None
        :return tuple[dict[str, float], dict[str, float]]: the last losses and accuracies
        """
        if len(data) != len(labels):
            error_msg = f"The lengths of data and labels must be same. However, {len(data)} vs {len(labels)}."
            raise ValueError(error_msg)

        if save_per_epoch and (model_dir_path is None):
            error_msg = f"If the save_per_epoch is True, then model_dir_path must be given, but it is None."
            raise ValueError(error_msg)

        if self.quclassi.parameter_values == dict():
            if not self.quclassi._is_built:
                self.quclassi._build()
            # If no parameter values are not, set them randomly.
            num_parameters_per_label = len(self.quclassi.trainable_parameters)
            self.quclassi.parameter_values = {
                label: np.random.rand(num_parameters_per_label) * (2 * np.pi)
                for label in self.quclassi.labels
            }

        # Initialise the histories.
        self.losses = {label: [] for label in self.quclassi.labels}
        self.accuracies = {label: [] for label in self.quclassi.labels}
        self.parameters = []

        # Transform the data by self._transformer.
        if self.quclassi._transformer is not None:
            data = self.quclassi._transformer(data)

        # Append 0 to each datum if their dimension is not the same as the using classical data size.
        if len(data[0]) != self.quclassi.using_classical_data_size:
            expanded_data = np.zeros(
                (len(data), self.quclassi.using_classical_data_size)
            )
            expanded_data[:, :-1] = data
            data = expanded_data.tolist()

        # Separate the dataset by their labels.
        data_separated_by_label = dict()
        for label in self.quclassi.labels:
            # Separate the data for training by their labels.
            target_indices = np.where(np.array(labels) == label)
            data_separated_by_label[label] = np.array(data)[target_indices].tolist()

        # Train the QuClassi.
        with tqdm(range(1, self.epochs + 1)) as tepoch:
            for epoch in tepoch:
                # Set the description.
                tepoch.set_description(f"Epoch {epoch}")

                with tqdm(self.quclassi.labels, leave=False) as tlabels:
                    for label in tlabels:
                        # Set the description.
                        tlabels.set_description(f"Label {label}")

                        # Train the QuClassi with the target data.
                        target_data = data_separated_by_label[label]
                        loss, accuracy = self._train_one_epoch(
                            data=target_data,
                            label=label,
                            epoch=epoch,
                            save=save_per_epoch,
                            model_dir_path=model_dir_path,
                        )

                        # Store the loss value and accuracies.
                        self.losses[label].append(loss)
                        self.accuracies[label].append(accuracy)

                    # Store the current parameters.
                    self.parameters.append(
                        copy.deepcopy(self.quclassi.parameter_values)
                    )

                    # Set the description.
                    loss_description = {
                        f"Loss_{label}": self.losses[label][-1]
                        for label in self.quclassi.labels
                    }
                    accuracy_description = {
                        f"Accuracy_{label}": self.accuracies[label][-1]
                        for label in self.quclassi.labels
                    }
                    description = {**loss_description, **accuracy_description}
                    tepoch.set_postfix(description)

        last_losses = {label: self.losses[label][-1] for label in self.quclassi.labels}
        last_accuracies = {
            label: self.accuracies[label][-1] for label in self.quclassi.labels
        }
        return last_losses, last_accuracies

    def _train_one_epoch(
        self,
        data: list[list[float]],
        label: str,
        epoch: int,
        save: bool,
        model_dir_path: str | None = None,
    ) -> tuple[float, float]:
        """Train the QuClassi only one epoch.

        :param list[list[float]] data: data to train the QuClassi
        :param str label: the label of the given data
        :param int epoch: the current epoch number
        :param bool save: if updated parameters should be saved
        :param str | None model_dir_path: a path to a directory to save parameters, defaults to None
        :return tuple[float, float]: the losses and accuracies
        """
        # Shuffle the data if needed.
        if self.shuffle:
            np.random.shuffle(data)

        # Adjust the size of the data if needed accordinf to the batch size.
        remainder = len(data) % self.batch_size
        if remainder != 0:
            data = np.concatenate((data, data[:remainder])).tolist()

        shift_value = np.pi / (2 * np.sqrt(epoch))

        # Iterate the number of batches.
        num_iterations = len(data) // self.batch_size
        with tqdm(range(num_iterations), leave=False) as tnum_iterations:
            for iteration in tnum_iterations:
                tnum_iterations.set_description(f"Iteration {iteration}")
                # Get the target data for this iteration.
                start_index = iteration * self.batch_size
                end_index = iteration * self.batch_size + self.batch_size
                target_data = data[start_index:end_index]

                # Train each parameter.
                num_parameters = len(self.quclassi.parameter_values[label])
                for parameter_index in range(num_parameters):
                    # Get the forward shifted parameter values.
                    forward_difference = self._get_difference(
                        data=target_data,
                        parameter_values=self.quclassi.parameter_values[label],
                        target_paramerter_index=parameter_index,
                        shift_value=shift_value,
                    )

                    # Get the backward shifted parameter values.
                    backward_difference = self._get_difference(
                        data=target_data,
                        parameter_values=self.quclassi.parameter_values[label],
                        target_paramerter_index=parameter_index,
                        shift_value=-shift_value,
                    )

                    # Update the current parameters.
                    diff = 0.5 * (forward_difference - backward_difference)
                    self.quclassi.parameter_values[label][parameter_index] -= (
                        diff * self.learning_rate
                    )

        if save and model_dir_path is not None:
            # If save is True and the model directory path is given,
            # save the current parameter values.
            parameters = {"initial_parameters": self.quclassi._parameter_values}
            # Save the parameters as the yaml file.
            num_digit = len(str(self.epochs))
            epoch_str = str(epoch).zfill(num_digit)
            filename = f"{epoch_str}_{QuClassiTrainer.PARAMETERS_PATH}"
            yaml_path = os.path.join(model_dir_path, filename)
            with open(yaml_path, "w", encoding=QuClassi.ENCODING) as yaml_file:
                yaml.dump(
                    parameters,
                    yaml_file,
                    default_flow_style=False,
                )

        # Get the probabilities of each datum.
        probabilities_list = [
            self.quclassi._get_probabilities(
                datum=datum,
                backend=self.backend,
                shots=self.shots,
                optimisation_level=self.optimisation_level,
                seed=self.seed,
            )
            for datum in data
        ]
        # Calculate the loss value.
        loss = CalculationUtils.calculate_cross_entropy(
            probabilities_list=probabilities_list, true_labels=[label] * len(data)
        )
        # Calculate the accuracy.
        predicted_labels = [
            max(probabilities, key=probabilities.get)
            for probabilities in probabilities_list
        ]
        accuracy = CalculationUtils.calculate_accuracy(
            predicted_labels=predicted_labels, true_labels=[label] * len(data)
        )

        return loss, accuracy

    def _get_difference(
        self,
        data: list[list[float]],
        parameter_values: list[float],
        target_paramerter_index: int,
        shift_value: float,
    ) -> float:
        """Get the difference computed by executing the QuClassi using shifted parameters.

        :param list[list[float]] data: data to be fed to the QuClassi
        :param list[float] parameter_values: parameter values to be used
        :param int target_paramerter_index: an index of the parameter to be shifted
        :param float shift_value: a shift value
        :return float: the difference
        """
        # Shift the parameter values.
        parameter_values[target_paramerter_index] += shift_value
        shifted_trainable_parameters = {
            trainable_parameter: shifted_parameter_value
            for trainable_parameter, shifted_parameter_value in zip(
                self.quclassi.trainable_parameters,
                parameter_values,
            )
        }
        # Get the fidelities.
        shifted_fidelities = self._get_fidelities(
            data=data,
            trainable_parameters=shifted_trainable_parameters,
        )
        # Calculate the average forward shifted fidelity.
        shifted_averaged_fidelity = np.average(shifted_fidelities)
        # Calculate the difference.
        difference = -CalculationUtils.safe_log_e(shifted_averaged_fidelity)

        parameter_values[target_paramerter_index] -= shift_value

        return difference

    def _get_fidelities(
        self,
        data: list[list[float]],
        trainable_parameters: dict[str, float],
    ) -> list[float]:
        """Get fidelities.

        :param list[list[float]] data: data to calculate the fidelities
        :param dict[str, float] trainable_parameters: parameters to be fed to the QuClassi
        :return list[float]: the fidelities
        """
        # Transplie the circuits.
        transpiled_circuit = self.quclassi._get_transpiled(
            optimisation_level=self.optimisation_level,
            backend=self.backend,
            seed=self.seed,
        )

        # Create primitive unified blocks.
        primitive_unified_blocks = []
        for datum in data:
            data_parameters = {
                data_parameter: d
                for data_parameter, d in zip(self.quclassi.data_parameters, datum)
            }
            primitive_unified_block = transpiled_circuit.assign_parameters(
                {**trainable_parameters, **data_parameters}
            )
            primitive_unified_blocks.append(primitive_unified_block)

        # Run the sampler.
        jobs = self.backend.run(primitive_unified_blocks, shots=self.shots)

        # Calculate the sequence of the fidelities.
        fidelities = []
        results = jobs.result().results
        for result in results:
            digit_result = {
                str(int(key, 16)): value for key, value in result.data.counts.items()
            }  # The keys are initially expressed in hex.
            fidelities.append(
                Postprocessor.calculate_fidelity_from_swap_test(digit_result)
            )

        return fidelities
