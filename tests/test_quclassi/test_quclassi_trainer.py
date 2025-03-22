import glob
import os

import numpy as np
import qiskit
import pytest

from quantum_machine_learning.quclassi.quclassi import QuClassi
from quantum_machine_learning.quclassi.quclassi_trainer import QuClassiTrainer


class TestQuClassiTrainer:
    @classmethod
    def setup_class(cls):
        classical_data_size = 2
        labels = ["Large", "Small"]
        cls.positive_data = np.array([[0.9, 0.8], [0.8, 0.9]])
        cls.negative_data = np.array([[0.1, 0.2], [0.2, 0.1]])
        cls.train_data = np.concatenate((cls.positive_data, cls.negative_data))
        cls.train_labels = np.array(["Large", "Large", "Small", "Small"])
        structure = "s"
        cls.model_dir_path = "./test/"
        cls.quclassi = QuClassi(classical_data_size=classical_data_size, labels=labels)
        cls.quclassi.build(structure)

        cls.epochs = 2
        cls.batch_size = 3
        cls.trained_paramters = {"layer0[0]": 1, "layer0[1]": 1}
        cls.initial_parameters = np.array([[1, 1], [0, 0]])

    @pytest.mark.quclassi
    def get_trainer(self):
        return QuClassiTrainer(
            quclassi=self.quclassi,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

    @pytest.mark.quclassi
    def test_init_with_invalid_initial_parameters(self):
        """Abnormal test;
        Create the instance with an invalid initial_parameters.

        Check if ValueError happens.
        """
        initial_parameters = np.array([1])
        with pytest.raises(ValueError):
            QuClassiTrainer(
                quclassi=self.quclassi, initial_parameters=initial_parameters
            )

    @pytest.mark.quclassi
    def test_init_with_valid_initial_parameters(self):
        """Abnormal test;
        Create the instance with a valid initial_parameters.

        Check if
        - the length of the member variable parameters_history of the return value is one
        - the only element of the member variable parameters_history of the return value is the same as self.initial_parameters.
        - the current_parameters of the return value is the same as self.initial_parameters.
        """
        quclassi_trainer = QuClassiTrainer(
            quclassi=self.quclassi, initial_parameters=self.initial_parameters
        )
        assert len(quclassi_trainer.parameters_history) == 1
        assert np.allclose(
            quclassi_trainer.parameters_history[0], self.initial_parameters
        )
        assert np.allclose(quclassi_trainer.current_parameters, self.initial_parameters)

    @pytest.mark.quclassi
    def test_train_with_evaluation(self):
        """Normal test;
        Run train with eval=True.

        Check if
        - the length of parameters_history is self.epochs + 1.
        - the length of train_accuracies is self.epochs.
        - the length of val_accuracies is self.epochs.
        - the current_parameters is not the same as the first element of the parameter_histories.
        """
        quclassi_trainer = self.get_trainer()
        quclassi_trainer.train(
            train_data=self.train_data,
            train_labels=self.train_labels,
            val_data=self.train_data,
            val_labels=self.train_labels,
            eval=True,
        )
        assert len(quclassi_trainer.parameters_history) == self.epochs + 1
        assert len(quclassi_trainer.train_accuracies) == self.epochs
        assert len(quclassi_trainer.val_accuracies) == self.epochs
        assert not np.allclose(
            quclassi_trainer.parameters_history[0], quclassi_trainer.current_parameters
        )

    @pytest.mark.quclassi
    def test_train_without_evaluation(self):
        """Normal test;
        Run train without eval=True.

        Check if
        - the length of parameters_history is self.epochs + 1.
        - the length of train_accuracies is 1.
        - the length of val_accuracies is 1.
        - the current_parameters is not the same as the first element of the parameter_histories.
        """
        quclassi_trainer = self.get_trainer()
        quclassi_trainer.train(
            train_data=self.train_data,
            train_labels=self.train_labels,
            val_data=self.train_data,
            val_labels=self.train_labels,
            eval=False,
        )
        assert len(quclassi_trainer.parameters_history) == self.epochs + 1
        assert len(quclassi_trainer.train_accuracies) == 1
        assert len(quclassi_trainer.val_accuracies) == 1
        assert not np.allclose(
            quclassi_trainer.parameters_history[0], quclassi_trainer.current_parameters
        )

    @pytest.mark.quclassi
    def test_train_one_epoch(self):
        """Normal test;
        Run train_one_epoch.

        Check if the current_parameters is not the same as the first element of the parameter_histories.
        """
        quclassi_trainer = self.get_trainer()
        quclassi_trainer.train_one_epoch(
            train_data=self.positive_data, label="Large", epoch=1
        )
        assert not np.allclose(
            quclassi_trainer.parameters_history[0], quclassi_trainer.current_parameters
        )

    @pytest.mark.quclassi
    def test_run_sampler(self):
        """Normal test;
        Run run_sampler.

        Check if
        - the type of the return value is qiskit.primitives.primitive_job.PrimitiveJob.
        - the return value has the function result() and its return value's length is the same as the length of self.train_data.
        """
        quclassi_trainer = self.get_trainer()
        jobs = quclassi_trainer.run_sampler(
            self.train_data, trained_parameters=self.trained_paramters
        )
        assert isinstance(jobs, qiskit.primitives.primitive_job.PrimitiveJob)
        assert len(jobs.result()) == len(self.train_data)

    @pytest.mark.quclassi
    def test_get_fidelities(self):
        """Normal test;
        Run get_fidelities.

        Check if
        - the length of the return value is the same as the length of self.train_data.
        - each element of the return value is between 0 and 1.
        """
        quclassi_trainer = self.get_trainer()
        fidelities = quclassi_trainer.get_fidelities(
            self.train_data, trained_parameters=self.trained_paramters
        )
        assert len(fidelities) == len(self.train_data)
        for fidelity in fidelities:
            assert 0 <= fidelity <= 1

    @pytest.mark.quclassi
    def test_save(self):
        """Normal test;
        Run save after running train.

        Check if there are (self.epochs + 1) + 4 .pkl files under self.model_dir_path.
        Note that, the "+4" comes from QuClassi.save function.
        """
        quclassi_trainer = self.get_trainer()
        quclassi_trainer.train(
            train_data=self.train_data,
            train_labels=self.train_labels,
            val_data=self.train_data,
            val_labels=self.train_labels,
            eval=False,
        )

        quclassi_trainer.save(self.model_dir_path)
        pkl_files = glob.glob(os.path.join(self.model_dir_path, "*.pkl"))
        assert len(pkl_files) == self.epochs + 1 + 4

        all_files = glob.glob(os.path.join(self.model_dir_path, "*"))
        for file in all_files:
            os.remove(file)
        os.rmdir(self.model_dir_path)

    @pytest.mark.quclassi
    @pytest.mark.parametrize(
        "predicted_labels_and_true_labels", [[[1], [2, 3]], [[1, 2], [3]]]
    )
    def test_calculate_accuracy_with_invalid_args(
        self, predicted_labels_and_true_labels
    ):
        """Abnormal test;
        Run calculate_accuracy with an invalid arguments.

        Check if ValueError happens.
        """
        (predicted_labels, true_labels) = predicted_labels_and_true_labels
        with pytest.raises(ValueError):
            QuClassiTrainer.calculate_accuracy(
                predicted_labels=predicted_labels, true_labels=true_labels
            )

    @pytest.mark.quclassi
    @pytest.mark.parametrize(
        "predicted_labels_and_true_labels_and_accuracy",
        [[[1], [1], 1], [[1, 2], [1, 3], 0.5], [[1, 2], [2, 1], 0]],
    )
    def test_calculate_accuracy_with_valid_args(
        self, predicted_labels_and_true_labels_and_accuracy
    ):
        """Abnormal test;
        Run calculate_accuracy with a valid arguments.

        Check if the return value, which is an accuracy, is correct.
        """
        (predicted_labels, true_labels, true_accuracy) = (
            predicted_labels_and_true_labels_and_accuracy
        )
        predicted_labels = np.array(predicted_labels)
        true_labels = np.array(true_labels)
        accuracy = QuClassiTrainer.calculate_accuracy(
            predicted_labels=predicted_labels, true_labels=true_labels
        )
        assert accuracy == true_accuracy
