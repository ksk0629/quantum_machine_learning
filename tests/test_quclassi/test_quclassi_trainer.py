import glob
import os
import tempfile

import qiskit_aer
import pytest
import yaml

from quantum_machine_learning.quclassi.quclassi import QuClassi
from quantum_machine_learning.quclassi.quclassi_trainer import QuClassiTrainer


class TestQuClassiTrainer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.quclassi
    def test_init(self):
        """Normal test;
        create an instance of QuClassiTrainer.

        Check if
        - its qulcassi is the same as the given quclassi.
        - its epochs is the same as the given epochs.
        - its learning_rate is the same as the given learning_rate.
        - its batch_size is the same as the given batch_size.
        - its backend is the same as the given backend.
        - its shots is the same as the given shots.
        - its seed is the same as the given seed.
        - its optimisation_level is the same as the given optimisation_level.
        - its losses is None.
        - its accuracies is None.
        - its parameters is None.
        - the above things hold after substituting new values.
        """
        classical_data_size = 3
        structure = "s"
        labels = ["a", "b"]
        initial_parameters = {"a": [[1, 2, 3, 4]], "b": [[3, 4, 5, 6]]}
        quclassi = QuClassi(
            classical_data_size=classical_data_size,
            structure=structure,
            labels=labels,
            initial_parameters=initial_parameters,
        )
        epochs = 1
        learning_rate = 0.1
        batch_size = 2
        shuffle = False
        backend = qiskit_aer.AerSimulator(seed_simulator=901)
        shots = 1
        seed = 91
        optimisation_level = 2
        quclassi_trainer = QuClassiTrainer(
            quclassi=quclassi,
            backend=backend,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            shuffle=shuffle,
            shots=shots,
            seed=seed,
            optimisation_level=optimisation_level,
        )
        assert quclassi_trainer.quclassi is quclassi
        assert quclassi_trainer.epochs == epochs
        assert quclassi_trainer.learning_rate == learning_rate
        assert quclassi_trainer.batch_size == batch_size
        assert quclassi_trainer.shuffle == shuffle
        assert quclassi_trainer.backend == backend
        assert quclassi_trainer.shots == shots
        assert quclassi_trainer.seed == seed
        assert quclassi_trainer.losses is None
        assert quclassi_trainer.accuracies is None
        assert quclassi_trainer.parameters is None

        new_value = None
        quclassi_trainer.quclassi = new_value
        quclassi_trainer.epochs = new_value
        quclassi_trainer.learning_rate = new_value
        quclassi_trainer.backend = new_value
        quclassi_trainer.batch_size = new_value
        quclassi_trainer.shuffle = new_value
        quclassi_trainer.shots = new_value
        quclassi_trainer.seed = new_value
        quclassi_trainer.optimisation_level = new_value
        assert quclassi_trainer.quclassi is new_value
        assert quclassi_trainer.epochs == new_value
        assert quclassi_trainer.learning_rate == new_value
        assert quclassi_trainer.batch_size == new_value
        assert quclassi_trainer.shuffle == new_value
        assert quclassi_trainer.backend == new_value
        assert quclassi_trainer.shots == new_value
        assert quclassi_trainer.seed == new_value
        assert quclassi_trainer.optimisation_level == new_value
        assert quclassi_trainer.losses is None
        assert quclassi_trainer.accuracies is None
        assert quclassi_trainer.parameters is None

    @pytest.mark.quclassi
    def test_train_without_saving(self):
        """Normal test;
        run train method without saving.

        Check if
        - the first return value, supposedly loss, is dict.
        - the keys of the first return value are the labels of the given QuClassi.
        - the values of the first return value are floats.
        """
        # Create data for training.
        data_0 = [1, 0]  # |0>
        data_1 = [0, 1]  # |1>
        num_data = 4
        data = []
        data_labels = []
        for _ in range(num_data):
            data.append(data_0)
            data_labels.append("0")
            data.append(data_1)
            data_labels.append("1")
        # Create an instance of QuClassi for the created data.
        structure = "s"
        labels = ["0", "1"]
        quclassi = QuClassi(
            classical_data_size=len(data_0),
            structure=structure,
            labels=labels,
        )
        # Create an instance of QuClassiTrainer.
        epochs = 2
        batch_size = 3
        backend = qiskit_aer.AerSimulator(seed_simulator=901)
        quclassi_trainer = QuClassiTrainer(
            quclassi=quclassi, backend=backend, epochs=epochs, batch_size=batch_size
        )
        # Train the QuClassi.
        loss, accuracy = quclassi_trainer.train(data=data, labels=data_labels)

        assert isinstance(loss, dict)
        assert set(loss.keys()) == set(quclassi.labels)
        assert all(isinstance(value, float) for value in loss.values())
        assert isinstance(accuracy, dict)
        assert set(accuracy.keys()) == set(quclassi.labels)
        assert all(isinstance(value, float) for value in accuracy.values())

    @pytest.mark.quclassi
    def test_train_with_saving(self):
        """Normal test;
        run train method with saving.

        Check if
        - epochs-parameters yaml files are in the specified directory.
        - each yaml file has a correct shape.
        - the first return value, supposedly loss, is dict.
        - the keys of the first return value are the labels of the given QuClassi.
        - the values of the first return value are floats.
        """
        # Create data for training.
        data_0 = [1, 0]  # |0>
        data_1 = [0, 1]  # |1>
        num_data = 1
        data = []
        data_labels = []
        for _ in range(num_data):
            data.append(data_0)
            data_labels.append("0")
            data.append(data_1)
            data_labels.append("1")
        # Create an instance of QuClassi for the created data.
        structure = "s"
        labels = ["0", "1"]
        quclassi = QuClassi(
            classical_data_size=len(data_0),
            structure=structure,
            labels=labels,
        )
        # Create an instance of QuClassiTrainer.
        epochs = 2
        backend = qiskit_aer.AerSimulator(seed_simulator=901)
        quclassi_trainer = QuClassiTrainer(
            quclassi=quclassi, backend=backend, epochs=epochs
        )
        # Train the QuClassi.
        with tempfile.TemporaryDirectory() as tmp_dir_path:
            loss, accuracy = quclassi_trainer.train(
                data=data,
                labels=data_labels,
                save_per_epoch=True,
                model_dir_path=tmp_dir_path,
            )

            target_file_name = os.path.join(tmp_dir_path, "./*.yaml")
            target_file_paths = glob.glob(target_file_name)
            assert len(target_file_paths) == epochs

            for target_file_path in target_file_paths:
                with open(
                    target_file_path, "r", encoding=QuClassi.ENCODING
                ) as yaml_file:
                    parameters = yaml.unsafe_load(yaml_file)
                assert set(parameters.keys()) == set(["initial_parameters"])
                parameters = parameters["initial_parameters"]
                assert set(parameters.keys()) == set(quclassi_trainer.quclassi.labels)
                for values in parameters.values():
                    assert len(values) == 2  # Because the structure is "s"

            assert isinstance(loss, dict)
            assert set(loss.keys()) == set(quclassi_trainer.quclassi.labels)
            assert all(isinstance(value, float) for value in loss.values())
            assert isinstance(accuracy, dict)
            assert set(accuracy.keys()) == set(quclassi_trainer.quclassi.labels)
            assert all(isinstance(value, float) for value in accuracy.values())

    @pytest.mark.quclassi
    def test_train_with_invalid_data(self):
        """Abnormal test;
        run train method with unbalance data and labels,
        which means that the lengths of the data and labels are not the same.

        Check if ValueError happens.
        """
        classical_data_size = 3
        structure = "s"
        labels = ["a", "b"]
        quclassi = QuClassi(
            classical_data_size=classical_data_size,
            structure=structure,
            labels=labels,
        )
        backend = qiskit_aer.AerSimulator(seed_simulator=901)
        quclassi_trainer = QuClassiTrainer(quclassi=quclassi, backend=backend)

        data = [[1, 2]]
        with pytest.raises(ValueError):
            quclassi_trainer.train(data=data, labels=labels)

    @pytest.mark.quclassi
    def test_train_without_giving_model_dir_while_saving(self):
        """Abnormal test;
        run train method with save_per_epoch is True and model_dir_path is None.

        Check if ValueError happens.
        """
        classical_data_size = 3
        structure = "s"
        labels = ["a", "b"]
        quclassi = QuClassi(
            classical_data_size=classical_data_size,
            structure=structure,
            labels=labels,
        )
        backend = qiskit_aer.AerSimulator(seed_simulator=901)
        quclassi_trainer = QuClassiTrainer(quclassi=quclassi, backend=backend)

        data = [[1, 2], [3, 4]]
        with pytest.raises(ValueError):
            quclassi_trainer.train(
                data=data, labels=labels, save_per_epoch=True, model_dir_path=None
            )
