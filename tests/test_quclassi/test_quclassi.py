import os

import numpy as np
import qiskit
import pytest

from src.quclassi.quclassi import QuClassi


class TestQuClassi:
    @classmethod
    def setup_class(cls):
        cls.classical_data_size = 5
        cls.labels = ["A", "B", "C"]
        cls.structure = "s"
        cls.model_dir_path = "./test/"
        cls.quclassi = QuClassi(
            classical_data_size=cls.classical_data_size, labels=cls.labels
        )

    def test_init(self):
        """Normal test;

        Check if self.quclassi has
        - classical_data_size as same as self.classical_data_size.
        - labels as same as self.labels.
        - num_data_qubits as the smallest integer that is greater than or equal to
          self.classical_data_size.
        - num_train_qubits as the smallest integer that is greater than or equal to
          self.classical_data_size.
        - num_qubits as same as double of the smallest integer that is greater than
          or equal to self.classical_data_size plus one.
        """
        assert self.quclassi.classical_data_size == self.classical_data_size
        assert self.quclassi.labels == self.labels
        num_data_qubits = int(np.ceil(self.classical_data_size / 2))
        assert self.quclassi.num_data_qubits == num_data_qubits
        num_train_qubits = int(np.ceil(self.classical_data_size / 2))
        assert self.quclassi.num_train_qubits == num_train_qubits
        assert self.quclassi.num_qubits == num_train_qubits + num_data_qubits + 1

    @pytest.mark.parametrize(
        "structure",
        ["s", "d", "e", "sd", "se", "de", "sde", "sss", "dseds"],
    )
    def test_build_with_valid_structure(self, structure):
        """Normal test;
        Run build function of self.quclassi with various structure.

        Check if
        - self.quclassi.circuit is an instance of qiskit.QuantumCircuit.
        - self.quclassi.trainable_parameters is qiskit.circuit.parametertable.ParameterView.
        - self.quclassi.data_parameters is qiskit.circuit.parametertable.ParameterView.
        - the length of self.quclassi.trainable_parameters is suitable,
          which varies depending on structure.
        - the length of self.quclassi.data_parameters is as same as
          self.quclassi.num_data_qubits * 2.
        """
        self.quclassi.build(structure)

        assert isinstance(self.quclassi.circuit, qiskit.QuantumCircuit)
        assert isinstance(
            self.quclassi.trainable_parameters,
            qiskit.circuit.parametertable.ParameterView,
        )
        assert isinstance(
            self.quclassi.data_parameters, qiskit.circuit.parametertable.ParameterView
        )
        assert len(self.quclassi.data_parameters) == self.quclassi.num_data_qubits * 2

        # Check the number of trainable_parameters
        num_trainable_parameters = 0
        num_trainable_parameters += (
            structure.count("s") * self.quclassi.num_train_qubits
        ) * 2
        num_trainable_parameters += (
            structure.count("d") * (self.quclassi.num_train_qubits - 1) * 2
        )
        num_trainable_parameters += (
            structure.count("e") * (self.quclassi.num_train_qubits - 1) * 2
        )
        assert len(self.quclassi.trainable_parameters) == num_trainable_parameters

    @pytest.mark.parametrize(
        "structure",
        ["as", "sac", "sdev", "a"],
    )
    def test_build_with_invalid_structure(self, structure):
        """Abnormal test;
        Run build function of self.quclassi with various structure.

        Case 1: structure contains any letters other than s, d or e.
        Check if VallueError happens.
        """
        with pytest.raises(ValueError):
            self.quclassi.build(structure)

    def test_get_basic_info_path(self):
        """Normal test;
        Run get_basic_info_path function.

        Check if the return value is self.model_dir_path/basic_info.pkl.
        """
        assert self.quclassi.get_basic_info_path(self.model_dir_path) == os.path.join(
            self.model_dir_path, "basic_info.pkl"
        )

    def test_get_circuit_path(self):
        """Normal test;
        Run get_circuit_path function.

        Check if the return value is self.model_dir_path/circuit.qpy.
        """
        assert self.quclassi.get_circuit_path(self.model_dir_path) == os.path.join(
            self.model_dir_path, "circuit.qpy"
        )

    def test_get_trainable_parameters_path(self):
        """Normal test;
        Run get_trainable_parameters_path function.

        Check if the return value is self.model_dir_path/trainable_parameters.pkl.
        """
        assert self.quclassi.get_trainable_parameters_path(
            self.model_dir_path
        ) == os.path.join(self.model_dir_path, "trainable_parameters.pkl")

    def test_get_data_parameters_path(self):
        """Normal test;
        Run get_data_parameters_path function.

        Check if the return value is self.model_dir_path/data_parameters.pkl.
        """
        assert self.quclassi.get_data_parameters_path(
            self.model_dir_path
        ) == os.path.join(self.model_dir_path, "data_parameters.pkl")

    def test_get_trained_parameters_path(self):
        """Normal test;
        Run get_trained_parameters_path function.

        Check if the return value is self.model_dir_path/trained_parameters.pkl.
        """
        assert self.quclassi.get_trained_parameters_path(
            self.model_dir_path
        ) == os.path.join(self.model_dir_path, "trained_parameters.pkl")

    def test_save(self):
        """Normal test;
        Run save function.

        Check if
        - there is basic_info.pkl.
        - there is circuit.qpy.
        - there is trainable_parameters.pkl.
        - there is data_parameters.pkl.
        - there is trained_parameters.pkl.
        - save does not work if there is the existing directory.
        """
        self.quclassi.build(self.structure)
        self.quclassi.save(self.model_dir_path)

        basic_info_path = os.path.join(self.model_dir_path, "basic_info.pkl")
        assert os.path.isfile(basic_info_path)
        os.remove(basic_info_path)

        circuit_path = os.path.join(self.model_dir_path, "circuit.qpy")
        assert os.path.isfile(circuit_path)
        os.remove(circuit_path)

        trainable_parameters_path = os.path.join(
            self.model_dir_path, "trainable_parameters.pkl"
        )
        assert os.path.isfile(trainable_parameters_path)
        os.remove(trainable_parameters_path)

        data_parameters_path = os.path.join(self.model_dir_path, "data_parameters.pkl")
        assert os.path.isfile(data_parameters_path)
        os.remove(data_parameters_path)
        trained_parameters_path = os.path.join(
            self.model_dir_path, "trained_parameters.pkl"
        )
        assert os.path.isfile(trained_parameters_path)
        os.remove(trained_parameters_path)

        with pytest.raises(OSError):
            self.quclassi.save(self.model_dir_path)

        os.rmdir(self.model_dir_path)
