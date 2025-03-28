import os
import tempfile

import numpy as np
import qiskit
import pytest

from quantum_machine_learning.quclassi.quclassi import QuClassi


class TestQuClassi:
    @classmethod
    def setup_class(cls):
        pass

    def test_init(self):
        """Normal test;
        create an instance of QuClassi with valid arguments.

        Check if
        - its classical_data_size is the same as the given classical_data_size.
        - its structure is the same as the given structure.
        - its labels are the same as the given labels.
        - its parameter values are the same as the given initial_parameters.
        - the things above are correct after substituting a new values.
        - the type of its trainable_parameters is qiskit.circuit.ParameterExpression.
        - the type of its data_parameters is qiskit.circuit.ParameterExpression.
        """
        classical_data_size = 3
        structure = "s"
        labels = ["a", "b"]
        initial_parameters = {"a": [[1, 2], [3, 4]], "b": [[3, 4], [5, 6]]}
        quclassi = QuClassi(
            classical_data_size=classical_data_size,
            structure=structure,
            labels=labels,
            initial_parameters=initial_parameters,
        )
        assert quclassi.classical_data_size == classical_data_size
        assert quclassi.structure == structure
        assert quclassi.labels == labels
        assert quclassi.parameter_values == initial_parameters

        new_classical_data_size = classical_data_size + 1
        quclassi.classical_data_size = new_classical_data_size
        assert quclassi.classical_data_size == new_classical_data_size

        new_structure = structure + "d" + "e"
        quclassi.structure = new_structure
        assert quclassi.structure == new_structure

        new_labels = labels + ["c"]
        quclassi.labels = new_labels
        assert quclassi.labels == new_labels

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

    def test_load(self):
        """Normal test;
        Run load.

        Check if the loaded QuClassi instance is the same as the quclassi that is saved.
        """
        self.quclassi.build(self.structure)
        self.quclassi.save(self.model_dir_path)

        loaded_quclassi = QuClassi.load(self.model_dir_path)
        assert self.quclassi.classical_data_size == loaded_quclassi.classical_data_size
        assert self.quclassi.labels == loaded_quclassi.labels
        assert self.quclassi.circuit == loaded_quclassi.circuit
        assert (
            self.quclassi.trainable_parameters == loaded_quclassi.trainable_parameters
        )
        assert self.quclassi.data_parameters == loaded_quclassi.data_parameters
        assert self.quclassi.trained_parameters == loaded_quclassi.trained_parameters

        basic_info_path = os.path.join(self.model_dir_path, "basic_info.pkl")
        os.remove(basic_info_path)

        circuit_path = os.path.join(self.model_dir_path, "circuit.qpy")
        os.remove(circuit_path)

        trainable_parameters_path = os.path.join(
            self.model_dir_path, "trainable_parameters.pkl"
        )
        os.remove(trainable_parameters_path)

        data_parameters_path = os.path.join(self.model_dir_path, "data_parameters.pkl")
        os.remove(data_parameters_path)
        trained_parameters_path = os.path.join(
            self.model_dir_path, "trained_parameters.pkl"
        )
        os.remove(trained_parameters_path)
        os.rmdir(self.model_dir_path)

    def test_get_fidelities_without_trained_parameters(self):
        """Abnormal test;
        Run get_fidelities function without setting the trained parameters.

        Check if ValueError happens.
        """
        if self.quclassi.trained_parameters is not None:
            self.quclassi.trained_parameters = None
        with pytest.raises(ValueError):
            self.quclassi.get_fidelities(self.data)

    def test_get_fidelities_with_trained_parameters(self):
        """Normal test;
        Run get_fidelities function after setting the trained_parameters.

        Check if
        - the type of the return value is dict.
        - the return value's keys coincide with self.quclassi.labels.
        - the return value's values are between 0 and 1.
        """
        self.quclassi.trained_parameters = self.trained_parameters
        fidelities = self.quclassi.get_fidelities(self.data)
        # Check the type of fidelieis.
        assert isinstance(fidelities, dict)
        # Check the keys of fidelities concide with self.quclassi.labels.
        assert set(self.quclassi.labels) == set(fidelities.keys())
        assert len(self.quclassi.labels) == len(fidelities.keys())
        # Check the values of fidelities are between 0 and 1.
        for fidelity in fidelities.values():
            assert 0 <= fidelity <= 1

    def test_classify_without_trained_parameters(self):
        """Abnormal test;
        Run classify function without setting the trained parameters.

        Check if ValueError happens.
        """
        if self.quclassi.trained_parameters is not None:
            self.quclassi.trained_parameters = None
        with pytest.raises(ValueError):
            self.quclassi.classify(self.data)

    def test_classify_with_trained_parameters(self):
        """Normal test;
        Run classify function after setting the trained_parameters.

        Check if the returned value is in self.labels.
        """
        self.quclassi.trained_parameters = self.trained_parameters
        label_classified = self.quclassi.classify(self.data)
        assert label_classified in self.labels
