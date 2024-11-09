import numpy as np
import qiskit
import pytest

from src.quclassi.quclassi import QuClassi


class TestQuClassi:
    @classmethod
    def setup_class(cls):
        cls.classical_data_size = 5
        cls.labels = ["A", "B", "C"]
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
