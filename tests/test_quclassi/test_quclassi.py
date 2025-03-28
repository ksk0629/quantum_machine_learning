import itertools
import os
import string
import tempfile

import qiskit
import pytest

from quantum_machine_learning.quclassi.quclassi import QuClassi


class TestQuClassi:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.quclassi
    def test_init(self):
        """Normal test;
        create an instance of QuClassi with valid arguments.

        Check if
        - its classical_data_size is the same as the given classical_data_size.
        - its using_classical_data_size is the nearest larger even number of classical_data_size.
        - its num_data_qubits is the half of its using_classical_data_size.
        - its num_train_qubits is the half of its using_classical_data_size.
        - its structure is the same as the given structure.
        - its labels are the same as the given labels.
        - its parameter values are the same as the given initial_parameters.
        - its classical_data_size is the same as the given new classical_data_size
          after substituting a new classical_data_size.
        - its using_classical_data_size is the nearest larger even number of
          new classical_data_size after substituting a new classical_data_size.
        - its num_data_qubits is the half of its new using_classical_data_size
          after substituting a new classical_data_size.
        - its num_train_qubits is the half of its new using_classical_data_size
          after substituting a new classical_data_size.
        - its structure is the same as the given new structure
          after substituting a new structure.
        - its labels is the same as the given new labels after substituting a new labels.
        - its parameter values are empty dict after substituting a new structure.
        - its parameter values are the same as the new parameter values
          after substituting new parameter values.
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
        assert quclassi.using_classical_data_size == classical_data_size + 1
        assert quclassi.num_data_qubits == quclassi.using_classical_data_size // 2
        assert quclassi.num_train_qubits == quclassi.using_classical_data_size // 2
        assert quclassi.structure == structure
        assert quclassi.labels == labels
        assert quclassi.parameter_values == initial_parameters

        new_classical_data_size = classical_data_size + 1
        quclassi.classical_data_size = new_classical_data_size
        assert quclassi.classical_data_size == new_classical_data_size
        assert quclassi.using_classical_data_size == new_classical_data_size
        assert quclassi.num_data_qubits == quclassi.using_classical_data_size // 2
        assert quclassi.num_train_qubits == quclassi.using_classical_data_size // 2

        new_structure = structure + "d" + "e"
        quclassi.structure = new_structure
        assert quclassi.structure == new_structure

        new_labels = labels + ["c"]
        quclassi.labels = new_labels
        assert quclassi.labels == new_labels
        assert quclassi.parameter_values == dict()

        new_parameter_values = initial_parameters
        new_parameter_values["c"] = [[7, 8], [9, 10]]
        quclassi.parameter_values = new_parameter_values
        assert quclassi.parameter_values == new_parameter_values

        assert isinstance(
            quclassi.trainable_parameters, qiskit.circuit.ParameterExpression
        )
        assert isinstance(quclassi.data_parameters, qiskit.circuit.ParameterExpression)

    @pytest.mark.quclassi
    @pytest.mark.parametrize(
        "all_letters",
        [
            "s",
            "d",
            "e",
            "sd",
            "se",
            "de",
            "sde",
            "ssde",
            "sdde",
            "sdee",
            "ssdde",
            "sddee",
            "ssddee",
        ],
    )
    def test_valid_structure(self, all_letters):
        """Normal test;
        Create an instance of QuClassi with several valid structures.

        Check if its structure is the same as the given structure.
        """
        classical_data_size = 3
        labels = ["a", "b"]
        for conb in itertools.permutations(all_letters):
            for structure_letters in conb:
                structure = "".join(structure_letters)
                quclassi = QuClassi(
                    classical_data_size=classical_data_size,
                    structure=structure,
                    labels=labels,
                )
                assert quclassi.structure == structure

    @pytest.mark.quclassi
    @pytest.mark.parametrize(
        "all_letters",
        [
            "s",
            "d",
            "e",
            "sd",
            "se",
            "de",
            "sde",
            "ssde",
            "sdde",
            "sdee",
            "ssdde",
            "sddee",
            "ssddee",
        ],
    )
    def test_invalid_structure(self, all_letters):
        """Abormal test;
        Create an instance of QuClassi with several invalid structures.

        Check if AttributeError happens.
        """
        classical_data_size = 3
        labels = ["a", "b"]
        for i in string.ascii_letters:  # Loop for all letters.
            if i == "s" or i == "d" or i == "e":
                # Continue if i is one of the valid letters.
                continue
            invalid_all_letters = all_letters + i
            for conb in itertools.permutations(invalid_all_letters):
                for structure_letters in conb:
                    structure = str(structure_letters)
                    with pytest.raises(AttributeError):
                        QuClassi(
                            classical_data_size=classical_data_size,
                            structure=structure,
                            labels=labels,
                        )

    @pytest.mark.quclassi
    @pytest.mark.parametrize("labels", [None, ["b"]])
    def test_invalid_parameter_values(self, labels):
        """Abnormal test;
        set invalid parameter values.

        Check if AttributeError happens.
        """
        classical_data_size = 3
        structure = "s"
        quclassi = QuClassi(
            classical_data_size=classical_data_size, labels=labels, structure=structure
        )

        parameter_values = {"a": [[1, 2], [3, 4]]}
        with pytest.raises(AttributeError):
            # If labels is None, AttributeError must happen.
            # Also, if the keys of the parameter values are not the same labels,
            # Attribute Error must happens.
            quclassi.parameter_values = parameter_values
