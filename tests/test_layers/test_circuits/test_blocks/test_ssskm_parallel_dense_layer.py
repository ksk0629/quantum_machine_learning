import pytest

from quantum_machine_learning.layers.blocks.ssskm_parallel_dense_layer import (
    SSSKMParallelDenseLayer,
)


class TestSSSKMParallelDenseLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_qubits", [1, 2])
    @pytest.mark.parametrize("num_reputations", [1, 2])
    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_init_with_defaults(self, num_qubits, num_reputations, num_layers):
        """Normal test:
        Createt an instance of SSSKMParallelDenseLayer.

        Check if
        1. its num_qubits is the same as the given num_qubits.
        2. its num_reputations is the same as the given num_reputations.
        3. its num_layers is the same as the given num_layers.
        4. its parameter_prefix is an empty strings.
        5. its _encoder_class is YAngle.
        6. its _transformer transforms the argument data into the data * 3.14...
        7. its _trainable_parameter_values is None.
        8. the length of its _dense_layers is the same as the given num_layers.
        9. the length of its _trainable_parameters is the same as the given num_layers.
        10. the length of its encoding_parameters is the same as the given num_qubits.
        11. its total_qubits is its num_qubits * its num_layers.
        """
        block = SSSKMParallelDenseLayer(
            num_qubits=num_qubits,
            num_reputations=num_reputations,
            num_layers=num_layers,
        )
        # 1. its num_qubits is the same as the given num_qubits.
        # 2. its num_reputations is the same as the given num_reputations.
        # 3. its num_layers is the same as the given num_layers.
        # 4. its parameter_prefix is an empty strings.
        # 5. its _encoder_class is YAngle.
        # 6. its _transformer transforms the argument data into the data * 3.14...
        # 7. its _trainable_parameter_values is None.
        # 8. the length of its _dense_layers is the same as the given num_layers.
        # 9. the length of its _trainable_parameters is the same as the given num_layers.
        # 10. the length of its encoding_parameters is the same as the given num_qubits.
        # 11. its total_qubits is its num_qubits * its num_layers.

    @pytest.mark.layer
    @pytest.mark.parametrize("num_qubits", [1, 2])
    @pytest.mark.parametrize("num_reputations", [1, 2])
    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_init_with_false_build(self, num_qubits, num_reputations, num_layers):
        """Normal test:
        Createt an instance of SSSKMParallelDenseLayer with build=False.

        Check if
        1. its num_qubits is the same as the given num_qubits.
        2. its num_reputations is the same as the given num_reputations.
        3. its num_layers is the same as the given num_layers.
        4. its parameter_prefix is an empty strings.
        5. its _encoder_class is YAngle.
        6. its _transformer transforms the argument data into the data * 3.14...
        7. its _trainable_parameter_values is None.
        8. its _dense_layers is an empty list.
        9. its _trainable_parameters is an empty list.
        10. the length of its encoding_parameters is the same as the given num_qubits.
        11. its total_qubits is its num_qubits * its num_layers.
        """
        block = SSSKMParallelDenseLayer(
            num_qubits=num_qubits,
            num_reputations=num_reputations,
            num_layers=num_layers,
            build=False,
        )
        # 1. its num_qubits is the same as the given num_qubits.
        # 2. its num_reputations is the same as the given num_reputations.
        # 3. its num_layers is the same as the given num_layers.
        # 4. its parameter_prefix is an empty strings.
        # 5. its _encoder_class is YAngle.
        # 6. its _transformer transforms the argument data into the data * 3.14...
        # 7. its _trainable_parameter_values is None.
        # 8. its _dense_layers is an empty list.
        # 9. its _trainable_parameters is an empty list.
        # 10. the length of its encoding_parameters is the same as the given num_qubits.
        # 11. its total_qubits is its num_qubits * its num_layers.

    def test_init_encoder_class(self):
        """Normal test:
        Create an instance of SSSKMParallelDenseLayer with encoder_class=XAngle.

        Check if
        1. its _encoder_class is XAngle.
        """
        # 1. its _encoder_class is XAngle.

    def test_init_transformer(self):
        """Normal test:
        Create an instance of SSSKMParallelDenseLayer with some transformer.

        Check if
        1. its _transformer is the same as the give one.
        2. the return value of its _transformer is the same one as of the given one.
        """
        # 1. its _transformer is the same as the give one.
        # 2. the return value of its _transformer is the same one as of the given one.

    def test_init_trainable_parameter_values(self):
        """Normal test:
        Create an instance of SSSKMParallelDenseLayer with some trainable_parameter_values.

        Check if
        1. its _trainable_parameter_values is the same as the given one.
        """
        # 1. its _trainable_parameter_values is the same as the given one.

    def test_num_qubits(self):
        """Normal test:
        Call the setter and getter of num_qubits.

        Check if
        1. its num_qubits is the same as the given one.
        2. each member of its _dense_layer has its-num_qubits num_qubits.
        3. its _encoder has its-num_qubits parameters.
        4. its total_qubits is the same as its num_qubits * its num_layers.
        5. its num_qubits is the new one after setting a new one.
        6. each member of its _dense_layer has its-num_qubits num_qubits.
        7. its _encoder has its-num_qubits parameters.
        8. its total_qubits is the same as its num_qubits * its num_layers.
        """
        # 1. its num_qubits is the same as the given one.
        # 2. each member of its _dense_layer has its-num_qubits num_qubits.
        # 3. its _encoder has its-num_qubits parameters.
        # 4. its total_qubits is the same as its num_qubits * its num_layers.
        # 5. its num_qubits is the new one after setting a new one.
        # 6. each member of its _dense_layer has its-num_qubits num_qubits.
        # 7. its _encoder has its-num_qubits parameters.
        # 8. its total_qubits is the same as its num_qubits * its num_layers.

    def test_num_reputations(self):
        """Normal test:
        Call the setter and getter of num_reputations.

        Check if
        1. its num_reputations is the same as the given one.
        2. its num_reputations is the new one after setting a new one.
        """
        # 1. its num_reputations is the same as the given one.
        # 2. its num_reputations is the new one after setting a new one.

    def test_num_layers(self):
        """Normal test:
        Call the setter and getter of num_layers.

        Check if
        1. its num_layers is the same as the given one.
        2. the number of its _dense_layer is the same as its num_layers.
        3. its total_qubits is the same as its num_qubits * its num_layers.
        4. its num_layers is the new one after setting a new one.
        5. the number of its _dense_layer is the same as its num_layers.
        6. its total_qubits is the same as its num_qubits * its num_layers.
        """
        # 1. its num_layers is the same as the given one.
        # 2. the number of its _dense_layer is the same as its num_layers.
        # 3. its total_qubits is the same as its num_qubits * its num_layers.
        # 4. its num_layers is the new one after setting a new one.
        # 5. the number of its _dense_layer is the same as its num_layers.
        # 6. its total_qubits is the same as its num_qubits * its num_layers.

    def test_parameter_prefix(self):
        """Normal test:
        Call the setter and getter of parameter_prefix.

        Check if
        1. its parameter_prefix is the same as the given one.
        2. its parameter_prefix is an empty list after setting None.
        3. its parameter_prefix is the new one after setting a new one.
        """
        # 1. its parameter_prefix is the same as the given one.
        # 2. its parameter_prefix is an empty list after setting None.
        # 3. its parameter_prefix is the new one after setting a new one.

    def test_invalid_run(self):
        """Abnormal test:
        Call run method with invalid data.

        Check if
        1. ValueError arises.
        """
        # 1. ValueError arises.

    def test_run(self):
        """Abnormal test:
        Call run method with valid data.

        Check if
        1. the length of the return value is the same as of the given data.
        """
        # 1. the length of the return value is the same as of the given data.
