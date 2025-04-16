import numpy as np
import pytest
import qiskit_aer
import qiskit_ibm_runtime

from quantum_machine_learning.layers.blocks.ssskm_parallel_dense_layer import (
    SSSKMParallelDenseLayer,
)
from quantum_machine_learning.layers.circuits.feature_maps.x_angle import XAngle
from quantum_machine_learning.layers.circuits.feature_maps.y_angle import YAngle


class TestSSSKMParallelDenseLayer:
    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.layer
    @pytest.mark.parametrize("num_qubits", [2, 3])
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
        assert block.num_qubits == num_qubits
        # 2. its num_reputations is the same as the given num_reputations.
        assert block.num_reputations == num_reputations
        # 3. its num_layers is the same as the given num_layers.
        assert block.num_layers == num_layers
        # 4. its parameter_prefix is an empty strings.
        assert block.parameter_prefix == ""
        # 5. its _encoder_class is YAngle.
        assert block._encoder_class == YAngle
        # 6. its _transformer transforms the argument data into the data * 3.14...
        data = [1, 2, 3, 4, 5]
        assert block._transformer(data) == [d * np.pi for d in data]
        # 7. its _trainable_parameter_values is None.
        assert block._trainable_parameter_values is None
        # 8. the length of its _dense_layers is the same as the given num_layers.
        assert len(block._dense_layers) == block.num_layers
        # 9. the length of its _trainable_parameters is the same as the given num_layers.
        assert len(block._trainable_parameters) == block.num_layers
        # 10. the length of its encoding_parameters is the same as the given num_qubits.
        assert len(block.encoding_parameters) == block.num_qubits
        # 11. its total_qubits is its num_qubits * its num_layers.
        assert block.total_qubits == block.num_qubits * block.num_layers

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
        5. its _encoder_class is None.
        6. its _transformer transforms the argument data into the data * 3.14...
        7. its _trainable_parameter_values is None.
        8. its _dense_layers is an empty list.
        9. its _trainable_parameters is an empty list.
        10. the length of its encoding_parameters is 0.
        11. its total_qubits is its num_qubits * its num_layers.
        """
        block = SSSKMParallelDenseLayer(
            num_qubits=num_qubits,
            num_reputations=num_reputations,
            num_layers=num_layers,
            build=False,
        )
        # 1. its num_qubits is the same as the given num_qubits.
        assert block.num_qubits == num_qubits
        # 2. its num_reputations is the same as the given num_reputations.
        assert block.num_reputations == num_reputations
        # 3. its num_layers is the same as the given num_layers.
        assert block.num_layers == num_layers
        # 4. its parameter_prefix is an empty strings.
        assert block.parameter_prefix == ""
        # 5. its _encoder_class is None.
        assert block._encoder is None
        # 6. its _transformer transforms the argument data into the data * 3.14...
        data = [1, 2, 3, 4, 5]
        assert block._transformer(data) == [d * np.pi for d in data]
        # 7. its _trainable_parameter_values is None.
        assert block._trainable_parameter_values is None
        # 8. its _dense_layers is an empty list.
        assert len(block._dense_layers) == 0
        # 9. its _trainable_parameters is an empty list.
        assert len(block._trainable_parameters) == 0
        # 10. the length of its encoding_parameters is 0.
        assert len(block.encoding_parameters) == 0
        # 11. its total_qubits is its num_qubits * its num_layers.
        assert block.total_qubits == block.num_qubits * block.num_layers

    @pytest.mark.layer
    def test_init_encoder_class(self):
        """Normal test:
        Create an instance of SSSKMParallelDenseLayer with encoder_class=XAngle.

        Check if
        1. its _encoder_class is XAngle.
        """
        block = SSSKMParallelDenseLayer(
            num_qubits=2, num_reputations=2, num_layers=2, encoder_class=XAngle
        )
        # 1. its _encoder_class is XAngle.
        assert block._encoder_class == XAngle

    @pytest.mark.layer
    def test_init_transformer(self):
        """Normal test:
        Create an instance of SSSKMParallelDenseLayer with some transformer.

        Check if
        1. its _transformer is the same as the give one.
        2. the return value of its _transformer is the same one as of the given one.
        """
        transformer = lambda data: [d * 2 for d in data]
        block = SSSKMParallelDenseLayer(
            num_qubits=2, num_reputations=2, num_layers=2, transformer=transformer
        )
        # 1. its _transformer is the same as the give one.
        assert block._transformer == transformer
        # 2. the return value of its _transformer is the same one as of the given one.
        data = [1, 2, 3, 4, 5]
        assert block._transformer(data) == transformer(data)

    @pytest.mark.layer
    def test_init_trainable_parameter_values(self):
        """Normal test:
        Create an instance of SSSKMParallelDenseLayer with some trainable_parameter_values.

        Check if
        1. its _trainable_parameter_values is the same as the given one.
        """
        num_qubits = 2
        num_reputations = 2
        num_layers = 2
        trainable_parameter_values = [
            [index] * (num_qubits * num_reputations * 3) for index in range(num_layers)
        ]
        block = SSSKMParallelDenseLayer(
            num_qubits=num_qubits,
            num_reputations=num_reputations,
            num_layers=num_layers,
            trainable_parameter_values=trainable_parameter_values,
        )
        # 1. its _trainable_parameter_values is the same as the given one.
        assert block._trainable_parameter_values == trainable_parameter_values

    @pytest.mark.layer
    def test_num_qubits(self):
        """Normal test:
        Call the setter and getter of num_qubits.

        Check if
        1. its num_qubits is the same as the given one.
        2. each member of its _dense_layers has its-num_qubits num_qubits.
        3. its _encoder has its-num_qubits parameters.
        4. its total_qubits is the same as its num_qubits * its num_layers.
        5. its num_qubits is the new one after setting a new one.
        6. each member of its _dense_layers has its-num_qubits num_qubits.
        7. its _encoder has its-num_qubits parameters.
        8. its total_qubits is the same as its num_qubits * its num_layers.
        """
        num_qubits = 2
        block = SSSKMParallelDenseLayer(
            num_qubits=num_qubits,
            num_reputations=2,
            num_layers=2,
        )
        # 1. its num_qubits is the same as the given one.
        assert block.num_qubits == num_qubits

        for dense_layer in block._dense_layers:
            # 2. each member of its _dense_layers has its-num_qubits num_qubits.
            assert dense_layer.num_qubits == block.num_qubits

        # 3. its _encoder has its-num_qubits parameters.
        assert block._encoder.num_parameters == block.num_qubits
        # 4. its total_qubits is the same as its num_qubits * its num_layers.
        assert block.total_qubits == block.num_qubits * block.num_layers
        # 5. its num_qubits is the new one after setting a new one.
        new_num_qubits = num_qubits * 2 + 1
        block.num_qubits = new_num_qubits
        assert block.num_qubits == new_num_qubits

        for dense_layer in block._dense_layers:
            # 2. each member of its _dense_layers has its-num_qubits num_qubits.
            assert dense_layer.num_qubits == block.num_qubits

        # 7. its _encoder has its-num_qubits parameters.
        assert block._encoder.num_parameters == block.num_qubits
        # 8. its total_qubits is the same as its num_qubits * its num_layers.
        assert block.total_qubits == block.num_qubits * block.num_layers

    @pytest.mark.layer
    def test_num_reputations(self):
        """Normal test:
        Call the setter and getter of num_reputations.

        Check if
        1. its num_reputations is the same as the given one.
        2. its num_reputations is the new one after setting a new one.
        """
        num_reputations = 2
        block = SSSKMParallelDenseLayer(
            num_qubits=2,
            num_reputations=num_reputations,
            num_layers=2,
        )
        # 1. its num_reputations is the same as the given one.
        assert block.num_reputations == num_reputations
        # 2. its num_reputations is the new one after setting a new one.
        new_num_reputations = num_reputations + 1
        block.num_reputations = new_num_reputations
        assert block.num_reputations == new_num_reputations

    @pytest.mark.layer
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
        num_layers = 2
        block = SSSKMParallelDenseLayer(
            num_qubits=2,
            num_reputations=2,
            num_layers=num_layers,
        )
        # 1. its num_layers is the same as the given one.
        assert block.num_layers == num_layers
        # 2. the number of its _dense_layer is the same as its num_layers.
        assert len(block._dense_layers) == block.num_layers
        # 3. its total_qubits is the same as its num_qubits * its num_layers.
        assert block.total_qubits == block.num_qubits * block.num_layers
        # 4. its num_layers is the new one after setting a new one.
        new_num_layers = num_layers + 1
        block.num_layers = new_num_layers
        assert block.num_layers == new_num_layers
        # 5. the number of its _dense_layer is the same as its num_layers.
        assert len(block._dense_layers) == block.num_layers
        # 6. its total_qubits is the same as its num_qubits * its num_layers.
        assert block.total_qubits == block.num_qubits * block.num_layers

    @pytest.mark.layer
    def test_parameter_prefix(self):
        """Normal test:
        Call the setter and getter of parameter_prefix.

        Check if
        1. its parameter_prefix is the same as the given one.
        2. its parameter_prefix is an empty list after setting None.
        3. its parameter_prefix is the new one after setting a new one.
        """
        parameter_prefix = "prefix"
        block = SSSKMParallelDenseLayer(
            num_qubits=2,
            num_reputations=2,
            num_layers=2,
            parameter_prefix=parameter_prefix,
        )
        # 1. its parameter_prefix is the same as the given one.
        assert block.parameter_prefix == parameter_prefix
        # 2. its parameter_prefix is an empty list after setting None.
        block.parameter_prefix = None
        assert block.parameter_prefix == ""
        # 3. its parameter_prefix is the new one after setting a new one.
        new_parameter_prefix = "new!"
        block.parameter_prefix = new_parameter_prefix
        assert block.parameter_prefix == new_parameter_prefix

    @pytest.mark.layer
    def test_invalid_run(self):
        """Abnormal test:
        Call run method with invalid data.

        Check if
        1. ValueError arises.
        """
        block = SSSKMParallelDenseLayer(
            num_qubits=2,
            num_reputations=2,
            num_layers=2,
        )
        data = [1, 2, 3]
        backend = qiskit_aer.AerSimulator()
        estimator_class = qiskit_ibm_runtime.EstimatorV2
        shots = 8192
        seed = 901
        optimisation_level = 3
        # 1. ValueError arises.
        with pytest.raises(ValueError):
            block.run(
                data=data,
                backend=backend,
                estimator_class=estimator_class,
                shots=shots,
                seed=seed,
                optimisation_level=optimisation_level,
            )

    @pytest.mark.layer
    def test_run(self):
        """Abnormal test:
        Call run method with valid data.

        Check if
        1. the length of the return value is the same as of the given data.
        """
        num_qubits = 2
        num_reputations = 2
        num_layers = 2
        trainable_parameter_values = [
            [index] * (num_qubits * num_reputations * 3) for index in range(num_layers)
        ]
        block = SSSKMParallelDenseLayer(
            num_qubits=num_qubits,
            num_reputations=num_reputations,
            num_layers=num_layers,
            trainable_parameter_values=trainable_parameter_values,
        )
        data = [1, 2, 3, 4]
        backend = qiskit_aer.AerSimulator()
        estimator_class = qiskit_ibm_runtime.EstimatorV2
        shots = 8192
        seed = 901
        optimisation_level = 3

        # 1. the length of the return value is the same as of the given data.
        processed_data = block.run(
            data=data,
            backend=backend,
            estimator_class=estimator_class,
            shots=shots,
            seed=seed,
            optimisation_level=optimisation_level,
        )
        assert len(processed_data) == len(data)
