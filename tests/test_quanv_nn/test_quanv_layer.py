import itertools
import os

import numpy as np
import pytest

from quantum_machine_learning.quanv_nn.quanv_layer import QuanvLayer


class TestQuanvLayer:
    @classmethod
    def setup_class(cls):
        cls.kernel_size = (2, 2)
        cls.num_filters = 3
        cls.model_dir_path = "./test/"
        cls.quanv_layer = QuanvLayer(
            kernel_size=cls.kernel_size, num_filters=cls.num_filters
        )

        cls.batch_data = np.array(
            [[1, 1, 1, 1], [1, 1, 0, 1], [1, 0, 0, 1]],
        )
        num_batch = len(cls.batch_data)
        cls.correct_output_shape = (cls.num_filters, num_batch)

    def test_init(self):
        """Normal test;
        Check if self.quanv_layer has
        - the same kernel_size as self.kernel_size.
        - the same num_filters as self.num_filters.
        - the same num_qubits as self.kernel_size[0] * self.kernel_size[1].
        - the list of num_filters qiskit.QuantumCircuit.
        """
        assert self.quanv_layer.kernel_size == self.kernel_size
        assert self.quanv_layer.num_filters == self.num_filters
        assert self.quanv_layer.num_qubits == self.kernel_size[0] * self.kernel_size[1]
        assert len(self.quanv_layer.filters) == self.num_filters

    def test_init_loaded(self):
        """Normal test;
        Run __init__ with the argument is_loaded being True.

        Check if the initialised instance has
        - the same kernel_size as self.kernel_size.
        - the same num_filters as self.num_filters.
        - empty list as self.filters.
        """
        quanv_layer = QuanvLayer(
            kernel_size=self.kernel_size, num_filters=self.num_filters, is_loaded=True
        )
        assert quanv_layer.kernel_size == self.kernel_size
        assert quanv_layer.num_filters == self.num_filters
        assert len(quanv_layer.filters) == 0

    def test_process_with_valid(self):
        """Normal test;
        Run process and __call__.

        Check if
        - the return value of process has the correct shape.
        - the return values of process and __call__ are the same.
        """
        processed_data_1 = self.quanv_layer.process(batch_data=self.batch_data)

        assert processed_data_1.shape == self.correct_output_shape

        processed_data_2 = self.quanv_layer(self.batch_data)
        assert np.allclose(processed_data_1, processed_data_2)

    def test_process_with_invalid_batch_data(self):
        """Abnormal test;
        Run process with invalid batch_data.

        Check if ValueError happens.
        """
        invalid_batch_data = np.array([[[1, 1, 1, 1], [1, 1, 0, 1], [1, 0, 0, 1]]])
        with pytest.raises(ValueError):
            self.quanv_layer.process(batch_data=invalid_batch_data)

    def test_process_with_invalid_data(self):
        """Abnormal test;
        Run process with batch_data having invalid data.

        Check if ValueError happens.
        """
        invalid_batch_data = np.array(
            [[1, 1, 1, 1, 9], [1, 1, 0, 1, 9], [1, 0, 0, 1, 9]],
        )
        with pytest.raises(ValueError):
            self.quanv_layer.process(batch_data=invalid_batch_data)

    def test_save(self):
        """Normal test;
        Run self.quanv_layer.save.

        Check if
        - there is basic_info.pkl.
        - there is circuit.qpy.
        """
        self.quanv_layer.save(model_dir_path=self.model_dir_path)

        basic_info_path = os.path.join(self.model_dir_path, "basic_info.pkl")
        assert os.path.isfile(basic_info_path)
        os.remove(basic_info_path)

        filters_path = os.path.join(self.model_dir_path, "circuit.qpy")
        assert os.path.isfile(filters_path)
        os.remove(filters_path)

        lookup_tables_path = os.path.join(self.model_dir_path, "lookup_tables.pkl")
        assert os.path.isfile(lookup_tables_path)
        os.remove(lookup_tables_path)

        os.rmdir(self.model_dir_path)

    def test_load(self):
        """Normal test;
        Run save and load.

        Check if
        - the loaded QuanvLayer has the same member variables.
        Note that, I have no idea but if you check each filter in filters,
        some of them are the same but some are not, but according to the draw(),
        they look the same even if == operator returns False.
        So, for now, comparison between each filter is not done here.
        """
        self.quanv_layer.save(model_dir_path=self.model_dir_path)

        quanv_layer = QuanvLayer.load(model_dir_path=self.model_dir_path)
        assert quanv_layer.kernel_size == self.quanv_layer.kernel_size
        assert quanv_layer.num_filters == self.quanv_layer.num_filters
        assert quanv_layer.num_qubits == self.quanv_layer.num_qubits
        assert len(quanv_layer.filters) == len(self.quanv_layer.filters)

        basic_info_path = os.path.join(self.model_dir_path, "basic_info.pkl")
        os.remove(basic_info_path)
        filters_path = os.path.join(self.model_dir_path, "circuit.qpy")
        os.remove(filters_path)
        lookup_tables_path = os.path.join(self.model_dir_path, "lookup_tables.pkl")
        os.remove(lookup_tables_path)
        os.rmdir(self.model_dir_path)

    def test_build_lookup_tables(self):
        """Normal test;
        run build_lookup_tables function.

        Check if
        - the length of self.quanv_layer.lookup_tables is the same as self.quanv_layer.num_filters.
        - the number of keys of each look-up table is the same as the number of the patterns.
        - no error happens when accessing the data in each look-up table.
        """
        pattern = [0, np.pi]
        all_patterns = np.array(
            list(itertools.product(pattern, repeat=self.quanv_layer.num_qubits))
        )
        self.quanv_layer.build_lookup_tables(patterns=all_patterns)

        assert len(self.quanv_layer.lookup_tables) == self.quanv_layer.num_filters

        for filter_index in range(self.quanv_layer.num_filters):
            assert len(self.quanv_layer.lookup_tables[filter_index].keys()) == len(
                all_patterns
            )
            for pattern in all_patterns:
                self.quanv_layer.lookup_tables[filter_index][tuple(pattern.tolist())]
