import os

import pytest

from quantum_machine_learning.path_getter.quclassi_path_getter import QuClassiPathGetter


class TestQuClassiPathGetter:

    @classmethod
    def setup_class(cls):
        cls.dir_path = "./../test"
        cls.prefix = "prefix"
        cls.postfix = "postfix"
        cls.path_getter = QuClassiPathGetter(dir_path=cls.dir_path)
        cls.path_getter_with_prefix_and_postfix = QuClassiPathGetter(
            dir_path=cls.dir_path, prefix=cls.prefix, postfix=cls.postfix
        )

    @pytest.mark.path_getter
    def test_basic_info(self):
        """Normal test;
        Run basic_info property with and without both prefix and postfix.

        Check if the return values are correct basic_info paths.
        """
        assert self.path_getter_with_prefix_and_postfix.basic_info == os.path.join(
            self.dir_path, f"{self.prefix}_basic_info_{self.postfix}.pkl"
        )
        assert self.path_getter.basic_info == os.path.join(
            self.dir_path, f"basic_info.pkl"
        )

    @pytest.mark.path_getter
    def test_circuit(self):
        """Normal test;
        Run circuit property with and without both prefix and postfix.

        Check if the return values are correct circuit paths.
        """
        assert self.path_getter_with_prefix_and_postfix.circuit == os.path.join(
            self.dir_path, f"{self.prefix}_circuit_{self.postfix}.qpy"
        )
        assert self.path_getter.circuit == os.path.join(self.dir_path, f"circuit.qpy")

    @pytest.mark.path_getter
    def test_trainable_parameters(self):
        """Normal test;
        Run trainable_parameters property with and without both prefix and postfix.

        Check if the return values are correct trainable_parameters paths.
        """
        assert (
            self.path_getter_with_prefix_and_postfix.trainable_parameters
            == os.path.join(
                self.dir_path, f"{self.prefix}_trainable_parameters_{self.postfix}.pkl"
            )
        )
        assert self.path_getter.trainable_parameters == os.path.join(
            self.dir_path, f"trainable_parameters.pkl"
        )

    @pytest.mark.path_getter
    def test_data_parameters(self):
        """Normal test;
        Run data_parameters property with and without both prefix and postfix.

        Check if the return values are correct data_parameters paths.
        """
        assert self.path_getter_with_prefix_and_postfix.data_parameters == os.path.join(
            self.dir_path, f"{self.prefix}_data_parameters_{self.postfix}.pkl"
        )
        assert self.path_getter.data_parameters == os.path.join(
            self.dir_path, f"data_parameters.pkl"
        )

    @pytest.mark.path_getter
    def test_trained_parameters(self):
        """Normal test;
        Run trained_parameters property with and without both prefix and postfix.

        Check if the return values are correct trained_parameters paths.
        """
        assert (
            self.path_getter_with_prefix_and_postfix.trained_parameters
            == os.path.join(
                self.dir_path, f"{self.prefix}_trained_parameters_{self.postfix}.pkl"
            )
        )
        assert self.path_getter.trained_parameters == os.path.join(
            self.dir_path, f"trained_parameters.pkl"
        )
