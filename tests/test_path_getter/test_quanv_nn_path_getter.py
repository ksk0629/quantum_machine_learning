import os

import pytest

from quantum_machine_learning.path_getter.quanv_nn_path_getter import QuanvNNPathGetter


class TestQuanvNNPathGetter:

    @classmethod
    def setup_class(cls):
        cls.dir_path = "./../test"
        cls.prefix = "prefix"
        cls.postfix = "postfix"
        cls.path_getter = QuanvNNPathGetter(dir_path=cls.dir_path)
        cls.path_getter_with_prefix_and_postfix = QuanvNNPathGetter(
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
    def test_classical_torch_model(self):
        """Normal test;
        Run classical_torch_model property with and without both prefix and postfix.

        Check if the return values are correct classical_torch_model paths.
        """
        assert (
            self.path_getter_with_prefix_and_postfix.classical_torch_model
            == os.path.join(
                self.dir_path, f"{self.prefix}_classical_torch_model_{self.postfix}.pth"
            )
        )
        assert self.path_getter.classical_torch_model == os.path.join(
            self.dir_path, f"classical_torch_model.pth"
        )
