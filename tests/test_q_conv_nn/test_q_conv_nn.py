import pytest

from quantum_machine_learning.q_conv_nn.q_conv_nn import QConvNN


class TestQConvNN:
    @classmethod
    def setup_class(cls):
        cls.q_conv_nn = QConvNN()
