import pytest

from src.quanv_nn.quanv_nn import QuanvNN


class TestQuanvNN:
    @classmethod
    def setup_class(cls):
        cls.quanv_nn = QuanvNN()
