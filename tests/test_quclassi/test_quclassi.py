import pytest

from src.quclassi.quclassi import QuClassi


class TestQuClassi:
    @classmethod
    def setup_class(cls):
        cls.quclassi = QuClassi()
