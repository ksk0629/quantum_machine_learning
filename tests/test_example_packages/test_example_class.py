import pytest

from src.example_package.example_class import ExampleClass


class TestExampleClassClass:
    @classmethod
    def setup_class(cls):
        cls.whole_number = 2
        cls.example_class = ExampleClass(cls.whole_number)
        assert cls.example_class.number == cls.whole_number

    @pytest.mark.example_class
    @pytest.mark.parametrize("whole_number", [1, -1, 2, -2, 0])
    def test_integer(
        self,
        whole_number,
        fixture_function,
        fixture_class,
        fixture_module,
        fixture_session,
    ):
        example_class = ExampleClass(whole_number)
        assert example_class.number == whole_number

    @pytest.mark.example_class
    @pytest.mark.parametrize("float_number", [0.1, -0.1, 0.2, -0.2, float(0.0)])
    def test_with_floats(
        self,
        float_number,
        fixture_function,
        fixture_class,
        fixture_module,
        fixture_session,
    ):
        with pytest.raises(TypeError):
            ExampleClass(float_number)
