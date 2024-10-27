import pytest

from src.example_package.example import add_one


class TestAddNumber:
    @pytest.mark.add_number
    @pytest.mark.parametrize(
        "positive_number, output", [(1, 2), (2, 3), (3, 4), (5, 6)]
    )
    def test_with_positive_numbers(
        self,
        positive_number,
        output,
        fixture_function,
        fixture_class,
        fixture_module,
        fixture_session,
    ):
        result = add_one(positive_number)
        assert result == output

    @pytest.mark.add_number
    @pytest.mark.parametrize(
        "negative_number, output", [(-4, -3), (-3, -2), (-2, -1), (-1, 0)]
    )
    def test_with_negative_number(
        self,
        negative_number,
        output,
        fixture_function,
        fixture_class,
        fixture_module,
        fixture_session,
    ):
        result = add_one(negative_number)
        assert result == output

    @pytest.mark.add_number
    def test_with_zero(
        self, fixture_function, fixture_class, fixture_module, fixture_session
    ):
        zero = 0
        result = add_one(zero)
        assert result == zero + 1
