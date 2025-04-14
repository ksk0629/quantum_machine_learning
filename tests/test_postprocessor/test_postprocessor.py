import pytest

from quantum_machine_learning_utils.postprocessor.postprocessor import Postprocessor


class TestPostprocessor:

    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.postprocessor
    @pytest.mark.parametrize(
        "result",
        [
            {"0": 1, "1": 1, "2": 1},
            {},
            {"0": 1, "-1": 1},
            {"1": 1, "3": 1},
            {"-2": 1},
        ],
    )
    def test_calculate_fidelity_from_swap_test_with_invalid_arg(self, result):
        """Abnormal test;
        Run calculate_fidelity_from_swap_test with an invalid argument.

        Check if ValueError happens.
        """
        with pytest.raises(ValueError):
            Postprocessor.calculate_fidelity_from_swap_test(result)

    @pytest.mark.postprocessor
    def test_calculate_fidelity_from_swap_test_with_zero(self):
        """Normal test;
        Run calculate_fidelity_from_swap_test with a result having only zero observed.

        Check if
        - the return value is 1.
        """
        result = {"0": 8192}
        fidelity = Postprocessor.calculate_fidelity_from_swap_test(result)
        assert fidelity == 1

    @pytest.mark.postprocessor
    def test_calculate_fidelity_from_swap_test_with_one(self):
        """Normal test;
        Run calculate_fidelity_from_swap_test with a result having only one observed.

        Check if
        - the return value is 0.
        """
        result = {"1": 8192}
        fidelity = Postprocessor.calculate_fidelity_from_swap_test(result)
        assert fidelity == 0

    @pytest.mark.postprocessor
    def test_calculate_fidelity_from_swap_test_with_half_zero_and_half_one(self):
        """Normal test;
        Run calculate_fidelity_from_swap_test with a result meaning the fidelity is 1/2.

        Check if
        - the return value is 1/2.
        """
        result = {"0": 3, "1": 1}
        fidelity = Postprocessor.calculate_fidelity_from_swap_test(result)
        assert fidelity == 1 / 2

    @pytest.mark.postprocessor
    def test_calculate_fidelity_from_swap_test_with_larger_zero(self):
        """Normal test;
        Run calculate_fidelity_from_swap_test with a result having the number of zeros more than of ones.

        Check if
        - the return value is 0.
        """
        result = {"0": 1, "1": 3}
        fidelity = Postprocessor.calculate_fidelity_from_swap_test(result)
        assert fidelity == 0

    @pytest.mark.postprocessor
    def test_count_one_bits_of_most_frequent_result_with_only_0(self):
        """Normal test;
        Run count_one_bits_of_most_frequent_result with a result having only zero.

        Check if
        - the return value is 0.
        """
        result = {"0": 3, "00": 5, "000": 4}
        num_ones = Postprocessor.count_one_bits_of_most_frequent_result(result)
        assert num_ones == 0

    @pytest.mark.postprocessor
    def test_count_one_bits_of_most_frequent_result(self):
        """Normal test;
        Run count_one_bits_of_most_frequent_result with a result having some results.

        Check if
        - the return value is correct.
        """
        result = {"0": 3, "10": 5, "110": 4}
        num_ones = Postprocessor.count_one_bits_of_most_frequent_result(result)
        assert num_ones == 1
