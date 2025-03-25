import pytest

from quantum_machine_learning.utils.circuit_utils import CircuitUtils


class TestCircuitUtils:

    @classmethod
    def setup_class(cls):
        pass

    @pytest.mark.utils
    @pytest.mark.parametrize(
        "parameter_dict", [{"x": 1.2, "1": 2}, {"y": 3.1}, {"layers!": 901}]
    )
    def test_get_parameter_dict(self, parameter_dict):
        """Normal test;
        run get_parameter_dict.

        Check if
        - the return value is the given parameter_dict.
        """
        parameter_names = list(parameter_dict.keys())
        parameters = list(parameter_dict.values())
        result = CircuitUtils.get_parameter_dict(
            parameter_names=parameter_names, parameters=parameters
        )

        assert result == parameter_dict
