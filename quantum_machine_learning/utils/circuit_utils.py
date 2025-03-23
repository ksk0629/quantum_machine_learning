class CircuitUtils:
    """Utils, for quantum circuits, class"""

    @staticmethod
    def get_parameter_dict(
        parameter_names: list[str], parameters: list[float]
    ) -> dict[str, float]:
        """Get the dictionary whose keys are names of the given parameters and
        the values are parameter values.

        :param list[str] parameter_names: names of parameters
        :param list[float] parameters: patameter values
        :return dict[str, float]: parameter dictionary
        """
        parameter_dict = {
            parameter_name: parameter
            for parameter_name, parameter in zip(parameter_names, parameters)
        }
        return parameter_dict
