from typing import Final


class Postprocessor:
    """Postprocessor class for the result of quantum circuit executions."""

    KEY_0: Final[str] = "0"
    KEY_1: Final[str] = "1"

    @staticmethod
    def calculate_fidelity_from_swap_test(result: dict[str, int]) -> float:
        """Calculate the quantum fidelity from the result of the CSWAP test.
        In the CSWAP test, the probability zero is obtained as
         P(0) = 1/2 - (fidelity)/2.
        Thus, the fidelity is calculated as
         2*P(0) = 1 - (fidelity) <=> (fidelity) = 1 - 2*P(0).

        :param dict[str, int] result: result of cswap test
        :raises ValueError: if the given result contains nothing or more than two keys
        :raises ValueError: if the given result is consisted with other thatn 0 and/or 1
        :return float: quantum fidelity
        """
        # Check the length of the given result.
        error_msg = f"""
            The argument result must be the dict whose keys are 
            {Postprocessor.KEY_0} and {Postprocessor.KEY_1} as the result of CSWAP test, 
            but {result}.
        """
        if len(result) > 2 or len(result) == 0:
            # If the length is more than two or equal to zero, throw the error.
            raise ValueError(error_msg)
        elif len(result) == 2:
            if not (Postprocessor.KEY_0 in result and Postprocessor.KEY_1 in result):
                # If the length is two and they are not "0" and "1", throw the error.
                raise ValueError(error_msg)
        elif (
            len(result) == 1
        ):  # For readability (obviously the number of keys are one if this code is running.)
            if not (Postprocessor.KEY_0 in result or Postprocessor.KEY_1 in result):
                # If the length is one and they are not eithery "0" or "1", throw the error.
                raise ValueError(error_msg)

        # Get the number of zeros.
        num_zeros = result[Postprocessor.KEY_0] if Postprocessor.KEY_0 in result else 0
        if num_zeros == 0:
            # If there is no zero in the result, means those two quantum states are orthogonal.
            return 0
        # Get the number of ones.
        num_ones = result[Postprocessor.KEY_1] if Postprocessor.KEY_1 in result else 0
        # Calculate the probability zero.
        probability_zero = num_zeros / (num_zeros + num_ones)
        # Calculate the fidelity from the probability.
        fidelity = 2 * probability_zero - 1
        if fidelity < 0:
            fidelity = 0

        return fidelity
