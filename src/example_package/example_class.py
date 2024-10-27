class ExampleClass:
    """Hold one integer with type check."""

    def __init__(self, number: int):
        """Set-up this cllass.

        :param int number: a whole number
        """
        self.number = number

    @property
    def number(self) -> int:
        """Retunr the private member variable number.

        :return int: private member variable number
        """
        return self.__number

    @number.setter
    def number(self, new_number: int):
        """Check the type of a given new_number and set it as the private member variable number.

        :param int new_number: new number
        :raises TypeError: if the type of a given new_number is not int
        """
        if not isinstance(new_number, int):
            msg = (
                f"The type of a given number must be int, but it is {type(new_number)}"
            )
            raise TypeError(msg)
        self.__number = new_number
