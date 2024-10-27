from src.example_package.example_class import ExampleClass


def add_one(number: int) -> int:
    """Add one to a given number.

    :param int number: whole number
    :return int: given number plus one
    """
    return number + 1


def add_one_with_example_class(number: int) -> int:
    """Add one to a given number using ExampleClass.

    :param int number: whole number
    :return int: given number plus one
    """
    example_class = ExampleClass(number)
    example_class.number = add_one(example_class.number)
    return example_class.number


def main():
    print(add_one(2))
