import pytest

from src.example_package.example_class import ExampleClass


@pytest.fixture(scope="function")
def fixture_function():
    print("fixture_function: function")


@pytest.fixture(scope="class")
def fixture_class():
    print("fixture_function: class")


@pytest.fixture(scope="module")
def fixture_module():
    print("fixture_function: module")


@pytest.fixture(scope="session")
def fixture_session():
    print("fixture_function: session")
