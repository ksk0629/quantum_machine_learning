from quantum_machine_learning.path_getter.base_path_getter import BasePathGetter


class QuanvNNPathGetter(BasePathGetter):
    """PathGetter class for QuanvNN."""

    def __init__(
        self, dir_path: str, prefix: None | str = None, postfix: None | str = None
    ):
        """Initialise this PathGetter.

        :param str dir_path: a path to the target directory
        :param None | str prefix: a prefix, defaults to None
        :param None | str postfix: a postfix, defaults to None
        """
        super().__init__(dir_path=dir_path, prefix=prefix, postfix=postfix)

    @property
    def basic_info(self) -> str:
        """Return a basic infomation file path.

        :return str: path to a basic information file
        """
        filename = "basic_info"
        extension = "pkl"
        return self._get_path(filename=filename, extension=extension)

    @property
    def circuit(self) -> str:
        """Return a circuit file path.

        :return str: path to a circuit file
        """
        filename = "circuit"
        extension = "qpy"
        return self._get_path(filename=filename, extension=extension)

    @property
    def classical_torch_model(self) -> str:
        """Return a classical_torch_model file path.

        :return str: path to a classical_torch_model file
        """
        filename = "classical_torch_model"
        extension = "pth"
        return self._get_path(filename=filename, extension=extension)
