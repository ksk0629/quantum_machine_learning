from abc import ABC
import os


class PathGetter(ABC):
    """A base class to get fixed paths."""

    def __init__(
        self, dir_path: str, prefix: None | str = None, postfix: None | str = None
    ):
        """Initialise this path getter.

        :param str dir_path: a path to the target directory
        :param None | str prefix: a prefix, defaults to None
        :param None | str postfix: a postfix, defaults to None
        """
        self._dir_path = dir_path
        self._prefix = prefix
        self._postfix = postfix

    def _get_path(self, filename: str, extension: str) -> str:
        """Get the path with the prefix and postfix.

        :param str filename: filename
        :param str extension: extension without dot
        :return str: path to the file
        """
        # FOR DEVELOPER: The extension must not start with ".".
        assert extension[0] != "."

        path = filename
        path = path if self._prefix is None else f"{self._prefix}_{path}"
        path = path if self._postfix is None else f"{path}_{self._postfix}"
        path = f"{path}.{extension}"

        return os.path.join(self._dir_path, path)
