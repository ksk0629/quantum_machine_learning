import numpy as np
from sklearn import datasets


def get_dataset(name: str) -> tuple[np.ndarray, list[str]]:
    """Get dataset from dataset name.

    :param str name: dataset name
    :return tuple[np.ndarray, list[str]]: dataset
    """
    match name:
        case "iris":
            return get_iris()
        case "wine":
            return get_wine()


def get_iris() -> tuple[np.ndarray, list[str]]:
    """Get the iris dataset.

    :return tuple[np.ndarray, list[str]]: iris dataset
    """
    iris = datasets.load_iris()
    data = iris.data
    labels = list(iris.target)
    for index in range(len(labels)):
        labels[index] = iris.target_names[labels[index]]

    return data, labels


def get_wine() -> tuple[np.ndarray, list[str]]:
    """Get the wine dataset.

    :return tuple[np.ndarray, list[str]]: wine dataset
    """
    wine_data = datasets.load_wine(return_X_y=True)
    data, labels = wine_data
    labels = [f"{label}" for label in labels]
    return data, labels
