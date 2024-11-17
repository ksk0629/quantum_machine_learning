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
        case "pos_neg":
            return get_pos_neg()


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


def get_pos_neg(
    dimension: int = 4, num_data: int = 100, highest=1
) -> tuple[np.ndarray, list[str]]:
    """Get a positive-negative numbers dataset,
    which is randomly generated.
    The label of each data is "positive" if the summation of the data is non-negative.
    Otherwise, it is "negative".

    :param int dimension: dimension of data, defaults to 4
    :param int num_data: number of data, defaults to 100
    :param int highest: highest number, defaults to 1
    :return tuple[np.ndarray, list[str]]: positive-negative numbers dataset
    """
    data = np.random.rand(num_data, dimension) * highest
    labels = ["positive" if np.sum(_d) >= 0 else "negative" for _d in data]

    assert len(data) == len(labels)

    return data, labels
