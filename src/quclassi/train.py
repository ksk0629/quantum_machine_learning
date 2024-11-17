from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.quclassi.quclassi import QuClassi
from src.quclassi.quclassi_trainer import QuClassiTrainer
import src.utils as utils


def preprocess_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    should_scale: bool,
    encoding_method: Callable,
    train_size: float = 0.75,
    seed: int = 901,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the dataset.

    :param np.ndarray data: data
    :param np.ndarray labels: labels
    :param bool should_scale: whether scale the data or not
    :param Callable encoding_method: encoding method
    :param float train_size: train data size, defaults to 0.75
    :param int seed: random seed, defaults to 901
    :param bool shuffle: whether shuffle the data or not, defaults to True
    :return tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: train data, train labels, validation data, validation labels
    """
    # Scale the data if needed.
    data = utils.scale_data(data) if should_scale else data

    # Split the data and labels.
    train_data, val_data, train_labels, val_labels = train_test_split(
        data,
        labels,
        train_size=train_size,
        random_state=seed,
        shuffle=shuffle,
        stratify=labels,
    )
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    # Normalise the data.
    train_data = utils.normalise_data(train_data)
    val_data = utils.normalise_data(val_data)

    # Encode the data.
    train_data = encoding_method(train_data)
    val_data = encoding_method(val_data)
    # Make the length of the each data even.
    if train_data.shape[1] % 2 != 0:
        new_shape = [0, 0]
        new_shape[0] = train_data.shape[0]
        new_shape[1] = train_data.shape[1] + 1
        train_data_new = np.zeros(new_shape)
        train_data = train_data_new
    if val_data.shape[1] % 2 != 0:
        new_shape = [0, 0]
        new_shape[0] = val_data.shape[0]
        new_shape[1] = val_data.shape[1] + 1
        val_data_new = np.zeros(new_shape)
        val_data = val_data_new

    return train_data, train_labels, val_data, val_labels


def train(
    data: np.ndarray,
    labels: np.ndarray,
    structure: str,
    trainer_options: dict,
    dataset_options: dict,
    eval: bool,
    model_dir_path: str,
):
    """Train QuClassi with the dataset.

    :param np.ndarray data: data
    :param np.ndarray labels: labels
    :param str structure: structure of QuClassi
    :param dict trainer_options: options for QuClassiTrainer
    :param dict dataset_options: options to create dataset
    :param bool eval: whether evaluate QuClassi each epoch or not
    :param str model_dir_path: path to model directory
    """
    train_data, train_labels, val_data, val_labels = preprocess_dataset(
        data=data, labels=labels, **dataset_options
    )

    quclassi = QuClassi(
        classical_data_size=len(train_data[0]), labels=np.unique(train_labels).tolist()
    )
    quclassi.build(structure)
    quclassi_trainer = QuClassiTrainer(quclassi=quclassi, **trainer_options)

    quclassi_trainer.train(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        eval=eval,
    )

    quclassi_trainer.save(model_dir_path=model_dir_path)
