import itertools
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split
import torch

from quantum_machine_learning.quanv_nn.quanv_layer import QuanvLayer
from quantum_machine_learning.quanv_nn.quanv_nn import QuanvNN
from quantum_machine_learning.torch_utils.torch_trainer import TorchTrainer
from quantum_machine_learning.torch_utils.plain_dataset import PlainDataset


def preprocess_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    encoding_method: Callable,
    train_size: float = 0.75,
    seed: int = 901,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the dataset.

    :param np.ndarray data: data
    :param np.ndarray labels: labels
    :param Callable encoding_method: encoding method
    :param float train_size: train data size, defaults to 0.75
    :param int seed: random seed, defaults to 901
    :param bool shuffle: whether shuffle the data or not, defaults to True
    :return tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: train data, train labels, validation data, validation labels
    """

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

    # Encode the data.
    train_data = encoding_method(train_data)
    val_data = encoding_method(val_data)

    return train_data, train_labels, val_data, val_labels


def train(
    data: np.ndarray,
    labels: np.ndarray,
    quanv_layer_options: dict,
    classical_model: torch.nn.Module,
    dataset_options: dict,
    epochs: int,
    model_dir_path: str,
    shots: int,
    is_lookup_mode: bool = True,
):
    """Train QuanvNN with the dataset.

    :param np.ndarray data: data
    :param np.ndarray labels: labels
    :param dict quanv_layer_options: options for QuanvLayers
    :param dict dataset_options: options to create dataset
    :param int epochs: number of epochs
    :param str model_dir_path: path to model directory
    :param int shots: number of shots
    :param bool is_lookup_mode: whether look-up tables should be built in advance
    """
    quanv_layer = QuanvLayer(**quanv_layer_options)
    if is_lookup_mode:
        # Make all possibly input patterns.
        pattern = [0, np.pi]
        patterns = np.array(
            list(itertools.product(pattern, repeat=quanv_layer.num_qubits))
        )
        quanv_layer.build_lookup_tables(patterns=patterns, shots=shots)
    quanv_nn = QuanvNN(quanv_layer=quanv_layer, classical_model=classical_model)

    torch_trainer = TorchTrainer(model=quanv_nn)

    train_data, train_labels, val_data, val_labels = preprocess_dataset(
        data=data, labels=labels, **dataset_options
    )
    train_dataset = PlainDataset(
        torch.Tensor(train_data), torch.Tensor(train_labels).type(torch.LongTensor)
    )
    val_dataset = PlainDataset(
        torch.Tensor(val_data), torch.Tensor(val_labels).type(torch.LongTensor)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=dataset_options["shuffle"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=dataset_options["shuffle"]
    )
    torch_trainer.train_and_eval(
        train_loader=train_loader,
        validation_loader=val_loader,
        epochs=epochs,
    )

    quanv_nn.save(model_dir_path=model_dir_path)
