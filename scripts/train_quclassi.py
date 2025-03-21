import argparse
import shutil
import yaml

from qiskit import primitives

from quantum_machine_learning.dataset import get_dataset
from quantum_machine_learning.quclassi.train import train
from quantum_machine_learning.utils import fix_seed, encode_through_arcsin

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and Evaluate QuClassi with iris dataset."
    )
    parser.add_argument(
        "-c",
        "--config_yaml_path",
        required=False,
        type=str,
        default="./config_iris.yaml",
    )
    args = parser.parse_args()

    config_yaml_path = args.config_yaml_path
    with open(config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)

    # Fix the seed.
    seed = config["general"]["seed"]
    fix_seed(seed)

    # Get the dataset.
    dataset_options = config["dataset_options"]
    dataset_options["encoding_method"] = encode_through_arcsin
    dataset_name = dataset_options["name"]
    del dataset_options["name"]
    data, labels = get_dataset(dataset_name)

    # Get options for QuClassi.
    structure = config["quclassi_options"]["structure"]

    # Get options for QuClassiTrainer
    trainer_options = config["trainer_options"]
    trainer_options["sampler"] = primitives.StatevectorSampler(seed=seed)
    eval = trainer_options["eval"]
    del trainer_options["eval"]
    model_dir_path = trainer_options["model_dir_path"]
    del trainer_options["model_dir_path"]

    train(
        data=data,
        labels=labels,
        structure=structure,
        trainer_options=trainer_options,
        dataset_options=dataset_options,
        eval=eval,
        model_dir_path=model_dir_path,
    )

    shutil.copy2(config_yaml_path, model_dir_path)
