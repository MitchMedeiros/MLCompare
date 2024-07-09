import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class KaggleDataset(BaseModel):
    """
    Represents a Kaggle dataset with all the fields required to download and prepare it for model evaluation.

    ### Attributes:
        username (str): The username of the Kaggle user who owns the dataset.
        dataset_name (str): The name of the Kaggle dataset.
        file_name (str): The name of the file associated with the dataset.
        save_name (str): The name to use for files saved from this dataset. Should be unique across datasets.
        target_column (str): The name of the target column in the dataset.
        columns_to_drop (list[str] | None): A list of column names to be dropped from the dataset.
    If None, no columns will be dropped.
        columns_to_encode (list[str] | None): A list of column names to be encoded in the dataset.
    If None, no columns will be encoded.
    """

    username: str
    dataset_name: str
    file_name: str
    target_column: str
    columns_to_drop: list[str] | None
    columns_to_encode: list[str] | None


class LocalDataset(BaseModel):
    """
    Represents a locally saved dataset with all the fields required to load and prepare it for model evaluation.

    ### Attributes:
        file_path (Path): The path to the local dataset file.
        save_name (str): The name to use for files saved from this dataset. Should be unique across datasets.
        target_column (str): The name of the target column in the dataset.
        columns_to_drop (list[str] | None): A list of column names to be dropped from the dataset.
    If None, no columns will be dropped.
        columns_to_encode (list[str] | None): A list of column names to be encoded in the dataset.
    If None, no columns will be encoded.
        save_name (str): The name to use for files saved from this dataset. Should be unique across datasets.
    If None, the file will be saved with the same name as the original file.
    """

    file_path: Path
    target_column: str
    columns_to_drop: list[str] | None
    columns_to_encode: list[str] | None
    save_name: str | None


def validate_dataset_params(
    dataset_params: list[dict[str, Any]] | Path
) -> list[KaggleDataset | LocalDataset]:
    """
    Creates a list of KaggleDataset and LocalDataset objects from a dictionary of parameters. You can
    provide the dictionary directly or a path to a .json file. The outer dictionary should have the
    dataset name as the key and a dictionary of parameters as the value for each dataset.

    This list returned by this function can be iterated over with the DataProcessor class to process each dataset.

    ### Args:
        The accessed keys are described below.

        Required keys for all dataset types:
            dataset_type: The type of dataset. Accepts 'kaggle' or 'local'.
            target_column: The name of the target column in the dataset.

        Additional required keys for 'kaggle' datasets:
            username: The Kaggle username of the dataset owner.
            dataset_name: The name of the Kaggle dataset.
            file_name: The name of the file to download from the dataset.

        Additional required keys for 'local' datasets:
            file_path: The path to the local dataset file. Should be a Path object. The file
            should be either within the data directory.

        Optional Keys:
            columns_to_drop: A list of column names to drop from the downloaded data.
            columns_to_encode: A list of column names to encode using a specific encoding method.

    ### Example Usage:
    ```python
    dataset_params = {
        "dataset1": {
            "dataset_type": "kaggle",
            "username": "nomadic-human",
            "dataset_name": "food-prices-by-country",
            "file_name": "food_prices.csv",
            "target_column": "cappuccino_price",
            "columns_to_drop": ["date", "continent"],
            "columns_to_encode": ["country"]
        },
        "dataset2": {
            "dataset_type": "local",
            "file_path": a_path_object,
            "target_column": "example",
            "columns_to_drop": ["column1", "column2"],
            "columns_to_encode": ["column3", "column4"]
        },
        "dataset3": {
            "dataset_type": "kaggle",
            "username": "dino-lover123",
            "dataset_name": "dinosaur-physiology",
            "file_name": "species.csv",
            "target_column": "skull_circumference",
        }
    }

    datasets = validate_datasets(dataset_params)
    ```
    """
    if isinstance(dataset_params, Path):
        try:
            with open(dataset_params, "r") as file:
                dataset_params = json.load(file)
        except FileNotFoundError as e:
            logger.exception(f"Could not find file: {dataset_params}")
            raise e

    assert isinstance(
        dataset_params, list
    ), "dataset_params must be a list of dictionaries or a path to .json file containing one."

    assert all(
        isinstance(params, dict) for params in dataset_params
    ), "Each list element in `dataset_params` must be a dictionary."

    dataset_validators = [
        (
            KaggleDataset(**params)
            if params.get("dataset_type") == "kaggle"
            else (
                LocalDataset(**params)
                if params.get("dataset_type") == "local"
                else None
            )
        )
        for params in dataset_params
    ]

    if None in dataset_validators:
        raise ValueError(
            "Invalid value for dataset_type for one or more entries. Must be 'kaggle' or 'local'."
        )

    return dataset_validators  # type: ignore
