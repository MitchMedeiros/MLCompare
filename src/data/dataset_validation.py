import json
import logging
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel

from .dataset_processor import DatasetType

logger = logging.getLogger(__name__)

Parameters: TypeAlias = dict[str, Any]
ParametersList: TypeAlias = list[Parameters]


class BaseDataset(BaseModel):
    """
    Base class for datasets, containing attributes related to data cleaning and reformatting.

    Attributes:
    -----------
        dataset_type (Literal["kaggle", "local"]): Type of dataset. Accepts 'kaggle' or 'local'.
        target_column (str): Column name for the target of the predictions.
        columns_to_drop (list[str] | None): List of column names to be dropped from the dataset.
        columns_to_encode (list[str] | None): List of column names to be one-hot encoded in the dataset.
    """

    dataset_type: Literal["kaggle", "local"]
    target_column: str
    columns_to_drop: list[str] | None
    columns_to_onehot_encode: list[str] | None


class KaggleDataset(BaseDataset):
    """
    Represents a Kaggle dataset with all the fields required to download and prepare it for model evaluation.

    Attributes:
    -----------
        dataset_type (Literal["kaggle", "local"]): Type of dataset. Accepts 'kaggle' or 'local'.
        username (str): Username of the Kaggle user who owns the dataset.
        dataset_name (str): Name of the Kaggle dataset.
        file_name (str): Name of the file to be downloaded from the dataset.
        target_column (str): Name of the target column in the dataset for training and prediction.
        columns_to_drop (list[str] | None): List of column names to be dropped from the dataset.
        If None, no columns will be dropped.
        columns_to_onehot_encode (list[str] | None): List of column names to be encoded in the dataset.
        If None, no columns will be encoded.
    """

    username: str
    dataset_name: str
    file_name: str


class LocalDataset(BaseDataset):
    """
    Represents a locally saved dataset with all the fields required to load and prepare it for model evaluation.

    Attributes:
    -----------
        dataset_type (Literal["kaggle", "local"]): Type of dataset. Accepts 'kaggle' or 'local'.
        file_path (Path): The path to the local dataset file.
        save_name (str): The name to use for files saved from this dataset. Should be unique across datasets.
        If None, the file will be saved with the same name as the original file.
        target_column (str): The name of the target column in the dataset for training and prediction.
        columns_to_drop (list[str] | None): A list of column names to be dropped from the dataset.
        If None, no columns will be dropped.
        columns_to_onehot_encode (list[str] | None): A list of column names to be encoded in the dataset.
        If None, no columns will be encoded.
    """

    file_path: Path
    save_name: str | None


def read_from_json_file(file_path: Path) -> ParametersList:
    """
    Read from a JSON file.

    Args:
    -----
        file_path (Path): Path to the JSON file containing dataset parameters.

    Returns:
    --------
        list[dict[str, Any]]: List of dataset parameters.

    Raises:
    -------
        FileNotFoundError: If the file is not found at the provided path.
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError as e:
        logger.exception(f"Could not find file: {file_path}")
        raise e


def create_dataset_instance(params: Parameters) -> DatasetType:
    """
    Creates an instance of KaggleDataset or LocalDataset based on the value of `dataset_type`
    within the provided parameters dict.

    Args:
    -----
        params (dict[str, Any]): Dictionary of dataset parameters.

    Returns:
    --------
        KaggleDataset | LocalDataset: An instance of KaggleDataset or LocalDataset.

    Raises:
    -------
        ValueError: If the value of `dataset_type` is not 'kaggle' or 'local'.
    """
    dataset_type = params.get("dataset_type")
    if dataset_type == "kaggle":
        return KaggleDataset(**params)
    elif dataset_type == "local":
        return LocalDataset(**params)
    else:
        raise ValueError("dataset_type must be either 'kaggle' or 'local'")


def validate_dataset_params(dataset_params: ParametersList | Path) -> list[DatasetType]:
    """
    Creates a list of KaggleDataset and LocalDataset objects from a dictionary of parameters.

    You can provide the dictionary directly or a path to a .json file. The list returned by
    this function can be iterated over with the DataProcessor class to process each dataset.

    Args:
    -----
        Required keys for all dataset types:
            dataset_type Literal["kaggle", "local"]: The type of dataset. Accepts 'kaggle' or 'local'.
            target_column (str): The name of the target column in the dataset.

        Additional required keys for 'kaggle' datasets:
            username (str): The Kaggle username of the dataset owner.
            dataset_name (str): The name of the Kaggle dataset.
            file_name (str): The name of the file to download from the dataset.

        Additional required keys for 'local' datasets:
            file_path (str): The path to the local dataset file. Should be a Path object. The file
            should be either within the data directory.

        Optional Keys:
            columns_to_drop (list[str]): A list of column names to drop from the downloaded data.
            columns_to_onehot_encode (list[str]): A list of column names to encode using a specific encoding method.
            save_name (str): The name to use for files saved from this dataset. Should be unique across datasets.

    Returns:
    --------
        list[DatasetType]: A list of DatasetType objects.

    Raises:
    -------
        AssertionError: If the dataset_params are not a list of dictionaries or a path to a .json file.
        AssertionError: If any list element in `dataset_params` is not a dictionary.

    Example Usage:
    --------------
    ```python
    dataset_params = [
        {
            "dataset_type": "kaggle",
            "username": "nomadic-human",
            "dataset_name": "food-prices-by-country",
            "file_name": "food_prices.csv",
            "target_column": "cappuccino_price",
            "columns_to_drop": ["date", "continent"],
            "columns_to_encode": ["country"]
        },
        {
            "dataset_type": "kaggle",
            "username": "dino-lover123",
            "dataset_name": "dinosaur-physiology",
            "file_name": "species.csv",
            "target_column": "skull_circumference",
        },
        {
            "dataset_type": "local",
            "file_path": a_path_object,
            "target_column": "example",
            "columns_to_drop": ["column1", "column2"],
            "columns_to_encode": ["column3", "column4"]
        }
    ]

    datasets = validate_datasets(dataset_params)
    ```
    """
    if isinstance(dataset_params, Path):
        dataset_params = read_from_json_file(dataset_params)

    assert isinstance(
        dataset_params, list
    ), "dataset_params must be a list of dictionaries or a path to .json file containing one."

    assert all(
        isinstance(params, dict) for params in dataset_params
    ), "Each list element in `dataset_params` must be a dictionary."

    dataset_validators = [create_dataset_instance(params) for params in dataset_params]
    return dataset_validators
