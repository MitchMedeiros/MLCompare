from __future__ import annotations as _annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
