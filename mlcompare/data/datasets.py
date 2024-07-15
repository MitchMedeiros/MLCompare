from __future__ import annotations as _annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Literal

from pydantic import BaseModel

from ..params_reader import ParamsReader
from ..types import DatasetType, ParamsInput

logger = logging.getLogger(__name__)


class BaseDataset(ABC, BaseModel):
    """
    Base class for datasets, containing attributes related to data cleaning and reformatting.

    Attributes:
    -----------
        save_name (str | None): The name to use for files saved from this dataset. Should be unique across datasets.
        target_column (str): Column name for the target of the predictions.
        columns_to_drop (list[str] | None): List of column names to be dropped from the dataset.
        columns_to_encode (list[str] | None): List of column names to be one-hot encoded in the dataset.
    """

    save_name: str | None = None
    target_column: str
    columns_to_drop: list[str] | None = None
    columns_to_onehot_encode: list[str] | None = None

    @abstractmethod
    def create_save_name(self):
        ...

    @abstractmethod
    def validate_data(self):
        ...


class LocalDataset(BaseDataset):
    """
    Represents a locally saved dataset with all the fields required to load and prepare it for model evaluation.

    Attributes:
    -----------
        file_path (str | Path): The path to the local dataset file.
        save_name (str | None): The name to use for files saved from this dataset. Should be unique across datasets.
        If None, the file will be saved with the same name as the original file.
        dataset_type (Literal["kaggle", "local"]): Type of dataset. Accepts 'kaggle' or 'local'.
        target_column (str): Column name for the target of the predictions.
        columns_to_drop (list[str] | None): List of column names to be dropped from the dataset.
        columns_to_encode (list[str] | None): List of column names to be one-hot encoded in the dataset.
    """

    file_path: str | Path

    def model_post_init(self, Any) -> None:
        # For explicitness; Pydantic already does this
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

        self.create_save_name()
        self.validate_data()

    def create_save_name(self):
        if self.save_name is None:
            self.save_name = self.file_path.stem

    def validate_data(self):
        try:
            self.file_path.resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")


class KaggleDataset(BaseDataset):
    """
    Represents a Kaggle dataset with all the fields required to download and prepare it for model evaluation.

    Attributes:
    -----------
        username (str): Username of the Kaggle user who owns the dataset.
        dataset_name (str): Name of the Kaggle dataset.
        file_name (str): Name of the file to be downloaded from the dataset.
        save_name (str | None): The name to use for files saved from this dataset. Should be unique across datasets.
        If None, the file will be named `username_dataset_name`.
        dataset_type (Literal["kaggle", "local"]): Type of dataset. Accepts 'kaggle' or 'local'.
        target_column (str): Column name for the target of the predictions.
        columns_to_drop (list[str] | None): List of column names to be dropped from the dataset.
        columns_to_encode (list[str] | None): List of column names to be one-hot encoded in the dataset.
    """

    username: str
    dataset_name: str
    file_name: str

    def model_post_init(self, Any) -> None:
        self.create_save_name()
        self.validate_data()

    def create_save_name(self):
        if self.save_name is None:
            self.save_name = self.username + "_" + self.dataset_name

    def validate_data(self):
        if not self.file_name.endswith(".csv"):
            raise ValueError("The dataset file should be in CSV format.")


class HuggingFaceDataset(BaseDataset):
    pass

    def create_save_name(self):
        pass

    def validate_data(self):
        pass


class OpenMLDataset(BaseDataset):
    pass

    def create_save_name(self):
        pass

    def validate_data(self):
        pass


class DatasetFactory:
    """
    Creates Dataset objects such as LocalDataset, KaggleDataset, etc. from a list of dictionaries.

    Attributes:
    -----------
        params_list (list[dict[str, Any]] | Path): List of dictionaries containing dataset parameters or a Path to
        a .json file with one. For a list of keys required in each dictionary, see below:

        Required keys for all dataset types:
            dataset_type Literal["kaggle", "local"]: Type of dataset. Accepts 'kaggle' or 'local'.
            target_column (str): Name of the target column in the dataset.

        Additional required keys for 'local' datasets:
            file_path (str): Path to the local dataset file. It can be relative or absolute.

        Additional required keys for 'kaggle' datasets:
            username (str): Kaggle username of the dataset owner.
            dataset_name (str): Name of the Kaggle dataset.
            file_name (str): Name of the file to download from the dataset.

        Optional Keys:
            save_name (str): Name to use for files saved from this dataset. Should be unique across datasets.
            columns_to_drop (list[str]): List of column names to drop from the downloaded data.
            columns_to_onehot_encode (list[str]): List of column names to encode using a specific encoding method.

    Raises:
    -------
        AssertionError: If `dataset_params` is not a list of dictionaries or a path to a .json file containing one.
    """

    def __init__(self, params_list: ParamsInput) -> None:
        self.params_list = ParamsReader.read(params_list)

    def __iter__(
        self,
    ) -> Generator[DatasetType, None, None]:
        """
        Makes the class iterable, yielding dataset instances one by one.

        Yields:
        -------
            BaseDataset: An instance of a dataset class.
        """
        for params in self.params_list:
            yield DatasetFactory.create(**params)

    @staticmethod
    def create(
        dataset_type: Literal["local", "kaggle", "hugging face", "openml"], **kwargs
    ) -> DatasetType:
        """
        Factory method to create a dataset instance based on the dataset type.

        Args:
        -----
            dataset_type (Literal["local", "kaggle", "hugging face", "openml"]): The type of dataset to create.
            **kwargs: Arbitrary keyword arguments to be passed to the dataset class constructor.

        Returns:
        --------
            BaseDataset: An instance of a dataset class (KaggleDataset or LocalDataset).

        Raises:
        -------
            ValueError: If an unknown dataset type is provided.
        """
        # dataset_type = dataset_type.lower()

        match dataset_type:
            case "local":
                return LocalDataset(**kwargs)
            case "kaggle":
                return KaggleDataset(**kwargs)
            case "hugging face":
                return HuggingFaceDataset(**kwargs)
            case "openml":
                return OpenMLDataset(**kwargs)
            case _:
                raise ValueError(f"Dataset type not implemented: {dataset_type}")
