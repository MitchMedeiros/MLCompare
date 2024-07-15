from __future__ import annotations as _annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Literal, TypeAlias

import pandas as pd
from pydantic import BaseModel

from ..params_reader import ParamsReader
from ..types import ParamsInput

logger = logging.getLogger(__name__)


class BaseDataset(ABC, BaseModel):
    """
    Base class for datasets, containing attributes related to data cleaning and reformatting.

    Attributes:
    -----------
        target_column (str): Column name for the target of the predictions.
        save_name (str | None): The name to use for files saved from this dataset. Should be unique across datasets.
        columns_to_drop (list[str] | None): List of column names to be dropped from the dataset.
        columns_to_encode (list[str] | None): List of column names to be one-hot encoded in the dataset.
    """

    target_column: str
    save_name: str | None = None
    columns_to_drop: list[str] | None = None
    columns_to_onehot_encode: list[str] | None = None

    @abstractmethod
    def validate_data(self) -> None:
        ...

    @abstractmethod
    def create_save_name(self) -> None:
        ...

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        ...


class LocalDataset(BaseDataset):
    """
    Represents a locally saved dataset with all the fields required to load and prepare it for model evaluation.

    Attributes:
    -----------
        file_path (str | Path): The path to the local dataset file.
        target_column (str): Column name for the target of the predictions.
        save_name (str | None): The name to use for files saved from this dataset. Should be unique across datasets.
        If None, the file will be saved with the same name as the original file.
        columns_to_drop (list[str] | None): List of column names to be dropped from the dataset.
        columns_to_encode (list[str] | None): List of column names to be one-hot encoded in the dataset.
    """

    file_path: str | Path

    def model_post_init(self, Any) -> None:
        # For explicitness; Pydantic already does this
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

        self.validate_data()
        self.create_save_name()

    def validate_data(self) -> None:
        try:
            self.file_path.resolve(strict=True)  # type: ignore
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def create_save_name(self) -> None:
        if self.save_name is None:
            self.save_name = self.file_path.stem  # type: ignore

    def get_data(self) -> pd.DataFrame:
        try:
            extension = self.file_path.suffix  # type: ignore
            match extension:
                case ".parquet":
                    df = pd.read_parquet(self.file_path)
                case ".csv":
                    df = pd.read_csv(self.file_path)
                case ".pkl":
                    df = pd.read_pickle(self.file_path)
                case ".json":
                    df = pd.read_json(self.file_path)
                case _:
                    raise ValueError(
                        "Data file must be a .parquet, .csv, .pkl, or .json file."
                    )
        except FileNotFoundError:
            logger.exception(f"File not found: {self.file_path}")
            raise

        logger.info(
            f"Data successfully loaded and converted to DataFrame:\n{df.head(3)}"
        )
        return df


class KaggleDataset(BaseDataset):
    """
    Represents a Kaggle dataset with all the fields required to download and prepare it for model evaluation.

    Attributes:
    -----------
        username (str): Username of the Kaggle user who owns the dataset.
        dataset_name (str): Name of the Kaggle dataset.
        file_name (str): Name of the file to be downloaded from the dataset.
        target_column (str): Column name for the target of the predictions.
        save_name (str | None): The name to use for files saved from this dataset. Should be unique across datasets.
        If None, the file will be named `username_dataset_name`.
        columns_to_drop (list[str] | None): List of column names to be dropped from the dataset.
        columns_to_encode (list[str] | None): List of column names to be one-hot encoded in the dataset.
    """

    username: str
    dataset_name: str
    file_name: str

    def model_post_init(self, Any) -> None:
        self.validate_data()
        self.create_save_name()

    def validate_data(self) -> None:
        if not self.file_name.endswith(".csv"):
            raise ValueError("The dataset file should be in CSV format.")

    def create_save_name(self) -> None:
        if self.save_name is None:
            self.save_name = self.username + "_" + self.dataset_name

    def get_data(self) -> pd.DataFrame:
        """
        Downloads a Kaggle dataset. Currently only implemented for CSV files.

        Returns:
            pd.DataFrame: The downloaded data as a Pandas DataFrame.

        Raises:
            ConnectionError: If unable to authenticate with Kaggle.
            ValueError: If there's no Kaggle dataset files for the provided user and dataset names.
            ValueError: If the file name provided doesn't match any of the files in the matched dataset.
        """
        from io import StringIO

        import kaggle
        from kaggle.api.kaggle_api_extended import ApiException

        try:
            data = kaggle.api.datasets_download_file(
                self.username,
                self.dataset_name,
                self.file_name,
            )

            file_like = StringIO(data)
            df = pd.read_csv(file_like)
            logger.info("Data successfully downloaded")
            return df
        except OSError:
            # Should never occur since empty environment variables are added in the `__init__.py`,
            # which should be sufficient for `dataset_downloads_file`.
            raise ConnectionRefusedError(
                "Unable to authenticate with Kaggle. Ensure that you have a Kaggle API key saved "
                "to the appropriate file or your username and password in your environment variables. "
                "See: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials"
            )
        except ApiException:
            try:
                dataset_files = kaggle.api.datasets_list_files(
                    self.username, self.dataset_name
                )
            except ApiException:
                raise ValueError(
                    "No Kaggle dataset files found using the provided username and dataset name."
                )

            if self.file_name not in [
                file["name"] for file in dataset_files["datasetFiles"]
            ]:
                raise ValueError(
                    f"Dataset: {self.username}/{self.dataset_name} was successfully found but doesn't "
                    f"contain any file named: {self.file_name}"
                )

        raise Exception("An unknown error occurred while downloading the dataset.")


class HuggingFaceDataset(BaseDataset):
    def create_save_name(self) -> None:
        pass

    def validate_data(self) -> None:
        pass

    def get_data(self) -> pd.DataFrame:
        return pd.DataFrame()


class OpenMLDataset(BaseDataset):
    def create_save_name(self) -> None:
        pass

    def validate_data(self) -> None:
        pass

    def get_data(self) -> pd.DataFrame:
        return pd.DataFrame()


DatasetType: TypeAlias = (
    LocalDataset | KaggleDataset | HuggingFaceDataset | OpenMLDataset
)


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
