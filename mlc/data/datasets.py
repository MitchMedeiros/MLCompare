from __future__ import annotations as _annotations

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Literal, TypeAlias

import pandas as pd
from pydantic import BaseModel, Field

from ..params_reader import ParamsInput, ParamsReader

logger = logging.getLogger(__name__)


def df_from_suffix(file_path: str | Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Takes a file path as input and creates a Pandas DataFrame based on the file type.

    Args:
    -----
        file_path (str | Path): Path to the data file.
        logger (logging.Logger): Logger object to log messages.

    Returns:
    --------
        pd.DataFrame: Data from the file as a Pandas DataFrame.

    Raises:
    -------
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file is not found.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    try:
        suffix = file_path.suffix
        match suffix:
            case ".parquet":
                df = pd.read_parquet(file_path)
            case ".csv":
                df = pd.read_csv(file_path)
            case ".pkl":
                df = pd.read_pickle(file_path)
            case ".json":
                df = pd.read_json(file_path)
            case _:
                raise ValueError(
                    "Data file must be a .parquet, .csv, .pkl, or .json file."
                )
        return df
    except FileNotFoundError:
        logger.exception(f"File not found: {file_path}")
        raise


class BaseDataset(ABC, BaseModel):
    """
    Base class for datasets, containing attributes related to data cleaning and reformatting.

    Attributes:
    -----------
        target (str): Column name for the target of the predictions.
        save_name (str | None): Name to use for files saved from this dataset. Should be unique across datasets.
        drop (list[str] | None): List of column names to be dropped from the dataset.
        onehot_encode (list[str] | None): List of column names to be one-hot encoded in the dataset.
    """

    target: str
    save_name: str | None = Field(None, alias="saveName")
    drop: list[str] | None = None
    onehot_encode: list[str] | None = Field(None, alias="onehotEncode")

    @abstractmethod
    def model_post_init(self, Any) -> None: ...

    @abstractmethod
    def create_save_name(self) -> None: ...

    @abstractmethod
    def get_data(self) -> pd.DataFrame: ...


class LocalDataset(BaseDataset):
    """
    Represents a locally saved dataset with all the fields required to load and prepare it for model evaluation.

    Attributes:
    -----------
        file_path (str | Path): Path to the local dataset file.
        target (str): Column name for the target of the predictions.
        save_name (str | None): Name to use for files saved from this dataset. Should be unique across datasets.
        If None, the file will be saved with the same name as the original file.
        drop (list[str] | None): List of column names to be dropped from the dataset.
        onehot_encode (list[str] | None): List of column names to be one-hot encoded in the dataset.
    """

    file_path: str | Path = Field(..., alias="path")

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
        df = df_from_suffix(self.file_path, logger)
        logger.info(
            f"Local data successfully loaded and converted to DataFrame:\n{df.head(3)}"
        )
        return df


class KaggleDataset(BaseDataset):
    """
    Represents a Kaggle dataset with all the fields required to download and prepare it for model evaluation.

    Attributes:
    -----------
        user (str): Username of the Kaggle user who owns the dataset.
        dataset (str): Name of the Kaggle dataset.
        file (str): Name of the file to be downloaded from the dataset.
        target (str): Column name for the target of the predictions.
        save_name (str | None): Name to use for files saved from this dataset. Should be unique across datasets.
        If None, the file will be named `user_dataset`.
        drop (list[str] | None): List of column names to be dropped from the dataset.
        onehot_encode (list[str] | None): List of column names to be one-hot encoded in the dataset.
    """

    user: str
    dataset: str
    file: str

    def model_post_init(self, Any) -> None:
        self.validate_data()
        self.create_save_name()

    def validate_data(self) -> None:
        if not self.file.endswith(".csv"):
            raise ValueError("The dataset file should be in CSV format.")

    def create_save_name(self) -> None:
        if self.save_name is None:
            self.save_name = self.user + "_" + self.dataset

    def get_data(self) -> pd.DataFrame:
        """
        Downloads a Kaggle dataset. Currently only implemented for CSV files.

        Returns:
            pd.DataFrame: Downloaded data as a Pandas DataFrame.

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
                self.user,
                self.dataset,
                self.file,
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
                dataset_files = kaggle.api.datasets_list_files(self.user, self.dataset)
            except ApiException:
                raise ValueError(
                    "No Kaggle dataset files found using the provided username and dataset name."
                )

            if self.file not in [
                file_metadata["name"] for file_metadata in dataset_files["datasetFiles"]
            ]:
                raise FileNotFoundError(
                    f"Dataset: {self.user}/{self.dataset} was successfully found but doesn't "
                    f"contain any file named: {self.file}"
                )

        raise Exception("An unknown error occurred while downloading the dataset.")


class HuggingFaceDataset(BaseDataset):
    repo: str
    file: str

    def model_post_init(self, Any) -> None:
        self.create_save_name()

    def create_save_name(self) -> None:
        if self.save_name is None:
            self.save_name = self.repo

    def get_data(self) -> pd.DataFrame:
        from huggingface_hub import hf_hub_download

        tmp_save_dir = Path("huggingface_pipeline_data")

        try:
            saved_data_path = hf_hub_download(
                repo_id=self.repo,
                filename=self.file,
                repo_type="dataset",
                local_dir=tmp_save_dir,
            )
            df = df_from_suffix(saved_data_path, logger)
        except Exception:
            shutil.rmtree(tmp_save_dir)
            raise

        logger.info(
            f"Hugging Face data successfully loaded and converted to DataFrame:\n{df.head(3)}"
        )
        shutil.rmtree(tmp_save_dir)
        return df


class OpenMLDataset(BaseDataset):
    id: int | str

    def model_post_init(self, Any) -> None:
        self.create_save_name()

    # Called within `get_data` since the dataset name won't be known until data is retrieved
    def create_save_name(self) -> None:
        if self.save_name is None:
            self.save_name = f"openml_dataset_{self.id}"

    def get_data(self) -> pd.DataFrame:
        from openml.datasets import get_dataset

        openml_data = get_dataset(
            self.id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        df = openml_data.get_data()[0]
        logger.info(
            f"OpenML data successfully loaded and converted to DataFrame:\n{df.head(3)}"
        )
        return df


DatasetType: TypeAlias = (
    LocalDataset | KaggleDataset | HuggingFaceDataset | OpenMLDataset
)


class DatasetFactory:
    """
    Creates Dataset objects such as LocalDataset, KaggleDataset, etc. from a list of dictionaries.

    Attributes:
    -----------
        params_list (list[dict[str, Any]] | Path): List of dictionaries containing dataset parameters or a
        path to a .json file with one. For a list of keys required in each dictionary, see below:

        Required keys for all dataset types:
            dataset_type Literal["kaggle", "local"]: Type of dataset. Accepts 'kaggle' or 'local'.
            target (str): Name of the target column in the dataset.

        Additional required keys for 'local' datasets:
            file_path (str): Path to the local dataset file. It can be relative or absolute.

        Additional required keys for 'kaggle' datasets:
            user (str): Kaggle username of the dataset owner.
            dataset (str): Name of the Kaggle dataset.
            file (str): Name of the file to download from the dataset.

        Optional Keys:
            save_name (str): Name to use for files saved from this dataset. Should be unique across datasets.
            drop (list[str]): List of column names to drop from the downloaded data.
            onehot_encode (list[str]): List of column names to encode using a specific encoding method.

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
        type: Literal["local", "kaggle", "hugging face", "huggingface", "openml"],
        **kwargs,
    ) -> DatasetType:
        """
        Factory method to create a dataset instance based on the dataset type.

        Args:
        -----
            dataset_type (Literal["local", "kaggle", "hugging face", "huggingface", "openml"]):
            The type of dataset to create.
            **kwargs: Keyword arguments passed to the dataset class constructors.

        Returns:
        --------
            BaseDataset: An instance of a dataset class (KaggleDataset or LocalDataset).

        Raises:
        -------
            ValueError: If an unknown dataset type is provided.
        """
        match type:
            case "local":
                return LocalDataset(**kwargs)
            case "kaggle":
                return KaggleDataset(**kwargs)
            case "hugging face" | "huggingface":
                return HuggingFaceDataset(**kwargs)
            case "openml":
                return OpenMLDataset(**kwargs)
            case _:
                raise ValueError(
                    f"type: {type} given in the dataset parameters is not supported. Valid options "
                    "are: 'local', 'kaggle', 'hugging face', or 'openml'."
                )
