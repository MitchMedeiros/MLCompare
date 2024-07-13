from __future__ import annotations as _annotations

import logging
import pickle
from io import StringIO
from pathlib import Path
from typing import Literal

import kaggle
import pandas as pd
from kaggle.rest import ApiException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ..types import DataFileSuffix, DatasetType, SplitDataTuple
from .datasets import KaggleDataset, LocalDataset
from .split_data import SplitData

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Processes validated datasets to prepare them for model training and evaluation.

    Attributes:
        dataset (DatasetType): The validated dataset to be processed.
        data_directory (Path): Directory to save files for `save_dataframe` and `split_and_save_data` methods.
    """

    def __init__(self, dataset: DatasetType, data_directory: Path) -> None:
        self._validate_init_params(dataset, data_directory)
        self.dataset = dataset
        self.data_directory = data_directory

        self.data = self._load_data()
        self.save_name = self._generate_save_name()
        self.target_column = dataset.target_column
        self.columns_to_drop = dataset.columns_to_drop
        self.columns_to_onehot_encode = dataset.columns_to_onehot_encode

    def _validate_init_params(self, dataset: DatasetType, data_directory: Path) -> None:
        if not isinstance(dataset, KaggleDataset | LocalDataset):
            raise ValueError("Data must be a KaggleDataset or LocalDataset object.")

        if not isinstance(data_directory, Path):
            raise ValueError("Data directory must be a Path object.")

    def _generate_save_name(self) -> str:
        if isinstance(self.dataset, LocalDataset):
            return self.dataset.file_path.stem
        elif isinstance(self.dataset, KaggleDataset):
            return f"{self.dataset.username}_{self.dataset.dataset_name}"

    def _load_data(self) -> pd.DataFrame:
        if isinstance(self.dataset, LocalDataset):
            df = self._read_from_path(self.dataset.file_path)
        elif isinstance(self.dataset, KaggleDataset):
            df = self._download_kaggle_data(
                self.dataset.username,
                self.dataset.dataset_name,
                self.dataset.file_name,
            )
        logger.info(
            f"Data successfully loaded and converted to DataFrame:\n{df.head(3)}"
        )
        return df

    def _read_from_path(self, file_path: Path) -> pd.DataFrame:
        try:
            extension = file_path.suffix
            if extension == ".parquet":
                df = pd.read_parquet(file_path)
            elif extension == ".csv":
                df = pd.read_csv(file_path)
            elif extension == ".pkl":
                df = pd.read_pickle(file_path)
            elif extension == ".json":
                df = pd.read_json(file_path)
            else:
                raise ValueError("Data file must be a Parquet, CSV, PKL, or JSON file.")
        except FileNotFoundError as e:
            logger.exception(f"File not found: {file_path}")
            raise e

        return df

    def _download_kaggle_data(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_name: str,
    ) -> pd.DataFrame:
        """
        Downloads a Kaggle dataset. Overwrites any existing data for the class instance.
        Currently only implemented for CSV files.

        Args:
            dataset_owner (str): The user under which the dataset is provided.
            dataset_name (str): The name of the dataset.
            file_name (str): The file to be downloaded from the dataset.

        Returns:
            pd.DataFrame: The downloaded data as a Pandas DataFrame.

        Raises:
            ConnectionError: If unable to authenticate with Kaggle.
            ValueError: If unable to find any Kaggle dataset files for the provided user and dataset names.
            ValueError: If the file name provided doesn't match any of the files in the dataset.
        """
        try:
            data = kaggle.api.datasets_download_file(
                dataset_owner,
                dataset_name,
                file_name,
            )
            logger.info("Data successfully downloaded")
        except OSError:
            # Should normally never occur since empty environment variables are added in the __init__.py,
            # which should be sufficient for dataset downloads.
            raise ConnectionRefusedError(
                "Unable to authenticate with Kaggle. Ensure that you have a Kaggle API key saved "
                "to the appropriate file or your username and password in your environment variables. "
                "See: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials"
            )
        except ApiException:
            try:
                dataset_files = kaggle.api.datasets_list_files(
                    dataset_owner, dataset_name
                )
            except ApiException:
                raise ValueError(
                    "No Kaggle dataset files found using the provided username and dataset name."
                )

            if file_name not in [
                file["name"] for file in dataset_files["datasetFiles"]
            ]:
                raise ValueError(
                    f"Dataset: {dataset_owner}/{dataset_name} was successfully found but doesn't "
                    f"contain any file named: {file_name}"
                )

        return pd.read_csv(StringIO(data))

    def has_missing_values(self, raise_exception: bool = True) -> bool:
        """
        Checks for missing values: NaN, "", and "." in the DataFrame and logs them.
        If `raise_exception` is True, raises a ValueError if any are found.

        Returns:
            bool: True if there are missing values, False otherwise.
        """
        df = self.data

        # Convert from numpy bool_ type to be safe
        has_nan = bool(df.isnull().values.any())
        has_empty_strings = bool((df == "").values.any())
        has_dot_values = bool((df == ".").values.any())

        missing_values = has_nan or has_empty_strings or has_dot_values

        if missing_values:
            logger.warning(
                f"Missing values found in DataFrame: {has_nan=}, {has_empty_strings=}, {has_dot_values=}."
                f"\nDataFrame:\n{df.head(3)}"
            )
            if raise_exception:
                raise ValueError()
        return missing_values

    def drop_columns(self) -> pd.DataFrame:
        """
        Drops the specified columns from the DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns dropped.
        """
        if self.columns_to_drop:
            df = self.data.drop(self.columns_to_drop, axis=1)
            logger.info(
                f"Columns: {self.columns_to_drop} successfully dropped:\n{df.head(3)}"
            )
            self.data = df
        return self.data

    def onehot_encode_columns(self) -> pd.DataFrame:
        """
        One-hot encodes the specified columns and replaces them in the DataFrame. Uses `self.columns_to_onehot_encode`,
        originating from the dataset parameters config.

        Returns:
            pd.DataFrame: The stored DataFrame with the specified columns replaced with one-hot encoded columns.
        """
        if self.columns_to_onehot_encode:
            df = self.data
            encoder = OneHotEncoder(sparse_output=False)
            encoded_array = encoder.fit_transform(df[self.columns_to_onehot_encode])

            encoded_columns_df = pd.DataFrame(
                encoded_array,
                columns=encoder.get_feature_names_out(self.columns_to_onehot_encode),
            )

            df = df.drop(columns=self.columns_to_onehot_encode).join(encoded_columns_df)
            logger.info(
                f"Columns: {self.columns_to_onehot_encode} successfully one-hot encoded:\n{df.head(3)}"
            )
            self.data = df
        return self.data

    def save_dataframe(
        self,
        file_format: DataFileSuffix = "parquet",
        file_name_ending: str = "",
    ) -> Path:
        """
        Saves the data to a file in the specified format.

        Args:
            file_format (DataFileSuffix, optional): The format to use when saving the data. Defaults to "parquet".
            file_name_ending (str, optional): String to append to the end of the file name. Defaults to "".

        Returns:
            Path: The path to the saved file.
        """
        file_path = self.data_directory / f"{self.save_name}{file_name_ending}"

        try:
            if file_format == "parquet":
                file_path = file_path.with_suffix(".parquet")
                self.data.to_parquet(file_path, index=False, compression="gzip")
            elif file_format == "csv":
                file_path = file_path.with_suffix(".csv")
                self.data.to_csv(file_path, index=False)
            elif file_format == "pickle":
                file_path = file_path.with_suffix(".pkl")
                self.data.to_pickle(file_path)
            elif file_format == "json":
                file_path = file_path.with_suffix(".json")
                self.data.to_json(file_path, orient="records")
            logger.info(f"Data saved to: {file_path}")
        except FileNotFoundError:
            logger.exception(f"Could not save dataset to {file_path}.")

        return file_path

    def split_data(self, test_size: float = 0.2) -> SplitDataTuple:
        """
        Separates the target column from the features and splits both into training and testing sets
        using scikit-learn's `train_test_split` function.

        Args:
            target_column (str): The column(s) to be used as the target variable(s).
            test_size (float, optional): The proportion of the data to be used for testing. Defaults to 0.2.

        Returns:
            SplitDataTuple:
                X_train (pd.DataFrame): Training data for features.
                X_test (pd.DataFrame): Testing data for features.
                y_train (pd.DataFrame | pd.Series): Training data for target variable(s).
                y_test (pd.DataFrame | pd.Series): Testing data for target variable(s).
        """
        if self.target_column is None:
            raise ValueError("No target column provided within the dataset parameters.")

        X = self.data.drop(columns=self.target_column)
        y = self.data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=0,
        )
        logger.info(
            f"Data successfully split: {X_train.shape=}, {X_test.shape=}, {y_train.shape=}, {y_test.shape=}"
        )
        return X_train, X_test, y_train, y_test

    def split_and_save_data(self, test_size: float = 0.2) -> Path:
        """
        Splits the data and saves it to a single pickle file as a SplitData object.

        Args:
            test_size (float, optional): Proportion of the data to be used for testing. Defaults to 0.2.

        Returns:
            Path: The path to the saved SplitData object.
        """
        X_train, X_test, y_train, y_test = self.split_data(test_size=test_size)

        split_data_obj = SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        save_path = self.data_directory / f"{self.save_name}_split.pkl"
        with open(save_path, "wb") as file:
            pickle.dump(split_data_obj, file)
        logger.info(f"Split data saved to: {save_path}")
        return save_path


def process_datasets(
    datasets: list[DatasetType],
    data_directory: Path,
    save_datasets: Literal["original", "cleaned", "both", "none"] = "both",
) -> list[Path]:
    """
    Downloads and processes data from multiple datasets that have been validated.

    Args:
        datasets (list[KaggleDataset | LocalDataset]): A list of datasets to process.
        data_directory (Path): Directory to save the original and processed data.
        save_datasets (Literal["original", "cleaned", "both", "none"], optional): Which datasets to save.
        Original is the unprocessed data, cleaned is after processing such as encoding. Defaults to "both".

    Returns:
        list[Path]: List of paths to the saved split data files for input into models.
    """
    split_data_paths = []
    for dataset in datasets:
        processor = DatasetProcessor(dataset, data_directory)

        if save_datasets in ["original", "both"]:
            processor.save_dataframe()

        processor.has_missing_values()
        processor.drop_columns()
        processor.onehot_encode_columns()

        if save_datasets in ["cleaned", "both"]:
            processor.save_dataframe(file_name_ending="_cleaned")

        save_path = processor.split_and_save_data()
        split_data_paths.append(save_path)

    return split_data_paths
