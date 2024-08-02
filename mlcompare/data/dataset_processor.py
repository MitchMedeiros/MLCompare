from __future__ import annotations as _annotations

import logging
import pickle
from pathlib import Path
from typing import Generator, Literal

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ..params_reader import ParamsInput
from .datasets import (
    DatasetFactory,
    DatasetType,
    HuggingFaceDataset,
    KaggleDataset,
    LocalDataset,
    OpenMLDataset,
)
from .split_data import SplitData, SplitDataTuple

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Processes validated datasets to prepare them for model training and evaluation.

    Attributes:
    -----------
        dataset (DatasetType): DatasetType object containing a `get_data()` method and attributes needed for data processing.
    """

    def __init__(self, dataset: DatasetType) -> None:
        if not isinstance(
            dataset, (KaggleDataset, LocalDataset, HuggingFaceDataset, OpenMLDataset)
        ):
            raise ValueError("Data must be a KaggleDataset or LocalDataset object.")

        self.data = dataset.get_data()
        self.target = dataset.target
        self.save_name = dataset.save_name
        self.drop = dataset.drop
        self.onehot_encode = dataset.onehot_encode

    def has_missing_values(
        self, drop_rows: bool = False, raise_exception: bool = True
    ) -> bool:
        """
        Checks for missing values: NaN, "", and "." in the DataFrame and either logs them, raises an
        exception, or drops the rows with missing values,

        Args:
        -----
            drop_rows (bool, optional): Whether to drop rows with missing values. Defaults to False.
            raise_exception (bool, optional): Whether to raise an exception if missing values are found.
            Defaults to True. Ignored if `drop_rows` is True.

        Returns:
        --------
            bool: True if there are missing values, False otherwise.

        Raises:
        -------
            ValueError: If missing values are found and `raise_exception` is True.
        """
        if not isinstance(drop_rows, bool):
            raise ValueError("`drop_rows` must be a boolean.")
        if not isinstance(raise_exception, bool):
            raise ValueError("`raise_exception` must be a boolean.")

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
            if drop_rows:
                df = df.dropna()
                logger.info(
                    f"Rows with missing values dropped. \nNew DataFrame length: {len(df)}"
                )
                self.data = df
            elif raise_exception:
                raise ValueError(
                    "Missing values found in DataFrame. Set `drop_rows=True` to drop them or `raise_exception=False` to continue processing."
                )
        return missing_values

    def drop_columns(self) -> pd.DataFrame:
        """
        Drops the specified columns from the DataFrame.

        Returns:
        --------
            pd.DataFrame: DataFrame with the specified columns dropped.
        """
        if self.drop:
            df = self.data.drop(self.drop, axis=1)
            logger.info(f"Columns: {self.drop} successfully dropped:\n{df.head(3)}")
            self.data = df
        return self.data

    def onehot_encode_columns(self) -> pd.DataFrame:
        """
        One-hot encodes the specified columns and replaces them in the DataFrame.

        Returns:
        --------
            pd.DataFrame: DataFrame with the specified columns replaced with one-hot encoded columns.
        """
        if self.onehot_encode:
            df = self.data
            encoder = OneHotEncoder(sparse_output=False)
            encoded_array = encoder.fit_transform(df[self.onehot_encode])

            encoded_columns_df = pd.DataFrame(
                encoded_array,
                columns=encoder.get_feature_names_out(self.onehot_encode),
            )

            df = df.drop(columns=self.onehot_encode).join(encoded_columns_df)
            logger.info(
                f"Columns: {self.onehot_encode} successfully one-hot encoded:\n{df.head(3)}"
            )
            self.data = df
        return self.data

    def save_dataframe(
        self,
        save_directory: Path | str,
        file_format: Literal["parquet", "csv", "json", "pickle"] = "parquet",
        file_name_ending: str = "",
    ) -> Path:
        """
        Saves the data to a file in the specified format.

        Args:
        -----
            save_directory (Path | str): Directory to save the data to.
            file_format (Literal["parquet", "csv", "json", "pickle"], optional): Format to use when
            saving the data. Defaults to "parquet".
            file_name_ending (str, optional): String to append to the end of the file name. Defaults to "".

        Returns:
        --------
            Path: Path to the saved data.
        """
        if not isinstance(file_format, str):
            raise ValueError("`file_format` must be a string.")
        if not isinstance(file_name_ending, str):
            raise ValueError("`file_name_ending` must be a string.")

        save_directory = validate_save_directory(save_directory)

        file_path = save_directory / f"{self.save_name}{file_name_ending}"

        try:
            match file_format:
                case "parquet":
                    file_path = file_path.with_suffix(".parquet")
                    self.data.to_parquet(file_path, index=False, compression="gzip")
                case "csv":
                    file_path = file_path.with_suffix(".csv")
                    self.data.to_csv(file_path, index=False)
                case "pickle":
                    file_path = file_path.with_suffix(".pkl")
                    self.data.to_pickle(file_path)
                case "json":
                    file_path = file_path.with_suffix(".json")
                    self.data.to_json(file_path, orient="records")
                case _:
                    raise ValueError(
                        "Invalid `file_format` provided. Must be one of: 'parquet', 'csv', 'json', 'pickle'."
                    )
            logger.info(f"Data saved to: {file_path}")
        except FileNotFoundError:
            logger.exception(f"Could not save dataset to {file_path}.")

        return file_path

    def split_data(self, test_size: float = 0.2) -> SplitDataTuple:
        """
        Separates the target column from the features and splits both into training and testing sets
        using scikit-learn's `train_test_split` function.

        Args:
        -----
            test_size (float, optional): Proportion of the data to be used for testing. Defaults to 0.2.

        Returns:
        --------
            SplitDataTuple:
                X_train (pd.DataFrame): Training data for features.
                X_test (pd.DataFrame): Testing data for features.
                y_train (pd.DataFrame | pd.Series): Training data for target variable(s).
                y_test (pd.DataFrame | pd.Series): Testing data for target variable(s).
        """
        if not isinstance(test_size, float):
            raise ValueError("`test_size` must be a float.")
        if test_size <= 0 or test_size >= 1:
            raise ValueError("`test_size` must be between 0 and 1.")

        X = self.data.drop(columns=self.target)
        y = self.data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0
        )
        logger.info(
            f"Data successfully split: {X_train.shape=}, {X_test.shape=}, {y_train.shape=}, {y_test.shape=}"
        )
        return X_train, X_test, y_train, y_test

    def split_and_save_data(
        self, save_directory: Path | str, test_size: float = 0.2
    ) -> Path:
        """
        Splits the data and saves it to a single pickle file as a SplitData object.

        Args:
        -----
            save_directory (Path | str): Directory to save the SplitData object to.
            test_size (float, optional): Proportion of the data to be used for testing. Defaults to 0.2.

        Returns:
        --------
            Path: Path to the saved SplitData object.
        """
        save_directory = validate_save_directory(save_directory)

        file_path = save_directory / f"{self.save_name}_split.pkl"

        X_train, X_test, y_train, y_test = self.split_data(test_size=test_size)

        split_data_obj = SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        with open(file_path, "wb") as file:
            pickle.dump(split_data_obj, file)
        logger.info(f"Split data saved to: {file_path}")
        return file_path

    def process_dataset(
        self,
        save_directory: Path | str,
        save_original: bool = True,
        save_processed: bool = True,
    ) -> SplitDataTuple:
        """
        Performs all data processing steps based on the parameters provided to `DatasetProcessor`.
        Optionally saves the original and processed data to files.

        Args:
        -----
            save_directory (Path | str): The directory to save the data to.
            save_original (bool): Whether to save the original data.
            save_processed (bool): Whether to save the processed, nonsplit data.

        Returns:
        --------
            SplitDataTuple:
                X_train (pd.DataFrame): Training data for features.
                X_test (pd.DataFrame): Testing data for features.
                y_train (pd.DataFrame | pd.Series): Training data for target variable(s).
                y_test (pd.DataFrame | pd.Series): Testing data for target variable(s).
        """
        if not isinstance(save_original, bool):
            raise ValueError("`save_original` must be a boolean.")
        if not isinstance(save_processed, bool):
            raise ValueError("`save_processed` must be a boolean.")

        if save_original:
            self.save_dataframe(save_directory=save_directory)

        self.has_missing_values()
        self.drop_columns()
        self.onehot_encode_columns()

        if save_processed:
            self.save_dataframe(
                save_directory=save_directory, file_name_ending="_processed"
            )

        return self.split_data()


def process_datasets(
    params_list: ParamsInput,
    save_directory: Path | str,
    save_original: bool = True,
    save_processed: bool = True,
) -> Generator[SplitDataTuple, None, None]:
    """
    Downloads and processes data from multiple datasets that have been validated.

    Args:
    -----
        params_list (ParamsInput): A list of dictionaries containing dataset parameters.
        save_directory (Path): Directory to save the data to.
        save_original (bool): Whether to save the original data.
        save_processed (bool): Whether to save the processed, nonsplit data.

    Returns:
    --------
        A Generator containing the split data for input into subsequent pipeline steps via iteration.
    """
    datasets = DatasetFactory(params_list)
    for dataset in datasets:
        try:
            processor = DatasetProcessor(dataset)
            split_data = processor.process_dataset(
                save_directory,
                save_original,
                save_processed,
            )
            yield split_data
        except Exception:
            logger.error("Failed to process dataset.")
            raise


def process_datasets_to_files(
    params_list: ParamsInput,
    save_directory: Path | str,
    save_original: bool = True,
    save_processed: bool = True,
) -> list[Path]:
    """
    Downloads and processes data from multiple datasets that have been validated.

    Args:
    -----
        datasets (list[KaggleDataset | LocalDataset]): A list of datasets to process.
        data_directory (Path): Directory to save the original and processed data.
        save_original (bool): Whether to save the original data.
        save_processed (bool): Whether to save the processed, nonsplit data.

    Returns:
    --------
        list[Path]: List of paths to the saved split data for input into subsequent pipeline steps.
    """
    save_directory = validate_save_directory(save_directory)

    split_data_paths = []
    datasets = DatasetFactory(params_list)
    for dataset in datasets:
        try:
            processor = DatasetProcessor(dataset)
            X_train, X_test, y_train, y_test = processor.process_dataset(
                save_directory,
                save_original,
                save_processed,
            )

            file_path = save_directory / f"{processor.save_name}_split.pkl"
            split_data_paths.append(file_path)
        except Exception:
            logger.error("Failed to process dataset.")
            raise

        split_data_obj = SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        with open(file_path, "wb") as file:
            pickle.dump(split_data_obj, file)
        logger.info(f"Split data saved to: {file_path}")

    return split_data_paths


def validate_save_directory(save_directory: Path | str) -> Path:
    """
    Validates the save directory and creates it if it does not exist.

    Args:
    -----
        save_directory (Path | str): Directory to save files to.

    Returns:
    --------
        Path: Path to the directory.
    """
    if not isinstance(save_directory, (Path)):
        if not isinstance(save_directory, str):
            raise ValueError("`save_directory` must be a string or Path object.")
        else:
            save_directory = Path(save_directory)

    save_directory.mkdir(exist_ok=True)
    return save_directory
