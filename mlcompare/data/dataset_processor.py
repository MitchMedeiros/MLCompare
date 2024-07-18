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
        dataset (DatasetType): A Dataset type object containing a `get_data()` method and attributes needed for data processing.
        data_directory (Path): Directory to save files to for the `save_dataframe` and `split_and_save_data` methods.
    """

    def __init__(self, dataset: DatasetType, data_directory: Path) -> None:
        if not isinstance(
            dataset, (KaggleDataset, LocalDataset, HuggingFaceDataset, OpenMLDataset)
        ):
            raise ValueError("Data must be a KaggleDataset or LocalDataset object.")

        if not isinstance(data_directory, Path):
            raise ValueError("Data directory must be a Path object.")

        self.data = dataset.get_data()
        self.target = dataset.target
        self.save_name = dataset.save_name
        self.drop = dataset.drop
        self.onehot_encode = dataset.onehot_encode

        self.data_directory = data_directory

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
            pd.DataFrame: The DataFrame with the specified columns dropped.
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
            pd.DataFrame: The stored DataFrame with the specified columns replaced with one-hot encoded columns.
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
        file_format: Literal["parquet", "csv", "json", "pickle"] = "parquet",
        file_name_ending: str = "",
    ) -> Path:
        """
        Saves the data to a file in the specified format.

        Args:
        -----
            file_format (Literal["parquet", "csv", "json", "pickle"], optional): The format to use when
            saving the data. Defaults to "parquet".
            file_name_ending (str, optional): String to append to the end of the file name. Defaults to "".

        Returns:
        --------
            Path: The path to the saved file.
        """
        file_path = self.data_directory / f"{self.save_name}{file_name_ending}"

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
                    raise ValueError("Invalid file format provided.")
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
            target (str): The column(s) to be used as the target variable(s).
            test_size (float, optional): The proportion of the data to be used for testing. Defaults to 0.2.

        Returns:
        --------
            SplitDataTuple:
                X_train (pd.DataFrame): Training data for features.
                X_test (pd.DataFrame): Testing data for features.
                y_train (pd.DataFrame | pd.Series): Training data for target variable(s).
                y_test (pd.DataFrame | pd.Series): Testing data for target variable(s).
        """
        if self.target is None:
            raise ValueError("No target column provided within the dataset parameters.")

        X = self.data.drop(columns=self.target)
        y = self.data[self.target]

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
        -----
            test_size (float, optional): Proportion of the data to be used for testing. Defaults to 0.2.

        Returns:
        --------
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

    def process_and_save_dataset(
        self, save_original: bool, save_cleaned: bool
    ) -> SplitDataTuple:
        if save_original:
            self.save_dataframe()

        self.has_missing_values()
        self.drop_columns()
        self.onehot_encode_columns()

        if save_cleaned:
            self.save_dataframe(file_name_ending="_cleaned")

        return self.split_data()


def process_datasets(
    params_list: ParamsInput,
    data_directory: Path,
    save_original: bool = True,
    save_cleaned: bool = True,
) -> Generator[SplitDataTuple, None, None]:
    """
    Downloads and processes data from multiple datasets that have been validated.

    Args:
    -----
        params_list (ParamsInput): A list of dictionaries containing dataset parameters.
        data_directory (Path): Directory to save the original, processed, and split data to.
        save_original (bool): Whether to save the original data.
        save_cleaned (bool): Whether to save the processed, nonsplit data.

    Returns:
    --------
        A generator containing the split data for input into subsequent pipeline steps via iteration.
    """
    datasets = DatasetFactory(params_list)
    for dataset in datasets:
        try:
            processor = DatasetProcessor(dataset, data_directory)
            split_data = processor.process_and_save_dataset(save_original, save_cleaned)
            yield split_data
        except Exception:
            logger.error("Failed to process dataset.")
            raise


def process_datasets_to_files(
    params_list: ParamsInput,
    data_directory: Path,
    save_original: bool = True,
    save_cleaned: bool = True,
) -> list[Path]:
    """
    Downloads and processes data from multiple datasets that have been validated.

    Args:
    -----
        datasets (list[KaggleDataset | LocalDataset]): A list of datasets to process.
        data_directory (Path): Directory to save the original and processed data.
        save_original (bool): Whether to save the original data.
        save_cleaned (bool): Whether to save the processed, nonsplit data.

    Returns:
    --------
        list[Path]: List of paths to the saved split data for input into subsequent pipeline steps.
    """
    split_data_paths = []
    datasets = DatasetFactory(params_list)
    for dataset in datasets:
        processor = DatasetProcessor(dataset, data_directory)

        if save_original:
            processor.save_dataframe()

        processor.has_missing_values()
        processor.drop_columns()
        processor.onehot_encode_columns()

        if save_cleaned:
            processor.save_dataframe(file_name_ending="_cleaned")

        save_path = processor.split_and_save_data()
        split_data_paths.append(save_path)

    return split_data_paths
