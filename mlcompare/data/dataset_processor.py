from __future__ import annotations as _annotations

import logging
import pickle
from pathlib import Path
from typing import Generator, Literal

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

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

sklearn.set_config(transform_output="pandas")


def validate_save_directory(save_directory: Path | str) -> Path:
    """
    Validates the existence of a directory and creates one if it doesn't exist.

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
        self.nan = dataset.nan
        self.onehot_encode = dataset.onehot_encode
        self.label_encode = dataset.label_encode
        self.ordinal_encode = dataset.ordinal_encode

        self.train_test_split()

    def train_test_split(self, test_size: float = 0.2) -> None:
        """
        Splits the data into training and testing sets. A wrapper around scikit-learn's `train_test_split` function.
        """
        if not isinstance(test_size, float):
            raise ValueError("`test_size` must be a float.")
        if test_size <= 0 or test_size >= 1:
            raise ValueError("`test_size` must be between 0 and 1.")

        try:
            X, y = train_test_split(self.data, test_size=test_size, random_state=0)

            logger.info(f"Data successfully split: {X.shape=}, {y.shape=}")
            self.train_data = X
            self.test_data = y
        except ValueError:
            logger.error(
                "Could not split the dataset into train and set sets since it is empty."
            )
            raise

    def handle_nan(self, raise_exception: bool = False) -> pd.DataFrame:
        """
        Checks for missing values: NaN, "", and "." in the DataFrame and either forward-fills, backwards-fills, drops them,
        or simply logs how many exist. Raises an exception instead is `raise_exception`=True.

        Args:
        -----
            raise_exception (bool, optional): Whether to raise an exception if missing values are found. Defaults to False.

        Returns:
        --------
            pd.DataFrame: DataFrame with the missing values either forward-filled, backward-filled,
            or dropped or neither if a method is provided for the dataset.

        Raises:
        -------
            ValueError: If missing values are found and `raise_exception` is True.
        """
        if not isinstance(raise_exception, bool):
            raise ValueError("`raise_exception` must be a boolean.")

        if self.nan:
            df = self.data

            # Convert from numpy bool_ type to be safe
            has_nan = bool(df.isna().values.any())
            has_empty_strings = bool((df == "").values.any())
            has_dot_values = bool((df == ".").values.any())
            missing_values = has_nan or has_empty_strings or has_dot_values

            if missing_values:
                logger.warning(
                    f"Missing values found in DataFrame: {has_nan=}, {has_empty_strings=}, {has_dot_values=}."
                    f"\nDataFrame:\n{df.head(3)}"
                )
                if raise_exception:
                    raise ValueError(
                        "Missing values found in DataFrame. Set `raise_exception=False` for `DatasetProcessor.handle_nan()` "
                        "to continue processing anyways."
                    )
                else:
                    df = df.replace(".", None)
                    df = df.replace("", None)

                    match self.nan:
                        case "ffill":
                            df = df.ffill()
                        case "bfill":
                            df = df.bfill()
                        case "drop":
                            df = df.dropna()
                        case _:
                            raise ValueError(
                                "Unexpected value for `nan` given. Allowed values are 'ffill', 'bfill', and 'drop'."
                            )

                    assert (
                        bool(df.isna().values.any()) is False
                    ), "handle_nan failed to remove all NaN values."
                    logger.info(
                        f"Rows with missing values dropped. \nNew DataFrame length: {len(df)}"
                    )
                    self.data = df

        return self.data

    def drop_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Drops the specified columns from the DataFrame.

        Returns:
        --------
            pd.DataFrame: DataFrame with the specified columns dropped.
        """
        if self.drop:
            self.train_data = self.train_data.drop(self.drop, axis=1)
            self.test_data = self.test_data.drop(self.drop, axis=1)

            logger.info(
                f"Columns: {self.drop} successfully dropped. Training split:\n{self.train_data.head(3)}"
            )

        return self.train_data, self.test_data

    def onehot_encode_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        One-hot encodes the specified columns and replaces them in the DataFrame.

        Returns:
        --------
            pd.DataFrame: DataFrame with the specified columns replaced with one-hot encoded columns.
        """
        if self.onehot_encode:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_train_columns = encoder.fit_transform(
                self.train_data[self.onehot_encode]
            )
            encoded_test_columns = encoder.transform(self.test_data[self.onehot_encode])

            self.train_data = self.train_data.drop(self.onehot_encode, axis=1).join(
                encoded_train_columns
            )
            self.test_data = self.test_data.drop(self.onehot_encode, axis=1).join(
                encoded_test_columns
            )

            logger.info(
                f"Columns: {self.onehot_encode} successfully one-hot encoded. Training split:\n{self.train_data.head(3)}"
            )

        return self.train_data, self.test_data

    def ordinal_encode_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ordinal encodes the specified columns and replaces them in the DataFrame.

        Returns:
        --------
            pd.DataFrame: DataFrame with the specified columns replaced with ordinal encoded columns.
        """
        if self.ordinal_encode:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            self.train_data[self.ordinal_encode] = encoder.fit_transform(
                self.train_data[self.ordinal_encode]
            )
            self.test_data[self.ordinal_encode] = encoder.transform(
                self.test_data[self.ordinal_encode]
            )

            logger.info(
                f"Columns: {self.ordinal_encode} successfully ordinal encoded. Training split:\n{self.train_data.head(3)}"
            )

        return self.train_data, self.test_data

    def label_encode_column(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Label encodes the specified columns and replaces them in the DataFrame.

        Returns:
        --------
            pd.DataFrame: DataFrame with the specified columns replaced with label encoded columns.
        """
        if self.label_encode:
            train_df = self.train_data.copy()
            test_df = self.test_data.copy()

            try:
                encoder = LabelEncoder()
                train_df[self.target] = encoder.fit_transform(train_df[self.target])
                test_df[self.target] = encoder.transform(test_df[self.target])
            except ValueError:
                logger.warning(
                    "Labels are present in the generated test split that are not present in the training split and, therefore, cannot be fit. \n"
                    "To resolve this, the label encoder will be fit on the entire dataset. This introduces data-leakage and may negatively impact the reliability of the results. \n"
                    "Consider using a larger dataset to address this."
                )
                combined_df = pd.concat([train_df, test_df])

                encoder.fit(combined_df[self.target])
                train_df[self.target] = encoder.transform(train_df[self.target])
                test_df[self.target] = encoder.transform(test_df[self.target])

            logger.info(
                f"Target: {self.target} successfully label encoded. Training split:\n{train_df.head(3)}"
            )
            self.train_data = train_df
            self.test_data = test_df
        return self.train_data, self.test_data

    def save_dataframe(
        self,
        save_directory: Path | str,
        file_format: Literal["parquet", "csv", "json", "pkl"] = "parquet",
        file_name_ending: str = "",
    ) -> Path:
        """
        Saves the data to a file in the specified format.

        Args:
        -----
            save_directory (Path | str): Directory to save the data to.
            file_format (Literal["parquet", "csv", "json", "pkl"], optional): Format to use when
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
        file_path = save_directory / f"{self.save_name}{file_name_ending}.{file_format}"

        file_count = 1
        while file_path.exists():
            file_path = (
                save_directory
                / f"{self.save_name}{file_name_ending}-{file_count}.{file_format}"
            )
            file_count += 1

        try:
            match file_format:
                case "parquet":
                    self.data.to_parquet(file_path, index=False, compression="gzip")
                case "csv":
                    self.data.to_csv(file_path, index=False)
                case "pkl":
                    self.data.to_pickle(file_path)
                case "json":
                    self.data.to_json(file_path, orient="records")
                case _:
                    raise ValueError(
                        "Invalid `file_format` provided. Must be one of: 'parquet', 'csv', 'json', 'pkl'."
                    )
            logger.info(f"Data saved to: {file_path}")
        except FileNotFoundError:
            logger.exception(f"Could not save dataset to {file_path}.")

        return file_path

    def split_target(self) -> SplitDataTuple:
        """
        Separates the target column from the features and splits both into training and testing sets
        using scikit-learn's `train_test_split` function.

        Args:
        -----
            test_size (float, optional): Proportion of the data to be used for testing. Defaults to 0.2.

        Returns:
        --------
            SplitDataTuple:
                pd.DataFrame: Training data for features.
                pd.DataFrame: Testing data for features.
                pd.DataFrame | pd.Series: Training data for target variable.
                pd.DataFrame | pd.Series: Testing data for target variable.
        """
        X_train = self.train_data
        y_train = X_train.pop(self.target)

        X_test = self.test_data
        y_test = X_test.pop(self.target)

        logger.info(
            f"Target successfully split from training and testing data: {X_train.shape=}, {X_test.shape=}, "
            f"{y_train.shape=}, {y_test.shape=}"
        )
        return X_train, X_test, y_train, y_test

    def split_and_save_data(self, save_directory: Path | str) -> Path:
        """
        Splits the data and saves it to a single pickle file as a SplitData object.

        Args:
        -----
            save_directory (Path | str): Directory to save the SplitData object to.

        Returns:
        --------
            Path: Path to the saved SplitData object.
        """
        X_train, X_test, y_train, y_test = self.split_target()
        split_data_obj = SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        save_directory = validate_save_directory(save_directory)
        file_path = save_directory / f"{self.save_name}-split.pkl"

        file_count = 1
        while file_path.exists():
            file_path = save_directory / f"{self.save_name}-split-{file_count}.pkl"
            file_count += 1

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
                pd.DataFrame: Training data for features.
                pd.DataFrame: Testing data for features.
                pd.DataFrame | pd.Series: Training data for target variable.
                pd.DataFrame | pd.Series: Testing data for target variable.
        """
        if not isinstance(save_original, bool):
            raise ValueError("`save_original` must be a boolean.")
        if not isinstance(save_processed, bool):
            raise ValueError("`save_processed` must be a boolean.")

        if save_original:
            self.save_dataframe(
                save_directory=save_directory, file_name_ending="-original"
            )

        self.drop_columns()
        self.handle_nan()
        self.onehot_encode_columns()
        self.ordinal_encode_columns()
        self.label_encode_column()

        if save_processed:
            self.save_dataframe(
                save_directory=save_directory, file_name_ending="-processed"
            )

        return self.split_target()


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

            file_path = save_directory / f"{processor.save_name}-split.pkl"
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
