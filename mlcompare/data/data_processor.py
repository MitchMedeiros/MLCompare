from __future__ import annotations as _annotations

import logging
import pickle
from io import StringIO
from pathlib import Path
from typing import Any

import kaggle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .split_data import SplitData, SplitDataTuple

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A class to download, load, process, and save data via Pandas.

    ### Initialization Parameters
    data: Accepts either a DataFrame, a Path object to a file containing data, or any valid input for a Pandas DataFrame.
    Can be omitted in favor of using the `download_kaggle_data` method or the dedicated `load_data` method.
    """

    def __init__(
        self,
        data: Any | pd.DataFrame | Path | None = None,
    ) -> None:
        if isinstance(data, Path):
            self.data = self.read_from_path(file_path=data)

        elif isinstance(data, pd.DataFrame):
            self.data = data

        elif data is None:
            self.data = pd.DataFrame()

        else:
            try:
                self.data = pd.DataFrame(data)

            except Exception as e:
                logger.exception(
                    f"Could not convert data with type: {type(data)} to DataFrame"
                )
                raise e

        assert isinstance(self.data, pd.DataFrame), "Data must be a Pandas DataFrame."

    def read_from_path(self, file_path: Path) -> pd.DataFrame:
        try:
            extension = file_path.suffix

            if extension == ".csv":
                df = pd.read_csv(file_path)

            elif extension == ".parquet":
                df = pd.read_parquet(file_path)

            elif extension == ".pkl":
                df = pd.read_pickle(file_path)

            elif extension == ".json":
                df = pd.read_json(file_path)

            else:
                raise ValueError("Data file must be a Parquet, CSV, PKL, or JSON file.")

            return df

        except FileNotFoundError as e:
            logger.exception(f"File not found: {file_path}")
            raise e

    def download_kaggle_data(
        self,
        dataset_owner: str,
        dataset_name: str,
        data_file_name: str,
    ) -> pd.DataFrame:
        """
        Downloads a Kaggle dataset. Overwrites any existing data for the class instance.
        Currently only supports CSV files.

        Args:
            dataset_owner (str): The user(s) under which the dataset is provided.
            dataset_name (str): The name(s) of the dataset.
            data_file_name (str): The file(s) to be downloaded from the dataset.

        Returns:
            pd.DataFrame: The downloaded data as a Pandas DataFrame.
        """
        try:
            data = kaggle.api.datasets_download_file(
                dataset_owner,
                dataset_name,
                data_file_name,
            )
            logger.info("Data successfully downloaded")

        except Exception as e:
            logger.exception("Exception when calling Kaggle Api")
            raise e

        data = StringIO(data)
        df = pd.read_csv(data)
        logger.info(f"String data converted to DataFrame: \n{df.head(3)}")

        self.data = df
        return df

    def has_missing_values(self, raise_exception: bool = True) -> bool:
        """
        Checks for missing values: NaN, "", and "." in the DataFrame and logs them.
        If `raise_exception` = True a ValueError will be raised if any are found.

        Returns:
            False - if no values are missing.
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
                f"\nDataFrame: \n{self.data.head(3)}"
            )
            if raise_exception:
                raise ValueError()

        return missing_values

    def drop_columns(self, columns_to_drop: list[str] | None) -> pd.DataFrame:
        """
        Drops the specified columns from the DataFrame.

        Args:
            columns_to_drop (list[str]): A list of column names to be dropped.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns dropped.
        """
        if columns_to_drop is not None:
            self.data = self.data.drop(columns_to_drop, axis=1)
            logger.info(f"Columns dropped: \n{self.data.head(3)}")

        return self.data

    def encode_columns(self, columns_to_encode: list[str] | None) -> pd.DataFrame:
        """
        Encodes the specified columns using one-hot encoding and returns the encoded DataFrame.

        Args:
            columns_to_encode (list[str]): A list of column names to be encoded.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns encoded using one-hot encoding.
        """
        if columns_to_encode is not None:
            df = self.data
            encoder = OneHotEncoder(sparse_output=False)
            encoded_array = encoder.fit_transform(df[columns_to_encode])

            # Convert the one-hot encoded ndarray to a DataFrame
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoder.get_feature_names_out(columns_to_encode),
            )

            # Drop the original columns and join the one-hot encoded columns
            df = df.drop(columns=columns_to_encode).join(encoded_df)
            logger.info(f"Data successfully encoded: \n{df.head(3)}")

            self.data = df

        return self.data

    def save_dataframe(self, file_path: Path, file_format: str = "pickle") -> None:
        """
        Save the data to a file.

        Args:
            file_path (Path): The path to save the data.

        Raises:
            ValueError: If no valid save path was provided.
        """
        try:
            if file_format == "pickle":
                self.data.to_pickle(file_path)  # type: ignore

            elif file_format == "csv":
                self.data.to_csv(file_path, index=False)

            elif file_format == "json":
                self.data.to_json(file_path, orient="records")

            elif file_format == "parquet":
                self.data.to_parquet(file_path, index=False, compression="gzip")

            logger.info(f"Data saved to: {file_path}")

        except FileNotFoundError:
            logger.exception(f"Could not save dataset to {file_path}.")

    def split_data(self, target_column: str, test_size: float = 0.2) -> SplitDataTuple:
        """
        Split the data into its features and target.

        Args:
            target_column (str): The column(s) to be used as the target variable(s).
            test_size (float, optional): The proportion of the data to be used for testing. Defaults to 0.2.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]: A tuple containing the training
            and testing data for features and target variables.
                X_train: Training data for features (pd.DataFrame)
                X_test: Testing data for features (pd.DataFrame)
                y_train: Training data for target variable(s) (pd.DataFrame | pd.Series)
                y_test: Testing data for target variable(s) (pd.DataFrame | pd.Series)
        """
        df = self.data
        X = df.drop([target_column], axis=1)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0
        )
        logger.info(
            f"Data successfully split: {X_train.shape=}, {X_test.shape=}, {y_train.shape=}, {y_test.shape=}"
        )

        return X_train, X_test, y_train, y_test

    def split_and_save_data(
        self,
        save_path: Path,
        target_column: str,
        test_size: float = 0.2,
    ) -> None:
        """
        Split the data and save it to a single pickle file as a SplitData object.

        Args:
            save_path (Path): The path to save the SplitData object to.
            target_column (str): The column(s) to be used as the target variable(s) or label(s).
            test_size (float, optional): The proportion of the data to be used for testing. Defaults to 0.2.

        Raises:
            ValueError: If no valid save path was provided.

        """
        X_train, X_test, y_train, y_test = self.split_data(
            target_column=target_column, test_size=test_size
        )

        split_data = SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        with open(save_path, "wb") as file:
            pickle.dump(split_data, file)


def split_and_save_data(
    data: Any | pd.DataFrame | Path,
    target_column: str,
    save_path: Path,
    test_size: float = 0.2,
) -> None:
    """
    A convenience function for DataProcessor(data).split_and_save_data().
    Splits the data and saves it to a single pickle file as a SplitData object.

    Args:
        data (Any | pd.DataFrame | Path): Data to be split. Can be the raw data, a DataFrame,
        or a Path object to a file containing data,
        target_column (str): Column(s) to be used as the target variable(s) or label(s).
        save_path (Path): Path to save the SplitData object to.
        test_size (float, optional): Proportion of the data to be used for testing. Defaults to 0.2.
    """
    processor = DataProcessor(data)
    processor.split_and_save_data(
        save_path=save_path,
        target_column=target_column,
        test_size=test_size,
    )
