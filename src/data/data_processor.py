import logging
import pickle
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import kaggle
import pandas as pd

# from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A class to download, load, process and save data mostly via Pandas.

    ### Initialization Parameters
    data: Accepts either a DataFrame, a Path to a file containing data, or a datatype excepted by pd.DataFrame.
    Can be omitted in favor of using the `download_kaggle_data` method or the dedicated `load_data` method.

    default_save_path: the default path to save the data to if none is specified in `save_data`.
    """

    def __init__(
        self,
        data: Any | pd.DataFrame | Path | None = None,
        default_save_path: Path | None = None,
    ) -> None:
        if isinstance(data, Path):
            try:
                extension = data.suffix

                if extension == ".csv":
                    df = pd.read_csv(data)

                elif extension == ".pkl":
                    df = pd.read_pickle(data)

                elif extension == ".json":
                    df = pd.read_json(data)

                else:
                    raise ValueError("Data file must be a CSV, PKL, or JSON file.")

            except FileNotFoundError:
                logger.exception(f"File not found: {data}")

        elif isinstance(data, pd.DataFrame):
            df = data

        elif data is None:
            df = pd.DataFrame()

        else:
            try:
                df = pd.DataFrame(data)
            except Exception as e:
                logger.exception(
                    f"Could not convert data with type: {type(data)} to DataFrame: {e}"
                )
                raise

        self.data = df
        self.default_save_path = default_save_path

    def download_kaggle_data(
        self,
        dataset_owner: str | Sequence[str],
        dataset_name: str | Sequence[str],
        data_file_name: str | Sequence[str],
    ) -> pd.DataFrame:
        """
        Downloads a Kaggle dataset. Overwrites any existing data for the class instance.
        Currently only supports CSV files.

        Args:
            dataset_owner (str or Sequence[str]): The user(s) under which the dataset is provided.
            dataset_name (str or Sequence[str]): The name(s) of the dataset.
            data_file_name (str or Sequence[str]): The file(s) to be downloaded from the dataset.

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
            logger.exception(f"Exception when calling Kaggle Api: {e}\n")
            raise

        data = StringIO(data)
        df = pd.read_csv(data)
        logger.info(f"String data converted to DataFrame: \n {df.head(3)}")

        self.data = df
        return df

    def drop_columns(self, columns_to_drop: list[str] | Sequence[str]) -> pd.DataFrame:
        """
        Drops the specified columns from the DataFrame.

        Args:
            columns_to_drop (list[str] | Sequence[str]): A list of column names to be dropped.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns dropped.
        """
        self.data = self.data.drop(columns_to_drop, axis=1)
        logger.info(f"Columns dropped: \n {self.data.head(3)}")

        return self.data

    def encode_columns(
        self,
        columns_to_encode: list[str] | Sequence[str],
    ) -> pd.DataFrame:
        """
        Encodes the specified columns using one-hot encoding and returns the encoded DataFrame.

        Args:
            columns_to_encode (list[str] | Sequence[str]): A list of column names to be encoded.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns encoded using one-hot encoding.
        """
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
        logger.info(f"Data successfully encoded: \n {df.head(3)}")

        self.data = df
        return df

    def split_data(
        self,
        target_column: str | Sequence[str],
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the data into its features and target.

        Args:
            target_column (str | Sequence[str]): The column(s) to be used as the target variable(s).

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training
            and testing data for features and target variables.
                X_train: Training data for features (pd.DataFrame)
                X_test: Testing data for features (pd.DataFrame)
                y_train: Training data for target variable(s) (pd.Series)
                y_test: Testing data for target variable(s) (pd.Series)
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

    def save_data(self, file_path: str | Path | None = None):
        """
        Save the data to a file.

        Args:
            file_path (str | Path | None, optional): The path to save the data. If None, the default save path will be used. Defaults to None.

        Raises:
            ValueError: If no valid save path was provided.
        """
        if self.default_save_path is not None and file_path is None:
            file_path = self.default_save_path

        elif self.default_save_path is None and file_path is None:
            raise ValueError("No valid save path was provided.")

        try:
            with open(file_path, "wb") as file:  # type: ignore
                pickle.dump(self.data, file)
            logger.info(f"Data saved to: {file_path}")

        except FileNotFoundError:
            logger.exception(f"Could not save dataset to {file_path}.")
