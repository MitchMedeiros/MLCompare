import logging
import pickle
from io import StringIO
from pathlib import Path
from typing import Literal

import kaggle
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .dataset_validation import KaggleDataset, LocalDataset

logger = logging.getLogger(__name__)


class DatasetProcessor:
    def __init__(
        self,
        dataset: KaggleDataset | LocalDataset,
        data_directory: Path,
    ) -> None:
        if isinstance(data_directory, Path):
            self.data_directory = data_directory
        else:
            raise ValueError("Data directory must be a Path object.")

        if isinstance(dataset, LocalDataset):
            file_path = dataset.file_path
            self.data = self._read_from_path(file_path)

            if dataset.save_name is not None:
                self.save_name = dataset.save_name
            else:
                self.save_name = file_path.stem

        elif isinstance(dataset, KaggleDataset):
            self.data = self._download_kaggle_data(
                dataset.username,
                dataset.dataset_name,
                dataset.file_name,
            )

            self.save_name = f"{dataset.username}_{dataset.dataset_name}"

        else:
            raise ValueError("Data must be a KaggleDataset or LocalDataset object.")

        self.target_column = dataset.target_column
        self.columns_to_drop = dataset.columns_to_drop
        self.columns_to_encode = dataset.columns_to_encode

    def _read_from_path(self, file_path: Path) -> pd.DataFrame:
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

    def _download_kaggle_data(
        self,
        dataset_owner: str,
        dataset_name: str,
        data_file_name: str,
    ) -> pd.DataFrame:
        """
        Downloads a Kaggle dataset. Overwrites any existing data for the class instance.
        Currently only tested on CSV files.

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

    def drop_columns(self) -> pd.DataFrame:
        """
        Drops the specified columns from the DataFrame.

        Args:
            columns_to_drop (list[str]): A list of column names to be dropped.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns dropped.
        """
        if self.columns_to_drop is not None:
            df = self.data.drop(self.columns_to_drop, axis=1)
            logger.info(f"Columns dropped: \n{self.data.head(3)}")

            self.data = df

        return self.data

    def encode_columns(self) -> pd.DataFrame:
        """
        Encodes the specified columns using one-hot encoding and returns the encoded DataFrame.

        Args:
            columns_to_encode (list[str]): A list of column names to be encoded.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns encoded using one-hot encoding.
        """
        if self.columns_to_encode is not None:
            df = self.data
            encoder = OneHotEncoder(sparse_output=False)
            encoded_array = encoder.fit_transform(df[self.columns_to_encode])

            # Convert the one-hot encoded ndarray to a DataFrame
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoder.get_feature_names_out(self.columns_to_encode),
            )

            # Drop the original columns and join the one-hot encoded columns
            df = df.drop(columns=self.columns_to_encode).join(encoded_df)
            logger.info(f"Data successfully encoded: \n{df.head(3)}")

            self.data = df

        return self.data

    def save_dataframe(
        self,
        file_format: Literal["pickle", "csv", "json", "parquet"] = "pickle",
    ) -> None:
        """
        Save the data to a file.

        Args:
            file_path (Path): The path to save the data.

        Raises:
            ValueError: If no valid save path was provided.
        """
        try:
            if file_format == "pickle":
                file_path = self.data_directory / f"{self.save_name}.pkl"
                self.data.to_pickle(file_path)

            elif file_format == "csv":
                file_path = self.data_directory / f"{self.save_name}.csv"
                self.data.to_csv(file_path, index=False)

            elif file_format == "json":
                file_path = self.data_directory / f"{self.save_name}.json"
                self.data.to_json(file_path, orient="records")

            elif file_format == "parquet":
                file_path = self.data_directory / f"{self.save_name}.parquet"
                self.data.to_parquet(file_path, index=False, compression="gzip")

            logger.info(f"Data saved to: {file_path}")

        except FileNotFoundError:
            logger.exception(f"Could not save dataset to {file_path}.")

    def split_data(
        self,
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the data into its features and target.

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
        if self.target_column is not None:
            df = self.data
            X = df.drop([self.target_column], axis=1)
            y = df[self.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0
            )
            logger.info(
                f"Data successfully split: {X_train.shape=}, {X_test.shape=}, {y_train.shape=}, {y_test.shape=}"
            )

            return X_train, X_test, y_train, y_test

        else:
            raise ValueError("No target column provided within the dataset parameters.")

    def split_and_save_data(
        self,
        test_size: float = 0.2,
    ) -> None:
        """
        Split the data and save it to a single pickle file as a SplitData object.

        Args:
            save_path (Path): The path to save the SplitData object to.
            target_column (str): The column(s) to be used as the target variable(s) or label(s).
            test_size (float, optional): The proportion of the data to be used for testing. Defaults to 0.2.
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


class SplitData(BaseModel):
    """
    A class to validate and hold the split data from sklearn.model_selection.train_test_split.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame | pd.Series
    y_test: pd.DataFrame | pd.Series

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_split_data(
    load_path: Path,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame | pd.Series, pd.DataFrame | pd.Series
]:
    """
    Load a SplitData object from a pickle file and return the data it was holding.

    Args:
        load_path (Path): The path to a pickle file of a SplitData object.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]: A tuple containing the training
        and testing data for features and target variables.
            X_train: Training data for features (pd.DataFrame)
            X_test: Testing data for features (pd.DataFrame)
            y_train: Training data for target variable(s) (pd.DataFrame | pd.Series)
            y_test: Testing data for target variable(s) (pd.DataFrame | pd.Series)
    """
    with open(load_path, "rb") as file:
        split_data = pickle.load(file)

    assert isinstance(split_data, SplitData), "Loaded data must be of type SplitData."

    return (
        split_data.X_train,
        split_data.X_test,
        split_data.y_train,
        split_data.y_test,
    )
