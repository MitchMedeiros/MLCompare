from __future__ import annotations as _annotations

import logging
import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from ..results_writer import ResultsWriter
from .datasets import (
    DatasetType,
    HuggingFaceDataset,
    KaggleDataset,
    LocalDataset,
    OpenMLDataset,
)
from .split_data import SplitData, SplitDataTuple

logger = logging.getLogger(__name__)

sklearn.set_config(transform_output="pandas")


class DatasetProcessor:
    """
    Processes validated datasets to prepare them for model training and evaluation.

    Attributes:
    -----------
        dataset (DatasetType): DatasetType object containing a `get_data()` method and attributes needed for
        data processing.
    """

    def __init__(self, dataset: DatasetType) -> None:
        if not isinstance(dataset, (KaggleDataset, LocalDataset, HuggingFaceDataset, OpenMLDataset)):
            raise TypeError(
                "`dataset` must be an instance of a KaggleDataset, LocalDataset, HuggingFaceDataset, or OpenMLDataset."
            )

        self.target = dataset.target
        self.save_name = dataset.save_name
        self.drop = dataset.drop
        self.nan = dataset.nan
        self.one_hot_encode = dataset.one_hot_encode
        self.ordinal_encode = dataset.ordinal_encode
        self.target_encode = dataset.target_encode
        self.label_encode = dataset.label_encode
        self.standard_scale = dataset.standard_scale
        self.min_max_scale = dataset.min_max_scale
        self.max_abs_scale = dataset.max_abs_scale
        self.robust_scale = dataset.robust_scale
        self.power_transform = dataset.power_transform
        self.quantile_transform = dataset.quantile_transform
        self.quantile_transform_normal = dataset.quantile_transform_normal
        self.normalize = dataset.normalize

        data = dataset.get_data()
        self._train_test_split_data(data)

    def _train_test_split_data(self, data: pd.DataFrame, test_size: float = 0.2) -> None:
        """Splits the data into train and test sets and saves them as attributes for future processing."""
        if not isinstance(test_size, float):
            raise TypeError("`test_size` must be a float.")
        if test_size <= 0 or test_size >= 1:
            raise TypeError("`test_size` must be between 0 and 1.")

        try:
            X, y = train_test_split(data, test_size=test_size, random_state=42)

            logger.info(f"Data successfully split: {X.shape=}, {y.shape=}. Training split:\n{X.head(3)}")
            self.train_data = X
            self.test_data = y
        except ValueError:
            logger.error(
                "Could not split the provided data into train and test sets since it contains 1 "
                "or fewer data points."
            )
            raise

    def handle_nan(self, raise_exception: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handles missing values in the data including: np.nan, None, "", and "." by either forward-filling
        (ffill), backward-filling (bfill), or dropping (drop) them based on the `nan` parameter.

        Args:
        -----
            raise_exception (bool, optional): Whether to raise an exception if missing values are found.
            Defaults to False.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the missing values in the
            specified columns either forward-filled, backward-filled, or dropped or neither if a
            method is provided for the dataset.

        Raises:
        -------
            ValueError: If missing values are found and `raise_exception` is True.
        """
        if not isinstance(raise_exception, bool):
            raise TypeError("`raise_exception` must be a boolean.")

        if self.nan:

            def handle_nan_for_split(df: pd.DataFrame) -> pd.DataFrame:
                # Convert from numpy bool_ type to be safe
                has_nan = bool(df.isna().values.any())
                has_empty_strings = bool((df == "").values.any())
                has_dot_values = bool((df == ".").values.any())
                missing_values = has_nan or has_empty_strings or has_dot_values

                if missing_values:
                    logger.info(
                        f"Missing values found in data: {has_nan=}, {has_empty_strings=}, {has_dot_values=}."
                        f"\nDataFrame:\n{df.head(3)}"
                    )
                    if raise_exception:
                        raise ValueError(
                            "Missing values found in data. Set `raise_exception=False` to continue processing."
                        )
                    else:
                        df = df.replace({"": None, ".": None})

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

                        if df.isna().any().any():
                            logger.warning("Not all NaN values were removed.")
                        else:
                            logger.info(f"Missing values handled. \nNew DataFrame length: {len(df)}")

                return df

            self.train_data = handle_nan_for_split(self.train_data)
            self.test_data = handle_nan_for_split(self.test_data)

        return self.train_data, self.test_data

    def drop_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Drops the columns specified with the `drop` parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the columns specified by the
            `drop` parameter dropped.
        """
        if self.drop:
            self.train_data = self.train_data.drop(self.drop, axis=1)
            self.test_data = self.test_data.drop(self.drop, axis=1)

            logger.info(
                f"Columns: {self.drop} successfully dropped. Training split:\n{self.train_data.head(3)}"
            )

        return self.train_data, self.test_data

    def one_hot_encode_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.OneHotEncoder` to the columns specified by the `onehotEncode`
        parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the specified columns encoded.
        """
        if self.one_hot_encode:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", max_categories=25)
            encoded_train_columns = encoder.fit_transform(self.train_data[self.one_hot_encode])
            encoded_test_columns = encoder.transform(self.test_data[self.one_hot_encode])

            self.train_data = self.train_data.drop(self.one_hot_encode, axis=1).join(
                encoded_train_columns
            )
            self.test_data = self.test_data.drop(self.one_hot_encode, axis=1).join(encoded_test_columns)

            logger.info(
                f"Columns: {self.one_hot_encode} successfully one-hot encoded. Training split:\n{self.train_data.head(3)}"
            )

        return self.train_data, self.test_data

    def ordinal_encode_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.OrdinalEncoder` to the columns specified by the `ordinalEncode`
        parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the specified columns encoded.
        """
        if self.ordinal_encode:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.train_data[self.ordinal_encode] = encoder.fit_transform(
                self.train_data[self.ordinal_encode]
            )
            self.test_data[self.ordinal_encode] = encoder.transform(self.test_data[self.ordinal_encode])

            logger.info(
                f"Columns: {self.ordinal_encode} successfully ordinal encoded. Training split:\n{self.train_data.head(3)}"
            )

        return self.train_data, self.test_data

    def target_encode_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.TargetEncoder` to the columns specified by the `targetEncode`
        parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the specified columns encoded.
        """
        if self.target_encode:
            encoder = TargetEncoder(random_state=42, cv=3)
            self.train_data[self.target_encode] = encoder.fit_transform(
                self.train_data[self.target_encode], self.train_data[self.target]
            )
            self.test_data[self.target_encode] = encoder.transform(self.test_data[self.target_encode])

            logger.info(
                f"Columns: {self.target_encode} successfully target encoded. Training split:\n{self.train_data.head(3)}"
            )

        return self.train_data, self.test_data

    def label_encode_column(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.LabelEncoder` to the target column.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the target column encoded.
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
                    "Labels are present in the generated test split that are not present in the training split "
                    "and, therefore, cannot be fit. To resolve this, the label encoder will be fit on the entire "
                    "dataset. This introduces data-leakage and may negatively impact the reliability of the "
                    "results. Consider using a larger dataset to address this."
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

    def _rescale_columns(
        self,
        scaler: (
            StandardScaler
            | MinMaxScaler
            | MaxAbsScaler
            | RobustScaler
            | PowerTransformer
            | QuantileTransformer
            | Normalizer
        ),
        columns: list[str],
    ) -> None:
        self.train_data[columns] = scaler.fit_transform(self.train_data[columns])
        self.test_data[columns] = scaler.transform(self.test_data[columns])

        logger.info(
            f"Columns: {columns} successfully regularized. Training split:\n{self.train_data.head(3)}"
        )

    def standard_scale_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.StandardScaler` to the columns specified by the `standardScale`
        parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the specified columns regularized.
        """
        if self.standard_scale:
            scaler = StandardScaler()
            self._rescale_columns(scaler=scaler, columns=self.standard_scale)

        return self.train_data, self.test_data

    def min_max_scale_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.MinMaxScaler` to the columns specified by the `minMaxScale` parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the specified columns regularized.
        """
        if self.min_max_scale:
            scaler = MinMaxScaler()
            self._rescale_columns(scaler=scaler, columns=self.min_max_scale)

        return self.train_data, self.test_data

    def max_abs_scale_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.MaxAbsScaler` to the columns specified by the `maxAbsScale` parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the specified columns regularized.
        """
        if self.max_abs_scale:
            scaler = MaxAbsScaler()
            self._rescale_columns(scaler=scaler, columns=self.max_abs_scale)

        return self.train_data, self.test_data

    def robust_scale_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.RobustScaler` to the columns specified by the `robustScale` parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the specified columns regularized.
        """
        if self.robust_scale:
            scaler = RobustScaler(quantile_range=(25, 75))
            self._rescale_columns(scaler=scaler, columns=self.robust_scale)

        return self.train_data, self.test_data

    def power_transform_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.PowerTransformer` using the Yeo-Johnson method to the columns specified
        by the `powerTransform` parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the specified columns regularized.
        """
        if self.power_transform:
            scaler = PowerTransformer(method="yeo-johnson")
            self._rescale_columns(scaler=scaler, columns=self.power_transform)

        return self.train_data, self.test_data

    def quantile_transform_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.QuantileTransformer` with `output_distribution = "uniform"`
        to the columns specified by the `quantileTransform` parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the specified columns regularized.
        """
        if self.quantile_transform:
            scaler = QuantileTransformer(output_distribution="uniform", random_state=42)
            self._rescale_columns(scaler=scaler, columns=self.quantile_transform)

        return self.train_data, self.test_data

    def quantile_transform_normal_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.QuantileTransformer` with `output_distribution = "normal"`
        to the columns specified by the `quantileTransformNormal` parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the specified columns regularized.
        """
        if self.quantile_transform_normal:
            scaler = QuantileTransformer(output_distribution="normal", random_state=42)
            self._rescale_columns(scaler=scaler, columns=self.quantile_transform_normal)

        return self.train_data, self.test_data

    def normalize_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `sklearn.preprocessing.Normalizer` to the columns specified by the `normalize` parameter.

        Returns:
        --------
            (pd.DataFrame, pd.DataFrame): Train and test split with the specified columns regularized.
        """
        if self.normalize:
            scaler = Normalizer(norm="l2")
            self._rescale_columns(scaler=scaler, columns=self.normalize)

        return self.train_data, self.test_data

    def save_data(
        self,
        writer: ResultsWriter,
        file_format: Literal["parquet", "csv", "json", "pkl"] = "parquet",
        file_name_ending: str = "",
        overwrite: bool = True,
    ) -> Path:
        """
        Recombined the train and test split and saves the data to a file using the specified format.

        Args:
        -----
            save_directory (str | Path): Directory to save the data to.
            file_format (Literal["parquet", "csv", "json", "pkl"], optional): Format to use when
            saving the data. Defaults to "parquet".
            file_name_ending (str, optional): String to append to the end of the file name in order to save the data
        multiple times. Defaults to "".

        Returns:
        --------
            Path: Path to the saved data.
        """
        if not isinstance(file_format, str):
            raise TypeError("`file_format` must be a string.")
        if not isinstance(file_name_ending, str):
            raise TypeError("`file_name_ending` must be a string.")

        file_path = writer.directory_name / f"{self.save_name}{file_name_ending}.{file_format}"
        file_path = writer.increment_name(file_path)

        try:
            df = pd.concat([self.train_data, self.test_data]).sort_index()
        except Exception:
            logger.error("Could not recombine the train and test data in order to save it.")
            raise

        try:
            match file_format:
                case "parquet":
                    df.to_parquet(file_path, index=False, compression="gzip")
                case "csv":
                    df.to_csv(file_path, index=False)
                case "pkl":
                    df.to_pickle(file_path)
                case "json":
                    df.to_json(file_path, orient="records")
                case _:
                    raise ValueError(
                        "Invalid `file_format` provided. Must be one of: 'parquet', 'csv', 'json', 'pkl'."
                    )
            logger.info(f"Data saved to: {file_path}")
        except FileNotFoundError:
            logger.error(f"Could not save data to {file_path}.")
            raise

        return file_path

    def split_target(self) -> SplitDataTuple:
        """
        Separates the target column from the features for both the train and test data.

        Returns:
        --------
            SplitDataTuple:
                pd.DataFrame: Training split features.
                pd.DataFrame: Testing split features.
                pd.DataFrame | pd.Series: Training split target values.
                pd.DataFrame | pd.Series: Testing split target values.
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

    def split_and_save_data(self, writer: ResultsWriter, overwrite: bool = True) -> Path:
        """
        Splits the data and saves it to a single pickle file as a SplitData object.

        Args:
        -----
            save_directory (str | Path): Directory to save the SplitData object to.

        Returns:
        --------
            Path: Path to the saved SplitData object.
        """
        X_train, X_test, y_train, y_test = self.split_target()
        split_data_obj = SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        file_path = writer.directory_name / f"{self.save_name}-split.pkl"
        file_path = writer.increment_name(file_path)

        with open(file_path, "wb") as file:
            pickle.dump(split_data_obj, file)
        logger.info(f"Split data saved to: {file_path}")
        return file_path

    def process_dataset(
        self,
        writer: ResultsWriter,
        save_original: bool = True,
        save_processed: bool = True,
        overwrite: bool = True,
    ) -> SplitDataTuple:
        """
        Performs all data processing steps based on the parameters provided to `DatasetProcessor`.
        Optionally saves the original and processed data to files.

        Args:
        -----
            save_directory (str | Path): The directory to save the data to.
            save_original (bool): Whether to save the original data.
            save_processed (bool): Whether to save the processed, nonsplit data.

        Returns:
        --------
            SplitDataTuple:
                pd.DataFrame: Training split features.
                pd.DataFrame: Testing split features.
                pd.DataFrame | pd.Series: Training split target values.
                pd.DataFrame | pd.Series: Testing split target values.
        """
        if not isinstance(save_original, bool):
            raise TypeError("`save_original` must be a boolean.")
        if not isinstance(save_processed, bool):
            raise TypeError("`save_processed` must be a boolean.")

        if save_original:
            self.save_data(writer, file_name_ending="-original", overwrite=overwrite)

        self.drop_columns()
        self.handle_nan()
        self.one_hot_encode_columns()
        self.ordinal_encode_columns()
        self.target_encode_columns()
        self.label_encode_column()
        self.standard_scale_columns()
        self.min_max_scale_columns()
        self.max_abs_scale_columns()
        self.robust_scale_columns()
        self.power_transform_columns()
        self.quantile_transform_columns()
        self.quantile_transform_normal_columns()
        self.normalize_columns()

        if save_processed:
            self.save_data(writer, file_name_ending="-processed", overwrite=overwrite)

        return self.split_target()
