import logging
import pickle
from io import StringIO
from pathlib import Path

import kaggle
import pandas as pd
from kaggle.rest import ApiException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def download_kaggle_data(
    dataset_owner: str, dataset_name: str, data_file_name: str, save_directory: Path
) -> StringIO:
    """
    Download the Kaggle dataset and save to a file. Currently only supports CSV files.

    ### Parameters
    data_id: the user under which the dataset is provided.

    dataset_name: the name of the dataset.

    data_file_name: the file to be downloaded from the dataset.

    save_directory: the directory to save the downloaded data in for future use.
    Will use the `data_file_name` as the file name.

    ### Returns
    The downloaded data as a StringIO object.
    """
    save_path = save_directory / data_file_name

    if not save_path.exists():
        try:
            data = kaggle.api.datasets_download_file(
                dataset_owner,
                dataset_name,
                data_file_name,
            )

            try:
                with open(save_path, "w") as file:
                    file.write(data)
                    logger.info(f"Dataset downloaded and saved to {save_path}.")

            except FileNotFoundError:
                logger.exception(f"Could not save dataset to {save_path}.")

        except ApiException as e:
            logger.exception(f"Exception when calling Kaggle Api: {e}\n")

    else:
        try:
            with open(save_path, "rb") as file:
                data = file.read()

            data = data.decode("utf-8")
            logger.info(f"Dataset download skipped. File: {save_path} used instead.")

        except Exception as e:
            logger.exception(f"Exception when trying to read existing data file: {e}")

    return StringIO(data)


def clean_and_encode_data(
    data: StringIO | Path,
    columns_to_drop: list[str] | None = None,
    columns_to_encode: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(data)

    if columns_to_drop is not None:
        df = df.drop(columns_to_drop, axis=1)

    if columns_to_encode is not None:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_array = encoder.fit_transform(df[columns_to_encode])

        # Convert the one-hot encoded ndarray to a DataFrame
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(columns_to_encode),
        )

        # Drop original columns and add the one-hot encoded columns
        df = df.drop(columns=columns_to_encode).join(encoded_df)

        logger.info(f"Data successfully cleaned and encoded: \n {df.head(5)}")

    return df


def split_and_save_data(
    df: pd.DataFrame, target_column: str, save_directory: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into its features and target, and save to a file for later model training."""
    X = df.drop([target_column], axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
    )
    logger.info(
        f"Data successfully split: {X_train.shape=}, {X_test.shape=}, {y_train.shape=}, {y_test.shape=}"
    )

    if save_directory is not None:
        save_path = save_directory / "train_test_split.pkl"

        with open(save_path, "wb") as file:
            pickle.dump(
                {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                },
                file,
            )
        logger.info(f"Split data saved to: {save_path}")

    return X_train, X_test, y_train, y_test


def process_kaggle_data():
    kaggle_user = "anthonytherrien"
    kaggle_dataset = "restaurant-revenue-prediction-dataset"
    kaggle_file_name = "restaurant_data.csv"

    parent_path = Path(__file__).parent.resolve()
    save_directory = parent_path / "saved_data"

    columns_to_drop = ["Name"]
    columns_to_encode = ["Location", "Cuisine", "Parking Availability"]
    target_column = "Revenue"

    raw_data = download_kaggle_data(
        kaggle_user,
        kaggle_dataset,
        kaggle_file_name,
        save_directory,
    )

    df = clean_and_encode_data(
        raw_data,
        columns_to_drop,
        columns_to_encode,
    )

    split_and_save_data(df, target_column, save_directory)
