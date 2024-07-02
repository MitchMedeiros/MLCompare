import logging
import os

import kaggle
import pandas as pd
from kaggle.rest import ApiException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def download_and_save_data(
    data_id: str, file_name: str, save_directory: str, overwrite: bool = False
) -> None:
    """
    Download and save any Kaggle dataset to a file.

    ### Parameters
    data_id: the string identifier of the dataset, in format <owner>/<dataset-name>.

    file_name: the name of the file. This is used for both downloading and saving.

    save_directory: the directory to save the file to.

    overwrite: overwrite a file at the same location if it already exists.
    """
    saved_file_path = os.path.join(save_directory, file_name)

    if not os.path.exists(saved_file_path) or overwrite:
        try:
            kaggle.api.dataset_download_file(
                data_id, file_name, save_directory, force=overwrite
            )

        except ApiException as e:
            logging.exception(f"Exception when calling Kaggle Api: {e}\n")

    else:
        logging.info(f"File already exists at {saved_file_path}")


def read_and_process_data(
    saved_file_path: str,
    columns_to_drop: list | None = None,
    columns_to_encode: list | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(saved_file_path)

    if columns_to_drop is not None:
        df = df.drop(columns_to_drop, axis=1)

    if columns_to_encode is not None:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_array = encoder.fit_transform(df[columns_to_encode])

        # Convert the one-hot encoded ndarray to a DataFrame
        encoded_df = pd.DataFrame(
            encoded_array, columns=encoder.get_feature_names_out(columns_to_encode)
        )

        # Drop original columns and add the one-hot encoded columns
        df = df.drop(columns=columns_to_encode).join(encoded_df)

    return df


def split_and_save_data(
    df: pd.DataFrame, target_column: str, save_directory: str
) -> None:
    """Split the data into features and target, and save them to separate files."""
    X = df.drop([target_column], axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    X.to_csv(os.path.join(save_directory, "features.csv"), index=False)
    y.to_csv(os.path.join(save_directory, "target.csv"), index=False)


# print(df.head(5))

if __name__ == "__main__":
    data_id = "anthonytherrien/restaurant-revenue-prediction-dataset"
    file_name = "restaurant_data.csv"
    save_directory = "data"

    columns_to_drop = ["Name"]
    columns_to_encode = ["Location", "Cuisine", "Parking Availability"]

    target_column = "Revenue"

    download_and_save_data(data_id, file_name, save_directory)
    df = read_and_process_data(os.path.join(save_directory, file_name))
    split_and_save_data(df, save_directory)
