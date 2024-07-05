from pydantic import BaseModel


class KaggleDataset(BaseModel):
    """
    Represents a Kaggle dataset.

    Attributes:
        username (str): The username of the Kaggle user who owns the dataset.
        dataset_name (str): The name of the Kaggle dataset.
        file_name (str): The name of the file associated with the dataset.
        target_column (str): The name of the target column in the dataset.
        columns_to_drop (list[str] | None): A list of column names to be dropped from the dataset.
            If None, no columns will be dropped.
        columns_to_encode (list[str] | None): A list of column names to be encoded in the dataset.
            If None, no columns will be encoded.
    """

    username: str
    dataset_name: str
    file_name: str
    target_column: str
    columns_to_drop: list[str] | None
    columns_to_encode: list[str] | None


# A dictionary containing parameters for each Kaggle dataset for downloading and processing.
kaggle_dataset_params = {
    "restaurant_revenue": {
        "username": "anthonytherrien",
        "dataset_name": "restaurant-revenue-prediction-dataset",
        "file_name": "restaurant_data.csv",
        "target_column": "Revenue",
        "columns_to_drop": ["Name"],
        "columns_to_encode": ["Location", "Cuisine", "Parking Availability"],
    },
}
