import logging
import sys
from pathlib import Path
from typing import Any

from data_processor import DataProcessor
from pydantic import BaseModel

src_dir = Path(__file__).resolve().parents[2].as_posix()
sys.path.append(src_dir)
import utils  # noqa: E402

logger = logging.getLogger(__name__)


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


def download_and_process_data(
    dataset_params: dict[str, dict[str, Any]],
    save_directory: Path,
    file_format: str,
) -> None:
    """
    Downloads and processes data from multiple Kaggle datasets.

    Args:
        dataset_params (Dict[str, Dict[str, Any]]): A dictionary containing parameters for each dataset.
            The keys are dataset names, and the values are dictionaries containing the following keys:
            - 'username': The Kaggle username of the dataset owner.
            - 'dataset_name': The name of the Kaggle dataset.
            - 'file_name': The name of the file to download from the dataset.
            - 'columns_to_drop' (optional): A list of column names to drop from the downloaded data.
            - 'columns_to_encode' (optional): A list of column names to encode using a specific encoding method.
        save_directory (Path): The directory where the processed data will be saved.
        file_format (str): The file format to use when saving the processed data.
    """
    # Validate the dataset parameters via Pydantic
    kaggle_datasets = [KaggleDataset(**params) for params in dataset_params.values()]

    for dataset in kaggle_datasets:
        processor = DataProcessor()

        processor.download_kaggle_data(
            dataset.username,
            dataset.dataset_name,
            dataset.file_name,
        )
        processor.drop_columns(dataset.columns_to_drop)
        processor.encode_columns(dataset.columns_to_encode)
        processor.save_data(save_directory / f"{dataset.dataset_name}.{file_format}")


if __name__ == "__main__":
    utils.setup_logging()

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

    save_directory = Path(__file__).parent.resolve() / "saved_data"
    file_format = "parquet"

    download_and_process_data(kaggle_dataset_params, save_directory, file_format)
