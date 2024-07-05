import logging
import sys
from pathlib import Path
from typing import Any

from data_processor import DataProcessor
from kaggle_dataset import KaggleDataset, kaggle_dataset_params

# Add the project source directory to the system path for importing utils
src_dir = Path(__file__).resolve().parents[2].as_posix()
sys.path.append(src_dir)
import utils  # noqa: E402

logger = logging.getLogger(__name__)


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

        raw_data_file_name = f"{dataset.dataset_name}.{file_format}"
        processed_data_file_name = f"{dataset.dataset_name}_cleaned.{file_format}"

        processor.save_data(save_directory / raw_data_file_name, file_format)
        processor.has_missing_values()
        processor.drop_columns(dataset.columns_to_drop)
        processor.encode_columns(dataset.columns_to_encode)
        processor.save_data(save_directory / processed_data_file_name, file_format)


if __name__ == "__main__":
    utils.setup_logging()

    save_directory = Path(__file__).parent.resolve() / "saved_data"
    file_format = "parquet"

    download_and_process_data(kaggle_dataset_params, save_directory, file_format)
