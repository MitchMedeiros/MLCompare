import logging
from pathlib import Path

# import src.training.random_forest as random_forest
import utils
from src.data.data_processor import DataProcessor

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    utils.setup_logging()

    save_directory = Path(__file__).parent.resolve() / "data" / "saved_data"

    kaggle_datasets = {
        "restaurant_revenue": {
            "username": "anthonytherrien",
            "dataset_name": "restaurant-revenue-prediction-dataset",
            "file_name": "restaurant_data.csv",
            "target_column": "Revenue",
            "columns_to_drop": ["Name"],
            "columns_to_encode": ["Location", "Cuisine", "Parking Availability"],
        },
    }

    for dataset in kaggle_datasets.values():
        processor = DataProcessor()
        processor.download_kaggle_data(
            dataset["username"],
            dataset["dataset_name"],
            dataset["file_name"],
        )
        processor.drop_columns(dataset["columns_to_drop"])
        processor.encode_columns(dataset["columns_to_encode"])
        processor.save_data(save_directory / f"{dataset['dataset_name']}_processed.pkl")
