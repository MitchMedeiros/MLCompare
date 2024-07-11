import logging
import sys
import unittest
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2].as_posix()
sys.path.append(root_dir)
import util  # noqa: E402

data_dir = Path(__file__).resolve().parents[1].as_posix()
sys.path.append(data_dir)
from data.dataset_validation import KaggleDataset, LocalDataset, validate_dataset_params

util.setup_logging()
logger = logging.getLogger("src.data.dataset_validation")


class TestDataProcessor(unittest.TestCase):
    current_dir = Path(__file__).parent.resolve()

    kaggle_dataset_params = {
        "dataset_type": "kaggle",
        "username": "anthonytherrien",
        "dataset_name": "restaurant-revenue-prediction-dataset",
        "file_name": "restaurant_data.csv",
        "target_column": "Revenue",
        "columns_to_drop": ["Name"],
        "columns_to_onehot_encode": ["Location", "Cuisine", "Parking Availability"],
    }

    def test_init_with_dict(self):
        dataset = KaggleDataset(
            dataset_type=self.kaggle_dataset_params["dataset_type"],
            username=self.kaggle_dataset_params["username"],
            dataset_name=self.kaggle_dataset_params["dataset_name"],
            file_name=self.kaggle_dataset_params["file_name"],
            target_column=self.kaggle_dataset_params["target_column"],
            columns_to_drop=self.kaggle_dataset_params["columns_to_drop"],
            columns_to_onehot_encode=self.kaggle_dataset_params[
                "columns_to_onehot_encode"
            ],
        )

        self.assertEqual(dataset.dataset_type, "kaggle")
        self.assertEqual(dataset.username, "anthonytherrien")
        self.assertEqual(dataset.dataset_name, "restaurant-revenue-prediction-dataset")
        self.assertEqual(dataset.file_name, "restaurant_data.csv")
        self.assertEqual(dataset.target_column, "Revenue")
        self.assertEqual(dataset.columns_to_drop, ["Name"])
        self.assertEqual(
            dataset.columns_to_onehot_encode,
            ["Location", "Cuisine", "Parking Availability"],
        )


if __name__ == "__main__":
    unittest.main()
