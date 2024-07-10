import sys
import unittest
from pathlib import Path

data_dir = Path(__file__).resolve().parents[1].as_posix()
sys.path.append(data_dir)
from data.dataset_validation import (  # noqa: E402
    KaggleDataset,
    LocalDataset,
    validate_dataset_params,
)


class TestDataProcessor(unittest.TestCase):
    current_dir = Path(__file__).parent.resolve()

    kaggle_dataset_params = {
        "username": "anthonytherrien",
        "dataset_name": "restaurant-revenue-prediction-dataset",
        "file_name": "restaurant_data.csv",
        "target_column": "Revenue",
        "columns_to_drop": ["Name"],
        "columns_to_encode": ["Location", "Cuisine", "Parking Availability"],
    }

    def test_init_with_dict(self):
        dataset = KaggleDataset(
            username=self.kaggle_dataset_params["username"],
            dataset_name=self.kaggle_dataset_params["dataset_name"],
            file_name=self.kaggle_dataset_params["file_name"],
            target_column=self.kaggle_dataset_params["target_column"],
            columns_to_drop=self.kaggle_dataset_params["columns_to_drop"],
            columns_to_encode=self.kaggle_dataset_params["columns_to_encode"],
        )

        self.assertEqual(dataset.username, "anthonytherrien")
        self.assertEqual(dataset.dataset_name, "restaurant-revenue-prediction-dataset")
        self.assertEqual(dataset.file_name, "restaurant_data.csv")
        self.assertEqual(dataset.target_column, "Revenue")
        self.assertEqual(dataset.columns_to_drop, ["Name"])
        self.assertEqual(
            dataset.columns_to_encode, ["Location", "Cuisine", "Parking Availability"]
        )


if __name__ == "__main__":
    unittest.main()
