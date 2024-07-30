import logging
import unittest
from pathlib import Path

import pandas as pd
from kaggle.rest import ApiException

from mlcompare import DataProcessor

data_processor_logger = logging.getLogger("mlcompare.data.data_processor")


class TestDataProcessor(unittest.TestCase):
    current_dir = Path(__file__).parent.resolve()
    two_column_data = {"A": [1, 2, 3], "B": [4, 5, 6]}

    def test_init_with_dataframe(self):
        data = pd.DataFrame(self.two_column_data)

        processor = DataProcessor(data=data)
        self.assertTrue(processor.data.equals(data))

    def test_init_with_path_csv(self):
        # Create a temporary CSV file for testing
        csv_path = self.current_dir / "test.csv"
        data = pd.DataFrame(self.two_column_data)
        data.to_csv(csv_path, index=False)

        processor = DataProcessor(data=csv_path)
        self.assertTrue(processor.data.equals(data))

        csv_path.unlink()

    def test_init_with_path_parquet(self):
        parquet_path = self.current_dir / "test.parquet"
        data = pd.DataFrame(self.two_column_data)
        data.to_parquet(parquet_path)

        processor = DataProcessor(data=parquet_path)
        self.assertTrue(processor.data.equals(data))

        parquet_path.unlink()

    def test_init_with_path_pkl(self):
        pkl_path = self.current_dir / "test.pkl"
        data = pd.DataFrame(self.two_column_data)
        data.to_pickle(pkl_path)

        processor = DataProcessor(data=pkl_path)
        self.assertTrue(processor.data.equals(data))

        pkl_path.unlink()

    def test_init_with_path_json(self):
        json_path = self.current_dir / "test.json"
        data = pd.DataFrame(self.two_column_data)
        data.to_json(json_path, orient="records")

        processor = DataProcessor(data=json_path)
        self.assertTrue(processor.data.equals(data))

        json_path.unlink()

    def test_init_with_unsupported_file_type(self):
        html_path = self.current_dir / "test.html"
        data = pd.DataFrame(self.two_column_data)
        data.to_html(html_path)

        with self.assertRaises(Exception):
            DataProcessor(data=html_path)

        html_path.unlink()

    def test_init_with_no_data(self):
        processor = DataProcessor()
        self.assertTrue(processor.data.empty)

    def test_init_with_invalid_data(self):
        with self.assertRaises(Exception):
            DataProcessor(data=123)

    def test_download_kaggle_data_success(self):
        owner = "anthonytherrien"
        dataset_name = "restaurant-revenue-prediction-dataset"
        file_name = "restaurant_data.csv"

        processor = DataProcessor()
        downloaded_data = processor.download_kaggle_data(owner, dataset_name, file_name)

        # Check if the downloaded data is a DataFrame that is not empty and that it was set as the data attribute
        self.assertTrue(isinstance(downloaded_data, pd.DataFrame))
        self.assertTrue(downloaded_data.equals(processor.data))
        self.assertTrue(not downloaded_data.empty)

    def test_download_kaggle_data_failure(self):
        owner = "asdf"
        dataset_name = "asdf"
        file_name = "asdf"

        with self.assertRaises(ApiException):
            processor = DataProcessor()
            processor.download_kaggle_data(owner, dataset_name, file_name)

    def test_drop_columns(self):
        data = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        processor = DataProcessor(data=data)
        processed_data = processor.drop_columns(["A", "C"])
        self.assertTrue(
            "A" not in processed_data.columns and "C" not in processed_data.columns
        )
        self.assertTrue("B" in processed_data.columns)

    def test_encode_columns(self):
        data = pd.DataFrame({"Category": ["A", "B", "A"], "Value": [1, 2, 3]})
        processor = DataProcessor(data=data)
        processed_data = processor.encode_columns(["Category"])
        self.assertTrue(
            "Category_A" in processed_data.columns
            and "Category_B" in processed_data.columns
        )
        self.assertEqual(processed_data["Category_A"].sum(), 2)
        self.assertEqual(processed_data["Category_B"].sum(), 1)

    def test_split_data(self):
        data = pd.DataFrame({"Feature": [1, 2, 3, 4], "Target": [5, 6, 7, 8]})
        processor = DataProcessor(data=data)
        X_train, X_test, y_train, y_test = processor.split_data("Target")
        self.assertEqual(len(X_train) + len(X_test), 4)
        self.assertEqual(len(y_train) + len(y_test), 4)
        self.assertTrue(isinstance(X_train, pd.DataFrame))
        self.assertTrue(isinstance(X_test, pd.DataFrame))
        self.assertTrue(isinstance(y_train, pd.Series))
        self.assertTrue(isinstance(y_test, pd.Series))

    def test_save_data_csv(self):
        save_path = self.current_dir / "test.csv"
        data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        processor = DataProcessor(data=data)

        processor.save_dataframe(file_path=save_path)
        self.assertTrue(save_path.exists())

        save_path.unlink()

    def test_save_data_pickle(self):
        save_path = self.current_dir / "test.pkl"
        data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        processor = DataProcessor(data=data)

        processor.save_dataframe(file_path=save_path)
        self.assertTrue(save_path.exists())

        save_path.unlink()

    def test_download_format_and_save_parquet_data_from_dict(self):
        kaggle_dataset_params = {
            "username": "anthonytherrien",
            "dataset_name": "restaurant-revenue-prediction-dataset",
            "file_name": "restaurant_data.csv",
            "target_column": "Revenue",
            "columns_to_drop": ["Name"],
            "columns_to_encode": ["Location", "Cuisine", "Parking Availability"],
        }
        file_format = "parquet"
        save_path = (
            self.current_dir / f"{kaggle_dataset_params['dataset_name']}.{file_format}"
        )
        processor = DataProcessor()

        processor.download_kaggle_data(
            kaggle_dataset_params["username"],
            kaggle_dataset_params["dataset_name"],
            kaggle_dataset_params["file_name"],
        )
        processor.drop_columns(kaggle_dataset_params["columns_to_drop"])
        processor.encode_columns(kaggle_dataset_params["columns_to_encode"])
        processor.save_dataframe(save_path, file_format)

        self.assertTrue(save_path.exists())
        df = pd.read_parquet(save_path)
        self.assertTrue(not df.empty)
        self.assertTrue("Name" not in df.columns)
        self.assertTrue("Location" not in df.columns)
        self.assertTrue("Cuisine" not in df.columns)

        save_path.unlink()

    def test_missing_values_no_missing_values(self):
        data = {"A": [1, 2, 3], "B": ["value", "value", "value"]}
        processor = DataProcessor(data=data)
        result = processor.has_missing_values()
        self.assertFalse(result)

    def test_missing_values_empty_dataframe(self):
        data = pd.DataFrame()
        processor = DataProcessor(data=data)
        result = processor.has_missing_values()
        self.assertFalse(result)

    def test_missing_values_none_value(self):
        data = {"A": [1, 2, None], "B": ["value", "value", "value"]}
        processor = DataProcessor(data=data)
        with self.assertRaises(ValueError):
            result = processor.has_missing_values()
            self.assertTrue(result)

    def test_missing_values_empty_strings(self):
        data = {"A": [1, 2, 3], "B": ["", "value", "value"]}
        processor = DataProcessor(data=data)
        with self.assertRaises(ValueError):
            result = processor.has_missing_values()
            self.assertTrue(result)

    def test_missing_values_dot_values(self):
        data = {"A": [1, 2, 3], "B": ["value", ".", "value"]}
        processor = DataProcessor(data=data)
        with self.assertRaises(ValueError):
            result = processor.has_missing_values()
            self.assertTrue(result)

    def test_multiple_missing_value_types(self):
        data = {"A": [1, 2, None], "B": ["", 3.5, "."], "C": [True, False, None]}
        processor = DataProcessor(data=data)
        with self.assertRaises(ValueError):
            result = processor.has_missing_values()
            self.assertTrue(result)

    def test_logs_warning_message(self):
        data = {"A": [1, 2, None], "B": ["", "value", "."]}
        processor = DataProcessor(data=data)
        with self.assertLogs(data_processor_logger, level="WARNING"):
            processor.has_missing_values(raise_exception=False)


if __name__ == "__main__":
    unittest.main()

"""Additional Tests"""
# class Test__Init__(unittest.TestCase):
#     # Initializes with a valid default save path
#     def test_initializes_with_valid_default_save_path(self):
#         save_path = Path("saved_data.pkl")
#         processor = DataProcessor(default_save_path=save_path)
#         self.assertEqual(processor.default_save_path, save_path)

#     # Initializes with an invalid file extension
#     def test_initializes_with_invalid_file_extension(self):
#         data_path = Path("test_data.txt")
#         with self.assertRaises(ValueError):
#             DataProcessor(data=data_path)

#     # Initializes with a non-existent file path
#     def test_initializes_with_non_existent_file_path(self):
#         data_path = Path("non_existent_file.csv")
#         with self.assertLogs(logger, level="ERROR") as log:
#             DataProcessor(data=data_path)
#             self.assertIn("File not found", log.output[0])

#     # Initializes with data that cannot be converted to a DataFrame
#     def test_initializes_with_unconvertible_data(self):
#         data = object()
#         with self.assertLogs(logger, level="ERROR") as log:
#             DataProcessor(data=data)
#             self.assertIn("Could not convert data", log.output[0])

#     # Initializes with a None default save path
#     def test_initializes_with_none_default_save_path(self):
#         processor = DataProcessor(default_save_path=None)
#         self.assertIsNone(processor.default_save_path)

#     # Initializes with a non-Path default save path
#     def test_initializes_with_non_path_default_save_path(self):
#         save_path = "saved_data.pkl"
#         processor = DataProcessor(default_save_path=save_path)
#         self.assertEqual(processor.default_save_path, save_path)

#     # Logs an error when file is not found
#     def test_logs_error_when_file_not_found(self):
#         data_path = Path("non_existent_file.csv")
#         with self.assertLogs(logger, level="ERROR") as log:
#             DataProcessor(data=data_path)
#             self.assertIn("File not found", log.output[0])

#     # Logs an error when data conversion fails
#     def test_logs_error_when_data_conversion_fails(self):
#         with self.assertLogs(logger, level="ERROR") as cm:
#             DataProcessor(data=123)
#         self.assertEqual(
#             cm.output,
#             [
#                 "ERROR:__main__:Could not convert data with type: <class 'int'> to DataFrame: "
#             ],
#         )

#     # Sets self.data to an empty DataFrame when no data is provided
#     def test_sets_data_to_empty_dataframe_when_no_data_provided(self):
#         processor = DataProcessor()
#         self.assertTrue(processor.data.empty)

#     # Raises ValueError for unsupported file extensions
#     def test_raises_value_error_for_unsupported_file_extensions(self):
#         with self.assertRaises(ValueError) as context:
#             DataProcessor(data=Path("data/test.txt"))
#         self.assertTrue(
#             "Data file must be a CSV, PKL, or JSON file." in str(context.exception)
#         )

#     # Handles mixed data types in the input data
#     def test_handles_mixed_data_types(self):
#         mixed_data = 123
#         processor = DataProcessor(data=mixed_data)
#         self.assertIsInstance(processor.data, pd.DataFrame)
