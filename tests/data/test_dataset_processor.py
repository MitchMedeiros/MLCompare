# import logging
# import unittest
# from pathlib import Path

# import pandas as pd
# from kaggle.rest import ApiException

# from mlcompare import DatasetProcessor

# logger = logging.getLogger("mlcompare.data.dataset_processor")


# kaggle_dataset_params = {
#     "type": "kaggle",
#     "user": "anthonytherrien",
#     "dataset": "restaurant-revenue-prediction-dataset",
#     "file": "restaurant_data.csv",
#     "target": "Revenue",
#     "drop": ["Name"],
#     "onehotEncode": ["Location", "Cuisine", "Parking Availability"],
# }


# class TestDatasetProcessor(unittest.TestCase):
#     current_dir = Path(__file__).parent.resolve()
#     two_column_data = {"A": [1, 2, 3], "B": [4, 5, 6]}
#     save_directory = "run_pipeline_results"

#     def test_init_with_dict(self):
#         data = self.two_column_data
#         processor = DatasetProcessor(dataset=data, data_directory=self.save_directory)
#         self.assertTrue(processor.data.equals(data))

#     def test_init_with_invalid_data(self):
#         # Initialize DatasetProcessor with an invalid data type
#         with self.assertRaises(Exception):
#             DatasetProcessor(dataset=123, data_directory=self.save_directory)

#     def test_init_with_dataframe(self):
#         data = pd.DataFrame(self.two_column_data)
#         processor = DatasetProcessor(dataset=data, data_directory=self.save_directory)
#         self.assertTrue(processor.data.equals(data))

#     def test_init_with_path_csv(self):
#         # Create a temporary CSV file for testing
#         csv_path = self.current_dir / "test.csv"
#         data = pd.DataFrame(self.two_column_data)
#         data.to_csv(csv_path, index=False)

#         processor = DatasetProcessor(
#             dataset=csv_path, data_directory=self.save_directory
#         )
#         self.assertTrue(processor.data.equals(data))

#         csv_path.unlink()

#     def test_init_with_path_parquet(self):
#         # Create a temporary pickle file for testing
#         parquet_path = self.current_dir / "test.parquet"
#         data = pd.DataFrame(self.two_column_data)
#         data.to_parquet(parquet_path)

#         processor = DatasetProcessor(
#             dataset=parquet_path, data_directory=self.save_directory
#         )
#         self.assertTrue(processor.data.equals(data))

#         parquet_path.unlink()

#     def test_init_with_path_pkl(self):
#         # Create a temporary pickle file for testing
#         pkl_path = self.current_dir / "test.pkl"
#         data = pd.DataFrame(self.two_column_data)
#         data.to_pickle(pkl_path)

#         processor = DatasetProcessor(
#             dataset=pkl_path, data_directory=self.save_directory
#         )
#         self.assertTrue(processor.data.equals(data))

#         pkl_path.unlink()

#     def test_init_with_path_json(self):
#         # Create a temporary JSON file for testing
#         json_path = self.current_dir / "test.json"
#         data = pd.DataFrame(self.two_column_data)
#         data.to_json(json_path, orient="records")

#         processor = DatasetProcessor(
#             dataset=json_path, data_directory=self.save_directory
#         )
#         self.assertTrue(processor.data.equals(data))

#         json_path.unlink()

#     def test_init_with_unsupported_file_type(self):
#         # Create a temporary JSON file for testing
#         html_path = self.current_dir / "test.html"
#         data = pd.DataFrame(self.two_column_data)
#         data.to_html(html_path)

#         with self.assertRaises(Exception):
#             DatasetProcessor(dataset=html_path, data_directory=self.save_directory)

#         html_path.unlink()

#     def test_download_kaggle_data_success(self):
#         owner = "anthonytherrien"
#         dataset_name = "restaurant-revenue-prediction-dataset"
#         file_name = "restaurant_data.csv"

#         processor = DatasetProcessor()
#         downloaded_data = processor._download_kaggle_data(
#             owner, dataset_name, file_name
#         )

#         # Check if the downloaded data is a DataFrame that is not empty and that it was set as the data attribute
#         self.assertTrue(isinstance(downloaded_data, pd.DataFrame))
#         self.assertTrue(downloaded_data.equals(processor.data))
#         self.assertTrue(not downloaded_data.empty)

#     def test_download_kaggle_data_failure(self):
#         owner = "asdf"
#         dataset_name = "asdf"
#         file_name = "asdf"

#         with self.assertRaises(ApiException):
#             processor = DatasetProcessor()
#             processor._download_kaggle_data(owner, dataset_name, file_name)

#     def test_drop_columns(self):
#         # Test dropping columns from the DataFrame
#         data = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
#         processor = DatasetProcessor(data=data)
#         processed_data = processor.drop_columns(["A", "C"])
#         self.assertTrue(
#             "A" not in processed_data.columns and "C" not in processed_data.columns
#         )
#         self.assertTrue("B" in processed_data.columns)

#     def test_encode_columns(self):
#         # Test encoding categorical columns
#         data = pd.DataFrame({"Category": ["A", "B", "A"], "Value": [1, 2, 3]})
#         processor = DatasetProcessor(data=data)
#         processed_data = processor.onehot_encode_columns(["Category"])
#         self.assertTrue(
#             "Category_A" in processed_data.columns
#             and "Category_B" in processed_data.columns
#         )
#         self.assertEqual(processed_data["Category_A"].sum(), 2)
#         self.assertEqual(processed_data["Category_B"].sum(), 1)

#     def test_split_data(self):
#         # Test splitting data into training and testing sets
#         data = pd.DataFrame({"Feature": [1, 2, 3, 4], "Target": [5, 6, 7, 8]})
#         processor = DatasetProcessor(data=data)
#         X_train, X_test, y_train, y_test = processor.split_data("Target")
#         self.assertEqual(len(X_train) + len(X_test), 4)
#         self.assertEqual(len(y_train) + len(y_test), 4)
#         self.assertTrue(isinstance(X_train, pd.DataFrame))
#         self.assertTrue(isinstance(X_test, pd.DataFrame))
#         self.assertTrue(isinstance(y_train, pd.Series))
#         self.assertTrue(isinstance(y_test, pd.Series))

#     def test_save_data_csv(self):
#         # Test saving data to a CSV file
#         save_path = self.current_dir / "test.csv"
#         data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
#         processor = DatasetProcessor(data=data)

#         processor.save_dataframe(file_path=save_path)
#         self.assertTrue(save_path.exists())

#         save_path.unlink()

#     def test_save_data_pickle(self):
#         # Test saving data to a pickle file
#         save_path = self.current_dir / "test.pkl"
#         data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
#         processor = DatasetProcessor(data=data)

#         processor.save_dataframe(file_path=save_path)
#         self.assertTrue(save_path.exists())

#         save_path.unlink()

#     def test_download_format_and_save_parquet_data_from_dict(self):
#         kaggle_dataset_params = {
#             "username": "anthonytherrien",
#             "dataset_name": "restaurant-revenue-prediction-dataset",
#             "file_name": "restaurant_data.csv",
#             "target_column": "Revenue",
#             "columns_to_drop": ["Name"],
#             "columns_to_encode": ["Location", "Cuisine", "Parking Availability"],
#         }
#         file_format = "parquet"
#         save_path = (
#             self.current_dir / f"{kaggle_dataset_params['dataset_name']}.{file_format}"
#         )
#         processor = DatasetProcessor()

#         processor._download_kaggle_data(
#             kaggle_dataset_params["username"],
#             kaggle_dataset_params["dataset_name"],
#             kaggle_dataset_params["file_name"],
#         )
#         processor.drop_columns(kaggle_dataset_params["columns_to_drop"])
#         processor.onehot_encode_columns(kaggle_dataset_params["columns_to_encode"])
#         processor.save_dataframe(save_path, file_format)

#         self.assertTrue(save_path.exists())
#         df = pd.read_parquet(save_path)
#         self.assertTrue(not df.empty)
#         self.assertTrue("Name" not in df.columns)
#         self.assertTrue("Location" not in df.columns)
#         self.assertTrue("Cuisine" not in df.columns)

#         save_path.unlink()

#     # Returns False when no missing values are present
#     def test_missing_values_no_missing_values(self):
#         data = {"A": [1, 2, 3], "B": ["value", "value", "value"]}
#         processor = DatasetProcessor(data=data)
#         result = processor.has_missing_values()
#         self.assertFalse(result)

#     # DataFrame is empty and should return False
#     def test_missing_values_empty_dataframe(self):
#         data = pd.DataFrame()
#         processor = DatasetProcessor(data=data)
#         result = processor.has_missing_values()
#         self.assertFalse(result)

#     # Detects NaN values in DataFrame and returns True
#     def test_missing_values_none_value(self):
#         data = {"A": [1, 2, None], "B": ["value", "value", "value"]}
#         processor = DatasetProcessor(data=data)
#         with self.assertRaises(ValueError):
#             result = processor.has_missing_values()
#             self.assertTrue(result)

#     # Detects empty strings in DataFrame and returns True
#     def test_missing_values_empty_strings(self):
#         data = {"A": [1, 2, 3], "B": ["", "value", "value"]}
#         processor = DatasetProcessor(data=data)
#         with self.assertRaises(ValueError):
#             result = processor.has_missing_values()
#             self.assertTrue(result)

#     # Detects "." values in DataFrame and returns True
#     def test_missing_values_dot_values(self):
#         data = {"A": [1, 2, 3], "B": ["value", ".", "value"]}
#         processor = DatasetProcessor(data=data)
#         with self.assertRaises(ValueError):
#             result = processor.has_missing_values()
#             self.assertTrue(result)

#     # DataFrame contains mixed data types
#     def test_multiple_missing_value_types(self):
#         data = {"A": [1, 2, None], "B": ["", 3.5, "."], "C": [True, False, None]}
#         processor = DatasetProcessor(data=data)
#         with self.assertRaises(ValueError):
#             result = processor.has_missing_values()
#             self.assertTrue(result)

#     # Logs a warning message when missing values are found
#     def test_logs_warning_message(self):
#         data = {"A": [1, 2, None], "B": ["", "value", "."]}
#         processor = DatasetProcessor(data=data)
#         with self.assertLogs(logger, level="WARNING"):
#             processor.has_missing_values(raise_exception=False)


# if __name__ == "__main__":
#     unittest.main()
