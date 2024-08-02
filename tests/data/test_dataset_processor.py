import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mlcompare import DatasetProcessor, load_split_data, process_datasets
from mlcompare.data.datasets import LocalDataset

logger = logging.getLogger("mlcompare.data.dataset_processor")


kaggle_dataset_params = {
    "type": "kaggle",
    "user": "anthonytherrien",
    "dataset": "restaurant-revenue-prediction-dataset",
    "file": "restaurant_data.csv",
    "target": "Revenue",
    "drop": ["Name"],
    "onehotEncode": ["Location", "Cuisine", "Parking Availability"],
}


def create_dataset_processor(
    data: dict, data_params: dict, data_path: str
) -> DatasetProcessor:
    path = Path(data_path)

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

    try:
        local_dataset = LocalDataset(**data_params)  # type: ignore
        processor = DatasetProcessor(dataset=local_dataset)
    finally:
        os.remove(f"{path}")

    return processor


class TestDatasetProcessor:
    # save_directory = "run_pipeline_results"
    data = {
        "A": [1, 2],
        "B": [3, 4],
        "C": [5, 6],
        "D": [7, 8],
        "E": [9, 10],
        "F": [11, 12],
    }
    data_params = {
        "path": "three_column.csv",
        "target": "F",
        "drop": ["A", "C"],
        "onehotEncode": ["B"],
    }
    data_path = "three_column.csv"

    def test_init(self):
        processor = create_dataset_processor(
            self.data,
            self.data_params,
            self.data_path,
        )

        assert processor.data.equals(pd.DataFrame(self.data)) is True

    def test_drop_columns(self):
        processor = create_dataset_processor(
            self.data,
            self.data_params,
            self.data_path,
        )
        processed_data = processor.drop_columns()

        assert "A" not in processed_data.columns and "C" not in processed_data.columns
        assert "F" in processed_data.columns

    def test_encode_columns(self):
        processor = create_dataset_processor(
            self.data,
            self.data_params,
            self.data_path,
        )
        processed_data = processor.onehot_encode_columns()

        assert "B_3" in processed_data.columns and "B_4" in processed_data.columns
        assert "B" not in processed_data.columns
        assert processed_data["B_3"].sum() == 1
        assert processed_data["B_4"].sum() == 1

    def test_missing_values_no_missing_values(self):
        processor = create_dataset_processor(
            self.data,
            self.data_params,
            self.data_path,
        )
        result = processor.has_missing_values()

        assert result is False

    def test_missing_values_empty_file(self):
        empty_data = {"A": [], "B": []}
        dataset_params = {
            "path": "empty_data.csv",
            "target": "A",
        }
        dataset_path = "empty_data.csv"

        processor = create_dataset_processor(
            empty_data,
            dataset_params,
            dataset_path,
        )
        result = processor.has_missing_values()
        assert result is False

    def test_missing_values_none_value(self):
        none_data = {"A": [1, 2, None], "B": ["value1", "value2", "value3"]}
        dataset_params = {
            "path": "none_data.csv",
            "target": "A",
        }
        dataset_path = "none_data.csv"

        processor = create_dataset_processor(
            none_data,
            dataset_params,
            dataset_path,
        )

        with pytest.raises(ValueError):
            processor.has_missing_values()

    def test_missing_values_empty_strings(self):
        none_data = {"A": [1, 2, 3], "B": ["", "value", "value"]}
        dataset_params = {
            "path": "none_data.csv",
            "target": "A",
        }
        dataset_path = "none_data.csv"

        processor = create_dataset_processor(
            none_data,
            dataset_params,
            dataset_path,
        )

        with pytest.raises(ValueError):
            processor.has_missing_values()

    def test_missing_values_dot_values(self):
        none_data = {"A": [1, 2, 3], "B": ["value", ".", "value"]}
        dataset_params = {
            "path": "none_data.csv",
            "target": "A",
        }
        dataset_path = "none_data.csv"

        processor = create_dataset_processor(
            none_data,
            dataset_params,
            dataset_path,
        )

        with pytest.raises(ValueError):
            processor.has_missing_values()

    def test_multiple_missing_value_types(self):
        none_data = {"A": [1, 2, None], "B": ["", 3.5, "."], "C": [True, False, None]}
        dataset_params = {
            "path": "none_data.csv",
            "target": "A",
        }
        dataset_path = "none_data.csv"

        processor = create_dataset_processor(
            none_data,
            dataset_params,
            dataset_path,
        )

        with pytest.raises(ValueError):
            processor.has_missing_values()

    def test_missing_values_logging(self, caplog):
        none_data = {"A": [1, 2, None], "B": ["", 3.5, "."], "C": [True, False, None]}
        dataset_params = {
            "path": "none_data.csv",
            "target": "A",
        }
        dataset_path = "none_data.csv"

        processor = create_dataset_processor(
            none_data,
            dataset_params,
            dataset_path,
        )
        processor.has_missing_values(raise_exception=False)
        assert "Missing values found in DataFrame" in caplog.text

    def test_split_data(self):
        processor = create_dataset_processor(
            self.data,
            self.data_params,
            self.data_path,
        )
        X_train, X_test, y_train, y_test = processor.split_data()

        assert len(X_train) + len(X_test) == len(self.data["A"])
        assert len(y_train) + len(y_test) == len(self.data["F"])
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_split_data_invalid_test_size(self):
        processor = create_dataset_processor(
            self.data,
            self.data_params,
            self.data_path,
        )

        with pytest.raises(ValueError):
            processor.split_data(test_size=1.1)

    def test_save_data_parquet(self):
        processor = create_dataset_processor(
            self.data,
            self.data_params,
            self.data_path,
        )

        try:
            processor.save_dataframe(save_directory="save_testing")
            assert Path("save_testing/three_column.parquet").exists()

            df = pd.read_parquet("save_testing/three_column.parquet")
            assert df.equals(pd.DataFrame(self.data)) is True
        finally:
            shutil.rmtree("save_testing")

    def test_split_and_save_data(self):
        processor = create_dataset_processor(
            self.data,
            self.data_params,
            self.data_path,
        )

        try:
            file_path = processor.split_and_save_data(save_directory="save_testing")
            assert file_path.exists()

            X_train, X_test, y_train, y_test = load_split_data(file_path)
            assert isinstance(X_train, pd.DataFrame)
            assert isinstance(y_train, pd.Series)
        finally:
            os.remove(file_path)

    def test_process_dataset(self):
        processor = create_dataset_processor(
            self.data,
            self.data_params,
            self.data_path,
        )

        try:
            processor.process_dataset(save_directory="save_testing")

            assert Path("save_testing/three_column.parquet").exists()
            assert Path("save_testing/three_column_processed.parquet").exists()
        finally:
            shutil.rmtree("save_testing")

    def test_process_dataset_invalid_save_original_type(self):
        processor = create_dataset_processor(
            self.data,
            self.data_params,
            self.data_path,
        )

        with pytest.raises(ValueError):
            processor.process_dataset(save_directory="save_testing", save_original=123)

    def test_process_dataset_invalid_save_processed_type(self):
        processor = create_dataset_processor(
            self.data,
            self.data_params,
            self.data_path,
        )

        with pytest.raises(ValueError):
            processor.process_dataset(save_directory="save_testing", save_processed=123)

    def test_process_datasets(self):
        params_list = [
            {"type": "local", "path": "test1.csv", "target": "C", "drop": ["A"]},
            {"type": "local", "path": "test2.csv", "target": "F", "drop": ["D"]},
        ]

        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        df.to_csv("test1.csv", index=False)
        df = pd.DataFrame({"D": [7, 8], "E": [9, 10], "F": [11, 12]})
        df.to_csv("test2.csv", index=False)

        try:
            split_datasets = process_datasets(
                params_list,
                save_directory="save_testing",
                save_original=False,
                save_processed=False,
            )

            for X_train, X_test, y_train, y_test in split_datasets:
                assert isinstance(X_train, pd.DataFrame)
                assert isinstance(X_test, pd.DataFrame)
                assert isinstance(y_train, pd.Series)
                assert isinstance(y_test, pd.Series)
                assert X_train.empty is False
                assert X_test.empty is False
                assert y_train.empty is False
                assert y_test.empty is False

        finally:
            os.remove("test1.csv")
            os.remove("test2.csv")
