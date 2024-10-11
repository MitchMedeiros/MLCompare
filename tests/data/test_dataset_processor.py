import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mlcompare import DatasetProcessor, load_split_data
from mlcompare.data.datasets import LocalDataset
from mlcompare.results_writer import ResultsWriter

logger = logging.getLogger("mlcompare.data.dataset_processor")


def create_dataset_processor(data: dict, data_params: dict) -> DatasetProcessor:
    """
    Utility function for testing that creates a csv file, a LocalDataset, and a DatasetProcessor.
    """
    path = Path(data_params["path"])
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

    try:
        local_dataset = LocalDataset(**data_params)  # type: ignore
        processor = DatasetProcessor(dataset=local_dataset)
    finally:
        os.remove(path)

    return processor


class TestDatasetProcessor:
    data = {
        "A": [1, 2, 9, 10, 11, 12, 20, 40, 8, 8, 8, 8],
        "B": [3, 4, 9, 10, 11, 12, 20, 40, 8, 8, 8, 8],
        "C": [5, 6, 9, 10, 11, 12, 20, 40, 8, 8, 8, 8],
        "D": [7, 13, 16, 18, 20, 10, 16, 10, 8, 8, 8, 8],
        "E": [9, 10, 11, 12, 42, 14, 11, 40, 8, 8, 8, 8],
        "F": [11, 11, 13, 13, 11, 11, 13, 13, 11, 13, 11, 13],
    }
    data_params = {
        "path": "integer_data.csv",
        "target": "F",
        "drop": ["A", "C"],
        "oneHotEncode": ["B", "D"],
        "nan": "drop",
    }

    def test_init(self):
        processor = create_dataset_processor(self.data, self.data_params)
        recombined_df = pd.concat([processor.train_data, processor.test_data]).sort_index()

        assert recombined_df.equals(pd.DataFrame(self.data)) is True

    def test_init_empty_file(self):
        empty_data = {"A": [], "B": []}
        dataset_params = {
            "path": "empty_data.csv",
            "target": "A",
            "nan": "drop",
        }

        with pytest.raises(ValueError):
            create_dataset_processor(empty_data, dataset_params)

    def test_drop_columns(self):
        processor = create_dataset_processor(self.data, self.data_params)
        train_data, test_data = processor.drop_columns()

        assert "A" not in train_data.columns and "C" not in train_data.columns
        assert "A" not in test_data.columns and "C" not in test_data.columns
        assert "F" in train_data.columns
        assert "F" in test_data.columns

    def test_one_hot_encode_columns(self):
        processor = create_dataset_processor(self.data, self.data_params)
        train_data, test_data = processor.one_hot_encode_columns()

        assert "B" not in train_data.columns and "B" not in test_data.columns
        assert "D" not in train_data.columns and "D" not in test_data.columns
        assert (
            "B_3" in train_data.columns
            or "B_4" in train_data.columns
            or "B_9" in train_data.columns
            or "B_10" in train_data.columns
            or "B_11" in train_data.columns
            or "B_12" in train_data.columns
        )

        if "B_3" in train_data.columns:
            assert train_data["B_3"].sum() == 1
        elif "B_4" in train_data.columns:
            assert train_data["B_4"].sum() == 1
        elif "B_9" in train_data.columns:
            assert train_data["B_9"].sum() == 1
        elif "B_10" in train_data.columns:
            assert train_data["B_10"].sum() == 1
        elif "B_11" in train_data.columns:
            assert train_data["B_11"].sum() == 1
        elif "B_12" in train_data.columns:
            assert train_data["B_12"].sum() == 1
        else:
            raise ValueError()

    def test_ordinal_encode_columns(self):
        processor = create_dataset_processor(
            self.data,
            {
                "path": "integer_data.csv",
                "target": "F",
                "ordinalEncode": ["D", "E"],
            },
        )
        train_data, test_data = processor.ordinal_encode_columns()

        assert train_data["E"].min() == 0
        assert train_data["E"].max() == len(processor.train_data["E"].unique()) - 1

    def test_target_encode_column(self):
        processor = create_dataset_processor(
            self.data,
            {
                "path": "integer_data.csv",
                "target": "F",
                "targetEncode": ["E"],
            },
        )
        train_data, test_data = processor.target_encode_columns()

        assert "E" in train_data.columns

    def test_label_encode_column(self):
        processor = create_dataset_processor(
            self.data,
            {
                "path": "integer_data.csv",
                "target": "F",
                "labelEncode": "yes",
            },
        )
        original_train_data_length = len(processor.train_data["F"].unique())
        train_data, test_data = processor.label_encode_column()

        assert train_data["F"].min() == 0
        assert train_data["F"].max() == original_train_data_length - 1

    def test_standard_scale_columns(self):
        processor = create_dataset_processor(
            self.data,
            {
                "path": "integer_data.csv",
                "target": "F",
                "standardScale": ["B"],
            },
        )
        train_data, test_data = processor.standard_scale_columns()

        assert round(train_data["B"].sum(), 3) == 0

    def test_min_max_scale_columns(self):
        processor = create_dataset_processor(
            self.data,
            {
                "path": "integer_data.csv",
                "target": "F",
                "minMaxScale": ["B"],
            },
        )
        train_data, test_data = processor.min_max_scale_columns()

        assert train_data["B"].max() == 1
        assert train_data["B"].min() == 0

    def test_max_abs_scale_columns(self):
        processor = create_dataset_processor(
            self.data,
            {
                "path": "integer_data.csv",
                "target": "F",
                "maxAbsScale": ["B"],
            },
        )
        original_train_data_sum = (
            processor.train_data["B"].apply(lambda x: x / processor.train_data["B"].max()).sum()
        )
        train_data, test_data = processor.max_abs_scale_columns()

        assert train_data["B"].max() == 1
        assert original_train_data_sum == train_data["B"].sum()

    def test_robust_scale_columns(self):
        processor = create_dataset_processor(
            self.data,
            {
                "path": "integer_data.csv",
                "target": "F",
                "robustScale": ["B"],
            },
        )
        train_data, test_data = processor.robust_scale_columns()

        assert "B" in train_data.columns

    def test_power_transform_columns(self):
        processor = create_dataset_processor(
            self.data,
            {
                "path": "integer_data.csv",
                "target": "F",
                "powerTransform": ["B"],
            },
        )
        train_data, test_data = processor.power_transform_columns()

        assert round(train_data["B"].sum(), 5) == 0

    def test_quantile_transform_columns(self):
        processor = create_dataset_processor(
            self.data,
            {
                "path": "integer_data.csv",
                "target": "F",
                "quantileTransform": ["B"],
            },
        )
        train_data, test_data = processor.quantile_transform_columns()

        assert train_data["B"].max() == 1
        assert train_data["B"].min() == 0

    def test_quantile_transform_normal_columns(self):
        processor = create_dataset_processor(
            self.data,
            {
                "path": "integer_data.csv",
                "target": "F",
                "quantileTransformNormal": ["D"],
            },
        )
        train_data, test_data = processor.quantile_transform_normal_columns()

        assert round(train_data["D"].min(), 5) == round(-train_data["D"].max(), 5)

    def test_handle_nan_no_missing_values(self):
        processor = create_dataset_processor(self.data, self.data_params)
        processor.handle_nan()

    def test_handle_nan_none_value(self):
        none_data = {"A": [1, 2, None], "B": ["value1", "value2", "value3"]}
        dataset_params = {
            "path": "none_data.csv",
            "target": "A",
            "nan": "drop",
        }

        processor1 = create_dataset_processor(none_data, dataset_params)
        processor2 = create_dataset_processor(none_data, dataset_params)
        processor1.handle_nan()

        with pytest.raises(ValueError):
            processor2.handle_nan(raise_exception=True)

    def test_handle_nan_empty_strings(self):
        none_data = {"A": [1, 2, 3], "B": ["", "value", "value"]}
        dataset_params = {
            "path": "none_data.csv",
            "target": "A",
            "nan": "drop",
        }

        processor1 = create_dataset_processor(none_data, dataset_params)
        processor2 = create_dataset_processor(none_data, dataset_params)
        processor1.handle_nan()

        with pytest.raises(ValueError):
            processor2.handle_nan(raise_exception=True)

    def test_handle_nan_dot_values(self):
        none_data = {"A": [1, 2, 3], "B": ["value", ".", "value"]}
        dataset_params = {
            "path": "none_data.csv",
            "target": "A",
            "nan": "drop",
        }

        processor1 = create_dataset_processor(none_data, dataset_params)
        processor2 = create_dataset_processor(none_data, dataset_params)
        processor1.handle_nan()

        with pytest.raises(ValueError):
            processor2.handle_nan(raise_exception=True)

    def test_multiple_missing_value_types(self):
        none_data = {"A": [1, 2, None], "B": ["", 3.5, "."], "C": [True, False, None]}
        dataset_params = {
            "path": "none_data.csv",
            "target": "A",
            "nan": "drop",
        }

        processor1 = create_dataset_processor(none_data, dataset_params)
        processor2 = create_dataset_processor(none_data, dataset_params)
        processor1.handle_nan()

        with pytest.raises(ValueError):
            processor2.handle_nan(raise_exception=True)

    def test_handle_nan_logging(self, caplog):
        none_data = {"A": [1, 2, None], "B": ["", 3.5, "."], "C": [True, False, None]}
        dataset_params = {
            "path": "none_data.csv",
            "target": "A",
            "nan": "drop",
        }

        processor = create_dataset_processor(none_data, dataset_params)
        processor.handle_nan()

        assert "Missing values found in data" in caplog.text

    def test_split_target(self):
        processor = create_dataset_processor(self.data, self.data_params)
        X_train, X_test, y_train, y_test = processor.split_target()

        assert len(X_train) + len(X_test) == len(self.data["A"])
        assert len(y_train) + len(y_test) == len(self.data["F"])
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_save_data_parquet(self):
        processor = create_dataset_processor(self.data, self.data_params)
        writer = ResultsWriter("save_testing")
        writer.create_directory()

        try:
            processor.save_data(writer)
            assert Path("save_testing/integer_data.parquet").exists()

            df = pd.read_parquet("save_testing/integer_data.parquet")
            assert df.equals(pd.DataFrame(self.data)) is True
        finally:
            shutil.rmtree("save_testing")

    def test_save_data_same_file_name(self):
        # Create two csv files in different parent directories with the same file name
        data_params1 = {"path": "test1/name_test.csv", "target": "C", "drop": ["A"]}
        data_params2 = {"path": "test2/name_test.csv", "target": "C", "drop": ["A"]}

        os.mkdir("test1")
        os.mkdir("test2")

        writer = ResultsWriter("name_save_testing")
        writer.create_directory()

        try:
            processor1 = create_dataset_processor(self.data, data_params1)
            processor2 = create_dataset_processor(self.data, data_params2)

            # Save the DataFrames to the same directory and check that they are saved with different names
            processor1.drop_columns()
            processor1.save_data(writer, overwrite=False)
            processor2.save_data(writer, overwrite=False)

            assert Path("name_save_testing/name_test.parquet").exists()
            assert Path("name_save_testing/name_test-1.parquet").exists()

        finally:
            shutil.rmtree("test1")
            shutil.rmtree("test2")
            shutil.rmtree("name_save_testing")

    def test_split_and_save_data(self):
        processor = create_dataset_processor(self.data, self.data_params)
        writer = ResultsWriter("save_testing")
        writer.create_directory()

        try:
            file_path = processor.split_and_save_data(writer)
            assert file_path.exists()

            X_train, X_test, y_train, y_test = load_split_data(file_path)
            assert isinstance(X_train, pd.DataFrame)
            assert isinstance(y_train, pd.Series)
        finally:
            shutil.rmtree("save_testing")

    def test_process_dataset(self):
        processor = create_dataset_processor(self.data, self.data_params)
        writer = ResultsWriter("save_testing")
        writer.create_directory()

        try:
            processor.process_dataset(writer)

            assert Path("save_testing/integer_data-original.parquet").exists()
            assert Path("save_testing/integer_data-processed.parquet").exists()
        finally:
            shutil.rmtree("save_testing")

    def test_process_dataset_invalid_save_original_type(self):
        processor = create_dataset_processor(self.data, self.data_params)

        with pytest.raises(TypeError):
            processor.process_dataset(save_directory="save_testing", save_original=123)

    def test_process_dataset_invalid_save_processed_type(self):
        processor = create_dataset_processor(self.data, self.data_params)

        with pytest.raises(TypeError):
            processor.process_dataset(save_directory="save_testing", save_processed=123)
