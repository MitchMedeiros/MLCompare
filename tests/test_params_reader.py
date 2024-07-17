import json
from pathlib import Path

import pytest

from mlcompare.params_reader import ParamsReader


class TestParamsReader:
    valid_str_path = "dataset_params.json"
    valid_path = Path(valid_str_path)
    valid_list = [
        {"param1": "value1"},
        {"param2": "value2"},
    ]
    invalid_json_path = "invalid_dataset_params.json"

    def test_valid_list(self):
        result = ParamsReader.read(self.valid_list)
        assert result == self.valid_list, "Should return the same list of dictionaries"

    def test_invalid_list(self):
        invalid_list = [
            {"param1": "value1"},
            "not_a_dict",
        ]
        with pytest.raises(AssertionError):
            ParamsReader.read(invalid_list)

    def test_invalid_input_type(self):
        invalid_input = 123
        with pytest.raises(AssertionError):
            ParamsReader.read(invalid_input)

    def test_valid_str_path(self):
        result = ParamsReader.read(self.valid_str_path)
        assert result == self.valid_list

    def test_valid_path(self):
        result = ParamsReader.read(self.valid_path)
        assert result == self.valid_list

    def test_invalid_str_path(self):
        with pytest.raises(FileNotFoundError):
            ParamsReader.read("an_invalid_path.json")

    def test_invalid_data_inside_json_file(self):
        with pytest.raises(AssertionError):
            ParamsReader.read(self.invalid_json_path)

    def test_non_json_file(self):
        with pytest.raises(json.JSONDecodeError):
            ParamsReader.read("local_dataset.csv")
