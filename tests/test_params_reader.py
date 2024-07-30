import json
import logging
from pathlib import Path

import pytest
from mlcompare.params_reader import ParamsReader

logger = logging.getLogger("mlcompare.params_reader")


class TestParamsReader:
    valid_list = [
        {"param1": "value1"},
        {"param2": "value2"},
    ]
    invalid_json_content = {"param1": "value1"}

    @pytest.fixture
    def valid_str_path(self, tmpdir):
        file_path = tmpdir.join("dataset_params.json")
        file_path.write(json.dumps(self.valid_list))
        return str(file_path)

    @pytest.fixture
    def invalid_json_path(self, tmpdir):
        file_path = tmpdir.join("invalid_dataset_params.json")
        file_path.write(json.dumps(self.invalid_json_content))
        return str(file_path)

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

    def test_valid_str_path(self, valid_str_path):
        result = ParamsReader.read(valid_str_path)
        assert result == self.valid_list

    def test_valid_path(self, valid_str_path):
        result = ParamsReader.read(Path(valid_str_path))
        assert result == self.valid_list

    def test_invalid_str_path(self):
        with pytest.raises(FileNotFoundError):
            ParamsReader.read("an_invalid_path.json")

    def test_invalid_data_inside_json_file(self, invalid_json_path):
        with pytest.raises(AssertionError):
            ParamsReader.read(invalid_json_path)

    def test_non_json_file(self, tmpdir):
        non_json_path = tmpdir.join("dataset.csv")
        non_json_path.write("param1,param2\nvalue1,value2")
        with pytest.raises(json.JSONDecodeError):
            ParamsReader.read(str(non_json_path))
