import logging
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from mlcompare.data.split_data import SplitData, load_split_data

logger = logging.getLogger("mlcompare.data.split_data")


@pytest.fixture
def split_data_fixture():
    X_train = pd.DataFrame({"feature": [1, 2, 3]})
    X_test = pd.DataFrame({"feature": [4, 5]})
    y_train = pd.Series([1, 0, 1])
    y_test = pd.Series([0, 1])
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def test_split_data_initialization(split_data_fixture):
    assert isinstance(split_data_fixture, SplitData)
    assert isinstance(split_data_fixture.X_train, pd.DataFrame)
    assert isinstance(split_data_fixture.y_train, pd.Series)


def test_load_split_data_valid(split_data_fixture):
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "split_data.pkl"
        with open(file_path, "wb") as file:
            pickle.dump(split_data_fixture, file)

        loaded_X_train, loaded_X_test, loaded_y_train, loaded_y_test = load_split_data(
            file_path
        )
        assert isinstance(loaded_X_train, pd.DataFrame)
        assert isinstance(loaded_y_train, pd.Series)


def test_load_split_data_invalid_type():
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "invalid_data.pkl"
        with open(file_path, "wb") as file:
            pickle.dump({"not": "SplitData"}, file)

        with pytest.raises(TypeError):
            load_split_data(file_path)


def test_load_split_data_non_existent_file():
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "non_existent.pkl"
        with pytest.raises(FileNotFoundError):
            load_split_data(file_path)
