import logging
import os

import pandas as pd

from mlcompare.processing import process_datasets  # process_models
from mlcompare.results_writer import ResultsWriter

logger = logging.getLogger("mlcompare.processing")


def test_process_datasets():
    params_list = [
        {"type": "local", "path": "test1.csv", "target": "C", "drop": ["A"]},
        {"type": "local", "path": "test2.csv", "target": "F", "drop": ["A"]},
    ]

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    df.to_csv("test1.csv", index=False)
    df = pd.DataFrame({"A": [7, 8], "E": [9, 10], "F": [11, 12]})
    df.to_csv("test2.csv", index=False)

    writer = ResultsWriter("save_testing")
    writer.create_directory()

    try:
        split_datasets = process_datasets(params_list, writer, save_original=False, save_processed=False)

        for X_train, X_test, y_train, y_test in split_datasets:
            assert isinstance(X_train, pd.DataFrame)
            assert isinstance(X_test, pd.DataFrame)
            assert isinstance(y_train, pd.Series)
            assert isinstance(y_test, pd.Series)
            assert "A" not in X_train.columns
            assert X_train.empty is False
            assert X_test.empty is False
            assert y_train.empty is False
            assert y_test.empty is False

    finally:
        os.remove("test1.csv")
        os.remove("test2.csv")
