from __future__ import annotations as _annotations

import logging
import pickle
from pathlib import Path
from typing import TypeAlias

import pandas as pd
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

SplitDataTuple: TypeAlias = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
]
"""Data for training and testing."""


class SplitData(BaseModel):
    """
    Validates and holds the split data from `sklearn.model_selection.train_test_split`.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame | pd.Series
    y_test: pd.DataFrame | pd.Series

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_split_data(load_path: Path) -> SplitDataTuple:
    """
    Loads a SplitData object from a pickle file and returns the data it was holding.

    Args:
    -----
        load_path (Path): Path to a pickle file of a SplitData object.

    Returns:
    --------
        SplitDataTuple: Tuple of length 4 containing the training and testing data split by features and target.
    """
    with open(load_path, "rb") as file:
        split_data = pickle.load(file)

    if not isinstance(split_data, SplitData):
        raise TypeError("Loaded data must be of type SplitData.")

    return split_data.X_train, split_data.X_test, split_data.y_train, split_data.y_test
