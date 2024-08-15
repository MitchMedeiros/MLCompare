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
"""A train-test split also split by features and target variable. Primarily used to both train and evaluate models."""


class SplitData(BaseModel):
    """
    Validates and stores train-test and feature-target split data.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame | pd.Series
    y_test: pd.DataFrame | pd.Series

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_split_data(load_path: str | Path) -> SplitDataTuple:
    """
    Loads a SplitData object from a pickle file and returns the data it was holding.

    Args:
    -----
        load_path (str | Path): Path to a pickle file contain a SplitData object.

    Returns:
    --------
        SplitDataTuple:
            pd.DataFrame: Training split features.
            pd.DataFrame: Testing split features.
            pd.DataFrame | pd.Series: Training split target values.
            pd.DataFrame | pd.Series: Testing split target values.
    """
    if not isinstance(load_path, (Path)):
        if not isinstance(load_path, str):
            raise ValueError("`load_path` must be a string or Path object.")
        else:
            load_path = Path(load_path)

    with open(load_path, "rb") as file:
        split_data = pickle.load(file)

    if not isinstance(split_data, SplitData):
        raise TypeError("Loaded data must be of type SplitData.")

    return split_data.X_train, split_data.X_test, split_data.y_train, split_data.y_test
