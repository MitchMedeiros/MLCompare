from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd

SplitDataTuple: TypeAlias = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
]
"""Data for training and testing."""

ParamsInput: TypeAlias = str | Path | list[dict[str, Any]]
"""User input for pipelines, containing information to load and process datasets or to create ml models."""
