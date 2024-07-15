from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd

from .data.datasets import (
    HuggingFaceDataset,
    KaggleDataset,
    LocalDataset,
    OpenMLDataset,
)
from .models.models import CustomModel, SklearnModel, XGBoostModel

DatasetType: TypeAlias = (
    LocalDataset | KaggleDataset | HuggingFaceDataset | OpenMLDataset
)
SplitDataTuple: TypeAlias = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
]

ParamsInput: TypeAlias = str | Path | list[dict[str, Any]]
"""asdf"""

MLModelTypes: TypeAlias = SklearnModel | XGBoostModel | CustomModel
