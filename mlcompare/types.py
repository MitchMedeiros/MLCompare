from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias

import pandas as pd

from .data.datasets import KaggleDataset, LocalDataset
from .models.models import CustomModel, SklearnModel, XGBoostModel

DatasetType: TypeAlias = KaggleDataset | LocalDataset
SplitDataTuple: TypeAlias = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
]

DatasetConfig: TypeAlias = Path | list[dict[str, Any]]
ModelConfig: TypeAlias = Path | list[dict[str, Any]]

DataFileSuffix: TypeAlias = Literal["parquet", "csv", "json", "pickle"]

MLModelTypes: TypeAlias = SklearnModel | XGBoostModel | CustomModel
