from __future__ import annotations as _annotations

from typing import TypeAlias

import pandas as pd

from .data.datasets import KaggleDataset, LocalDataset

DatasetType: TypeAlias = KaggleDataset | LocalDataset
SplitDataTuple: TypeAlias = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
]
