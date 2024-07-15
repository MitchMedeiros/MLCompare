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

ParamsList: TypeAlias = list[dict[str, Any]]

ParamsInput: TypeAlias = str | Path | ParamsList
"""str | Path | list[dict[str, Any]] - User inputs for pipelines"""
