import os

from .datasets import (
    BaseDataset,
    DatasetFactory,
    HuggingFaceDataset,
    KaggleDataset,
    LocalDataset,
    OpenMLDataset,
)
from .split_data import SplitData

__all__ = [
    "BaseDataset",
    "DatasetFactory",
    "HuggingFaceDataset",
    "KaggleDataset",
    "LocalDataset",
    "OpenMLDataset",
    "SplitData",
]

# Set empty environment variables for downloading data from Kaggle
# To avoid importing the library throwing an error
os.environ["KAGGLE_USERNAME"] = ""
os.environ["KAGGLE_KEY"] = ""
