from .data.data_processor import DataProcessor
from .data.dataset_processor import DatasetProcessor
from .data.datasets import DatasetFactory
from .data.split_data import SplitData, load_split_data
from .models.models import ModelFactory, process_models
from .params_reader import ParamsReader
from .pipelines import data_exploration_pipeline, full_pipeline

__version__ = "1.0.0"
__title__ = "MLCompare"
__author__ = "Mitchell Medeiros"
__license__ = "MIT"
__description__ = (
    "Quickly compare machine learning models from different libraries across datasets."
)
__docformat__ = "restructuredtext"
__docs_url__ = ""
__github_url__ = "https://github.com/MitchMedeiros/mlcompare"
__github_issues_url__ = "https://github.com/MitchMedeiros/mlcompare/issues"
__doc__ = f"""
MLCompare is a Python library for running comparison pipelines, focused on being straight-forward and 
flexible. It supports multiple popular ML libraries, can perform dataset retrieval, processing, and 
results visualization. It also support providing your own models and model pipelines. See the docs:
{__docs_url__} for more info."
"""
__keywords__ = [
    "machine learning",
    "ML",
    "AI",
    "model comparison",
    "model evaluation",
    "ML pipeline",
    "machine learning automation",
]

__all__ = [
    "DataProcessor",
    "DatasetProcessor",
    "DatasetFactory",
    "ModelFactory",
    "process_models",
    "ParamsReader",
    "full_pipeline",
    "SplitData",
    "load_split_data",
    "data_exploration_pipeline",
]
