from .data.dataset_processor import DatasetProcessor
from .data.datasets import DatasetFactory
from .data.split_data import SplitData, load_split_data
from .models.models import ModelFactory
from .params_reader import ParamsReader
from .pipelines import data_exploration_pipeline, data_pipeline, full_pipeline
from .processing import (
    evaluate_prediction,
    process_and_split_datasets,
    process_datasets,
    process_models,
)
from .results_writer import ResultsWriter

__version__ = "1.2.2"
__title__ = "MLCompare"
__author__ = "Mitchell Medeiros"
__license__ = "MIT"
__description__ = "Quickly compare machine learning models from different libraries across datasets."
__docformat__ = "restructuredtext"
__docs_url__ = "https://mlcompare.readthedocs.io/en/latest/api_reference"
__github_url__ = "https://github.com/MitchMedeiros/MLCompare"
__github_issues_url__ = "https://github.com/MitchMedeiros/MLCompare/issues"
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
    "ResultsWriter",
    "ParamsReader",
    "DatasetProcessor",
    "DatasetFactory",
    "ModelFactory",
    "evaluate_prediction",
    "process_models",
    "process_datasets",
    "process_and_split_datasets",
    "data_exploration_pipeline",
    "data_pipeline",
    "full_pipeline",
    "SplitData",
    "load_split_data",
]
