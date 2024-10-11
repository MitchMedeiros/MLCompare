from __future__ import annotations as _annotations

import logging
from pathlib import Path
from typing import Literal

from .params_reader import ParamsInput
from .processing import process_datasets, process_models
from .results_writer import ResultsWriter

logger = logging.getLogger(__name__)


def data_exploration_pipeline():
    pass


def data_pipeline(
    dataset_params: ParamsInput,
    save_original: bool = True,
    save_processed: bool = True,
    save_directory: str | Path | None = None,
) -> None:
    """
    A pipeline which only performs data retrieval and/or processing.

    Args:
    -----
        dataset_params (ParamsInput): Parameters for loading and processing datasets.
        save_original (bool, optional): Save original datasets. Defaults to True.
        save_processed (bool, optional): Save processed datasets. Defaults to True.
        save_directory (str | Path, optional): Directory to save results to. Defaults to "mlcompare-results-Y-m-dTH-M-S"
    """
    writer = ResultsWriter(save_directory)
    writer.create_directory()

    split_data = process_datasets(dataset_params, writer, save_original, save_processed)
    for data in split_data:
        pass


def full_pipeline(
    dataset_params: ParamsInput,
    model_params: ParamsInput,
    task_type: Literal["classification", "regression"],
    save_models: Literal["all", "best", "none"] = "none",
    save_original: bool = True,
    save_processed: bool = True,
    save_directory: str | Path | None = None,
) -> None:
    """
    A pipeline with data retrieval, processing, model training and model evaluation.

    Args:
    -----
        dataset_params (ParamsInput): List containing dataset information.
        model_params (ParamsInput): List containing model information.
        task_type (Literal["classification", "regression"]): Type of machine learning task to be performed.
        save_models (Literal["all", "best", "none"], optional): Save all models, only the best model, or no models. Defaults to "none".
        save_original (bool, optional): Save original datasets. Defaults to True.
        save_processed (bool, optional): Save processed datasets. Defaults to True.
        save_directory (str | Path, optional): Directory to save data, models, and results to. Defaults to "mlc-y-m-DTH-M-S-MS".
    """
    writer = ResultsWriter(save_directory)
    writer.create_directory()

    split_data = process_datasets(dataset_params, writer, save_original, save_processed)
    for data in split_data:
        process_models(model_params, data, writer, task_type, save_models)
