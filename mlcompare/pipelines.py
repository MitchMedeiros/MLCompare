from __future__ import annotations as _annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .data.dataset_processor import process_datasets
from .models.models import process_models
from .params_reader import ParamsInput

logger = logging.getLogger(__name__)


def prepare_files(save_directory: str | Path = Path("pipeline_results")):
    """
    Prepare the directory for saving results by creating it if it doesn't exist and removing past model results.

    Args:
    -----
        save_directory (str | Path, optional): Directory to save results to. Defaults to Path("pipeline_results").
    """
    if isinstance(save_directory, str):
        save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True)

    model_results_file = save_directory / "model_results.json"
    if model_results_file.exists():
        model_results_file.unlink()


def data_exploration_pipeline():
    pass


def data_pipeline(
    dataset_params: ParamsInput,
    save_original_data: bool = True,
    save_cleaned_data: bool = True,
    save_directory: str | Path = Path("pipeline_results"),
) -> None:
    """
    A pipeline for data retrieval and/or processing without any model training or evaluation.

    Args:
    -----
        dataset_params (ParamsInput): Parameters for loading and processing datasets.
        save_original_data (bool, optional): Save original datasets. Defaults to True.
        save_cleaned_data (bool, optional): Save cleaned datasets. Defaults to True.
        save_directory (str | Path, optional): Directory to save results to. Defaults to Path("pipeline_results").
    """
    if isinstance(save_directory, str):
        save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True)

    process_datasets(dataset_params, save_directory, save_original_data, save_cleaned_data)


def full_pipeline(
    dataset_params: ParamsInput,
    model_params: ParamsInput,
    task_type: Literal["classification", "regression"],
    custom_models: list[Any] | None = None,
    save_original_data: bool = True,
    save_processed_data: bool = True,
    save_directory: str | Path | None = None,
) -> None:
    """
    Executes a full pipeline for training and evaluating multiple models on multiple datasets.

    Args:
    -----
        dataset_params (ParamsInput): List containing dataset information.
        model_params (ParamsInput): List containing model information.
        task_type (Literal["classification", "regression"]): Type of task to be performed.
        custom_models (list[Any], optional): List of custom models to include in the pipeline. Defaults to None.
        save_original_data (bool, optional): Save original datasets. Defaults to True.
        save_processed_data (bool, optional): Save cleaned datasets. Defaults to True.
        save_directory (str | Path, optional): Directory to save results to. Defaults to Path("pipeline_results").
    """
    if save_directory is None:
        current_datetime = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_directory = Path(f"mlcompare-results-{current_datetime}")
    # prepare_files(save_directory)

    split_data = process_datasets(
        dataset_params, save_directory, save_original_data, save_processed_data
    )
    for data in split_data:
        process_models(model_params, data, task_type, save_directory)
        # pass custom models here ^^^
