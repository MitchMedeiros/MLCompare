from __future__ import annotations as _annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .params_reader import ParamsInput
from .processing import process_datasets, process_models
from .results_writer import ResultsWriter

logger = logging.getLogger(__name__)


def prepare_directory(save_directory: str | Path | None) -> Path:
    """
    Prepare the directory for saving results by creating it if it doesn't exist and removing past model results.

    Args:
    -----
        save_directory (str | Path | None): Directory to save results to. Uses "mlcompare-results-Y-m-dTH-M-S" if set to None.
    """
    if isinstance(save_directory, str) or isinstance(save_directory, Path):
        save_directory = Path(save_directory)

        if not save_directory.exists():
            save_directory.mkdir()
        else:
            model_results = save_directory / "model_results.json"
            if model_results.exists():
                model_results.unlink()
    elif save_directory is None:
        current_datetime = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]
        save_directory = Path(f"mlcompare-results-{current_datetime}")
    else:
        raise TypeError("save_directory must be a string or Path object.")

    return save_directory


def data_exploration_pipeline():
    pass


def data_pipeline(
    dataset_params: ParamsInput,
    save_original_data: bool = True,
    save_processed_data: bool = True,
    save_directory: str | Path | None = None,
) -> None:
    """
    A pipeline which only performs data retrieval and/or processing.

    Args:
    -----
        dataset_params (ParamsInput): Parameters for loading and processing datasets.
        save_original_data (bool, optional): Save original datasets. Defaults to True.
        save_processed_data (bool, optional): Save processed datasets. Defaults to True.
        save_directory (str | Path, optional): Directory to save results to. Defaults to "mlcompare-results-Y-m-dTH-M-S"
    """
    writer = ResultsWriter(save_directory)
    results_directory = writer.create_directory()

    split_data = process_datasets(
        dataset_params,
        results_directory,
        save_original_data,
        save_processed_data,
    )
    for data in split_data:
        pass


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
    A pipeline with data retrieval, processing, model training and model evaluation.

    Args:
    -----
        dataset_params (ParamsInput): List containing dataset information.
        model_params (ParamsInput): List containing model information.
        task_type (Literal["classification", "regression"]): Type of task to be performed.
        custom_models (list[Any], optional): List of custom models to include in the pipeline. Defaults to None.
        save_original_data (bool, optional): Save original datasets. Defaults to True.
        save_processed_data (bool, optional): Save processed datasets. Defaults to True.
        save_directory (str | Path, optional): Directory to save results to. Defaults to "mlcompare-results-Y-m-dTH-M-S"
    """
    writer = ResultsWriter(save_directory)
    results_directory = writer.create_directory()

    split_data = process_datasets(
        dataset_params,
        results_directory,
        save_original_data,
        save_processed_data,
    )
    for data in split_data:
        process_models(model_params, data, task_type, results_directory)
        # pass custom models here ^^^
