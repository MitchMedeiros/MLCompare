from __future__ import annotations as _annotations

import logging
from pathlib import Path
from typing import Any

from .data.dataset_processor import process_datasets
from .models.models import process_models
from .params_reader import ParamsInput

logger = logging.getLogger(__name__)


def full_pipeline(
    dataset_params: ParamsInput,
    model_params: ParamsInput,
    custom_models: list[Any] | None = None,
    save_original_data: bool = True,
    save_cleaned_data: bool = True,
    save_directory: str | Path = Path("pipeline_results"),
) -> None:
    """
    Executes a full pipeline for training and evaluating multiple models on multiple different datasets.
    To change the datasets or models used, simply modify the dictionary entries in the
    dataset_parameters.json and model_parameters.json files.

    Note that this can also be used to compare the performance of multiple models on a single dataset
    or the performance of just a single model across multiple datasets.
    """
    if isinstance(save_directory, str):
        save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True)

    split_data = process_datasets(
        dataset_params,
        save_directory,
        save_original_data,
        save_cleaned_data,
    )
    for data in split_data:
        process_models(model_params, data, save_directory)
        # pass custom models here ^^^


def data_exploration_pipeline():
    pass
