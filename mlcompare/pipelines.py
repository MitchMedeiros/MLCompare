from __future__ import annotations as _annotations

import logging
from pathlib import Path
from typing import Any, Literal

from .data.dataset_processor import process_datasets
from .models.models import evaluate_models
from .types import ParamsInput

logger = logging.getLogger(__name__)


def run_pipeline(
    dataset_params: ParamsInput,
    model_params: ParamsInput,
    custom_models: list[Any] | None = None,
    save_datasets: Literal["original", "cleaned", "both", "none"] = "both",
    save_directory: str | Path = Path("run_pipeline_results"),
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

    split_data = process_datasets(dataset_params, save_directory)
    for data in split_data:
        evaluate_models(model_params, split_data, save_directory)  # type: ignore


# def file_based_pipeline(
#     dataset_params: ParamsInput,
#     model_params: ParamsInput,
#     custom_models: list[Any] | None = None,
#     save_datasets: Literal["original", "cleaned", "both", "none"] = "both",
#     save_directory: str | Path = Path("run_pipeline_results"),
# ) -> None:
#     """
#     Executes a full pipeline for training and evaluating multiple models on multiple different datasets.
#     To change the datasets or models used, simply modify the dictionary entries in the
#     dataset_parameters.json and model_parameters.json files.

#     Note that this can also be used to compare the performance of multiple models on a single dataset
#     or the performance of just a single model across multiple datasets.
#     """
#     if isinstance(save_directory, str):
#         save_directory = Path(save_directory)
#     save_directory.mkdir(exist_ok=True)

#     split_data_paths = process_datasets(dataset_params, save_directory)
#     models = validate_model_params(model_params)

#     model_results_path = save_directory / "model_results.json"
#     for data_path in split_data_paths:
#         model_results = train_and_predict(models, data_path)
#         model_results_path.write_text(json.dumps(model_results, indent=4))
