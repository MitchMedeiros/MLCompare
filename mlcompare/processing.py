from __future__ import annotations as _annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Generator, Literal

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from .data.dataset_processor import DatasetProcessor
from .data.datasets import DatasetFactory
from .data.split_data import SplitData, SplitDataTuple
from .models.models import ModelFactory
from .params_reader import ParamsInput
from .results_writer import ResultsWriter

logger = logging.getLogger(__name__)


def process_datasets(
    params_list: ParamsInput,
    save_directory: str | Path,
    save_original: bool = True,
    save_processed: bool = True,
) -> Generator[SplitDataTuple, None, None]:
    """
    Downloads and processes data from multiple datasets that have been validated.

    Args:
    -----
        params_list (ParamsInput): List of dictionaries containing dataset parameters.
        save_directory (str | Path): Directory to save the data to.
        save_original (bool): Whether to save the original data.
        save_processed (bool): Whether to save the processed, nonsplit data.

    Returns:
    --------
        A Generator containing the split data for input into subsequent pipeline steps via iteration.
    """
    datasets = DatasetFactory(params_list)
    for dataset in datasets:
        try:
            processor = DatasetProcessor(dataset)
            split_data = processor.process_dataset(save_directory, save_original, save_processed)
            yield split_data
        except Exception:
            logger.error("Failed to process dataset.")
            raise


def process_datasets_to_files(
    params_list: ParamsInput,
    save_directory: str | Path,
    save_original: bool = True,
    save_processed: bool = True,
) -> list[Path]:
    """
    Downloads and processes data from multiple datasets that have been validated.

    Args:
    -----
        datasets (list[KaggleDataset | LocalDataset]): List of datasets to process.
        data_directory (str | Path): Directory to save the original and processed data.
        save_original (bool): Whether to save the original data.
        save_processed (bool): Whether to save the processed, nonsplit data.

    Returns:
    --------
        list[Path]: List of paths to the saved split data for input into subsequent pipeline steps.
    """
    writer = ResultsWriter(save_directory)
    save_directory = writer.create_directory()

    split_data_paths = []
    datasets = DatasetFactory(params_list)
    for dataset in datasets:
        try:
            processor = DatasetProcessor(dataset)
            X_train, X_test, y_train, y_test = processor.process_dataset(
                save_directory, save_original, save_processed
            )

            file_path = save_directory / f"{processor.save_name}-split.pkl"
            split_data_paths.append(file_path)
        except Exception:
            logger.error("Failed to process dataset.")
            raise

        split_data_obj = SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        with open(file_path, "wb") as file:
            pickle.dump(split_data_obj, file)
        logger.info(f"Split data saved to: {file_path}")

    return split_data_paths


def evaluate_prediction(
    y_test,
    y_pred,
    model_name: str,
    task_type: Literal["classification", "regression"],
    data_split: Literal["train", "test"] = "test",
) -> dict[str, Any]:
    """
    Evaluate the predictions of a model using several metrics from sklearn.metrics.

    Args:
    -----
        y_test: True target values.
        y_pred: Predicted target values.
        model_name (str): Name of the model.
        task_type (Literal["classification", "regression"]): Type of data the model is making predictions for.
        data_split (Literal["train", "test"]): Data split used for evaluation.

    Returns:
    --------
        dict[str, Any]: A dictionary containing the evaluation metrics.
    """
    if task_type not in ["classification", "regression"]:
        raise ValueError("Task type must be one of 'classification' or 'regression'.")

    if task_type == "regression":
        determined_task_type = "regression"
    else:
        if y_test.dropna().nunique() <= 2:
            determined_task_type = "binary"
        else:
            determined_task_type = "multiclass"

    match determined_task_type:
        case "binary":
            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)

            return {
                "model": model_name,
                "data split": data_split,
                "accuracy": accuracy,  # balanced classification
                "balanced accuracy": balanced_accuracy,  # imbalanced classification
                "F1": f1,  # imbalanced classification
                "recall": recall,  # classification focused on minimizing false negatives
                "precision": precision,  # classification focused on minimizing false positives
            }
        case "multiclass":
            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            f1_macro = f1_score(y_test, y_pred, average="macro")
            recall_weighted = recall_score(y_test, y_pred, average="weighted")
            recall_macro = recall_score(y_test, y_pred, average="macro")
            precision_weighted = precision_score(y_test, y_pred, average="weighted")
            precision_macro = precision_score(y_test, y_pred, average="macro")

            return {
                "model": model_name,
                "data split": data_split,
                "accuracy": accuracy,
                "balanced accuracy": balanced_accuracy,
                "F1 weighted-average": f1_weighted,
                "F1 macro-average": f1_macro,
                "recall weighted-average": recall_weighted,
                "recall macro-average": recall_macro,
                "precision weighted-average": precision_weighted,
                "precision macro-average": precision_macro,
            }
        case "regression":
            r2 = r2_score(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)

            return {
                "model": model_name,
                "data split": data_split,
                "R2": r2,
                "RMSE": rmse,
            }
        case _:
            raise ValueError("Task type must be one of 'binary', 'multiclass', or 'regression'.")


def append_json(results: dict[str, float], save_directory: str | Path) -> None:
    """
    Append the results of model evaluation to a JSON file.

    Args:
    -----
        results (dict[str, float]): Results of the model evaluation.
    """
    writer = ResultsWriter(save_directory)
    save_directory = writer.create_directory()
    file_path = save_directory / "model_results.json"

    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    except json.JSONDecodeError:
        data = []

    if not isinstance(data, list):
        raise ValueError("The existing data in the JSON file is not a list.")

    if isinstance(results, dict):
        data.append(results)
    else:
        raise TypeError("`results` should be a dictionary.")

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def process_models(
    params_list: ParamsInput,
    split_data: SplitDataTuple,
    task_type: Literal["classification", "regression"],
    save_directory: str | Path,
) -> None:
    """
    Train and evaluate models on a dataset.

    Args:
    -----
        params_list (ParamsInput): List of dictionaries containing model parameters.
        split_data (SplitDataTuple): Tuple containing the training and testing data split by features and target.
        task_type (Literal["classification", "regression"]): Type of data the model is making predictions for.

    Raises:
    -------
        Exception: If a model fails to process.
    """
    X_train, X_test, y_train, y_test = split_data

    models = ModelFactory(params_list)
    for model in models:
        try:
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            model_results = evaluate_prediction(
                y_test,
                y_pred,
                model._ml_model.__class__.__name__,
                task_type,
            )
            append_json(model_results, save_directory)
        except Exception:
            logger.error(f"Failed to process model: {model._ml_model.__class__.__name__}")
            raise
