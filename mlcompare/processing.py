from __future__ import annotations as _annotations

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
    writer: ResultsWriter,
    save_original: bool = True,
    save_processed: bool = True,
) -> Generator[SplitDataTuple, None, None]:
    """
    Downloads and processes data from multiple datasets that have been validated.

    Args:
    -----
        params_list (ParamsInput): List of dictionaries containing dataset parameters.
        writer: ResultsWriter for save directory management.
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
            split_data = processor.process_dataset(writer, save_original, save_processed)
            yield split_data
        except Exception:
            logger.error("Failed to process dataset.")
            raise


def process_and_split_datasets(
    params_list: ParamsInput,
    writer: ResultsWriter,
    save_original: bool = True,
    save_processed: bool = True,
) -> list[Path]:
    """
    Downloads and processes data from multiple datasets that have been validated.

    Args:
    -----
        datasets (list[KaggleDataset | LocalDataset]): List of datasets to process.
        writer: ResultsWriter for save directory management.
        save_original (bool): Whether to save the original data.
        save_processed (bool): Whether to save the processed, nonsplit data.

    Returns:
    --------
        list[Path]: List of paths to the saved split data for input into subsequent pipeline steps.
    """
    split_data_paths = []

    datasets = DatasetFactory(params_list)
    for dataset in datasets:
        try:
            processor = DatasetProcessor(dataset)
            X_train, X_test, y_train, y_test = processor.process_dataset(
                writer, save_original, save_processed
            )

            file_path = writer.directory_name / f"{processor.save_name}-split.pkl"
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
                "accuracy": accuracy,
                "balanced accuracy": balanced_accuracy,
                "F1": f1,
                "recall": recall,
                "precision": precision,
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


def process_models(
    params_list: ParamsInput,
    split_data: SplitDataTuple,
    writer: ResultsWriter,
    task_type: Literal["classification", "regression"],
    save_models: Literal["all", "best", "none"] = "none",
) -> None:
    """
    Train and evaluate models on a dataset.

    Args:
    -----
        params_list (ParamsInput): List of dictionaries containing model parameters.
        split_data (SplitDataTuple): Tuple containing the training and testing data split by features and target.
        task_type (Literal["classification", "regression"]): Type of data the model is making predictions for.
        writer: ResultsWriter for save directory management.
        save_models (Literal["all", "best", "none"]): Save all models, only the best model, or none.

    Raises:
    -------
        Exception: If a model fails to process.
    """
    X_train, X_test, y_train, y_test = split_data

    best_accuracy = 0
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
            writer.append_model_results(model_results)

            if save_models == "all":
                model.save(writer.directory_name)
            elif save_models == "best":
                if model_results["accuracy"] > best_accuracy:
                    for file in writer.directory_name.glob("*.pkl"):
                        if file.root in model_results["model"]:
                            file.unlink()

                    model.save(writer.directory_name)
                    best_accuracy = model_results["accuracy"]
        except Exception:
            logger.error(f"Failed to process model: {model._ml_model.__class__.__name__}")
            raise


def load_model():
    """
    Loads a trained model from a pickle file,
    """
    pass
