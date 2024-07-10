import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

import config
from models.model_validation import SklearnModel, XGBoostModel, validate_model_params

from data.dataset_processor import load_split_data, process_datasets
from data.dataset_validation import validate_dataset_params

# Add the project root directory to the system path for importing utils
root_dir = Path(__file__).resolve().parents[1].as_posix()
sys.path.append(root_dir)
import utils  # noqa: E402

logger = logging.getLogger(__name__)


def train_and_predict(
    models: list[SklearnModel | XGBoostModel],
    split_data_path: Path,
    save_results_path: Path,
) -> dict:
    """
    Train and perform predictions using a list of models and save their performance metrics to a file.
    Data can be provided as a single dataset or as a train-test split. If a single dataset is provided,
    the data will be split into training and testing sets. If both nonsplit_data and
    split_data are provided, split_data will be used.

    Args:
        models (list[SklearnModel | XGBoostModel]): A list of models to process.
        split_data_path (Path): The path to a pickle file containing a SplitData object.
        save_results_path (Path): The path to save the results of the model evaluation.
    """
    try:
        X_train, X_test, y_train, y_test = load_split_data(split_data_path)
    except FileNotFoundError:
        logger.error(
            f"No file or incorrect path when attempting to load split data from: {split_data_path}"
        )
        raise

    # y_test = y_test.to_numpy().ravel()
    # print(y_test)

    model_results_dict = {}
    for model in models:
        model.train(X_train, y_train)
        prediction = model.predict(X_test)
        results = model.evaluate(y_test, prediction)
        model_results_dict[model.__class__.__name__] = results

    return model_results_dict


def run_pipeline(
    dataset_params: list[dict[str, Any]] | Path | None = None,
    model_params: list[dict[str, Any]] | Path | None = None,
    custom_models: list[Any] | None = None,
    save_data: Literal["original", "cleaned", "both", "none"] = "both",
) -> None:
    """
    Executes a full pipeline for training and evaluating multiple models on multiple different datasets.
    To change the datasets or models used, simply modify the dictionary entries in the
    dataset_parameters.json and model_parameters.json files.

    Note that this can also be used to compare the performance of multiple models on a single dataset
    or the performance of just a single model across multiple datasets.
    """
    utils.setup_logging()

    saved_data_directory = config.SAVED_DATA_DIRECTORY
    model_directory = config.MODEL_DIRECTORY
    model_results_path = config.MODEL_RESULTS_PATH

    if dataset_params is None:
        dataset_params = config.DATA_PARAMS_PATH
    if model_params is None:
        model_params = config.MODEL_PARAMS_PATH

    datasets = validate_dataset_params(dataset_params)
    split_data_paths = process_datasets(datasets, saved_data_directory)

    validated_models = validate_model_params(model_params)
    for data_path in split_data_paths:
        model_results = train_and_predict(
            validated_models,
            data_path,
            model_directory / "model_results.json",
        )
        model_results_path.write_text(json.dumps(model_results, indent=4))


if __name__ == "__main__":
    run_pipeline()
