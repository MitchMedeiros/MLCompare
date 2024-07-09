import json
import logging
import sys
from pathlib import Path

import config
from models.model_validation import SklearnModel, XGBoostModel, validate_model_params

from data.dataset_processor import load_split_data, process_datasets
from data.dataset_validation import validate_dataset_params

# Add the project root directory to the system path for importing utils
root_dir = Path(__file__).resolve().parents[1].as_posix()
sys.path.append(root_dir)
import utils  # noqa: E402

logger = logging.getLogger(__name__)


def train_predict_and_save(
    models: list[SklearnModel | XGBoostModel],
    split_data_path: Path,
    save_results_path: Path,
) -> None:
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

    save_results_path.write_text(json.dumps(model_results_dict, indent=4))


def full_pipeline(custom_models: list | None = None) -> None:
    """
    Executes a full pipeline for training and evaluating multiple models on multiple different datasets.
    To change the datasets or models used, simply modify the dictionary entries in the
    dataset_parameters.json and model_parameters.json files.

    Note that this can also be used to compare the performance of multiple models on a single dataset
    or the performance of just a single model across multiple datasets.
    """
    utils.setup_logging()

    saved_data_directory = config.SAVED_DATA_DIRECTORY
    dataset_params_path = config.DATA_PARAMS_PATH
    model_directory = config.MODEL_DIRECTORY
    model_params_path = config.MODEL_PARAMS_PATH

    datasets = validate_dataset_params(dataset_params_path)
    split_data_paths = process_datasets(datasets, saved_data_directory)

    # Convert the parameters into a list of model objects
    models = validate_model_params(model_params_path)

    for data_path in split_data_paths:
        train_predict_and_save(
            models, data_path, model_directory / "model_results.json"
        )


if __name__ == "__main__":
    full_pipeline()
