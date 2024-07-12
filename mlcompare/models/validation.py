from __future__ import annotations as _annotations

import json
import logging
from pathlib import Path

from ..data.split_data import load_split_data
from ..types import MLModelTypes, ModelConfig
from .models import CustomModel, SklearnModel, XGBoostModel, evaluate_prediction

logger = logging.getLogger(__name__)


def validate_model_params(model_config: ModelConfig) -> list[MLModelTypes]:
    if isinstance(model_config, Path):
        try:
            with open(model_config) as file:
                model_config = json.load(file)
        except FileNotFoundError as e:
            logger.error(f"Could not find file: {model_config}")
            raise e

    assert isinstance(
        model_config, list
    ), "model_config must be a list of dictionaries or a path to .json file containing one."

    assert all(
        isinstance(model, dict) for model in model_config
    ), "Each list element in `model_config` must be a dictionary."

    initialized_models: list[MLModelTypes] = []
    for model in model_config:
        if model["library"] in ["sklearn", "scikit-learn"]:
            initialized_models.append(SklearnModel(**model))
        elif model["library"] in ["xgboost", "xgb"]:
            initialized_models.append(XGBoostModel(**model))
        elif model["library"] == "custom":
            initialized_models.append(CustomModel(**model))
        else:
            raise ValueError(
                f"Library {model['library']} is not supported. Valid library names "
                "are: 'sklearn', 'xgboost', 'pytorch', or 'tensorflow'. If your model is not "
                "in one of these libraries use 'custom' and provide a value for 'custom_function' "
                "that takes in train-test split data and returns an nd.array or pd.Series of "
                "predictions. See the documentation for more details."
            )
    return initialized_models


def train_and_predict(models: list[MLModelTypes], split_data_path: Path) -> dict:
    """
    Train and perform predictions using a list of models and save their performance metrics to a file.
    Data can be provided as a single dataset or as a train-test split. If a single dataset is provided,
    the data will be split into training and testing sets. If both nonsplit_data and
    split_data are provided, split_data will be used.

    Args:
        models (list[MLModelTypes]): A list of models to process.
        split_data_path (Path): The path to a pickle file containing a SplitData object.
    """
    try:
        X_train, X_test, y_train, y_test = load_split_data(split_data_path)
    except FileNotFoundError:
        logger.error(
            f"No file or incorrect path when attempting to load split data from: {split_data_path}"
        )
        raise

    model_results_dict = {}
    for model in models:
        if isinstance(model, CustomModel):
            pass

        else:
            model.train(X_train, y_train)
            prediction = model.predict(X_test)
            results = evaluate_prediction(y_test, prediction)
            model_results_dict[model.__class__.__name__] = results

    return model_results_dict
