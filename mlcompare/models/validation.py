from __future__ import annotations as _annotations

import json
import logging
from pathlib import Path

from ..types import MLModelTypes, ModelConfig
from .models import CustomModel, SklearnModel, XGBoostModel

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
