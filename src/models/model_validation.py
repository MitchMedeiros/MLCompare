import logging
from importlib import import_module
from typing import Any, Literal

from pydantic import BaseModel, PrivateAttr, ValidationError
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class CustomModel(BaseModel):
    library: Literal["custom"]
    custom_function: Any


class LibraryModel(BaseModel):
    library: Literal[
        "sklearn", "scikit-learn", "xgboost", "xgb", "pytorch", "torch", "tensorflow"
    ]
    module: str
    name: str
    params: dict[str, Any] | None = None
    _initialized_model: Any = PrivateAttr(None)

    def __init__(self, **data: Any):
        try:
            super().__init__(**data)
        except ValidationError as e:
            logger.error(
                "Could not validate model configuration. Please check the supplied dictionary contains \
            'library', 'module', and 'name' fields for each model."
            )
            raise e

        self._init_model()

    def _init_model(self):
        try:
            model_module = import_module(f"{self.module}")
        except ImportError as e:
            logger.error(f"Could not import module {self.module}")
            raise e

        try:
            model_class = getattr(model_module, self.name)
        except AttributeError as e:
            logger.error(f"Could not find class {self.name} in module {self.module}")
            raise e

        try:
            initialized_model = model_class(**self.params)
        except Exception as e:
            logger.error(
                f"Could not initialize model {self.name} with params {self.params}"
            )
            raise e

        self._initialized_model = initialized_model

    def evaluate(y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return {"r2_score": r2, "rmse": rmse}


class SklearnModel(LibraryModel):
    def train(self, X_train, y_train):
        self._initialized_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._initialized_model.predict(X_test)


class XGBoostModel(LibraryModel):
    def train(self, X_train, y_train):
        self._initialized_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._initialized_model.predict(X_test)


# class PytorchModel(LibraryModel):
#     def train(self, X_train, y_train):
#         self.model.fit(X_train, y_train)

#     def predict(self, X_test):
#         return self.model.predict(X_test)


# class TensorflowModel(LibraryModel):
#     def train(self, X_train, y_train):
#         self.model.fit(X_train, y_train)

#     def predict(self, X_test):
#         return self.model.predict(X_test)


def validate_model_params(
    model_config: list,
) -> list[SklearnModel | XGBoostModel | CustomModel]:
    initialized_models: list[SklearnModel | XGBoostModel | CustomModel] = []

    for model_values in model_config:
        if model_values["library"] in ["sklearn", "scikit-learn"]:
            skl_model = SklearnModel(**model_values)
            initialized_models.append(skl_model)

        elif model_values["library"] in ["xgboost", "xgb"]:
            xgb_model = XGBoostModel(**model_values)
            initialized_models.append(xgb_model)

        else:
            raise ValueError(
                f"Library {model_values['library']} is not implemented. Valid library names \
            are: 'sklearn', 'xgboost', 'pytorch', or 'tensorflow'. If your model is not \
            in one of these libraries use custom and provide a custom_function that takes in \
            train-test split data and returns a prediction nd.array or pd.Series."
            )

    return initialized_models
