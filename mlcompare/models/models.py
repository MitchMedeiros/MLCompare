from __future__ import annotations as _annotations

import logging
from importlib import import_module
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

SklearnLibraryNames: TypeAlias = Literal["sklearn", "scikit-learn"]
XGBoostLibraryNames: TypeAlias = Literal["xgboost", "xgb"]
PytorchLibraryNames: TypeAlias = Literal["pytorch", "torch"]
TensorflowLibraryNames: TypeAlias = Literal["tensorflow", "tf"]
LibraryNames: TypeAlias = (
    SklearnLibraryNames
    | XGBoostLibraryNames
    | PytorchLibraryNames
    | TensorflowLibraryNames
)
CustomNames: TypeAlias = Literal["custom"]


class MLModel(BaseModel):
    def evaluate(self, y_test, y_pred) -> dict[str, float]:
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return {"r2_score": r2, "rmse": rmse}


class CustomModel(MLModel):
    library: CustomNames
    custom_function: Any


class LibraryModel(MLModel):
    library: LibraryNames
    module: str
    name: str
    params: dict | None = None

    def model_post_init(self, Any) -> None:
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

        self.initialized_model = initialized_model


class SklearnModel(LibraryModel):
    def train(self, X_train, y_train) -> None:
        self.initialized_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.initialized_model.predict(X_test)


class XGBoostModel(LibraryModel):
    def train(self, X_train, y_train) -> None:
        self.initialized_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.initialized_model.predict(X_test)


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
