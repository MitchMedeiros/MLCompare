import logging
from pathlib import Path

import pytest
from pydantic import ValidationError
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

from mlcompare.models.models import (
    LibraryModel,
    ModelFactory,
    SklearnModel,
    XGBoostModel,
)

logger = logging.getLogger("mlcompare.models.models")


# Abstract base class with an abstract method `validate_data` shouldn't be instantiable
class TestLibraryModel:
    def test_init(self):
        with pytest.raises(TypeError):
            LibraryModel(name="target")


# Minimal implementation of LibraryModel for testing
class LibraryModelChild(LibraryModel):
    def model_post_init(self, Any):
        pass

    def train(self, X_train, y_train) -> None:
        pass

    def predict(self, X_test):
        pass


class TestLibraryModelChild:
    def test_init(self):
        LibraryModelChild(
            name="aa",
            module="bb",
            params={"cc": "dd"},
        )

    def test_instantiate_sklearn_model(self):
        model = LibraryModelChild(
            module="ensemble",
            name="RandomForestClassifier",
        )
        model.instantiate_model("sklearn")
        assert isinstance(model._ml_model, RandomForestClassifier)

    def test_instantiate_xgboost_model(self):
        model = LibraryModelChild(
            module="",
            name="XGBRegressor",
        )
        model.instantiate_model("xgboost")
        assert isinstance(model._ml_model, XGBRegressor)
