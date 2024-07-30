import logging
from typing import Literal

import pandas as pd
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
    _library: Literal["sklearn", "xgboost", "torch", "tensorflow"]

    def train(self, X_train, y_train) -> None:
        pass

    def predict(self, X_test):
        pass


class TestLibraryModelChild:
    def test_init(self):
        LibraryModelChild(name="a", module="b", params={"c": "d"})

    def test_init_with_nones(self):
        LibraryModelChild(name="a", module=None, params=None)

    def test_init_with_empty_values(self):
        LibraryModelChild(name="a", module="", params={})

    def test_no_name(self):
        with pytest.raises(ValidationError):
            LibraryModelChild()

    def test_invalid_name_type(self):
        with pytest.raises(ValidationError):
            LibraryModelChild(name=123)

    def test_invalid_module_type(self):
        with pytest.raises(ValidationError):
            LibraryModelChild(name="a", module=123)

    def test_invalid_params_type(self):
        with pytest.raises(ValidationError):
            LibraryModelChild(name="a", params=123)

    # # More general test of child class instantiation
    # def test_instantiate_sklearn_model(self):
    #     model = LibraryModelChild(
    #         module="ensemble",
    #         name="RandomForestClassifier",
    #         _library="sklearn",
    #     )
    #     model.instantiate_model()
    #     assert isinstance(model._ml_model, RandomForestClassifier)

    # # More general test of child class instantiation
    # def test_instantiate_xgboost_model(self):
    #     model = LibraryModelChild(
    #         module="",
    #         name="XGBRegressor",
    #         _library="xgboost",
    #     )
    #     model.instantiate_model()
    #     assert isinstance(model._ml_model, XGBRegressor)


class TestSklearnModel:
    X_train_df = pd.DataFrame({"A": [1, 2, 4, 6]})
    y_train_sr = pd.Series([1, 2, 3, 4])
    X_test_df = pd.DataFrame({"A": [10, 20, 40, 60]})
    y_test_sr = pd.Series([10, 20, 30, 40])

    model_module = "ensemble"
    model_name = "RandomForestClassifier"
    model_type = RandomForestClassifier

    def test_init(self):
        skl_model = SklearnModel(
            name=self.model_name,
            module=self.model_module,
            params={"n_estimators": 50, "max_depth": 5, "n_jobs": -1},
        )
        assert isinstance(skl_model._ml_model, self.model_type)
        assert skl_model._ml_model.n_estimators == 50
        assert skl_model._ml_model.max_depth == 5

    def test_init_no_module(self):
        skl_model = SklearnModel(
            name=self.model_name,
            params={"n_estimators": 50, "max_depth": 5, "n_jobs": -1},
        )
        assert isinstance(skl_model._ml_model, self.model_type)
        assert skl_model._ml_model.n_estimators == 50
        assert skl_model._ml_model.max_depth == 5

    def test_invalid_module(self):
        with pytest.raises(ModuleNotFoundError):
            SklearnModel(
                name=self.model_name,
                module="wrong_name",
                params={"n_estimators": 50, "max_depth": 5, "n_jobs": -1},
            )

    def test_train(self):
        model = SklearnModel(
            name=self.model_name,
            module=self.model_module,
            params={},
        )
        model.train(self.X_train_df, self.y_train_sr)

    def test_train_predict(self):
        model = SklearnModel(
            name=self.model_name,
            module=self.model_module,
            params={},
        )
        model.train(self.X_train_df, self.y_train_sr)
        model.predict(self.X_test_df)


class TestXGBoostModel:
    X_train_df = pd.DataFrame({"A": [1, 2, 4, 6]})
    y_train_sr = pd.Series([1, 2, 3, 4])
    X_test_df = pd.DataFrame({"A": [10, 20, 40, 60]})
    y_test_sr = pd.Series([10, 20, 30, 40])

    model_module = ""
    model_name = "XGBRegressor"
    model_type = XGBRegressor

    def test_init(self):
        xgb_model = XGBoostModel(
            name=self.model_name,
            module=self.model_module,
            params={"n_estimators": 50, "max_depth": 5},
        )
        assert isinstance(xgb_model._ml_model, self.model_type)
        assert xgb_model._ml_model.n_estimators == 50
        assert xgb_model._ml_model.max_depth == 5

    def test_invalid_module(self):
        with pytest.raises(ModuleNotFoundError):
            XGBoostModel(
                name=self.model_name,
                module="wrong_name",
                params={"n_estimators": 50, "max_depth": 5},
            )

    def test_train(self):
        model = XGBoostModel(
            name=self.model_name,
            module=self.model_module,
            params={},
        )
        model.train(self.X_train_df, self.y_train_sr)

    def test_train_predict(self):
        model = XGBoostModel(
            name=self.model_name,
            module=self.model_module,
            params={},
        )
        model.train(self.X_train_df, self.y_train_sr)
        model.predict(self.X_test_df)


class TestModelFactory:
    X_train_df = pd.DataFrame({"A": [1, 2, 4, 6]})
    y_train_sr = pd.Series([1, 2, 3, 4])
    X_test_df = pd.DataFrame({"A": [10, 20, 40, 60]})
    y_test_sr = pd.Series([10, 20, 30, 40])

    skl_params_list = [
        {
            "library": "sklearn",
            "module": "ensemble",
            "name": "RandomForestClassifier",
            "params": {},
        }
    ]
    xgb_params_list = [
        {
            "library": "xgboost",
            "name": "XGBRegressor",
            "params": {},
        },
    ]
    mixed_params_list = [
        {
            "library": "sklearn",
            "module": "ensemble",
            "name": "RandomForestClassifier",
        },
        {
            "library": "xgboost",
            "name": "XGBRegressor",
        },
    ]

    def test_create_skl_model_directly(self):
        model_factory = ModelFactory(self.skl_params_list)
        model_factory.create(**self.skl_params_list[0])

    def test_create_xgb_model_directly(self):
        model_factory = ModelFactory(self.xgb_params_list)
        model_factory.create(**self.xgb_params_list[0])

    def test_iter_skl_model(self):
        model_count = 0
        model_factory = ModelFactory(self.skl_params_list)

        for model in model_factory:
            model_count += 1
            assert isinstance(model, SklearnModel)
        assert model_count == len(self.skl_params_list)

    def test_iter_xgb_model(self):
        model_count = 0
        model_factory = ModelFactory(self.xgb_params_list)

        for model in model_factory:
            model_count += 1
            assert isinstance(model, XGBoostModel)
        assert model_count == len(self.xgb_params_list)

    def test_iter_mixed_libraries(self):
        model_count = 0
        model_factory = ModelFactory(self.mixed_params_list)

        for model in model_factory:
            model_count += 1
            assert isinstance(model, (SklearnModel, XGBoostModel))
        assert model_count == len(self.mixed_params_list)

    def test_iter_train_mixed_libraries(self):
        model_factory = ModelFactory(self.mixed_params_list)

        for model in model_factory:
            model.train(self.X_train_df, self.y_train_sr)

    def test_iter_train_predict_mixed_libraries(self):
        model_factory = ModelFactory(self.mixed_params_list)

        for model in model_factory:
            model.train(self.X_train_df, self.y_train_sr)
            model.predict(self.X_test_df)

    def test_invalid_library(self):
        with pytest.raises(ValueError):
            model_factory = ModelFactory(
                [
                    {
                        "library": "asdf",
                        "module": "ensemble",
                        "name": "RandomForestClassifier",
                    }
                ]
            )
            for model in model_factory:
                pass

    def test_invalid_library_type(self):
        with pytest.raises(AssertionError):
            model_factory = ModelFactory(
                [
                    {
                        "library": 123,
                        "module": "ensemble",
                        "name": "RandomForestClassifier",
                    }
                ]
            )

            for model in model_factory:
                pass
