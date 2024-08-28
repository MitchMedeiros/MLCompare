import logging
from pathlib import Path

import pytest

from mlcompare import data_pipeline, full_pipeline

logger = logging.getLogger("mlcompare")


def test_full_pipeline_regression():
    datasets = [
        {
            "type": "kaggle",
            "user": "anthonytherrien",
            "dataset": "restaurant-revenue-prediction-dataset",
            "file": "restaurant_data.csv",
            "target": "Revenue",
            "drop": ["Name"],
            "oneHotEncode": ["Location", "Cuisine", "Parking Availability"],
            "standardScale": [
                "Seating Capacity",
                "Average Meal Price",
                "Rating",
                "Marketing Budget",
                "Social Media Followers",
                "Chef Experience Years",
                "Avg Review Length",
                "Ambience Score",
                "Service Quality Score",
                "Weekend Reservations",
                "Weekday Reservations",
                "Revenue",
            ],
        },
        # {
        #     "type": "kaggle",
        #     "user": "anthonytherrien",
        #     "dataset": "restaurant-revenue-prediction-dataset",
        #     "file": "restaurant_data.csv",
        #     "target": "Revenue",
        #     "drop": ["Name"],
        #     "oneHotEncode": ["Location", "Cuisine", "Parking Availability"],
        # },
    ]

    models = [
        {
            "library": "sklearn",
            "name": "LinearRegression",
        },
        # {
        #     "library": "sklearn",
        #     "module": "linear_model",
        #     "name": "LinearRegression",
        # },
        # {
        #     "library": "sklearn",
        #     "name": "RandomForestRegressor",
        # },
        # {
        #     "library": "sklearn",
        #     "module": "ensemble",
        #     "name": "RandomForestRegressor",
        #     "params": {"n_estimators": 100},
        # },
        # {
        #     "library": "xgboost",
        #     "name": "XGBRegressor",
        # },
        # {
        #     "library": "xgboost",
        #     "module": "sklearn",
        #     "name": "XGBRegressor",
        #     "params": {"n_estimators": 100},
        # },
    ]

    full_pipeline(datasets, models, "regression")


def test_full_pipeline_classification():
    datasets = [
        {
            "type": "kaggle",
            "user": "anthonytherrien",
            "dataset": "restaurant-revenue-prediction-dataset",
            "file": "restaurant_data.csv",
            "target": "Revenue",
            "drop": ["Name"],
            "oneHotEncode": ["Location", "Cuisine", "Parking Availability"],
        }
    ]

    models = [
        {
            "library": "sklearn",
            "module": "sklearn.linear_model",
            "name": "LinearRegression",
            "params": {},
        },
        {
            "library": "sklearn",
            "module": "sklearn.ensemble",
            "name": "RandomForestRegressor",
            "params": {"n_estimators": 100},
        },
        {
            "library": "xgboost",
            "module": "xgboost",
            "name": "XGBRegressor",
            "params": {},
        },
    ]

    # full_pipeline(datasets, models, "classification")


def test_data_pipeline():
    # data_pipeline()
    pass
