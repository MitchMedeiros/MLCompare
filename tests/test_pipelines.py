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
            "robustScale": [
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
                "Number of Reviews",
            ],
        },
    ]

    models = [
        {
            "library": "sklearn",
            "name": "LinearRegression",
        },
        {
            "library": "sklearn",
            "name": "RandomForestRegressor",
            "params": {"n_estimators": 100},
        },
        {
            "library": "xgboost",
            "name": "XGBRegressor",
            "params": {
                "n_estimators": 1000,
                "max_depth": 7,
                "eta": 0.1,
                "subsample": 0.7,
                "colsample_bytree": 0.8,
            },
        },
    ]

    full_pipeline(datasets, models, "regression")


def test_full_pipeline_classification():
    datasets = [
        {
            "type": "kaggle",
            "user": "iabhishekofficial",
            "dataset": "mobile-price-classification",
            "file": "train.csv",
            "target": "price_range",
        }
    ]

    models = [
        {
            "library": "sklearn",
            "name": "LinearSVC",
        },
        {
            "library": "sklearn",
            "name": "RandomForestClassifier",
            "params": {"n_estimators": 100},
        },
        {
            "library": "xgboost",
            "name": "XGBClassifier",
        },
    ]

    full_pipeline(datasets, models, "classification")


# def test_data_pipeline():
#     # data_pipeline()
#     pass
