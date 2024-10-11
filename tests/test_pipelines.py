import logging
import os
import shutil
from pathlib import Path

import pytest

from mlcompare import data_pipeline, full_pipeline

logger = logging.getLogger("mlcompare.pipelines")


@pytest.fixture(scope="function")
def setup_cleanup_directories(request):
    initial_dirs = set(os.listdir("."))

    def cleanup():
        current_dirs = set(os.listdir("."))
        new_dirs = current_dirs - initial_dirs

        for directory in new_dirs:
            if directory.startswith("mlc-2") or directory.startswith("mlcompare-results-"):
                shutil.rmtree(directory, ignore_errors=True)

    request.addfinalizer(cleanup)


def test_full_pipeline_regression(setup_cleanup_directories):
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
            "params": {"n_estimators": 10},
        },
        {
            "library": "xgboost",
            "name": "XGBRegressor",
            "params": {"n_estimators": 10},
        },
    ]

    full_pipeline(datasets, models, "regression")


def test_full_pipeline_classification(setup_cleanup_directories):
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
            "params": {"n_estimators": 10},
        },
        {
            "library": "xgboost",
            "name": "XGBClassifier",
            "params": {"n_estimators": 10},
        },
    ]

    full_pipeline(datasets, models, "classification")


def test_full_pipeline_custom_directory(setup_cleanup_directories):
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
    ]

    save_directory = Path("mlcompare-results-custom")
    full_pipeline(datasets, models, "classification", save_directory=save_directory)

    assert save_directory.exists()
    assert Path(
        save_directory / "iabhishekofficial_mobile-price-classification-original.parquet"
    ).exists()
    assert Path(
        save_directory / "iabhishekofficial_mobile-price-classification-processed.parquet"
    ).exists()


def test_data_pipeline(setup_cleanup_directories):
    datasets = [
        {
            "type": "kaggle",
            "user": "iabhishekofficial",
            "dataset": "mobile-price-classification",
            "file": "train.csv",
            "target": "price_range",
        }
    ]

    data_pipeline(datasets)
