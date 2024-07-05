import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def load_model(model_path: Path) -> RandomForestRegressor:
    """
    Load a pretrained model from a pickle file for testing.

    model_path: Path to the pickle file.
    """
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


def load_presplit_data(
    data_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data that has been split into train and test intervals from a pickle file.

    data_path: Path to the pickle file.
    """
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    X_train, X_test, y_train, y_test = (
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"],
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    logger.info("Model trained.")

    return model


def test_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[float, float]:
    prediction = model.predict(X_test)

    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)
    logger.info(f"Model Mean Squared Error: {mse}")
    logger.info(f"Model R2 Score: {r2}")

    return mse, r2


def save_model(model: RandomForestRegressor, model_path: Path) -> None:
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    logger.info(f"Model saved to: {model_path}")


def train_and_save_model():
    src_path = Path(__file__).parent.parent.resolve()
    data_path = src_path / "data" / "saved_data" / "train_test_split.pkl"
    model_path = src_path / "models" / "random_forest_regressor.pkl"

    X_train, X_test, y_train, y_test = load_presplit_data(data_path)

    model = train_model(X_train, y_train)
    test_model(model, X_test, y_test)
    save_model(model, model_path)
