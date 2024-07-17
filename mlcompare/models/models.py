from __future__ import annotations as _annotations

import json
import logging
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Any, Generator, Literal, TypeAlias

from pydantic import BaseModel
from sklearn.metrics import mean_squared_error, r2_score

from ..params_reader import ParamsReader
from ..types import ParamsInput, SplitDataTuple

logger = logging.getLogger(__name__)

LibraryNames: TypeAlias = Literal[
    "sklearn",
    "scikit-learn",
    "skl",
    "xgboost",
    "xgb",
    "pytorch",
    "torch",
    "tensorflow",
    "tf",
]


class LibraryModel(ABC, BaseModel):
    name: str
    module: str | None = None
    params: dict | None = None
    _ml_model: Any = None

    def instantiate_model(
        self,
        library_name: Literal[
            "sklearn",
            "xgboost",
            "torch",
            "tensorflow",
        ],
    ) -> None:
        if self.module:
            full_import = f"{library_name}.{self.module}"
        else:
            full_import = library_name

        try:
            model_module = import_module(full_import)
        except ImportError:
            logger.error(f"Could not import module {full_import}")
            raise

        try:
            model_class = getattr(model_module, self.name)
        except AttributeError:
            logger.error(f"Could not find class {self.name} in module {self.module}")
            raise

        try:
            if self.params:
                ml_model = model_class(**self.params)
            else:
                ml_model = model_class()
        except Exception:
            logger.error(
                f"Could not initialize model {self.name} with params {self.params}"
            )
            raise

        self._ml_model = ml_model

    @abstractmethod
    def model_post_init(self, Any): ...

    @abstractmethod
    def train(self, X_train, y_train) -> None: ...

    @abstractmethod
    def predict(self, X_test): ...


class SklearnModel(LibraryModel):
    def model_post_init(self, Any):
        self.instantiate_model("sklearn")

    def train(self, X_train, y_train) -> None:
        self._ml_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._ml_model.predict(X_test)


class XGBoostModel(LibraryModel):
    def model_post_init(self, Any):
        self.instantiate_model("xgboost")

    def train(self, X_train, y_train) -> None:
        self._ml_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._ml_model.predict(X_test)


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


MLModelType: TypeAlias = SklearnModel | XGBoostModel


class ModelFactory:
    """
    Creates Model objects such as SklearnModel, XGBoostModel, etc. from a list of dictionaries.

    Attributes:
    -----------
        params_list (list[dict[str, Any]] | Path): List of dictionaries containing dataset parameters or a Path to
        a .json file with one. For a list of keys required in each dictionary, see below:

        Required keys:
        - `library` (Literal["sklearn", "xgboost", "pytorch", "tensorflow", "custom"]): The library to use.
        - `module` (str): The module containing the model class.
        - `name` (str): The name of the model class.

        Optional keys:
        - `params` (dict | None): The parameters to pass to the model class constructor

    Raises:
    -------
        AssertionError: If `dataset_params` is not a list of dictionaries or a path to a .json file containing one.
    """

    def __init__(self, params_list: ParamsInput) -> None:
        self.params_list = ParamsReader.read(params_list)

    def __iter__(
        self,
    ) -> Generator[MLModelType, None, None]:
        """
        Makes the class iterable, yielding dataset instances one by one.

        Yields:
        -------
            MLModelType: An instance of a LibraryModel child class.
        """
        for params in self.params_list:
            yield ModelFactory.create(**params)

    @staticmethod
    def create(library: LibraryNames, **kwargs) -> MLModelType:
        """
        Factory method to create a dataset instance based on the dataset type.

        Args:
        -----
            library (LibraryNames): The type of dataset to create.
            **kwargs: Arbitrary keyword arguments to be passed to the dataset class constructor.

        Returns:
        --------
            BaseDataset: An instance of a dataset class (KaggleDataset or LocalDataset).

        Raises:
        -------
            ValueError: If an unknown dataset type is provided.
        """
        # library = library.lower()

        match library:
            case "sklearn" | "scikit-learn" | "skl":
                return SklearnModel(**kwargs)
            case "xgboost" | "xgb":
                return XGBoostModel(**kwargs)
            case _:
                raise ValueError(
                    f"Library: {library} is not supported. Valid library names "
                    "are: 'sklearn', 'xgboost', 'pytorch', or 'tensorflow'. If your model is not "
                    "in one of these libraries use 'custom' and provide a value for 'custom_function' "
                    "that takes in train-test split data and returns an nd.array or pd.Series of "
                    "predictions. See the documentation for more details."
                )


def evaluate_prediction(y_test, y_pred) -> dict[str, float]:
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    return {"r2_score": r2, "rmse": rmse}


def append_model_results(results: dict[str, float], save_directory: Path) -> None:
    """
    Append the results of a model evaluation to a file.

    Args:
    -----
        results (dict[str, float]): The results of the model evaluation.
        save_directory (Path): The directory to save the results to.
    """
    file_path = save_directory / "model_results.json"

    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []

    if not isinstance(data, list):
        raise ValueError("The existing data in the JSON file is not a list")

    if isinstance(results, dict):
        data.append(results)
    else:
        raise ValueError("`results` should be a dictionary")

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def process_models(
    params_list: ParamsInput,
    split_data: SplitDataTuple,
    save_directory: Path,
) -> None:
    """
    Train and evaluate models on a dataset.

    Args:
    -----
        params_list (ParamsInput): A list of dictionaries containing model parameters.
        split_data (SplitDataTuple): A tuple containing the training and testing data.
        save_directory (Path): The directory to save the results to.

    Raises:
    -------
        Exception: If a model fails to process.
    """
    X_train, X_test, y_train, y_test = split_data

    models = ModelFactory(params_list)
    for model in models:
        try:
            model.train(X_train, y_train)
            prediction = model.predict(X_test)

            model_results = evaluate_prediction(y_test, prediction)
            append_model_results(model_results, save_directory)
        except Exception:
            logger.error("Failed to process model.")
            raise


# def train_and_predict(models: list[MLModelTypes], split_data_path: Path) -> dict:
#     """
#     Train and perform predictions using a list of models and save their performance metrics to a file.
#     Data can be provided as a single dataset or as a train-test split. If a single dataset is provided,
#     the data will be split into training and testing sets. If both nonsplit_data and
#     split_data are provided, split_data will be used.

#     Args:
#     -----
#         models (list[MLModelTypes]): A list of models to process.
#         split_data_path (Path): The path to a pickle file containing a SplitData object.

#     Returns:
#     --------
#         dict: A dictionary containing the performance metrics of each model.

#     Raises:
#     -------
#         FileNotFoundError: If the split_data_path does not exist.
#     """
#     try:
#         X_train, X_test, y_train, y_test = load_split_data(split_data_path)
#     except FileNotFoundError:
#         logger.error(
#             f"No file or incorrect path when attempting to load split data from: {split_data_path}"
#         )
#         raise

#     model_results_dict = {}
#     for model in models:
#         if isinstance(model, CustomModel):
#             pass

#         else:
#             model.train(X_train, y_train)
#             prediction = model.predict(X_test)
#             results = evaluate_prediction(y_test, prediction)
#             model_results_dict[model.__class__.__name__] = results

#     return model_results_dict
