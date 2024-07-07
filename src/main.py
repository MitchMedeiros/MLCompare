import json
import logging
import sys
from pathlib import Path

import config
import pandas as pd
from models.model_validation import SklearnModel, XGBoostModel, validate_model_params
from sklearn.model_selection import train_test_split

from data.data_processor import DataProcessor
from data.dataset_validation import KaggleDataset, LocalDataset, validate_dataset_params

# Add the project root directory to the system path for importing utils
root_dir = Path(__file__).resolve().parents[1].as_posix()
sys.path.append(root_dir)
import utils  # noqa: E402

logger = logging.getLogger(__name__)


def process_datasets(
    datasets: list[KaggleDataset | LocalDataset],
    save_directory: Path,
    file_format: str,
) -> None:
    """
    Downloads and processes data from multiple datasets.

    Args:
        dataset_params (dict[str, dict[str, Any]]): A dictionary containing parameters for each dataset.
            The keys are dataset names, and the values are dictionaries containing the following keys:
            - 'username': The Kaggle username of the dataset owner.
            - 'dataset_name': The name of the Kaggle dataset.
            - 'file_name': The name of the file to download from the dataset.
            - 'columns_to_drop' (optional): A list of column names to drop from the downloaded data.
            - 'columns_to_encode' (optional): A list of column names to encode using a specific encoding method.
        save_directory (Path): The directory where the processed data will be saved.
        file_format (str): The file format to use when saving the processed data.
    """
    for dataset in datasets:
        if isinstance(dataset, LocalDataset):
            processor = DataProcessor(dataset.file_path)

        elif isinstance(dataset, KaggleDataset):
            processor = DataProcessor()

            processor.download_kaggle_data(
                dataset.username,
                dataset.dataset_name,
                dataset.file_name,
            )

        else:
            raise ValueError(
                "Dataset must be an instance of KaggleDataset or LocalDataset."
            )

        raw_data_file_name = f"{dataset.dataset_name}.{file_format}"
        processed_data_file_name = f"{dataset.dataset_name}_cleaned.{file_format}"

        processor.save_data(save_directory / raw_data_file_name, file_format)
        processor.has_missing_values()
        processor.drop_columns(dataset.columns_to_drop)
        processor.encode_columns(dataset.columns_to_encode)
        processor.save_data(save_directory / processed_data_file_name, file_format)
        X_train, X_test, y_train, y_test = processor.split_data()


def train_predict_and_save(
    models: list[SklearnModel | XGBoostModel],
    save_results_path: Path,
    nonsplit_data=Path,
    split_data=None,
    test_size: float = 0.2,
) -> None:
    """
    Train and perform predictions using a list of models and save their performance metrics to a file.
    Data can be provided as a single dataset or as a train-test split. If a single dataset is provided,
    the data will be split into training and testing sets. If both nonsplit_data and
    split_data are provided, split_data will be used.

    Args:
        models (list[SklearnModel | XGBoostModel]): A list of models to process.
    """
    assert nonsplit_data is not None or split_data is not None, (
        "Either nonsplit_data or split_data must be provided. "
        "If both are provided, split_data will be used."
    )

    if split_data is None:
        df = pd.read_parquet(nonsplit_data)

        X_train, X_test, y_train, y_test = train_test_split(
            df, test_size=0.2, random_state=0
        )

    model_results_dict = {}

    for model in models:
        model.train(X_train, y_train)
        prediction = model.predict(X_test)
        results = model.evaluate(y_test, prediction)
        model_results_dict[model.__class__.__name__] = results

    save_results_path.write_text(json.dumps(model_results_dict, indent=4))


def main():
    utils.setup_logging()

    data_dir = Path(__file__).resolve().parent / "data"
    dataset_save_dir = data_dir / "saved_data"
    # dataset_params_path = data_dir / "dataset_parameters.json"
    # dataset_save_format = config.DATASET_SAVE_FORMAT

    # datasets = validate_dataset_params(dataset_params_path)
    # process_datasets(datasets, dataset_save_dir, dataset_save_format)

    ########################

    model_dir = Path(__file__).resolve().parent / "models"
    model_params_path = model_dir / "model_parameters.json"

    # Convert the parameters into a list of model objects
    models = validate_model_params(model_params_path)
    print(type(models[2]))

    # Train and evaluate the models
    train_predict_and_save(
        models,
        model_dir / "model_results.json",
        nonsplit_data=dataset_save_dir
        / "restaurant-revenue-prediction-dataset_cleaned.parquet",
    )


if __name__ == "__main__":
    main()
