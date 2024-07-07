import json
import logging
import sys
from pathlib import Path

import config
from models.model_validation import SklearnModel, XGBoostModel, validate_model_params

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


def process_models(
    models: list[SklearnModel | XGBoostModel], data, split_data, results_path: Path
):
    """
    Processes a list of models by training and evaluating them.

    Args:
        models (list[SklearnModel | XGBoostModel]): A list of models to process.
    """
    model_results_dict = {}
    for model in models:
        model.train()
        prediction = model.predict()
        results = model.evaluate(prediction)
        model_results_dict[model.__class__.__name__] = results

    results_path.write_text(json.dumps(model_results_dict, indent=4))


def main(model_params):
    utils.setup_logging()

    data_dir = Path(__file__).parent.resolve() / "data"
    dataset_save_dir = data_dir / "saved_data"
    dataset_params_path = data_dir / "dataset_parameters.json"
    dataset_save_format = config.DATASET_SAVE_FORMAT

    datasets = validate_dataset_params(dataset_params_path)
    process_datasets(datasets, dataset_save_dir, dataset_save_format)

    models = validate_model_params(model_params)
    process_models(models)


if __name__ == "__main__":
    model_params = {}

    main(model_params)
