from pathlib import Path

SAVED_DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "saved_data"
DATA_PARAMS_PATH = Path(__file__).resolve().parent / "data" / "dataset_parameters.json"

MODEL_DIRECTORY = Path(__file__).resolve().parent / "models" / "saved_data"
MODEL_PARAMS_PATH = Path(__file__).resolve().parent / "models" / "model_parameters.json"
MODEL_RESULTS_PATH = MODEL_DIRECTORY / "model_results.json"
