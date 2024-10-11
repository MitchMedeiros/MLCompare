from __future__ import annotations as _annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ResultsWriter:
    def __init__(self, directory_name: str | Path | None = None) -> None:
        if isinstance(directory_name, type(None)):
            directory_name = self.generate_default_directory()
        elif isinstance(directory_name, str):
            directory_name = Path(directory_name)
        elif not isinstance(directory_name, Path):
            raise TypeError("save_directory must be a string or Path object.")

        self.directory_name = directory_name

    @staticmethod
    def generate_default_directory() -> Path:
        current_datetime = datetime.now().strftime("%y-%m-%dT%H-%M-%S-%f")[:-4]
        default_name = Path(f"mlc-{current_datetime}")
        return default_name

    @staticmethod
    def increment_name(name: str | Path) -> Path:
        """
        Increments the final Path component for a file or directory until it is unique.

        Args:
        -----
            name (str | Path): Name of the file or directory to increment.

        Returns:
        --------
            Path: Incremented Path for a file or directory.
        """
        if isinstance(name, str):
            name = Path(name)
        if not isinstance(name, Path):
            raise TypeError("name must be a string or Path object.")

        count = 1
        while name.exists():
            name = name.with_name(f"{name.stem}-{count}{name.suffix}")
            count += 1
        return name

    def clear_model_results(self) -> None:
        model_results = self.directory_name / "model_results.json"
        if model_results.exists():
            model_results.unlink()

    def create_directory(self, overwrite: bool = False) -> Path:
        """
        Create a directory to save results to and returns the directory path.

        Args:
        -----
            overwrite (bool, optional): Overwrite the directory if it already exists. Defaults to False.

        Returns:
        --------
            Path: Path to the directory created.

        """
        if overwrite is False:
            incremented_directory = self.increment_name(self.directory_name)
            incremented_directory.mkdir()
            self.directory_name = incremented_directory
        else:
            self.directory_name.mkdir(exist_ok=True)
            self.clear_model_results()

        return self.directory_name

    def append_model_results(self, results: dict[str, float]) -> None:
        """
        Append the results of model evaluation to a JSON file.

        Args:
        -----
            results (dict[str, float]): Results of the model evaluation.
        """
        if self.directory_name is None:
            self.directory_name = self.generate_default_directory()

        file_path = self.directory_name / "model_results.json"

        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            data = []
        except json.JSONDecodeError:
            data = []

        if not isinstance(data, list):
            raise ValueError("The existing data in the model_results.json file is not a list.")

        if isinstance(results, dict):
            data.append(results)
        else:
            raise TypeError("The list elements in model_results.json are not dictionaries.")

        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
        logger.info(f"Model results appended to: {file_path}")
