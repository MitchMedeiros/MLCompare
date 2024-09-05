from __future__ import annotations as _annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ResultsWriter:
    def __init__(self, directory_name: str | Path | None = None) -> None:
        if isinstance(directory_name, str):
            directory_name = Path(directory_name)
        elif isinstance(directory_name, (Path, type(None))):
            pass
        else:
            raise TypeError("save_directory must be a string or Path object.")

        self.directory_name = directory_name

    def increment_name(self, name: str | Path) -> Path:
        """
        Increments the name the final Path component for a file or directory until it is unique.

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
        if self.directory_name is not None:
            model_results = self.directory_name / "model_results.json"
            if model_results.exists():
                model_results.unlink()
        else:
            raise ValueError("A directory name has not been set.")

    def generate_default_directory(self) -> Path:
        current_datetime = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]
        default_name = Path(f"mlcompare-results-{current_datetime}")
        return default_name

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
        if self.directory_name is None:
            self.directory_name = self.generate_default_directory()

        if overwrite is False:
            incremented_directory = self.increment_name(self.directory_name)
            incremented_directory.mkdir()
            self.directory_name = incremented_directory
        else:
            self.directory_name.mkdir(exist_ok=True)
            self.clear_model_results()

        return self.directory_name
