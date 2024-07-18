from __future__ import annotations as _annotations

import json
import logging
from pathlib import Path
from typing import Any, TypeAlias

logger = logging.getLogger(__name__)

ParamsInput: TypeAlias = str | Path | list[dict[str, Any]]
"""User input for pipelines, containing information to load and process datasets or to create ml models."""


class ParamsReader:
    """
    Reads and validates a list of parameters by calling `.read()`.

    Attributes:
    -----------
        params_list (list[dict[str, Any]] | str | Path]): List of dictionaries or a path to a JSON file.
    """

    @staticmethod
    def read(params_list: ParamsInput) -> list[dict[str, Any]]:
        """
        Reads and validates a list of parameters.

        Args:
        -----
            params_list (list[dict[str, Any]] | str | Path]): List of dictionaries or a path to a JSON file.

        Returns:
        --------
            list[dict[str, Any]]: List of dictionaries containing parameters.

        Raises:
        -------
            TypeError: If the params_list is not a list of dictionaries or a Path to a JSON file.
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If there is an error decoding the JSON file.
        """
        if isinstance(params_list, str):
            params_list = Path(params_list)

        if isinstance(params_list, Path):
            params_list = ParamsReader._load_params_from_file(params_list)

        ParamsReader._validate_params_list_type(params_list)
        return params_list

    @staticmethod
    def _load_params_from_file(file_path: Path) -> list[dict[str, Any]]:
        """
        Loads parameters from a JSON file.

        Args:
        -----
            file_path (Path): The path to the JSON file.

        Returns:
        --------
            List[dict[str, Any]]: A list of dictionaries containing parameters.

        Raises:
        -------
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If there is an error decoding the JSON file.
        """
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            logger.exception(f"Could not find file: {file_path}")
            raise
        except json.JSONDecodeError:
            logger.exception(f"Error decoding JSON from file: {file_path}")
            raise
        except Exception:
            logger.exception(f"Unexpected error loading params from file: {file_path}")
            raise

    @staticmethod
    def _validate_params_list_type(params_list: Any) -> None:
        assert isinstance(
            params_list, list
        ), "`params_list` must be a list of dictionaries or a path to .json file containing one."

        assert all(
            isinstance(params, dict) for params in params_list
        ), "Each list element of `params_list` must be a dictionary."
