import json
import logging.config
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
default_config_path = Path(__file__).parent / "log_conf.json"


def setup_logging(config_file: Path = default_config_path) -> None:
    """
    Setup logging configuration from a JSON file.

    config_file: Path to the logging configuration file.
    """
    with open(config_file, "r") as file:
        config = json.load(file)

    # Create loggers, handlers, and formatters from the configuration
    logging.config.dictConfig(config)
    logger.info("Logging configuration loaded.")

    # Function to log uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger = logging.getLogger("root")
        logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception
    logger.info("Uncaught exception handler set.")


setup_logging()
