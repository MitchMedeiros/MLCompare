import json
import logging.config
import sys


def setup_logging(config_file: str):
    """
    Setup logging configuration from a JSON file.

    config_file: Path to the logging configuration file.
    """
    with open(config_file, "r") as f:
        config = json.load(f)

    # Create loggers, handlers, and formatters from the configuration
    logging.config.dictConfig(config)

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


setup_logging(config_file="logging_config.json")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    pass
