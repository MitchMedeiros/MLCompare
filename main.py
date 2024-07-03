import logging

import src.data.data_processing as data_processing
import src.training.random_forest as random_forest
import utils

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    utils.setup_logging()

    data_processing.process_kaggle_data()
    random_forest.train_and_save_model()
