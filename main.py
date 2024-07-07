import logging

import utils
from src.models.model_validation import validate_models_from_dict

logger = logging.getLogger(__name__)


def main():
    ml_config = [
        {
            "library": "sklearn",
            "module": "sklearn.ensemble",
            "name": "RandomForestClassifie",
            "params": {
                "n_estimators": 100,
                "max_depth": 2,
                "random_state": 0,
            },
        }
    ]

    model = validate_models_from_dict(ml_config)
    print(type(model[0]))


if __name__ == "__main__":
    utils.setup_logging()
    main()
