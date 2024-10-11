## v1.2.2 (2024-08-02)

[GitHub release](https://github.com/MitchMedeiros/MLCompare/tag/v1.2.2)

### Pipelines
- Added the ability to save all or only the most accurate model with `full_pipeline`

### Files
- Changed the default naming for the directory where data, models, and results are saved
- Cleaned up the implementation of directory creation using ResultsWriter

## v1.2.0 (2024-08-02)

[GitHub release](https://github.com/MitchMedeiros/MLCompare/tag/v1.2.0)

### Pipelines
- Created a `data_pipeline` function for performing only data retrieval and processing
- Expanded the generated model performance metrics and added a required argument to `full_pipeline` for specifying whether the pipeline is being used for regression or classification tasks

### DatasetProcessor
- Refactored the class to store the train-test split data for easier processing
- Added a `handle_nan` method which can drop, forward-fill, and backward-fill missing values
- Added label encoding ordinal encoding, and target encoding methods
- Added several scaling and transformation methods from sklearn: StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, and Normalize

### Documentation
- Created a new homepage
- Updated the layout of the API Reference page
- Added content to the Release Notes page
- Improved various docstrings
- Made multiple updates to the README including adding a "Planned Additions" section

### Other
- Added a `ResultsWriter` class, responsible for directory and file naming and creation throughout pipelines
    - Implemented directory and file name incrementing to prevent overwrites
    - Changed the default directory name to use the current timestamp to ensure uniqueness
    - Improved how saving model results is handled
- Removed the `DataProcessor` class in favor of pipelines
- Migrated several high-level functions being used within pipelines to a new module: `processing.py`
- Improved unit test coverage

## v1.1.0 (2024-08-02)

[GitHub release](https://github.com/MitchMedeiros/MLCompare/tag/v1.1.0)

- Refactored DatasetProcessor, moving save_directory from a class attribute to a method argument
- Added type validation to several methods within DatasetProcessor
- Updated docstrings for the dataset_processor module
- Updated unit tests for DatasetProcessor
- Added optimal device selection for PyTorch models as default behavior
- Corrected a logging issue with model processing

## v1.0.1 (2024-07-31)

[GitHub release](https://github.com/MitchMedeiros/MLCompare/tag/v1.0.1)

- Updated the project versioning to dynamically use the version in mlcompare/__init__.py
- Modified the package attributes displayed on PyPi including adding links to documentation
- Added the link to the documentation to the library __init__
- Created a GitHub action for publishing newly tagged versions to PyPi

## v1.0.0 (2024-07-31)

[GitHub release](https://github.com/MitchMedeiros/MLCompare/tag/v1.0.0)

Initial Release
