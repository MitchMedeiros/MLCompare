Release Notes
=============

This is the list of changes to MLCompare between each release. For full details,
see the `commit logs <https://github.com/MitchMedeiros/MLCompare/commits/>`_.

Version 1.1.0
-------------

- Refactored DatasetProcessor, moving save_directory from a class attribute to a method argument
- Added type validation to several methods within DatasetProcessor
- Updated docstrings for the dataset_processor module
- Updated unit tests for DatasetProcessor
- Added optimal device selection for PyTorch models as default behavior
- Corrected a logging issue with model processing

Version 1.0.1
-------------

- Updated the project versioning to dynamically use the version in mlcompare/__init__.py
- Modified the package attributes displayed on PyPi including adding links to documentation
- Added the link to the documentation to the library __init__
- Created a GitHub action for publishing newly tagged versions to PyPi

Version 1.0.0
-------------

Initial Release
