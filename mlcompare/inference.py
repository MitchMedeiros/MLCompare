from __future__ import annotations as _annotations

import logging

logger = logging.getLogger(__name__)


def inference_function():
    pass


# def inference_pipeline(
#     dataset_params: ParamsInput,
#     model_paths: list[str | Path],
#     save_original: bool = True,
#     save_processed: bool = True,
#     save_directory: str | Path | None = None,
# ) -> None:
#     """
#     Perform data processing and model inference using one or more pretrained models.

#     Args:
#     -----
#         dataset_params (ParamsInput): List containing dataset information.
#         model_params (ParamsInput): List containing model information.
#         save_original (bool, optional): Save original datasets. Defaults to True.
#         save_processed (bool, optional): Save processed datasets. Defaults to True.
#         save_directory (str | Path, optional): Directory to save results to. Defaults to "mlcompare-results-Y-m-dTH-M-S"
#     """
#     writer = ResultsWriter(save_directory)
#     writer.create_directory()

#     split_data = process_datasets(dataset_params, writer, save_original, save_processed)
#     for data in split_data:
#         for path in model_paths:
#             with open(path, "rb") as file:
#                 model = pickle.load(file)
#                 prediction = model.predict(data)
#                 print(type(prediction))
#                 print(prediction)
# with open(results_directory / "inference_results.csv", "w") as file:
# writer.write_results(prediction)
