import argparse
import os
from typing import Any, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd

from apple_stock_extremes_ml import app_config, logger
from apple_stock_extremes_ml.data_handler import DataHandler
from apple_stock_extremes_ml.models import RFPredictor
from apple_stock_extremes_ml.models.tcnn_tuning import TCNNHyperparameterTuner
from apple_stock_extremes_ml.train_tcnn import TCNNTrainer


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Perform tasks with predefined options."
    )

    # Add the `--tasks` argument with predefined choices
    parser.add_argument(
        "--task",
        choices=[
            "data-analysis",
            "train-rf",
            "train-tcnn",
            "end-to-end",
            "tune-tcnn",
            "predict-rf",
            "predict-tcnn",
        ],
        required=True,
        help="Specify the task to perform. Choices: rf, tcnn, delete, search.",
    )

    return parser.parse_args()


def train_tcnn(
    data_handler: DataHandler, normalized_data: pd.DataFrame, app_config: Dict[str, Any]
) -> None:
    """
    Wrapper function to train tcnn model and log results in MLFLOW

    :param data_handler: The claass instance to handle data
    :param normalized_data: The data after applying normalization
    :param app_config: The dictionary with the configuration values
    :return:
    """

    # Set experiment for TCNN
    mlflow.set_experiment("TCNN_Experiment")
    with mlflow.start_run():
        logger.info("Started TCNN Run")
        tcnn_trainer = TCNNTrainer(
            app_config, normalized_data, data_handler.modeling_data
        )

        # Log parameters, metrics or model specific to the TCNN here
        mlflow.log_param("model_type", "TCNN")

        # Train TCNN model
        tcnn_trainer.model_train()


def train_rf(data_handler: DataHandler, app_config: Dict[str, Any]) -> None:
    """
    Wrapper function to train RF model and log results in MLFLOW

    :param data_handler: The class instance to handle data
    :param app_config:  The dictionary with the configuration values
    :return:
    """
    mlflow.set_experiment("Random_Forest_Experiment")
    if mlflow.active_run() is None:
        with mlflow.start_run():
            logger.info("Started Random Forest Run")
            rf = RFPredictor(app_config, data_handler.modeling_data)

            # Log parameters, metrics or model specific to the Random Forest here
            mlflow.log_param("model_type", "Random Forest")

            # Train Random Forest model
            rf.train_rf()


def analyse_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_handler = DataHandler(app_config["data"])
    logger.info(data_handler)
    data_handler.check_quality()
    data_handler.define_extreme_events()
    data_handler.split_data()
    normalized_data = data_handler.normalize_data()

    splits = ["train", "val", "test"]
    columns = data_handler.modeling_data["train_data"].columns
    for s in splits:
        for c in columns:
            outliers = data_handler.detect_outliers(
                data_handler.modeling_data[f"{s}_data"], c[0]
            )
            if len(outliers) > 0:
                if (
                    s == "test" or c[0] == "Daily_Return"
                ):  # keep real world conditions in test
                    continue
                data_handler.handle_outliers(
                    data_handler.modeling_data[f"{s}_data"], outliers, c[0]
                )

        data_handler.data_statistics(
            data_handler.modeling_data[f"{s}_data"],
            data_handler.modeling_data[f"{s}_target"],
            s,
        )
    return data_handler, normalized_data


if __name__ == "__main__":
    task = setup_parser().task
    os.environ["ENABLE_LOGGING"] = "True"
    match task:
        case "data-analysis":
            analyse_data()
        case "train-tcnn":
            data_handler, normalized_data = analyse_data()
            train_tcnn(data_handler, normalized_data, app_config)

        case "train-rf":
            data_handler, normalized_data = analyse_data()
            train_rf(data_handler, app_config)

        case "end-to-end":
            data_handler, normalized_data = analyse_data()
            train_rf(data_handler, app_config)
            train_tcnn(data_handler, normalized_data, app_config)

        case "tune-tcnn":
            data_handler, normalized_data = analyse_data()
            os.environ["ENABLE_LOGGING"] = "False"
            tcnn_trainer = TCNNTrainer(
                app_config, normalized_data, data_handler.modeling_data
            )
            tuner = TCNNHyperparameterTuner(tcnn_trainer)
            tuner.tune(n_trials=3)
        case "predict-rf":
            data_handler, normalized_data = analyse_data()

            rf = RFPredictor(app_config, data_handler.modeling_data)
            model = rf.model_load(app_config["api_predict"]["rf_model"])
            x = np.array(
                data_handler.modeling_data["test_data"][0:10].T, dtype=np.float32
            ).reshape(1, 6, 10)

            result_rf = rf.rf_predict(model, x)
            logger.info(result_rf)
        case "predict-tcnn":
            # Predicts the class of the first element of the test dataset. Ideally the sample to
            # predict the class for has to be passed as query/parameter

            data_handler, normalized_data = analyse_data()

            tcnn_trainer = TCNNTrainer(
                app_config, normalized_data, data_handler.modeling_data
            )

            # load model
            model = tcnn_trainer.model_load(app_config["api_predict"]["tcnn_model"])
            x = np.array(
                data_handler.modeling_data["test_data"][0:10].T, dtype=np.float32
            ).reshape(1, 6, 10)
            result = tcnn_trainer.model_predict(model, x)
            logger.info(result)
        case _:
            print(
                "The supported tasks are the following:\n"
                "- data-analysis\n"
                "- train-tcnn\n"
                "- train-rf\n"
                "- end-to-end\n"
                "- tune-tcnn\n"
                "- predict-rf\n"
                "- predict-tcnn"
            )
