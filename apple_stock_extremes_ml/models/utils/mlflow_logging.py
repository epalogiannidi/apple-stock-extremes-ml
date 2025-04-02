import os
from functools import wraps

import mlflow
import torch

from apple_stock_extremes_ml import logger


class MlflowLogging:
    """
    Defines all the decorators that can be used for logging
    model and experiment related information to the mlflow
    """

    @staticmethod
    def log_params(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if (
                not os.getenv("ENABLE_LOGGING", "False").lower() == "true"
            ):  # Skip logging
                return function(*args, **kwargs)
            """Log parameters"""
            result = function(*args, **kwargs)

            mlflow.log_params(result.params)

            return result

        return wrapper

    @staticmethod
    def log_ml_params(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if (
                not os.getenv("ENABLE_LOGGING", "False").lower() == "true"
            ):  # Skip logging
                return function(*args, **kwargs)
            """Log parameters"""
            result = function(*args, **kwargs)

            mlflow.log_params(args[0].params)

            return result

        return wrapper

    @staticmethod
    def log_ml_model(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if (
                not os.getenv("ENABLE_LOGGING", "False").lower() == "true"
            ):  # Skip logging
                return function(*args, **kwargs)
            """Log model"""
            result = function(*args, **kwargs)

            mlflow.sklearn.log_model(result.model, args[0].params["model_name"])

            logger.info(f"Model saved as {args[0].params['model_name']} in MLflow!")

        return wrapper

    @staticmethod
    def log_model(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if (
                not os.getenv("ENABLE_LOGGING", "False").lower() == "true"
            ):  # Skip logging
                return function(*args, **kwargs)
            """Log model"""
            result = function(*args, **kwargs)
            input_example = torch.randn(
                1, 6, 10
            )  # Example: (batch_size=1, num_features=6, sequence_length=10)

            mlflow.pytorch.log_model(
                result.model,
                args[0].params["model_name"],
                input_example=input_example.numpy(),
            )

            logger.info(f"Model saved as {args[0].params['model_name']} in MLflow!")

        return wrapper

    @staticmethod
    def log_metrics(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if (
                not os.getenv("ENABLE_LOGGING", "False").lower() == "true"
            ):  # Skip logging
                return function(*args, **kwargs)
            """Log parameters"""
            result = function(*args, **kwargs)

            mlflow.log_metrics(result)

            return result

        return wrapper

    @staticmethod
    def log_plot(plot_filename):
        if not os.getenv("ENABLE_LOGGING", "False").lower() == "true":  # Skip logging
            return
        mlflow.log_artifact(plot_filename, artifact_path="plots")
