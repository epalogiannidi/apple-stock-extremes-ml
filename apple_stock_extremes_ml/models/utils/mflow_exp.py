import time
from abc import ABC, abstractmethod
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel


class MlflowExperiment(ABC):
    """
    Abstract class for defining mlflow experiments.
    It should be inherited by any model class that aims to use mlflow
    for tracking experiments. It defines the properties and the methods
    that the models should implement.

    Attributes
    ----------
    tracking_uri:
        The name of the directory to be created for storing the run information
    experiment_name:
        The name of the experiment
    experiment_description:
        A brief description for the experiment
    run description:
        A brief description for the run
    data:
        The data that are related to the experiment
    """

    def __init__(
        self,
        tracking_uri: str = "mlruns",
        experiment_name: str = f"exp-{time.strftime('%m%d%Y-%H:%M:%S')}",
        experiment_description: str = "",
        run_description: str = "",
    ):
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._experiment_description = experiment_description
        self._run_description = run_description

        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        mlflow.set_experiment_tag("mlflow.note.content", self._experiment_description)
        mlflow.set_tag("mlflow.note.content", self._run_description)

        self._artifact_uri = mlflow.get_artifact_uri()

    @property
    def experiment_name(self):
        return self._experiment_name

    @experiment_name.setter
    def experiment_name(self, experiment_name):
        if not isinstance(experiment_name, str):
            raise TypeError("Experiment name should be a string")
        self._experiment_name = experiment_name
        mlflow.set_experiment(self._experiment_name)

    @property
    def artifact_uri(self):
        return self._artifact_uri

    @property
    def experiment_description(self):
        return self._experiment_description

    @experiment_description.setter
    def experiment_description(self, description):
        if not isinstance(description, str):
            raise TypeError("Description should be a string")
        self._experiment_description = description

    @property
    def run_description(self):
        return self._run_description

    @run_description.setter
    def run_description(self, run_description):
        if not isinstance(run_description, str):
            raise TypeError("Description should be a string")
        self._run_description = run_description

    @property
    def tracking_uri(self):
        return self._tracking_uri

    @tracking_uri.setter
    def tracking_uri(self, tracking_uri):
        if not isinstance(tracking_uri, str):
            raise TypeError("Tracking uri should be a string.")
        self._tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self._tracking_uri)

    @abstractmethod
    def model_train(self):
        raise NotImplementedError("Not implemented error.")

    @abstractmethod
    def model_evaluate(
        self, data: pd.DataFrame, labels: pd.DataFrame, epoch: int, label: str
    ):
        raise NotImplementedError("Not implemented error.")

    @abstractmethod
    def model_predict(self, model: PyFuncModel, x: np.ndarray):
        raise NotImplementedError("Not implemented error.")

    @abstractmethod
    def model_save(self):
        raise NotImplementedError("Not implemented error.")

    @staticmethod
    @abstractmethod
    def model_load(model: str):
        raise NotImplementedError("Not implemented error.")

    @staticmethod
    def model_end_run():
        """
        Ends the currently active run
        """
        mlflow.end_run()

    @staticmethod
    def model_start_run(description: str):
        """
        Start a new mlflow run
        Parameters
        ----------
        description: str
            The run description
        """
        mlflow.start_run(description=description)

    @staticmethod
    def get_info() -> Dict[str, str]:
        """
        Retrieves the information of the active run
        Returns
        -------
        The dictionary with the information of the active run
        """
        run = mlflow.active_run()
        if run:
            run_info = {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "tracking_uri": mlflow.get_tracking_uri(),
            }
        else:
            run_info = {"tracking_uri": mlflow.get_tracking_uri()}
        return run_info
