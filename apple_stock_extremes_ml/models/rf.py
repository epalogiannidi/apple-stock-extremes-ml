from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
from mlflow.pyfunc import PyFuncModel
from sklearn.ensemble import RandomForestClassifier

from apple_stock_extremes_ml.models.utils import (Evaluator, MlflowLogging,
                                                  ModelTrainReturnObject)


class RFPredictor:
    """
    Random Forest Predictor

    """

    def __init__(self, config: Dict[str, Any], modeling_data: Dict[str, pd.DataFrame]):
        self.history = config["modeling"]["history"]
        self.ticker = config["data"]["ticker"]
        self.train_data, self.train_target = self.prepare_data(
            modeling_data["train_data"], modeling_data["train_target"]
        )
        self.val_data, self.val_target = self.prepare_data(
            modeling_data["val_data"], modeling_data["val_target"]
        )
        self.test_data, self.test_target = self.prepare_data(
            modeling_data["test_data"], modeling_data["test_target"]
        )
        self.params = config["modeling"]["rf"]
        self.classes = config["data"]["classes"]
        self.classes_weights = {0: 1, 1: 1.1}

        if self.params["class_weights"]:
            self.rf = RandomForestClassifier(
                n_estimators=self.params["trees"],
                random_state=42,
                class_weight=self.classes_weights,
            )
        else:
            self.rf = RandomForestClassifier(
                n_estimators=self.params["trees"], random_state=42
            )

    def __repr__(self):
        return (
            f"{__name__} is a Random Forest Predictor that uses features "
            f"for the past 10 days to predict if an extreme event happens in the 11th day."
        )

    def prepare_data(
        self, train_features: pd.DataFrame, train_labels: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepares the data so that they are in the appropriate format for training the classifier

        :param train_features: The training dataset
        :param train_labels: The target labels of the training dataset
        :return: The training dataset and target labels in the approapriate format
        """
        features = pd.DataFrame()
        for i in range(1, self.history + 1):
            shifted_df = train_features.shift(i)
            shifted_df.columns = [f"{col}_{i}" for col in train_features.columns]
            features = pd.concat([features, shifted_df], axis=1)

        # Drop the rows with NaN values (because the first 10 rows won't have full data)
        features.dropna(inplace=True)
        return features, train_labels.iloc[self.history :]

    @MlflowLogging.log_ml_params
    @MlflowLogging.log_ml_model
    def train_rf(self) -> ModelTrainReturnObject:
        """
        Trains a Random Forest classifier on the training set and evaluates the trained model on the train,
        validation and test set.
        :return: The trained model information
        """

        self.rf.fit(self.train_data, self.train_target.squeeze())

        train_pred = self.rf.predict(self.train_data)

        # Evaluate on validation and test set
        val_pred = self.rf.predict(self.val_data)
        test_pred = self.rf.predict(self.test_data)

        Evaluator(
            self.train_target[self.ticker].tolist(),
            train_pred,
            "train",
            self.params["model_name"],
            0,
        )()
        Evaluator(
            self.val_target[self.ticker].tolist(),
            val_pred,
            "validation",
            self.params["model_name"],
            0,
        )()
        Evaluator(
            self.test_target[self.ticker].tolist(),
            test_pred,
            "test",
            self.params["model_name"],
            0,
        )()

        return ModelTrainReturnObject(
            model=self.rf, epoch=0, desc=self.params["model_name"], params=self.params
        )

    def rf_predict(self, model: PyFuncModel, x) -> Dict[str, Any]:
        """
        Given a trained model and an input sample it returns the predicted information

        :param model: A loaded from  mlflow
        :param x: The input sample
        :return: A dictionary with the prediction information
        """
        result = model.predict(x.reshape(1, -1))
        return {
            "predicted_class": result[0],
            "prediction_label": self.classes[result[0]],
        }

    @staticmethod
    def model_load(model: str) -> PyFuncModel:
        """
        Loads a model saved on MLFLOW based on a uri

        :param model: the uri for the model to be loaded
        :return:
        """
        return mlflow.pyfunc.load_model(model)
