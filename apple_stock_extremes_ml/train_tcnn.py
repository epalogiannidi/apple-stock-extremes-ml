import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.pyfunc import PyFuncModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from apple_stock_extremes_ml import PLOTS_DIR, logger
from apple_stock_extremes_ml.models import TCNN
from apple_stock_extremes_ml.models.utils import (EarlyStopper, Evaluator,
                                                  MlflowExperiment,
                                                  MlflowLogging,
                                                  ModelTrainReturnObject)

ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "True").lower() == "true"


class StockDataset(Dataset):
    """
    Creates a dataset based on the input data that is in the appropriate form entering the neural network
    """

    def __init__(self, features, target, sequence_length=10):
        """ """
        self.sequence_length = sequence_length
        self.features = features.values
        self.target = target.values

        # Number of sequences we can extract
        self.num_samples = len(features) - sequence_length

    def __len__(self):
        """Returns the total number of samples available."""
        return self.num_samples

    def __getitem__(self, index):
        """
        Returns a single sequence of stock data.

        :param index: The start index of the sequence.
        :return: Tuple (X, y) where:
                 - X: Stock features of shape (num_features, sequence_length)
                 - y: Target label for the next day
        """
        # Extract sequence of stock features (past 10 days)
        X = self.features[
            index : index + self.sequence_length
        ]  # Shape: (sequence_length, num_features)
        X = torch.tensor(
            X, dtype=torch.float32
        ).T  # Transpose to (num_features, sequence_length)

        # Target label for the next day
        y = self.target[index + self.sequence_length]
        y = torch.tensor(y, dtype=torch.float32)

        return X, y


class TCNNTrainer(MlflowExperiment):
    """
    Neural network trainer class that is responsible for organizing the training procedure.
    It inherits an Abstract class that acts as interface in order to define the core functions of a model
    """

    def __init__(
        self,
        config: Dict[str, Any],
        normalized_data: Dict[str, pd.DataFrame],
        modeling_data: Dict[str, pd.DataFrame],
    ):
        super().__init__()
        self.history = config["modeling"]["history"]
        self.ticker = config["data"]["ticker"]
        self.classes = config["data"]["classes"]
        self.params = config["modeling"]["tcnn"]
        self.train_data = normalized_data["train_data"]
        self.train_target = modeling_data["train_target"]
        self.val_data = normalized_data["val_data"]
        self.val_target = modeling_data["val_target"]
        self.test_data = normalized_data["test_data"]
        self.test_target = modeling_data["val_target"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tcnn = TCNN(
            self.train_data.shape[1],
            self.params["channels"],
            self.params["kernel"],
            sequence_length=10,
            dropout=self.params["dropout"],
            attention=self.params["attention"],
        ).to(self.device)
        # if self.params["attention"]:
        #     self.tcnn: nn.Module = TCNNWithAttention(
        #         self.train_data.shape[1],
        #         self.params["channels"],
        #         self.params["kernel"],
        #         sequence_length=10,
        #         dropout=self.params["dropout"],
        #     ).to(self.device)
        # else:
        #     self.tcnn = TCNN(
        #         self.train_data.shape[1],
        #         self.params["channels"],
        #         self.params["kernel"],
        #         sequence_length=10,
        #         dropout=self.params["dropout"],
        #     ).to(self.device)
        self.batch_size = self.params["bs"]
        self.epochs = self.params["epochs"]
        self.lr = self.params["lr"]
        self.current_info = {"epoch": 0, "train_loss": 1000.0, "val_loss": 1000.0}
        self.criterion = (
            nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 3.0]),
                label_smoothing=self.params["label_smoothing"],
            )
            if self.params["weights"]
            else nn.CrossEntropyLoss(label_smoothing=self.params["label_smoothing"])
        )
        self.optimizer = optim.AdamW(self.tcnn.parameters(), lr=self.params["lr"])

        self.early_stopper = EarlyStopper(patience=5)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=5, factor=0.05
        )

    def model_evaluate(
        self, data: pd.DataFrame, labels: pd.DataFrame, epoch: int, label: str
    ) -> Tuple[float, float]:
        """
        Validates a model on unseen data (during training)

        :param data: The validation dataset
        :param labels: The target labels
        :param epoch: The current epoch
        :param label: An indicator about the set (train, val, etc)
        :return: Validation loss and validation accuracy
        """
        dataset = StockDataset(data, labels, self.history)
        val_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        val_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            groundtruths = []
            predictions = []

            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                outputs = self.tcnn(batch_X)
                loss = self.criterion(outputs, batch_y.view((-1)).long())
                val_loss += loss.item()

                preds = nn.functional.softmax(outputs, dim=1).argmax(dim=1)
                correct += (preds == batch_y.view(-1)).sum().item()
                total += batch_y.size(0)

                groundtruths.extend(batch_y.view((-1)).long().tolist())
                predictions.extend(preds.tolist())

        Evaluator(groundtruths, predictions, label, self.params["model_name"], epoch)()

        return val_loss / len(val_loader), (correct / total)

    @MlflowLogging.log_params
    def model_train(self) -> ModelTrainReturnObject:
        """
        Main training function. The model is trained for # epochs and validated on unseen data.
        :return: The trained model information
        """
        dataset = StockDataset(self.train_data, self.train_target, self.history)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # Fetch one batch

        epochs_lst, train_loss_lst, val_loss_lst = [], [], []

        # Training loop
        for epoch in tqdm(range(self.epochs), total=self.epochs):
            epochs_lst.append(epoch)
            groundtruths = []
            predictions = []

            self.tcnn.train()
            total_loss, correct, total = 0, 0, 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.tcnn(batch_X)
                loss = self.criterion(outputs, batch_y.view((-1)).long())
                loss.backward()

                self.optimizer.step()

                preds = nn.functional.softmax(outputs, dim=1).argmax(dim=1)
                correct += (preds == batch_y.view(-1)).sum().item()
                total += batch_y.size(0)
                total_loss += loss.item()

                groundtruths.extend(batch_y.view((-1)).long().tolist())
                predictions.extend(preds.tolist())

            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total

            # Validation step
            self.tcnn.eval()
            avg_val_loss, val_accuracy = self.model_evaluate(
                self.val_data, self.val_target, epoch, "val"
            )
            avg_test_loss, test_accuracy = self.model_evaluate(
                self.test_data, self.test_target, epoch, "test"
            )

            self.current_info["epoch"] = epoch
            self.current_info["train_loss"] = avg_train_loss
            self.current_info["val_loss"] = avg_val_loss

            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f},"
                f"TestLoss: {avg_test_loss:.4f}, "
                f"Test Accuracy: {test_accuracy:.4f},"
                f" Patience: {self.early_stopper.counter}, Current LR: {current_lr}"
            )

            self.lr_scheduler.step(avg_val_loss)

            if ENABLE_LOGGING:
                mlflow.log_metric("Train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("Train_accuracy", train_accuracy, step=epoch)
                mlflow.log_metric("Val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("Val_accuracy", val_accuracy, step=epoch)
            val_loss_lst.append(avg_val_loss)
            train_loss_lst.append(avg_train_loss)

            Evaluator(
                groundtruths, predictions, "train", self.params["model_name"], epoch
            )()
            if self.early_stopper(self.current_info["val_loss"]):
                break

        logger.info("Training Complete!")

        if ENABLE_LOGGING:  # Skip logging

            # Plot using Matplotlib
            plt.figure(figsize=(8, 5))
            plt.plot(epochs_lst, train_loss_lst, label="Train Loss", marker="o")
            plt.plot(epochs_lst, val_loss_lst, label="Validation Loss", marker="o")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training vs Validation Loss")
            plt.legend()
            plt.grid()
            plt.savefig(f"{PLOTS_DIR}/{self.params['model_name']}_train_val_loss.png")
            plt.close()

            MlflowLogging.log_plot(
                f"{PLOTS_DIR}/{self.params['model_name']}_train_val_loss.png"
            )

            self.model_save(self.tcnn, epoch)

        return ModelTrainReturnObject(
            model=self.tcnn,
            epoch=epoch,
            desc=self.params["model_name"],
            params=self.params,
        )

    def model_predict(self, model: PyFuncModel, x: np.ndarray) -> Dict[str, Any]:
        """
        Given a trained model and an input sample it returns the predicted information

        :param model: A model loaded from mlflow
        :param x: The input sample
        :return: A dictionary with the prediction information
        """

        with torch.no_grad():
            result = model.predict(x)
            probabilities = nn.functional.softmax(torch.tensor(result).detach(), dim=1)

            prediction = probabilities.argmax(dim=1).item()

        return {
            "prediction_probabilities": np.array(probabilities).tolist()[0],
            "predicted_class": prediction,
            "prediction_label": self.classes[prediction],
        }

    @MlflowLogging.log_model
    def model_save(self, model: TCNN, epoch: int) -> ModelTrainReturnObject:
        """
        Saves a trained model on MLFLOW

        :param model: trained model that can be either tcnn or tcnn with attention
        :param epoch: the indicator of the last epoch
        :return:
        """
        if ENABLE_LOGGING:
            return ModelTrainReturnObject(model=model, epoch=epoch)
        else:
            raise RuntimeError("Logging is not enabled")

    @staticmethod
    def model_load(model: str) -> PyFuncModel:
        """
        Loads a model saved on MLFLOW based on a uri
        :param model: the uri for the model to be loaded
        :return:
        """
        return mlflow.pyfunc.load_model(model)
