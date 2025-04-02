import os
from typing import List

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)

from apple_stock_extremes_ml import PLOTS_DIR, logger


class Evaluator:
    def __init__(
        self,
        groundtruth: List[int],
        predictions: List[int],
        label: str,
        model: str,
        epoch,
    ):
        self.groundtruth = groundtruth
        self.predictions = predictions
        self.label = label
        self.model = model
        self.path = f"{PLOTS_DIR}/eval_metrics/{model}"
        self.metrics = ["precision", "recall", "f1-score"]
        self.epoch = epoch

    def __call__(self):
        # lazy import to quickly skip circular import issue
        from apple_stock_extremes_ml.models.utils import (MlflowLogging,
                                                          ModelMetricsObject)

        os.makedirs(self.path, exist_ok=True)
        cl_report = classification_report(
            self.groundtruth, self.predictions, output_dict=True, zero_division=0
        )
        conf_matrix = confusion_matrix(self.groundtruth, self.predictions)
        fpr, tpr, thresholds = roc_curve(self.groundtruth, self.predictions)
        roc_auc = auc(fpr, tpr)

        logger.info(f"{self.label} Classification report:\n {cl_report}")
        logger.info(f"{self.label} Confusion matrix:\n {conf_matrix}")
        logger.info(f"{self.label} ROC-AUC:\n {roc_auc}")

        class_names = ["No Extreme Event", "Extreme Event"]

        # Plot the confusion matrix using seaborn heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"{self.path}/{self.label}_rf_confusion_matrix.png", format="png")
        plt.close()

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot(
            [0, 1], [0, 1], color="gray", linestyle="--"
        )  # Diagonal line (random classifier)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{self.label} ROC")
        plt.legend(loc="lower right")
        plt.savefig(f"{self.path}/{self.label}_rf_roc.png", format="png")
        plt.close()

        if os.getenv("ENABLE_LOGGING", "False").lower() == "true":
            MlflowLogging.log_plot(f"{self.path}/{self.label}_rf_confusion_matrix.png")
            MlflowLogging.log_plot(f"{self.path}/{self.label}_rf_roc.png")

            for k in cl_report.keys():
                if k == "accuracy":
                    continue
                for m in self.metrics:
                    mlflow.log_metric(
                        f"{self.label}_{m}_{k}", cl_report[k][m], step=self.epoch
                    )
            mlflow.log_metric(
                f"{self.label}_accuracy", cl_report["accuracy"], step=self.epoch
            )
            mlflow.log_metric(f"{self.label}_roc_auc", roc_auc, step=self.epoch)

        return ModelMetricsObject(cl_report, conf_matrix, roc_auc)
