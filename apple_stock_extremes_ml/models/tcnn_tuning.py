from typing import Any, Dict

import optuna
import torch.optim as optim
from tqdm import tqdm

from apple_stock_extremes_ml import logger
from apple_stock_extremes_ml.models import TCNN


class TCNNHyperparameterTuner:
    """
    Class for tuning the hyperparameters of the tcnn architecture
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device

    def objective(self, trial) -> float:
        """
        The objective function to tune the hyperparameters
        :param trial: The current trial
        :return:
        """
        # Sample hyperparameters
        out_channels = trial.suggest_categorical("out_channels", [16, 32, 64])
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        ls = trial.suggest_float("label_smoothing", 0.0, 0.5)

        # Update model with new hyperparameters
        self.trainer.tcnn = TCNN(
            num_features=self.trainer.train_data.shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            sequence_length=10,
        ).to(self.device)

        self.trainer.params["label_smoothing"] = ls
        self.trainer.tcnn.dropout.p = dropout_rate
        self.trainer.optimizer = optim.Adam(self.trainer.tcnn.parameters(), lr=lr)

        self.trainer.model_train()

        # Evaluate on validation set
        val_loss, val_accuracy = self.trainer.model_evaluate(
            self.trainer.val_data, self.trainer.val_target, epoch=0, label="val"
        )

        return val_loss  # Optimize for accuracy (can switch to F1-score)

    def tune(self, n_trials=30) -> Dict[str, Any]:
        study = optuna.create_study(direction="minimize")
        # Create tqdm progress bar
        with tqdm(total=n_trials, desc="Hyperparameter Tuning", unit="trial") as pbar:

            def tqdm_callback(study, trial):
                pbar.update(1)  # Update progress bar by 1 step

            study.optimize(self.objective, n_trials=n_trials, callbacks=[tqdm_callback])

        logger.info(f"\nBest Hyperparameters: {study.best_params}")
        return study.best_params
