import torch
from unittest.mock import MagicMock

from apple_stock_extremes_ml.models.tcnn_tuning import TCNNHyperparameterTuner
from apple_stock_extremes_ml.train_tcnn import TCNNTrainer, StockDataset


def test_tcnn_tuning_initialization():
    _ = TCNNHyperparameterTuner


def test_stock_dataset(mock_data):
    features, target = mock_data
    dataset = StockDataset(features, target, sequence_length=10)

    assert len(dataset) == 90  # (100 - 10)
