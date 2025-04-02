import sys
import os

import torch
import pandas as pd

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from apple_stock_extremes_ml.data_handler import DataHandler


@pytest.fixture
def version():
    return "1.0.0"


@pytest.fixture
def val_loss():
    return 1.6


@pytest.fixture
def mock_data():
    features = torch.randn(100, 5)
    target = torch.randint(0, 2, (100,))

    return features, target


@pytest.fixture
def mock_train_data():
    return {
        "train_data": torch.randn(100, 5),
        "train_target": torch.randint(0, 2, (100,)),
        "val_data": torch.randn(20, 5),
        "val_target": torch.randint(0, 2, (20,)),
        "test_data": torch.randn(20, 5),
        "test_target": torch.randint(0, 2, (20,)),
    }


@pytest.fixture
def mock_config():
    return {
        "modeling": {
            "history": 10,
            "rf": {"trees": 100, "class_weights": True, "model_name": "rf_model"},
        },
        "data": {"ticker": "AAPL", "classes": ["No Extreme Event", "Extreme Event"]},
    }


@pytest.fixture
def mock_config_tcnn():
    return {
        "modeling": {
            "history": 10,
            "tcnn": {
                "model_name": "tcnn_baseline",
                "bs": 8,
                "channels": 32,
                "epochs": 100,
                "lr": 0.001,
                "label_smoothing": 0,
                "kernel": 3,
                "dropout": 0.3,
                "attention": True,
                "weights": False,
            },
        },
        "data": {"ticker": "AAPL", "classes": ["No Extreme Event", "Extreme Event"]},
    }


test_config = {
    "ticker": "AAPL",
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "ffill_holidays": True,
    "extreme_event_percentage": 2.0,
    "keep_features": ["Open", "Volume", "Adj Close", "High", "Low", "Close"],
}


@pytest.fixture
def data_handler():
    return DataHandler(test_config)


@pytest.fixture
def inconsistent_data(data_handler):
    """
    Creates a dataset with an intentional inconsistency (Low > High)
    to test the quality check function.
    """
    data = data_handler.download_data()
    data = data.copy(deep=True)  # Avoid modifying the original dataset
    data.iloc[10, data.columns.get_loc(("Low", "AAPL"))] = (
        data.iloc[10, data.columns.get_loc(("High", "AAPL"))] + 1
    )
    data = pd.concat([data, data.iloc[[5]]], axis=0)  # Duplicate the 10th row
    return data


@pytest.fixture
def duplicated_data(data_handler):
    """
    Creates a dataset with a duplicated row
    to test the quality check function.
    """
    data = data_handler.download_data()
    data = data.copy(deep=True)  # Avoid modifying the original dataset
    data = pd.concat([data, data.iloc[[5]]], axis=0)  # Duplicate the 10th row
    return data
