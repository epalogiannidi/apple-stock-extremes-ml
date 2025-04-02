import mlflow
import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from apple_stock_extremes_ml.models.rf import RFPredictor
from sklearn.ensemble import RandomForestClassifier
from apple_stock_extremes_ml.models.utils import ModelTrainReturnObject


# @pytest.fixture
# def mock_config():
#     return {
#         "modeling": {
#             "history": 10,
#             "rf": {
#                 "trees": 100,
#                 "class_weights": True,
#                 "model_name": "rf_model"
#             }
#         },
#         "data": {
#             "ticker": "AAPL",
#             "classes": ["No Extreme Event", "Extreme Event"]
#         }
#     }


@pytest.fixture
def mock_modeling_data():
    # Mock data, normally this would be loaded from a CSV or other source
    train_data = pd.DataFrame(
        np.random.randn(100, 6),
        columns=["Open", "Volume", "Daily_Return", "High", "Low", "Close"],
    )
    train_target = pd.Series(np.random.randint(0, 2, size=100), name="target")

    val_data = pd.DataFrame(
        np.random.randn(50, 6),
        columns=["Open", "Volume", "Daily_Return", "High", "Low", "Close"],
    )
    val_target = pd.Series(np.random.randint(0, 2, size=50), name="target")

    test_data = pd.DataFrame(
        np.random.randn(30, 6),
        columns=["Open", "Volume", "Daily_Return", "High", "Low", "Close"],
    )
    test_target = pd.Series(np.random.randint(0, 2, size=30), name="target")

    return {
        "train_data": train_data,
        "train_target": train_target,
        "val_data": val_data,
        "val_target": val_target,
        "test_data": test_data,
        "test_target": test_target,
    }


@pytest.fixture
def rf_predictor(mock_config, mock_modeling_data):
    return RFPredictor(mock_config, mock_modeling_data)


def test_rf_predictor_initialization(rf_predictor):
    # Test if the predictor initializes correctly
    assert isinstance(rf_predictor, RFPredictor)
    assert rf_predictor.history == 10
    assert rf_predictor.ticker == "AAPL"
    assert isinstance(rf_predictor.rf, RandomForestClassifier)


def test_prepare_data(rf_predictor, mock_modeling_data):
    # Test the prepare_data method
    train_data, train_target = rf_predictor.prepare_data(
        mock_modeling_data["train_data"], mock_modeling_data["train_target"]
    )

    # Assert that the prepared data has the correct shape
    assert (
        train_data.shape[0]
        == mock_modeling_data["train_data"].shape[0] - rf_predictor.history
    )
    assert (
        train_data.shape[1] == rf_predictor.history * 6
    )  # Number of features * history


def test_rf_predict(rf_predictor):
    # Test the rf_predict method with mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])  # Mock prediction

    prediction = rf_predictor.rf_predict(
        mock_model, np.random.randn(1, 60)
    )  # Assume 10-day history with 6 features

    assert prediction["predicted_class"] == 1
    assert prediction["prediction_label"] == rf_predictor.classes[1]


def test_model_load(rf_predictor):
    # Test the model_load method
    mock_model = MagicMock()
    mlflow.pyfunc.load_model = MagicMock(return_value=mock_model)

    model_uri = "mock_model_uri"
    loaded_model = rf_predictor.model_load(model_uri)

    assert loaded_model == mock_model
    mlflow.pyfunc.load_model.assert_called_once_with(model_uri)
