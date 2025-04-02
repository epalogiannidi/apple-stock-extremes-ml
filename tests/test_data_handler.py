import pytest
import pandas as pd
from apple_stock_extremes_ml.exceptions import DataInconsistencyException

# Test data download
def test_download_data(data_handler):
    data = data_handler.download_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_check_quality(data_handler):
    data_handler.check_quality()
    assert data_handler.data.isnull().sum().sum() == 0  # No missing values
    assert data_handler.data.index.duplicated().sum() == 0  # No duplicates
    assert (
        data_handler.data[("Low", "AAPL")] <= data_handler.data[("High", "AAPL")]
    ).all()  # No inconsistencies


def test_inconsistent_data_handling(data_handler, inconsistent_data):
    data_handler.data = inconsistent_data  # Replace with test data

    with pytest.raises(DataInconsistencyException):  # Expecting an error
        data_handler.check_quality()


def test_duplicates_data_handling(data_handler, duplicated_data):
    data_handler.data = duplicated_data  # Replace with test data

    initial_length = len(data_handler.data)
    data_handler.check_quality()  # This should remove duplicates
    new_length = len(data_handler.data)

    assert new_length == initial_length - 1, "Duplicate row was not removed"


# Test daily return calculation
def test_calculate_daily_return(data_handler):
    data_handler.calculate_daily_return()
    assert ("Daily_Return", "AAPL") in data_handler.data.columns
    assert not data_handler.data[("Daily_Return", "AAPL")].isnull().any()


def test_define_extreme_events(data_handler):
    data_handler.define_extreme_events()
    assert ("Extreme_Event", "AAPL") in data_handler.data.columns
    assert ("Extreme_Event_Tomorrow", "AAPL") in data_handler.data.columns

    assert not data_handler.data[("Extreme_Event_Tomorrow", "AAPL")].isnull().any()
