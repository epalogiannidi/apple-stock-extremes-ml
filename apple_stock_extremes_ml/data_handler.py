from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from apple_stock_extremes_ml import PLOTS_DIR, logger
from apple_stock_extremes_ml.exceptions import DataInconsistencyException


class DataHandler:
    """
    Handles the data and specifically the following functions:

    - Download data
    - Handle holidays
    - Quality check
    - Feature enhancement
    - Dataset statistics
    - Dataset split

    """

    def __init__(self, config: Dict[str, Any]):
        self.data_config = config
        self.original_data = self.download_data()
        self.original_data = self.original_data[config["keep_features"]]
        self.ffill_holidays = config["ffill_holidays"]
        self.data = self.ffill_hodidays()
        self.extract_date_features()
        self.eep = config["extreme_event_percentage"]
        self.modeling_data = {
            "train_data": pd.DataFrame.empty,
            "train_target": pd.DataFrame.empty,
            "val_data": pd.DataFrame.empty,
            "val_target": pd.DataFrame.empty,
            "test_data": pd.DataFrame.empty,
            "test_target": pd.DataFrame.empty,
        }
        self.scaler_minmax = MinMaxScaler()
        self.scaler_standard = StandardScaler()
        self.price_columns = list(
            set(x[0] for x in self.original_data.columns).intersection(
                {"Open", "High", "Low", "Close"}
            )
        )

    def __repr__(self):
        return (
            f"{__name__} is a function designed to handle data. It loads a configuration file and based on the loaded"
            f"values applies the desired functionalities that include: data download, quality checks, "
            f"feature engineering, statistics and dataset split."
        )

    def download_data(self) -> pd.DataFrame:
        """
        Download the data and store them in a dataframe, given a configuration file

        :return: the initial stock prices data
        """
        return yf.download(
            self.data_config["ticker"],
            start=self.data_config["start_date"],
            end=self.data_config["end_date"],
            auto_adjust=False,
        )

    def ffill_hodidays(self) -> pd.DataFrame:
        """
        Decides whether the holidays will be full-filled or ignored as in the original dataset.

        :return: pd.DataFrame
            The data that contain also holidays, full-filled with the previous trading day prices
        """
        data = self.original_data.copy(deep=True)

        if self.ffill_holidays:
            data.index = pd.to_datetime(data.index)
            full_range = pd.date_range(
                start=data.index.min(), end=data.index.max(), freq="B"
            )  # 'B' = Business Days
            data = data.reindex(full_range, method="ffill")
        return data

    def extract_date_features(self) -> None:
        """
        Extracts date related features.
        :return: None
        """
        self.data[("date_Date", self.data_config["ticker"])] = pd.to_datetime(
            self.data.index
        )  # Ensure Date is a datetime type

        self.data[("date_Year", self.data_config["ticker"])] = self.data[
            ("date_Date", self.data_config["ticker"])
        ].dt.year
        self.data[("date_Month", self.data_config["ticker"])] = self.data[
            ("date_Date", self.data_config["ticker"])
        ].dt.month
        self.data[("date_Day", self.data_config["ticker"])] = self.data[
            ("date_Date", self.data_config["ticker"])
        ].dt.day
        self.data[("date_DayOfWeek", self.data_config["ticker"])] = self.data[
            ("date_Date", self.data_config["ticker"])
        ].dt.weekday  # Monday = 0, Sunday = 6
        self.data[("date_WeekOfYear", self.data_config["ticker"])] = (
            self.data[("date_Date", self.data_config["ticker"])].dt.isocalendar().week
        )
        self.data[("date_Quarter", self.data_config["ticker"])] = self.data[
            ("date_Date", self.data_config["ticker"])
        ].dt.quarter
        self.data[("date_IsMonthStart", self.data_config["ticker"])] = self.data[
            ("date_Date", self.data_config["ticker"])
        ].dt.is_month_start.astype(int)
        self.data[("date_IsMonthEnd", self.data_config["ticker"])] = self.data[
            ("date_Date", self.data_config["ticker"])
        ].dt.is_month_end.astype(int)
        self.data[("date_DayOfWeek_sin", self.data_config["ticker"])] = np.sin(
            2 * np.pi * self.data[("date_DayOfWeek", self.data_config["ticker"])] / 7
        )
        self.data[("date_DayOfWeek_cos", self.data_config["ticker"])] = np.cos(
            2 * np.pi * self.data[("date_DayOfWeek", self.data_config["ticker"])] / 7
        )

    def check_quality(self):
        """
        Applies quality checks to the original dataset. These checks include:
            - missing values: drop missing values if any
            - duplicates: drop duplicates if any
            - outliers detection and box plots
            - basic data inconsistencies checks: raises Error if this happen

        :return: None
        """
        # 1. missing values
        missing_values = self.data.isnull().sum()
        logger.info(f"Missing values: {missing_values}")
        self.data.dropna(inplace=True)

        # 2. duplicates
        duplicates = self.data[self.data.index.duplicated()]
        logger.info(f"Duplicates found: {len(duplicates)}")
        if len(duplicates) > 0:
            self.data.drop_duplicates(inplace=True)

        # 3. check for outliers
        columns = self.data.xs(
            self.data_config["ticker"], level="Ticker", axis=1
        ).columns.tolist()

        for c in columns:
            if not c.startswith("date_"):
                logger.info(f"{c}: {self.data[c].describe()}")
                _ = self.detect_outliers(self.data, c)
                self.box_plot(c)

        # 4. check data inconsistency
        # Check if High < Low on any day
        if "High" in self.data.columns:
            inconsistent_data = (
                self.data[("Low", self.data_config["ticker"])]
                > self.data[("High", self.data_config["ticker"])]
            ).sum()
            logger.info(f"Inconsistent data where Low > High: {inconsistent_data}")
            if inconsistent_data > 0:
                raise DataInconsistencyException(
                    f"Inconsistent data where Low > High: {inconsistent_data}"
                )

    def box_plot(self, feature: str) -> None:
        """
        Saves a box plot per column to visualize the data distributions and outliers

        :param feature: str
            The feature to check for outliers

        :return: None
        """
        self.data[feature].plot(kind="box")
        plt.savefig(f"{PLOTS_DIR}/{feature}_boxplot.png", format="png")

    def detect_outliers(self, data: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Detect outliers based on IQR

        :param data: pd.Dataframe
            The data to apply outlier check
        :param feature: str
            The column to check for outliers
        :return: pd.Dataframe
            A dataframe with the outliers
        """
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = data[(feature, self.data_config["ticker"])].quantile(0.25)
        Q3 = data[(feature, self.data_config["ticker"])].quantile(0.75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[
            (data[(feature, self.data_config["ticker"])] < lower_bound)
            | (data[(feature, self.data_config["ticker"])] > upper_bound)
        ]
        logger.info(
            f"Outliers of: {feature}: {len(outliers)}. Lower bound: {lower_bound}, Upper bound: {upper_bound}"
        )
        return outliers

    def calculate_daily_return(self) -> None:
        """
        Compute the daily return and remove rows with nan values.
        First row contains nan, because it doesn't have a previous Adj Close value

        :return: None
        """
        self.data[("Daily_Return", self.data_config["ticker"])] = (
            self.data[("Adj Close", self.data_config["ticker"])].pct_change() * 100
        )
        self.data.dropna(inplace=True)

    def define_extreme_events(self) -> None:
        """
        Define the labels, i.e., extreme events that are either 0 or 1.

        :return: None
        """
        self.calculate_daily_return()

        self.data[("Extreme_Event", self.data_config["ticker"])] = (
            abs(self.data[("Daily_Return", self.data_config["ticker"])]) > self.eep
        ).astype(int)

        self.data[("Extreme_Event_Tomorrow", self.data_config["ticker"])] = (
            self.data[("Extreme_Event", self.data_config["ticker"])]
            .shift(-1)
            .fillna(0)
            .astype(int)
        )
        # drop last line because the actual prediction is Nan
        self.data = self.data.drop(self.data.index[-1])

    def split_data(self) -> None:
        """
        Prepares the data for the modeling part by applying a twofold splitting.
        First, it separates between the features and the targets. Then, it splits in
        train, validation and test sets. The split percentages are loaded from the configuration file
        and no shuffling is applied on the data, in order to keep the temporal order.
        Validation contains only future values compared to train, and test contains only future values
        compared to validation and test

        :return: None
        """
        self.target = self.data[self.data_config["target"]]
        self.data = self.data[self.data_config["basic_features"]]

        total_rows = len(self.data)

        train_end = int(self.data_config["train_split_per"] * total_rows)
        val_end = int(
            (self.data_config["train_split_per"] + self.data_config["val_split_per"])
            * total_rows
        )

        self.modeling_data = {
            "train_data": self.data.iloc[:train_end],
            "train_target": self.target.iloc[:train_end],
            "val_data": self.data.iloc[train_end:val_end],
            "val_target": self.target.iloc[train_end:val_end],
            "test_data": self.data.iloc[val_end:],
            "test_target": self.target.iloc[val_end:],
        }

    def normalize_data(self) -> Dict[str, pd.DataFrame]:
        # Open, High, Low, Close
        normalized_data = {
            "train_data": self.modeling_data["train_data"].copy(deep=True),
            "val_data": self.modeling_data["val_data"].copy(deep=True),
            "test_data": self.modeling_data["test_data"].copy(deep=True),
        }

        self.scaler_minmax.fit(self.modeling_data["train_data"][self.price_columns])
        normalized_data["train_data"][
            self.price_columns
        ] = self.scaler_minmax.transform(
            self.modeling_data["train_data"][self.price_columns]
        )
        normalized_data["val_data"][self.price_columns] = self.scaler_minmax.transform(
            self.modeling_data["val_data"][self.price_columns]
        )
        normalized_data["test_data"][self.price_columns] = self.scaler_minmax.transform(
            self.modeling_data["test_data"][self.price_columns]
        )

        # Daily_Return
        self.scaler_standard.fit(self.modeling_data["train_data"]["Daily_Return"])
        normalized_data["train_data"]["Daily_Return"] = self.scaler_standard.transform(
            self.modeling_data["train_data"]["Daily_Return"]
        )
        normalized_data["val_data"]["Daily_Return"] = self.scaler_standard.transform(
            self.modeling_data["val_data"]["Daily_Return"]
        )
        normalized_data["test_data"]["Daily_Return"] = self.scaler_standard.transform(
            self.modeling_data["test_data"]["Daily_Return"]
        )

        # Volume
        normalized_data["train_data"]["Volume"] = np.log1p(
            self.modeling_data["train_data"]["Volume"]
        )
        normalized_data["val_data"]["Volume"] = np.log1p(
            self.modeling_data["val_data"]["Volume"]
        )
        normalized_data["test_data"]["Volume"] = np.log1p(
            self.modeling_data["test_data"]["Volume"]
        )

        return normalized_data

    def data_statistics(
        self, features_df: pd.DataFrame, labels_df: pd.DataFrame, label: str
    ) -> None:
        """
        Computes correlation between the features and the target and saves a plot with the correlation heatmap.

        :param features_df: The features dataset
        :param labels_df: The dataset with the target values (labels)
        :param label: A label to distinguish between the different cases, e.g., train, validation, test
        :return:
        """
        df_numeric = features_df.select_dtypes(include=["float64", "int64"])
        labels_numeric = labels_df.select_dtypes(include=["float64", "int64"])
        labels_numeric.columns = ["Target"]
        df_numeric = pd.concat([df_numeric, labels_numeric], axis=1)
        correlation_matrix = df_numeric.corr()

        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Correlation Heatmap: {label} features")
        plt.xticks(rotation=15)

        plt.savefig(
            f"{PLOTS_DIR}/{label}_features_outliers_impute_{self.data_config['impute_outliers']}_corr_heatmap.png",
            format="png",
        )

        """ Plot target distributions """
        plt.figure(figsize=(10, 6))

        # Plot the distribution of the target variable in the train set
        plt.hist(labels_df, bins=2, edgecolor="black", alpha=0.7)
        plt.title(f"{label} Set Distribution")
        plt.xticks([0, 1])
        plt.legend(["No Extreme Event", "Extreme Event"])
        plt.ylabel("Frequency")

        plt.savefig(f"{PLOTS_DIR}/{label}_target_distribution.png", format="png")

        eep = float(f"{100 * (labels_df.sum()/len(features_df)).iloc[0]:.2f}")
        logger.info(
            f"{label} target distribution: \n Extreme event: {eep}%\n "
            f"No Extreme event = {100 - eep}%"
        )

    def handle_outliers(
        self, data: pd.DataFrame, outliers: pd.DataFrame, feature: str
    ) -> None:
        """

        :param data:
        :param outliers:
        :param feature:
        :return:
        """
        logger.info(f"Handling {len(outliers)} outliers for feature: {feature}")
        if self.data_config["impute_outliers"] == "median":
            # find the median based on the train set
            median = self.modeling_data["train_data"][
                (feature, self.data_config["ticker"])
            ].median()
            logger.info(
                f"Outliers will be replaced with median value computed using the training set: {median}"
            )
            data.loc[outliers.index, (feature, self.data_config["ticker"])] = median
        elif self.data_config["impute_outliers"] == "None":
            pass
        else:
            raise ValueError("Not supported outliers handling.")
