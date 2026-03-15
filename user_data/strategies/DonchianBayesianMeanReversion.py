import logging

import pandas as pd
import talib
from pandas import DataFrame

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class DonchianBayesianMeanReversion(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.10}
    stoploss = -0.05
    timeframe = "15m"
    can_short = True
    window = 50
    entry_std = 2.5
    exit_std = 0.25

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["donchian_high"] = dataframe["high"].rolling(window=self.window).max()
        dataframe["donchian_low"] = dataframe["low"].rolling(window=self.window).min()
        dataframe["donchian_mid"] = (dataframe["donchian_high"] + dataframe["donchian_low"]) / 2
        dataframe["mean"] = talib.SMA(dataframe["close"], timeperiod=self.window)
        dataframe["std"] = talib.STDDEV(dataframe["close"], timeperiod=self.window)
        dataframe["entry_band_long"] = dataframe["mean"] - self.entry_std * dataframe["std"]
        dataframe["entry_band_short"] = dataframe["mean"] + self.entry_std * dataframe["std"]
        dataframe["zscore"] = (
            (dataframe["close"] - dataframe["mean"])
            / (dataframe["std"] + 1e-10)  # Avoid division by zero
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        long_condition = (
            (dataframe["zscore"] < -self.entry_std)
            & (dataframe["volume"] > 0)  # Ensure volume exists
        )
        conditions.append({"enter_long": long_condition, "enter_tag": "long"})

        short_condition = (
            (dataframe["zscore"] > self.entry_std)
            & (dataframe["volume"] > 0)  # Ensure volume exists
        )
        conditions.append({"enter_short": short_condition, "enter_tag": "short"})

        if conditions:
            df_cond = pd.concat([pd.DataFrame(c) for c in conditions], axis=1, join="outer")
            dataframe = pd.concat([dataframe, df_cond], axis=1)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        long_exit_condition = dataframe["zscore"] > -self.exit_std
        conditions.append({"exit_long": long_exit_condition, "exit_tag": "exit_long"})
        short_exit_condition = dataframe["zscore"] < self.exit_std
        conditions.append({"exit_short": short_exit_condition, "exit_tag": "exit_short"})

        if conditions:
            df_cond = pd.concat([pd.DataFrame(c) for c in conditions], axis=1, join="outer")
            dataframe = pd.concat([dataframe, df_cond], axis=1)

        return dataframe

    @property
    def plot_config(self):
        """
        Configure plot overlays and subplots for visualization.
        """
        return {
            "main_plot": {
                "close": {"color": "blue", "type": "candle"},
                "mean": {"color": "orange", "type": "line"},
                "donchian_high": {"color": "gray", "type": "line"},
                "donchian_low": {"color": "gray", "type": "line"},
                "entry_band_long": {"color": "green", "type": "line"},
                "entry_band_short": {"color": "red", "type": "line"}
            },
            "subplots": {
                "zscore": {
                    "zscore": {"color": "purple", "type": "line"},
                    "entry_std": {"color": "red", "type": "line"},
                    "exit_std": {"color": "green", "type": "line"},
                    "-entry_std": {"color": "red", "type": "line"},
                    "-exit_std": {"color": "green", "type": "line"},
                },
            },
        }
