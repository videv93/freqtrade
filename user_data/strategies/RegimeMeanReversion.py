import numpy as np
import ruptures as rpt
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class RegimeMeanReversion(IStrategy):
    # Strategy Parameters
    ticker_interval = "1h"
    can_short = True  # Enable shorting for crypto

    # ROI and Stoploss (Mandatory)
    minimal_roi = {"0": 10.0}  # We use signal-based exits
    stoploss = -0.05  # 5% safety net

    # Lookback window for Change Point Detection
    cpd_window = 500

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. Initialize columns
        dataframe["target_mean"] = np.nan
        dataframe["target_std"] = np.nan

        # 2. Rolling CPD (Note: This can be slow in backtesting)
        # We loop to find the last change point for each 'regime'
        stride = 20
        for i in range(self.cpd_window, len(dataframe), stride):
            # Slice the window
            window_data = dataframe["close"].iloc[i - self.cpd_window : i].values

            # Detect change points in the window
            algo = rpt.Pelt(model="rbf").fit(window_data)
            result = algo.predict(pen=10)

            # Get the index of the most recent change point (within the window)
            last_cp_in_window = result[-2]

            # Calculate Mean and Std Dev ONLY from that CP to the current candle
            current_regime = window_data[last_cp_in_window:]

            dataframe.loc[dataframe.index[i : i + stride], "target_mean"] = np.mean(current_regime)
            dataframe.loc[dataframe.index[i : i + stride], "target_std"] = np.std(current_regime)

        # 3. Calculate Strategy Bands
        dataframe["long_entry"] = dataframe["target_mean"] - (2 * dataframe["target_std"])
        dataframe["short_entry"] = dataframe["target_mean"] + (2 * dataframe["target_std"])

        # Exit levels (0.25 sigma from the mean)
        dataframe["long_exit"] = dataframe["target_mean"] - (0.25 * dataframe["target_std"])
        dataframe["short_exit"] = dataframe["target_mean"] + (0.25 * dataframe["target_std"])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe["close"] < dataframe["long_entry"]), "enter_long"] = 1
        dataframe.loc[(dataframe["close"] > dataframe["short_entry"]), "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit Long: Price has recovered to within 0.25 sigma of mean
        dataframe.loc[(dataframe["close"] > dataframe["long_exit"]), "exit_long"] = 1
        # Exit Short: Price has dropped to within 0.25 sigma of mean
        dataframe.loc[(dataframe["close"] < dataframe["short_exit"]), "exit_short"] = 1
        return dataframe
