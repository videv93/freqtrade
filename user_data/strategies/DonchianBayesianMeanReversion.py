import logging
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
import talib
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.persistence import Trade
from pandas import DataFrame
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DonchianBayesianMeanReversion(IStrategy):
    """
    Donchian + Bayesian Changepoint Mean Reversion Strategy
    
    This strategy combines:
    - Donchian Channel for extremes detection
    - Adaptive Mean & Volatility (SMA + StdDev)
    - Bayesian Changepoint Detection (price deviation + vol spike + crossover)
    - Regime Tracking (bars_in_regime, stabilization check)
    - Mean Reversion Entry/Exit Signals
    
    Parameters based on Pine Script original strategy.
    """
    
    INTERFACE_VERSION = 3
    
    # ============ Strategy Configuration ============
    minimal_roi = {
        "0": 0.10  # 10% profit target as fallback
    }
    
    stoploss = -0.05  # 5% stop loss
    
    timeframe = '1h'  # Same as Kelly strategy
    can_short = True
    
    # ============ Strategy Parameters ============
    # Donchian Channel
    lookback = 44  # Donchian period
    
    # Volatility & Mean
    volatility_window = 20
    
    # Bayesian Changepoint Detection
    changepoint_threshold = 0.75  # Probability threshold (0-1)
    
    # Mean Reversion Bands
    entry_std = 1.5  # Entry band standard deviations
    exit_std = 0.5   # Exit band standard deviations
    
    # Regime Detection
    regime_stabilization_bars = 50
    regime_change_window = 5
    
    # Trade Management
    exit_on_changepoint = True  # Emergency exit when changepoint detected
    
    def informative_pairs(self):
        return []
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators:
        1. Donchian Channel
        2. Adaptive Mean (SMA)
        3. Volatility (StdDev)
        4. Price Deviation from Mean
        5. Bayesian Changepoint Detection
        6. Regime Tracking
        """
        
        # ========== 1. DONCHIAN CHANNEL ==========
        # Highest high over lookback period
        dataframe['donchian_high'] = dataframe['high'].rolling(
            window=self.lookback
        ).max()
        
        # Lowest low over lookback period
        dataframe['donchian_low'] = dataframe['low'].rolling(
            window=self.lookback
        ).min()
        
        # Donchian midpoint
        dataframe['donchian_mid'] = (
            dataframe['donchian_high'] + dataframe['donchian_low']
        ) / 2
        
        # ========== 2. ADAPTIVE MEAN & VOLATILITY ==========
        # SMA as adaptive mean
        dataframe['adaptive_mean'] = talib.SMA(
            dataframe['close'], 
            timeperiod=self.volatility_window
        )
        
        # Standard deviation (volatility)
        dataframe['volatility'] = talib.STDDEV(
            dataframe['close'],
            timeperiod=self.volatility_window
        )
        
        # ========== 3. PRICE DEVIATION FROM MEAN ==========
        # Distance from mean (in standard deviations)
        dataframe['distance_from_mean'] = (
            (dataframe['close'] - dataframe['adaptive_mean']) / 
            (dataframe['volatility'] + 1e-10)  # Avoid division by zero
        )
        
        # ========== 4. BAYESIAN CHANGEPOINT DETECTION ==========
        # This is a simplified version of Bayesian changepoint detection
        # We use: price deviation spike + volatility spike + crossover
        
        # Price deviation from Donchian midpoint
        dataframe['price_deviation_donchian'] = (
            dataframe['close'] - dataframe['donchian_mid']
        ) / (dataframe['donchian_high'] - dataframe['donchian_low'] + 1e-10)
        
        # Volatility spike detection (current vol > mean vol)
        dataframe['volatility_ma'] = talib.SMA(
            dataframe['volatility'],
            timeperiod=self.regime_change_window
        )
        dataframe['vol_spike'] = (
            dataframe['volatility'] > 
            (dataframe['volatility_ma'] * 1.2)
        ).astype(int)
        
        # Extreme price deviation (far from mean)
        dataframe['extreme_deviation'] = (
            np.abs(dataframe['distance_from_mean']) > self.entry_std
        ).astype(int)
        
        # Crossover probability: price crossing mean (close vs SMA)
        dataframe['price_above_mean'] = (
            dataframe['close'] > dataframe['adaptive_mean']
        ).astype(int)
        dataframe['price_above_mean_prev'] = dataframe['price_above_mean'].shift(1)
        dataframe['mean_crossover'] = (
            dataframe['price_above_mean'] != dataframe['price_above_mean_prev']
        ).astype(int)
        
        # Bayesian Changepoint Score (0-1 probability)
        # Combines: vol spike + extreme deviation + crossover
        dataframe['changepoint_score'] = (
            (dataframe['vol_spike'] * 0.3) +  # 30% weight
            (dataframe['extreme_deviation'] * 0.4) +  # 40% weight
            (dataframe['mean_crossover'] * 0.3)  # 30% weight
        )
        
        # Binary changepoint detection
        dataframe['changepoint_detected'] = (
            dataframe['changepoint_score'] >= self.changepoint_threshold
        ).astype(int)
        
        # ========== 5. REGIME TRACKING ==========
        # Determine regime: uptrend or downtrend
        dataframe['trend'] = 0
        close_prices = dataframe['close'].values
        
        # Simple trend: price above or below Donchian midpoint
        dataframe['regime'] = (
            (dataframe['close'] > dataframe['donchian_mid']).astype(int)
        )
        
        # Track bars in current regime
        dataframe['bars_in_regime'] = self._calculate_bars_in_regime(
            dataframe['regime'].values
        )
        
        # Regime stabilization: at least N bars in same regime
        dataframe['regime_stable'] = (
            dataframe['bars_in_regime'] >= self.regime_stabilization_bars
        ).astype(int)
        
        # ========== 6. ENTRY & EXIT SIGNALS (Preliminary) ==========
        # Mean reversion entry signals
        dataframe['entry_long_signal'] = (
            (dataframe['distance_from_mean'] < -self.entry_std) &
            (dataframe['regime_stable'] == 1)
        ).astype(int)
        
        dataframe['entry_short_signal'] = (
            (dataframe['distance_from_mean'] > self.entry_std) &
            (dataframe['regime_stable'] == 1)
        ).astype(int)
        
        # Exit signals
        dataframe['exit_long_signal'] = (
            dataframe['distance_from_mean'] > -self.exit_std
        ).astype(int)
        
        dataframe['exit_short_signal'] = (
            dataframe['distance_from_mean'] < self.exit_std
        ).astype(int)
        
        return dataframe
    
    @staticmethod
    def _calculate_bars_in_regime(regime: np.ndarray) -> np.ndarray:
        """
        Calculate the number of consecutive bars in the same regime.
        """
        bars_in_regime = np.ones(len(regime), dtype=int)
        
        for i in range(1, len(regime)):
            if regime[i] == regime[i-1]:
                bars_in_regime[i] = bars_in_regime[i-1] + 1
            else:
                bars_in_regime[i] = 1
        
        return bars_in_regime
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signal logic:
        - Long: distance_from_mean < -entry_std AND regime_stable
        - Short: distance_from_mean > entry_std AND regime_stable
        """
        
        conditions = []
        
        # ========== LONG ENTRY ==========
        long_condition = (
            (dataframe['distance_from_mean'] < -self.entry_std) &
            (dataframe['regime_stable'] == 1) &
            (dataframe['volume'] > 0)  # Ensure volume exists
        )
        
        conditions.append({
            'enter_long': long_condition,
            'enter_tag': 'long_mean_reversion'
        })
        
        # ========== SHORT ENTRY ==========
        short_condition = (
            (dataframe['distance_from_mean'] > self.entry_std) &
            (dataframe['regime_stable'] == 1) &
            (dataframe['volume'] > 0)  # Ensure volume exists
        )
        
        conditions.append({
            'enter_short': short_condition,
            'enter_tag': 'short_mean_reversion'
        })
        
        if conditions:
            df_cond = pd.concat(
                [pd.DataFrame(c) for c in conditions],
                axis=1,
                join='outer'
            )
            dataframe = pd.concat([dataframe, df_cond], axis=1)
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signal logic:
        - Long Exit: distance_from_mean > -exit_std
        - Short Exit: distance_from_mean < exit_std
        - Emergency Exit: changepoint detected during position
        """
        
        conditions = []
        
        # ========== LONG EXIT ==========
        long_exit_condition = (
            (dataframe['distance_from_mean'] > -self.exit_std) |
            (self.exit_on_changepoint & (dataframe['changepoint_detected'] == 1))
        )
        
        conditions.append({
            'exit_long': long_exit_condition,
            'exit_tag': 'long_mean_reversion_exit'
        })
        
        # ========== SHORT EXIT ==========
        short_exit_condition = (
            (dataframe['distance_from_mean'] < self.exit_std) |
            (self.exit_on_changepoint & (dataframe['changepoint_detected'] == 1))
        )
        
        conditions.append({
            'exit_short': short_exit_condition,
            'exit_tag': 'short_mean_reversion_exit'
        })
        
        if conditions:
            df_cond = pd.concat(
                [pd.DataFrame(c) for c in conditions],
                axis=1,
                join='outer'
            )
            dataframe = pd.concat([dataframe, df_cond], axis=1)
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss implementation with emergency exit on changepoint.
        """
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe is None or len(dataframe) == 0:
            return -1
        
        last_candle = dataframe.iloc[-1]
        
        # Emergency exit if changepoint detected during position
        if self.exit_on_changepoint and last_candle.get('changepoint_detected', 0) == 1:
            logger.info(
                f"Changepoint detected for {pair}. Initiating emergency exit. "
                f"Changepoint Score: {last_candle.get('changepoint_score', 0):.3f}"
            )
            return -1  # Immediate exit
        
        # Default stoploss
        return self.stoploss
    
    def bot_loop_start(self, **kwargs) -> None:
        """
        Log regime state and changepoint signals for monitoring.
        """
        pass
    
    def log_signal_state(self, pair: str) -> None:
        """
        Helper to log current signal state for debugging.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe is None or len(dataframe) == 0:
            return
        
        last = dataframe.iloc[-1]
        
        logger.info(
            f"Signal State - {pair} | "
            f"Close: {last['close']:.4f} | "
            f"Mean: {last.get('adaptive_mean', 0):.4f} | "
            f"Distance: {last.get('distance_from_mean', 0):.3f}σ | "
            f"Regime: {'UP' if last.get('regime', 0) == 1 else 'DOWN'} "
            f"({int(last.get('bars_in_regime', 0))} bars) | "
            f"Stable: {'YES' if last.get('regime_stable', 0) == 1 else 'NO'} | "
            f"Changepoint: {last.get('changepoint_score', 0):.3f} | "
            f"Vol Spike: {'YES' if last.get('vol_spike', 0) == 1 else 'NO'}"
        )
    
    @property
    def plot_config(self):
        """
        Configure plot overlays and subplots for visualization.
        """
        return {
            "main_plot": {
                "close": {
                    "color": "blue",
                    "type": "candle"
                },
                "adaptive_mean": {
                    "color": "orange",
                    "type": "line"
                },
                "donchian_high": {
                    "color": "gray",
                    "type": "line"
                },
                "donchian_low": {
                    "color": "gray",
                    "type": "line"
                }
            },
            "subplots": {
                "Distance from Mean": {
                    "distance_from_mean": {
                        "color": "purple",
                        "type": "line"
                    },
                    "entry_std": {
                        "color": "red",
                        "type": "line"
                    },
                    "exit_std": {
                        "color": "green",
                        "type": "line"
                    },
                    "-entry_std": {
                        "color": "red",
                        "type": "line"
                    },
                    "-exit_std": {
                        "color": "green",
                        "type": "line"
                    }
                },
                "Changepoint Detection": {
                    "changepoint_score": {
                        "color": "red",
                        "type": "line"
                    },
                    "changepoint_threshold": {
                        "color": "orange",
                        "type": "line"
                    }
                },
                "Volatility": {
                    "volatility": {
                        "color": "blue",
                        "type": "line"
                    },
                    "volatility_ma": {
                        "color": "orange",
                        "type": "line"
                    }
                },
                "Regime": {
                    "bars_in_regime": {
                        "color": "green",
                        "type": "bar"
                    },
                    "regime_stabilization_bars": {
                        "color": "red",
                        "type": "line"
                    }
                }
            }
        }
