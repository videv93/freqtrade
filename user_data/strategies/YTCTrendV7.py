# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy import (
    IStrategy,
    CategoricalParameter,
    IntParameter,
)
import talib.abstract as ta


class YTCTrendV7(IStrategy):
    """
    Vectorized version of YTC Trend + Setup Detector V5 - Look-ahead bias FIXED

    This version eliminates look-ahead bias by:
    1. Proper pivot detection with confirmation lag (no future data)
    2. Vectorized trend detection based on pivot patterns
    3. Forward-filling state information correctly

    Key improvements:
    - NO look-ahead bias - pivots confirmed only after 'order' bars
    - Results match live trading behavior
    - More memory efficient
    - Easier to maintain and debug

    Trade-offs:
    - Pivot signals delayed by 'order' bars (realistic behavior)
    - May have fewer trades than the flawed version
    - More conservative, but honest backtests
    """

    # Strategy interface version - Required
    INTERFACE_VERSION = 3

    # Minimal ROI - Backup only, primary exits via S/R targets and custom logic
    # Set to 10% to allow trades room to reach HTF S/R levels
    minimal_roi = {"360": 0, "240": 0.01, "120": 0.02, "60": 0.03, "0": 0.04}

    # Stoploss - Backup only, primary stops via custom_stoploss trailing behind pivots
    # Set to 4% to allow trades room to breathe while custom stops handle protection

    stoploss = -0.04
    # Trailing stop - Conservative configuration
    trailing_stop = True
    trailing_stop_positive_offset = 0.04  # Activate at 4% profit
    trailing_stop_positive = 0.02  # Trail 2% below peak
    trailing_only_offset_is_reached = True  # Only activate after offset is reached

    # Timeframe
    timeframe = "15m"

    # Higher timeframe for S/R detection
    informative_timeframe = "4h"

    # Short
    can_short = True

    # Use exit signals
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # --- Hyperparameters ---

    # == Pivot Settings ==
    pivot_lookback = IntParameter(3, 10, default=2, space="buy", optimize=True)

    # == Setup Detection ==
    pb_swing_count = IntParameter(0, 3, default=0, space="buy", optimize=True)
    cpb_swing_count = IntParameter(3, 10, default=3, space="buy", optimize=True)

    # == MA Filter ==
    ma_length = IntParameter(10, 50, default=20, space="buy", optimize=True)
    use_ma_filter = CategoricalParameter([True, False], default=False, space="buy", optimize=True)

    # == Entry Type Controls ==
    use_pb_entries = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    use_cpb_entries = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    use_tst_entries = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    use_bof_entries = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    use_bpb_entries = CategoricalParameter([True, False], default=True, space="buy", optimize=True)

    # == Exit Controls ==
    exit_at_sr = CategoricalParameter([True, False], default=False, space="sell", optimize=True)
    sr_exit_threshold = IntParameter(1, 20, default=5, space="sell", optimize=True)  # In 0.1% units
    max_holding_bars = IntParameter(16, 128, default=96, space="sell", optimize=True)  # Max 8h hold

    # == Performance Optimization ==
    # Set to False to disable detailed pattern tracking in entry tags (saves memory)
    track_individual_patterns = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False
    )

    # --- Strategy Logic ---

    def informative_pairs(self):
        """
        Define additional informative pairs to be cached from the exchange.
        These pairs are fetched to get additional information for the strategy.

        Returns list of tuples in the format (pair, timeframe)
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Vectorized indicator calculation
        """

        # Get hyperparameters
        pivot_lb = self.pivot_lookback.value

        # --- Basic Indicators ---
        # Only calculate MA if it's actually used
        if self.use_ma_filter.value:
            ma_len = self.ma_length.value
            dataframe["ma"] = ta.SMA(dataframe, timeperiod=ma_len)

        # --- Vectorized Pivot Detection (main timeframe) ---
        dataframe = self._detect_pivots_vectorized(dataframe, pivot_lb)

        # --- Support/Resistance Detection (from higher timeframe) ---
        dataframe = self._detect_support_resistance_htf(dataframe, metadata)

        # --- Trend Detection ---
        dataframe = self._detect_trend_vectorized(dataframe)

        # --- Swing High/Low Tracking (AFTER trend - counts swings within current trend only) ---
        dataframe = self._track_swing_structure(dataframe)

        # --- Candlestick Pattern Detection ---
        dataframe = self._detect_candlestick_patterns(dataframe)

        # --- Setup Detection ---
        dataframe = self._detect_setups_vectorized(dataframe)

        return dataframe

    def _detect_pivots_vectorized(self, dataframe: DataFrame, order: int) -> DataFrame:
        """
        Detect pivot highs and lows WITHOUT look-ahead bias (VECTORIZED)

        A pivot high at bar i is confirmed at bar i+order when:
        - high[i] > high[i-order:i] AND high[i] > high[i+1:i+order+1]

        This introduces a lag of 'order' bars, which is correct behavior for live trading.
        Pivots are only marked after confirmation, not in real-time.

        Fully vectorized for performance using numpy operations.
        """
        high_array = dataframe["high"].values
        low_array = dataframe["low"].values
        n = len(dataframe)

        # Initialize output arrays
        pivot_high_values = np.full(n, np.nan, dtype=float)
        pivot_low_values = np.full(n, np.nan, dtype=float)
        is_pivot_high = np.zeros(n, dtype=bool)
        is_pivot_low = np.zeros(n, dtype=bool)

        # Only process bars that have enough history and future data
        # A pivot can only be detected at indices [order, n-order-1]
        valid_range_start = order
        valid_range_end = n - order - 1

        if valid_range_end > valid_range_start:
            # Create indices for potential pivots
            pivot_indices = np.arange(valid_range_start, valid_range_end + 1)

            # Vectorized pivot high detection
            # For each potential pivot at index i:
            # - Check if high[i] > max(high[i-order : i])
            # - Check if high[i] > max(high[i+1 : i+order+1])
            pivot_highs = high_array[pivot_indices]
            pivot_lows = low_array[pivot_indices]

            # Calculate rolling maximums for left window (excluding current bar)
            # Using pandas rolling for efficiency, then converting to numpy
            left_max_high = (
                pd.Series(high_array).rolling(window=order, min_periods=order).max().shift(1).values
            )
            left_min_low = (
                pd.Series(low_array).rolling(window=order, min_periods=order).min().shift(1).values
            )

            # Calculate rolling maximums for right window (future bars) - FULLY VECTORIZED
            # We need to look ahead, but we'll shift the results later to avoid lookahead
            right_max_high = np.full(n, np.nan, dtype=float)
            right_min_low = np.full(n, np.nan, dtype=float)

            # Use advanced indexing to create all right windows at once
            # For each pivot_idx, get indices [pivot_idx+1 : pivot_idx+order+1]
            if len(pivot_indices) > 0:
                # Create index matrix: each row is a pivot index, each column is offset from pivot+1
                right_indices = np.arange(order).reshape(1, -1) + pivot_indices.reshape(-1, 1) + 1
                # Mask for valid indices (within bounds)
                valid_mask = right_indices < n

                # Extract values for all windows using advanced indexing
                # Use np.take or direct indexing with masked arrays
                right_window_highs = np.full((len(pivot_indices), order), np.nan)
                right_window_lows = np.full((len(pivot_indices), order), np.nan)

                # Fill valid positions directly using boolean indexing
                right_window_highs[valid_mask] = high_array[right_indices[valid_mask]]
                right_window_lows[valid_mask] = low_array[right_indices[valid_mask]]

                # Calculate max/min across each window (axis=1), ignoring NaN
                # Only consider windows where we have all 'order' values
                complete_windows = np.sum(valid_mask, axis=1) == order
                right_max_high[pivot_indices[complete_windows]] = np.nanmax(
                    right_window_highs[complete_windows], axis=1
                )
                right_min_low[pivot_indices[complete_windows]] = np.nanmin(
                    right_window_lows[complete_windows], axis=1
                )

            # Vectorized comparison: pivot is higher/lower than both windows
            # Only check pivots where we have complete windows on both sides
            # For pivot highs: high[i] > max(left_window) AND high[i] > max(right_window)
            high_is_pivot = (
                (pivot_highs > left_max_high[pivot_indices])
                & (pivot_highs > right_max_high[pivot_indices])
                & ~np.isnan(left_max_high[pivot_indices])
                & ~np.isnan(right_max_high[pivot_indices])  # Valid right window
            )

            # For pivot lows: low[i] < min(left_window) AND low[i] < min(right_window)
            low_is_pivot = (
                (pivot_lows < left_min_low[pivot_indices])
                & (pivot_lows < right_min_low[pivot_indices])
                & ~np.isnan(left_min_low[pivot_indices])
                & ~np.isnan(right_min_low[pivot_indices])  # Valid right window
            )

            # Set pivot values where conditions are met
            pivot_high_values[pivot_indices[high_is_pivot]] = pivot_highs[high_is_pivot]
            pivot_low_values[pivot_indices[low_is_pivot]] = pivot_lows[low_is_pivot]
            is_pivot_high[pivot_indices[high_is_pivot]] = True
            is_pivot_low[pivot_indices[low_is_pivot]] = True

            # Filter pivots to enforce minimum distance of order + 1 between consecutive pivots
            # This prevents too many pivot points from being detected
            min_distance = 2 * order

            # Filter pivot highs
            pivot_high_indices = np.where(is_pivot_high)[0]
            if len(pivot_high_indices) > 1:
                # Keep track of which pivots to keep
                keep_pivots = np.ones(len(pivot_high_indices), dtype=bool)
                last_kept_idx = 0

                for i in range(1, len(pivot_high_indices)):
                    # Check distance from last kept pivot
                    if pivot_high_indices[i] - pivot_high_indices[last_kept_idx] < min_distance:
                        # Too close - keep the higher pivot
                        if (
                            pivot_high_values[pivot_high_indices[i]]
                            > pivot_high_values[pivot_high_indices[last_kept_idx]]
                        ):
                            # New pivot is higher, remove old one and keep new one
                            keep_pivots[last_kept_idx] = False
                            last_kept_idx = i
                        else:
                            # Old pivot is higher, remove new one
                            keep_pivots[i] = False
                    else:
                        # Far enough, keep both
                        last_kept_idx = i

                # Remove filtered pivots
                removed_indices = pivot_high_indices[~keep_pivots]
                is_pivot_high[removed_indices] = False
                pivot_high_values[removed_indices] = np.nan

            # Filter pivot lows
            pivot_low_indices = np.where(is_pivot_low)[0]
            if len(pivot_low_indices) > 1:
                # Keep track of which pivots to keep
                keep_pivots = np.ones(len(pivot_low_indices), dtype=bool)
                last_kept_idx = 0

                for i in range(1, len(pivot_low_indices)):
                    # Check distance from last kept pivot
                    if pivot_low_indices[i] - pivot_low_indices[last_kept_idx] < min_distance:
                        # Too close - keep the lower pivot
                        if (
                            pivot_low_values[pivot_low_indices[i]]
                            < pivot_low_values[pivot_low_indices[last_kept_idx]]
                        ):
                            # New pivot is lower, remove old one and keep new one
                            keep_pivots[last_kept_idx] = False
                            last_kept_idx = i
                        else:
                            # Old pivot is lower, remove new one
                            keep_pivots[i] = False
                    else:
                        # Far enough, keep both
                        last_kept_idx = i

                # Remove filtered pivots
                removed_indices = pivot_low_indices[~keep_pivots]
                is_pivot_low[removed_indices] = False
                pivot_low_values[removed_indices] = np.nan

        # Assign to dataframe
        dataframe["pivot_high"] = pivot_high_values
        dataframe["pivot_low"] = pivot_low_values
        dataframe["is_pivot_high"] = is_pivot_high
        dataframe["is_pivot_low"] = is_pivot_low

        # CRITICAL FIX: Shift pivot markers forward by 'order' bars
        # This ensures they appear at CONFIRMATION time, not pivot time
        # Without this shift, all downstream indicators have lookahead bias
        dataframe["pivot_high"] = dataframe["pivot_high"].shift(order)
        dataframe["pivot_low"] = dataframe["pivot_low"].shift(order)
        dataframe["is_pivot_high"] = (
            dataframe["is_pivot_high"].shift(order).fillna(False).astype(bool)
        )
        dataframe["is_pivot_low"] = (
            dataframe["is_pivot_low"].shift(order).fillna(False).astype(bool)
        )

        return dataframe

    def _track_swing_structure(self, dataframe: DataFrame) -> DataFrame:
        """
        Track swing highs and lows in a vectorized manner

        swing_count represents the number of trend-confirming pivots WITHIN THE CURRENT TREND.
        - In uptrend: counts each higher_high and higher_low (HH, HL, HH = 3 swings)
        - In downtrend: counts each lower_high and lower_low (LH, LL, LH = 3 swings)

        NOTE: This MUST be called AFTER trend detection to properly count swings per trend.
        """

        # Initialize higher/lower pivot tracking columns
        dataframe["higher_high"] = False
        dataframe["higher_low"] = False
        dataframe["lower_high"] = False
        dataframe["lower_low"] = False

        # Get indices of pivot highs and lows (for vectorized operations)
        # Filter dataframes to only pivot points
        pivot_highs_df = dataframe[dataframe["is_pivot_high"]].copy()
        pivot_lows_df = dataframe[dataframe["is_pivot_low"]].copy()

        # Vectorized approach: Compare consecutive pivot highs
        if len(pivot_highs_df) > 1:
            # Get current and previous pivot high values using shift
            current_highs = pivot_highs_df["pivot_high"]
            previous_highs = pivot_highs_df["pivot_high"].shift(1)

            # Compare current vs previous
            higher_high_mask = current_highs > previous_highs
            lower_high_mask = current_highs < previous_highs

            # Apply to original dataframe using the indices
            dataframe.loc[higher_high_mask.index[higher_high_mask], "higher_high"] = True
            dataframe.loc[lower_high_mask.index[lower_high_mask], "lower_high"] = True

        # Vectorized approach: Compare consecutive pivot lows
        if len(pivot_lows_df) > 1:
            # Get current and previous pivot low values using shift
            current_lows = pivot_lows_df["pivot_low"]
            previous_lows = pivot_lows_df["pivot_low"].shift(1)

            # Compare current vs previous
            higher_low_mask = current_lows > previous_lows
            lower_low_mask = current_lows < previous_lows

            # Apply to original dataframe using the indices
            dataframe.loc[higher_low_mask.index[higher_low_mask], "higher_low"] = True
            dataframe.loc[lower_low_mask.index[lower_low_mask], "lower_low"] = True

        # Create trend group IDs (increments when trend changes)
        dataframe["trend_changed"] = (dataframe["trend"] != dataframe["trend"].shift(1)).fillna(
            True
        )
        dataframe["trend_group"] = dataframe["trend_changed"].cumsum()

        # Count swings based on trend direction:
        # - Uptrend: count both higher_highs AND higher_lows
        # - Downtrend: count both lower_highs AND lower_lows
        dataframe["swing_event"] = False
        dataframe.loc[
            (dataframe["trend"] == "Uptrend")
            & (dataframe["higher_high"] | dataframe["higher_low"]),
            "swing_event",
        ] = True
        dataframe.loc[
            (dataframe["trend"] == "Downtrend")
            & (dataframe["lower_high"] | dataframe["lower_low"]),
            "swing_event",
        ] = True

        # Count swing events within each trend group
        dataframe["swing_count"] = dataframe.groupby("trend_group")["swing_event"].cumsum()

        # Clean up temporary columns including unused swing structure columns
        dataframe.drop(
            columns=[
                "trend_changed",
                "trend_group",
                "swing_event",
                "higher_high",
                "higher_low",
                "lower_high",
                "lower_low",
            ],
            inplace=True,
        )

        return dataframe

    def _detect_candlestick_patterns(self, dataframe: DataFrame) -> DataFrame:
        """
        Detect key candlestick reversal patterns using TA-Lib (YTC methodology)

        Bullish patterns (for long entries):
        - Hammer
        - Dragonfly Doji
        - Bullish Engulfing
        - Morning Star
        - Inside Bar (manual detection)

        Bearish patterns (for short entries):
        - Shooting Star
        - Gravestone Doji
        - Bearish Engulfing
        - Evening Star
        - Inside Bar (manual detection)
        """
        # --- BULLISH PATTERNS (TA-Lib returns 100 for pattern, 0 for no pattern) ---
        hammer = ta.CDLHAMMER(dataframe)
        dragonfly_doji = ta.CDLDRAGONFLYDOJI(dataframe)
        morning_star = ta.CDLMORNINGSTAR(dataframe)

        # Engulfing pattern returns 100 for bullish, -100 for bearish
        engulfing = ta.CDLENGULFING(dataframe)
        bullish_engulfing = engulfing > 0

        # --- BEARISH PATTERNS (TA-Lib returns -100 for bearish pattern, 0 for no pattern) ---
        shooting_star = ta.CDLSHOOTINGSTAR(dataframe)
        gravestone_doji = ta.CDLGRAVESTONEDOJI(dataframe)
        evening_star = ta.CDLEVENINGSTAR(dataframe)
        bearish_engulfing = engulfing < 0

        # --- INSIDE BAR PATTERN (manual detection - breakout pattern) ---
        # Detect inside bars: current bar contained within previous bar
        is_inside_bar = (dataframe["high"] < dataframe["high"].shift(1)) & (
            dataframe["low"] > dataframe["low"].shift(1)
        )

        # Vectorized approach to track mother bar levels
        # Create sequence groups: every time we enter/exit inside bar status, new sequence
        sequence_change = is_inside_bar != is_inside_bar.shift(1)
        sequence_id = sequence_change.cumsum()

        # Detect start of inside bar sequences (first inside bar after non-inside bar)
        inside_bar_start = is_inside_bar & ~is_inside_bar.shift(1).fillna(False)

        # Create series to hold mother bar values at sequence start positions
        mother_high_at_start = pd.Series(np.nan, index=dataframe.index)
        mother_low_at_start = pd.Series(np.nan, index=dataframe.index)

        # At sequence start positions, get the previous bar's high/low (mother bar)
        mother_high_at_start[inside_bar_start] = dataframe["high"].shift(1)[inside_bar_start]
        mother_low_at_start[inside_bar_start] = dataframe["low"].shift(1)[inside_bar_start]

        # Forward fill mother bar values through each sequence
        dataframe["mother_bar_high"] = mother_high_at_start.groupby(sequence_id).ffill()
        dataframe["mother_bar_low"] = mother_low_at_start.groupby(sequence_id).ffill()

        # Clear values for non-inside bars
        dataframe.loc[~is_inside_bar, "mother_bar_high"] = np.nan
        dataframe.loc[~is_inside_bar, "mother_bar_low"] = np.nan

        # Inside bar bullish breakout: close breaks above mother bar high
        inside_bar_bullish = (
            (~dataframe["mother_bar_high"].isna())
            & (dataframe["close"] > dataframe["mother_bar_high"])
            & (dataframe["close"].shift(1) <= dataframe["mother_bar_high"].shift(1))
        )

        # Inside bar bearish breakout: close breaks below mother bar low
        inside_bar_bearish = (
            (~dataframe["mother_bar_low"].isna())
            & (dataframe["close"] < dataframe["mother_bar_low"])
            & (dataframe["close"].shift(1) >= dataframe["mother_bar_low"].shift(1))
        )

        # Clean up temporary mother bar columns (no longer needed after pattern detection)
        dataframe.drop(columns=["mother_bar_high", "mother_bar_low"], inplace=True)

        # Store individual pattern flags only if detailed tracking is enabled
        if self.track_individual_patterns.value:
            dataframe["hammer"] = hammer != 0
            dataframe["dragonfly_doji"] = dragonfly_doji != 0
            dataframe["bullish_engulfing"] = bullish_engulfing
            dataframe["morning_star"] = morning_star != 0
            dataframe["shooting_star"] = shooting_star != 0
            dataframe["gravestone_doji"] = gravestone_doji != 0
            dataframe["bearish_engulfing"] = bearish_engulfing
            dataframe["evening_star"] = evening_star != 0
            dataframe["inside_bar_bullish"] = inside_bar_bullish
            dataframe["inside_bar_bearish"] = inside_bar_bearish

        # Combine bullish patterns (any pattern detected)
        dataframe["bullish_reversal_pattern"] = (
            (hammer != 0)
            | (dragonfly_doji != 0)
            | bullish_engulfing
            | (morning_star != 0)
            | inside_bar_bullish
        )

        # Combine bearish patterns (any pattern detected)
        dataframe["bearish_reversal_pattern"] = (
            (shooting_star != 0)
            | (gravestone_doji != 0)
            | bearish_engulfing
            | (evening_star != 0)
            | inside_bar_bearish
        )

        return dataframe

    def _detect_support_resistance_htf(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Detect support and resistance levels from higher timeframe (4h) pivot points

        YTC methodology uses higher timeframe swing highs/lows as S/R levels for better quality.
        This method:
        1. Fetches 4h data
        2. Detects pivots on 4h timeframe
        3. Uses recent 4h pivot highs as resistance
        4. Uses recent 4h pivot lows as support
        5. Merges S/R levels to main timeframe
        """

        use_htf_sr = False

        # Fetch higher timeframe data (4h)
        if self.dp:
            try:
                informative = self.dp.get_pair_dataframe(
                    pair=metadata["pair"], timeframe=self.informative_timeframe
                )

                # Check if we have valid data with required columns
                if (
                    informative is not None
                    and len(informative) > 0
                    and "high" in informative.columns
                ):
                    # Detect pivots on higher timeframe
                    pivot_lb = self.pivot_lookback.value
                    informative = self._detect_pivots_vectorized(informative.copy(), pivot_lb)

                    # Get the most recent pivot high and low from HTF
                    # These will be our S/R levels
                    informative["htf_resistance"] = informative["pivot_high"]
                    informative["htf_support"] = informative["pivot_low"]

                    # Forward fill to get the most recent pivot high/low at each point
                    informative["htf_resistance"] = informative["htf_resistance"].ffill()
                    informative["htf_support"] = informative["htf_support"].ffill()

                    # Merge HTF data to main timeframe
                    # Rename columns to avoid conflicts
                    informative_merge = informative[
                        ["date", "htf_resistance", "htf_support"]
                    ].copy()
                    informative_merge.columns = ["date", "resistance", "support"]

                    # Merge using merge_asof to align timestamps
                    dataframe = pd.merge_asof(
                        dataframe, informative_merge, on="date", direction="backward"
                    )
                    use_htf_sr = True
            except Exception as e:
                # If HTF data fetch fails, fall back to main timeframe
                pass

        # Fallback: Use main timeframe pivots as S/R if HTF not available
        if not use_htf_sr:
            dataframe["resistance"] = dataframe["pivot_high"].ffill()
            dataframe["support"] = dataframe["pivot_low"].ffill()

        # Calculate distance to S/R levels
        dataframe["dist_to_resistance"] = np.nan
        dataframe["dist_to_support"] = np.nan

        # Only calculate distance where S/R levels exist
        has_resistance = ~dataframe["resistance"].isna()
        has_support = ~dataframe["support"].isna()

        dataframe.loc[has_resistance, "dist_to_resistance"] = (
            (dataframe.loc[has_resistance, "resistance"] - dataframe.loc[has_resistance, "close"])
            / dataframe.loc[has_resistance, "close"]
            * 100
        )

        dataframe.loc[has_support, "dist_to_support"] = (
            (dataframe.loc[has_support, "close"] - dataframe.loc[has_support, "support"])
            / dataframe.loc[has_support, "close"]
            * 100
        )

        # Mark when price is near S/R (within threshold - use absolute value to catch both sides)
        sr_threshold = self.sr_exit_threshold.value / 10.0  # Convert to percentage
        dataframe["near_resistance"] = (~dataframe["dist_to_resistance"].isna()) & (
            dataframe["dist_to_resistance"].abs() <= sr_threshold
        )
        dataframe["near_support"] = (~dataframe["dist_to_support"].isna()) & (
            dataframe["dist_to_support"].abs() <= sr_threshold
        )

        # Mark when price breaks S/R
        dataframe["broke_resistance"] = False
        dataframe["broke_support"] = False

        has_both = has_resistance & has_support
        dataframe.loc[has_both, "broke_resistance"] = (
            dataframe.loc[has_both, "close"] > dataframe.loc[has_both, "resistance"]
        )
        dataframe.loc[has_both, "broke_support"] = (
            dataframe.loc[has_both, "close"] < dataframe.loc[has_both, "support"]
        )

        return dataframe

    def _detect_trend_vectorized(self, dataframe: DataFrame) -> DataFrame:
        """
        Vectorized trend detection based on pivot patterns

        Logic follows YTC methodology:
        1. Track last 4 pivots (ph1, ph2, pl1, pl2) with their bar indices
        2. Check for trend breaks first (2-bar confirmation)
        3. If no break, check for trend establishment/continuation
        4. Detect weakening patterns

        Uptrend: Higher highs AND higher lows
        Downtrend: Lower highs AND lower lows
        """
        n = len(dataframe)

        # Initialize trend tracking arrays
        trend = np.full(n, "Sideways", dtype=object)
        weakening = np.zeros(n, dtype=bool)

        # Initialize key level tracking
        uptrend_reversal_level = np.full(n, np.nan, dtype=float)  # leadingPL_to_highestPH
        downtrend_reversal_level = np.full(n, np.nan, dtype=float)  # leadingPH_to_lowestPL
        ph2_values = np.full(n, np.nan, dtype=float)  # Previous pivot high
        pl2_values = np.full(n, np.nan, dtype=float)  # Previous pivot low

        # State variables
        current_trend = "Sideways"
        trend_changed_bar = -1
        highest_ph = np.nan
        leading_pl_to_highest_ph = np.nan
        lowest_pl = np.nan
        leading_ph_to_lowest_pl = np.nan
        max_holding_bars_value = self.max_holding_bars.value
        is_weakening = False  # Track if trend is currently weakening

        # Track last 4 pivots
        ph1, ph2 = np.nan, np.nan
        pl1, pl2 = np.nan, np.nan
        ph1_bar, ph2_bar = -1, -1
        pl1_bar, pl2_bar = -1, -1

        # Get pivot data
        pivot_highs = dataframe["pivot_high"].values
        pivot_lows = dataframe["pivot_low"].values
        is_pivot_high = dataframe["is_pivot_high"].values
        is_pivot_low = dataframe["is_pivot_low"].values
        close_prices = dataframe["close"].values

        # Iterate through bars to build trend state
        for i in range(n):
            # Update pivot tracking when new pivots form
            if is_pivot_high[i]:
                # Shift pivot history
                ph2, ph2_bar = ph1, ph1_bar
                ph1, ph1_bar = pivot_highs[i], i

            if is_pivot_low[i]:
                # Shift pivot history
                pl2, pl2_bar = pl1, pl1_bar
                pl1, pl1_bar = pivot_lows[i], i

            # ===== IF NO TREND BREAK, CHECK FOR TREND ESTABLISHMENT =====
            if current_trend == "Sideways" or i - trend_changed_bar > max_holding_bars_value:
                # Need 4 pivots to establish trend
                has_pivots = (
                    not np.isnan(ph1)
                    and not np.isnan(ph2)
                    and not np.isnan(pl1)
                    and not np.isnan(pl2)
                    and ph1_bar >= 0
                    and ph2_bar >= 0
                    and pl1_bar >= 0
                    and pl2_bar >= 0
                )

                if has_pivots:
                    # UPTREND DETECTION: pl2 < ph2 < pl1 < ph1
                    uptrend_time_order = pl2_bar < ph2_bar < pl1_bar < ph1_bar
                    if uptrend_time_order:
                        higher_highs = ph1 > ph2
                        higher_lows = pl1 > pl2

                        if higher_highs and higher_lows:
                            current_trend = "Uptrend"

                            # Initialize uptrend structure
                            highest_ph = ph1
                            leading_pl_to_highest_ph = pl1
                            lowest_pl = np.nan
                            leading_ph_to_lowest_pl = np.nan
                            trend_changed_bar = pl2_bar
                            is_weakening = True  # Trend changed - start as weakening

                    # DOWNTREND DETECTION: ph2 < pl2 < ph1 < pl1
                    downtrend_time_order = ph2_bar < pl2_bar < ph1_bar < pl1_bar
                    if downtrend_time_order:
                        lower_highs = ph1 < ph2
                        lower_lows = pl1 < pl2

                        if lower_highs and lower_lows:
                            current_trend = "Downtrend"

                            # Initialize downtrend structure
                            lowest_pl = pl1
                            leading_ph_to_lowest_pl = ph1
                            highest_ph = np.nan
                            leading_pl_to_highest_ph = np.nan
                            trend_changed_bar = ph2_bar
                            is_weakening = True  # Trend changed - start as weakening

            # ===== CHECK FOR TREND BREAKS FIRST =====
            trend_reversed = False

            # UPTREND BREAK: Price closes below key level for 2+ bars
            if current_trend == "Uptrend" and not np.isnan(leading_pl_to_highest_ph):
                if close_prices[i] < leading_pl_to_highest_ph and is_pivot_low[i]:
                    current_trend = "Downtrend"
                    is_weakening = False  # Reset weakness on trend reversal

                    # Initialize downtrend structure
                    lowest_pl = pl1
                    leading_ph_to_lowest_pl = ph1
                    highest_ph = np.nan
                    leading_pl_to_highest_ph = np.nan
                    trend_changed_bar = ph2_bar

            # DOWNTREND BREAK: Price closes above key level for 2+ bars
            if current_trend == "Downtrend" and not np.isnan(leading_ph_to_lowest_pl):
                if close_prices[i] > leading_ph_to_lowest_pl and is_pivot_high[i]:
                    current_trend = "Uptrend"
                    is_weakening = False  # Reset weakness on trend reversal

                    # Initialize uptrend structure
                    highest_ph = ph1
                    leading_pl_to_highest_ph = pl1
                    lowest_pl = np.nan
                    leading_ph_to_lowest_pl = np.nan
                    trend_changed_bar = pl2_bar

            # Update trend structure when new pivots form in established trend
            if current_trend == "Uptrend" and is_pivot_high[i]:
                if pivot_highs[i] > highest_ph or np.isnan(highest_ph):
                    highest_ph = pivot_highs[i]
                    # Update the leading low to current structure
                    leading_pl_to_highest_ph = pl1
                    is_weakening = False  # New higher high - trend no longer weakening
                elif pivot_highs[i] < highest_ph:
                    is_weakening = True  # Lower high - trend weakening
            if current_trend == "Downtrend" and is_pivot_low[i]:
                if pivot_lows[i] < lowest_pl or np.isnan(lowest_pl):
                    lowest_pl = pivot_lows[i]
                    # Update the leading high to current structure
                    leading_ph_to_lowest_pl = ph1
                    is_weakening = False  # New lower low - trend no longer weakening
                elif pivot_lows[i] > lowest_pl:
                    is_weakening = True  # Higher low - trend weakening

            # Store current state
            trend[i] = current_trend
            weakening[i] = is_weakening  # Store current weakening state
            uptrend_reversal_level[i] = leading_pl_to_highest_ph
            downtrend_reversal_level[i] = leading_ph_to_lowest_pl
            ph2_values[i] = ph2  # Store previous pivot high
            pl2_values[i] = pl2  # Store previous pivot low

        # Assign to dataframe
        dataframe["trend"] = trend
        dataframe["trend_weakening"] = weakening
        dataframe["uptrend_reversal_level"] = uptrend_reversal_level
        dataframe["downtrend_reversal_level"] = downtrend_reversal_level
        dataframe["ph2"] = ph2_values
        dataframe["pl2"] = pl2_values

        return dataframe

    def _detect_setups_vectorized(self, dataframe: DataFrame) -> DataFrame:
        """
        Vectorized setup detection (PB, CPB, TST, BOF, BPB)

        YTC Setup Types:
        - PB: Simple pullback in trend
        - CPB: Complex pullback in trend
        - TST: Test of S/R expected to hold
        - BOF: Breakout failure (price breaks S/R then reverses)
        - BPB: Breakout pullback (price breaks S/R, holds, pulls back for continuation)
        """
        pb_count = self.pb_swing_count.value
        cpb_count = self.cpb_swing_count.value
        use_ma = self.use_ma_filter.value

        # Track most recent pivot high and low at each bar (forward-fill)
        dataframe["last_pivot_high"] = dataframe["pivot_high"].ffill()
        dataframe["last_pivot_low"] = dataframe["pivot_low"].ffill()

        # Initialize setup columns (direction-specific to prevent opposite entries)
        dataframe["is_pb_long"] = False
        dataframe["is_pb_short"] = False
        dataframe["is_cpb_long"] = False
        dataframe["is_cpb_short"] = False
        dataframe["is_tst_long"] = False
        dataframe["is_tst_short"] = False
        dataframe["is_bof_long"] = False
        dataframe["is_bof_short"] = False
        dataframe["is_bpb_long"] = False
        dataframe["is_bpb_short"] = False

        # ========== TREND-BASED SETUPS (PB & CPB) ==========
        # PB, CPB Long Setup (Continuation in Uptrend)
        # When trend is weakening, require price to break previous pivot low then recover
        # Detect recent break of last pivot low (close went below it)
        dataframe["broke_last_pivot_low"] = (~dataframe["last_pivot_low"].isna()) & (
            dataframe["close"] < dataframe["last_pivot_low"]
        )

        # PB, CPB Short Setup (Continuation in Downtrend)
        # When trend is weakening, require price to break previous pivot high then recover
        # Detect recent break of last pivot high (close went above it)
        dataframe["broke_last_pivot_high"] = (~dataframe["last_pivot_high"].isna()) & (
            dataframe["close"] > dataframe["last_pivot_high"]
        )

        # PB Long Setup (Pullback in Uptrend)
        # Limited to swing_count <= 3 to catch early trend entries only
        # Requires bullish reversal candlestick pattern
        pb_long_condition = (
            (dataframe["trend"] == "Uptrend")
            & (dataframe["swing_count"] >= pb_count)
            & (dataframe["swing_count"] < cpb_count)  # Early trend only (max 3 swings)
            & (dataframe["bullish_reversal_pattern"])  # Bullish candlestick pattern
            & (~dataframe["near_resistance"])
            & (
                # If trend NOT weakening: require price in pullback zone (between support and ph2)
                (
                    (~dataframe["trend_weakening"])
                    & (~dataframe["uptrend_reversal_level"].isna())
                    & (~dataframe["ph2"].isna())
                    & (dataframe["close"] >= dataframe["uptrend_reversal_level"])
                    & (dataframe["close"] <= dataframe["ph2"])
                )
                |
                # If trend weakening: require break and recovery of last pivot low
                ((dataframe["trend_weakening"]) & (dataframe["broke_last_pivot_low"]))
            )
        )

        if use_ma:
            pb_long_condition = pb_long_condition & (dataframe["close"] < dataframe["ma"])

        # PB Short Setup (Pullback in Downtrend)
        # Limited to swing_count <= 3 to catch early trend entries only
        # Requires bearish reversal candlestick pattern
        pb_short_condition = (
            (dataframe["trend"] == "Downtrend")
            & (dataframe["swing_count"] >= pb_count)
            & (dataframe["swing_count"] <= cpb_count)  # Early trend only (max 3 swings)
            & (dataframe["bearish_reversal_pattern"])  # Bearish candlestick pattern
            & (~dataframe["near_support"])
            & (
                # If trend NOT weakening: require price in pullback zone (between resistance and pl2)
                (
                    (~dataframe["trend_weakening"])
                    & (~dataframe["downtrend_reversal_level"].isna())
                    & (~dataframe["pl2"].isna())
                    & (dataframe["close"] <= dataframe["downtrend_reversal_level"])
                    & (dataframe["close"] >= dataframe["pl2"])
                )
                |
                # If trend weakening: require break and recovery of last pivot high
                ((dataframe["trend_weakening"]) & (dataframe["broke_last_pivot_high"]))
            )
        )

        if use_ma:
            pb_short_condition = pb_short_condition & (dataframe["close"] > dataframe["ma"])

        cpb_long_condition = (
            (dataframe["trend"] == "Uptrend")
            & (dataframe["swing_count"] >= cpb_count)
            & (dataframe["bullish_reversal_pattern"])  # Bullish candlestick pattern
            & (~dataframe["near_resistance"])
            & (
                # If trend NOT weakening: require price in pullback zone (between support and ph2)
                (
                    (~dataframe["trend_weakening"])
                    & (~dataframe["uptrend_reversal_level"].isna())
                    & (~dataframe["ph2"].isna())
                    & (dataframe["close"] >= dataframe["uptrend_reversal_level"])
                    & (dataframe["close"] <= dataframe["ph2"])
                )
                |
                # If trend weakening: require break and recovery of last pivot low
                ((dataframe["trend_weakening"]) & (dataframe["broke_last_pivot_low"]))
            )
        )

        if use_ma:
            cpb_long_condition = cpb_long_condition & (dataframe["close"] < dataframe["ma"])

        cpb_short_condition = (
            (dataframe["trend"] == "Downtrend")
            & (dataframe["swing_count"] >= cpb_count)
            & (dataframe["bearish_reversal_pattern"])  # Bearish candlestick pattern
            & (~dataframe["near_support"])
            & (
                # If trend NOT weakening: require price in pullback zone (between resistance and pl2)
                (
                    (~dataframe["trend_weakening"])
                    & (~dataframe["downtrend_reversal_level"].isna())
                    & (~dataframe["pl2"].isna())
                    & (dataframe["close"] <= dataframe["downtrend_reversal_level"])
                    & (dataframe["close"] >= dataframe["pl2"])
                )
                |
                # If trend weakening: require break and recovery of last pivot high
                ((dataframe["trend_weakening"]) & (dataframe["broke_last_pivot_high"]))
            )
        )

        if use_ma:
            cpb_short_condition = cpb_short_condition & (dataframe["close"] > dataframe["ma"])

        # ========== S/R-BASED SETUPS (TST, BOF, BPB) ==========

        # TST Long Setup (Test of Support expected to hold)
        tst_long_condition = (
            (dataframe["near_support"])  # Price near support
            & (dataframe["is_pivot_low"])  # Formed a pivot low (price stalled/reversed)
            & (~dataframe["broke_support"])  # Support held (not broken)
        )

        # TST Short Setup (Test of Resistance expected to hold)
        tst_short_condition = (
            (dataframe["near_resistance"])  # Price near resistance
            & (dataframe["is_pivot_high"])  # Formed a pivot high (price stalled/reversed)
            & (~dataframe["broke_resistance"])  # Resistance held (not broken)
        )

        # BOF Long Setup (Breakout Failure at Support - price breaks down then reverses up)
        # Detect: broke support on previous candle, now reversing back above support
        dataframe["prev_broke_support"] = (
            dataframe["broke_support"].shift(1).fillna(False).astype(bool)
        )
        bof_long_condition = (
            (dataframe["prev_broke_support"])  # Previously broke support
            & (~dataframe["broke_support"])  # Now back above support
            & (dataframe["close"] > dataframe["support"])  # Confirmed back above
        )

        # BOF Short Setup (Breakout Failure at Resistance - price breaks up then reverses down)
        dataframe["prev_broke_resistance"] = (
            dataframe["broke_resistance"].shift(1).fillna(False).astype(bool)
        )
        bof_short_condition = (
            (dataframe["prev_broke_resistance"])  # Previously broke resistance
            & (~dataframe["broke_resistance"])  # Now back below resistance
            & (dataframe["close"] < dataframe["resistance"])  # Confirmed back below
        )

        # BPB Long Setup (Breakout Pullback at Resistance - breaks up, pulls back, continues up)
        # After breaking resistance, price pulls back to retest it as new support
        # Only in uptrend - this is a continuation pattern
        dataframe["resistance_broken_recently"] = (
            dataframe["broke_resistance"]
            | dataframe["broke_resistance"].shift(1)
            | dataframe["broke_resistance"].shift(2)
        ).fillna(False)

        bpb_long_condition = (
            (dataframe["resistance_broken_recently"])  # Resistance was broken recently
            & (dataframe["near_resistance"])  # Pulled back to test old resistance (now support)
            & (dataframe["is_pivot_low"])  # Formed a pivot low (pullback complete)
            & (dataframe["close"] > dataframe["resistance"])  # Still above old resistance
        )

        # BPB Short Setup (Breakout Pullback at Support - breaks down, pulls back, continues down)
        # Only in downtrend - this is a continuation pattern
        dataframe["support_broken_recently"] = (
            dataframe["broke_support"]
            | dataframe["broke_support"].shift(1)
            | dataframe["broke_support"].shift(2)
        ).fillna(False)

        bpb_short_condition = (
            (dataframe["support_broken_recently"])  # Support was broken recently
            & (dataframe["near_support"])  # Pulled back to test old support (now resistance)
            & (dataframe["is_pivot_high"])  # Formed a pivot high (pullback complete)
            & (dataframe["close"] < dataframe["support"])  # Still below old support
        )

        # Mark all setups with direction-specific columns
        dataframe.loc[pb_long_condition, "is_pb_long"] = True
        dataframe.loc[pb_short_condition, "is_pb_short"] = True
        dataframe.loc[cpb_long_condition, "is_cpb_long"] = True
        dataframe.loc[cpb_short_condition, "is_cpb_short"] = True
        dataframe.loc[tst_long_condition, "is_tst_long"] = True
        dataframe.loc[tst_short_condition, "is_tst_short"] = True
        dataframe.loc[bof_long_condition, "is_bof_long"] = True
        dataframe.loc[bof_short_condition, "is_bof_short"] = True
        dataframe.loc[bpb_long_condition, "is_bpb_long"] = True
        dataframe.loc[bpb_short_condition, "is_bpb_short"] = True

        # Clean up temporary columns
        dataframe.drop(
            columns=[
                "prev_broke_support",
                "prev_broke_resistance",
                "resistance_broken_recently",
                "support_broken_recently",
                "broke_last_pivot_low",
                "broke_last_pivot_high",
                "last_pivot_high",
                "last_pivot_low",
            ],
            inplace=True,
        )

        return dataframe

    def _get_pattern_name(self, row, direction: str) -> str:
        """
        Get the name of the candlestick pattern for a given row

        Args:
            row: DataFrame row
            direction: 'long' or 'short'

        Returns:
            Pattern name abbreviation (e.g., 'hmr', 'eng', 'mstr', 'sstr', 'ib', etc.)
        """
        if direction == "long":
            # Check bullish patterns in priority order
            if row.get("hammer", False):
                return "hmr"
            elif row.get("bullish_engulfing", False):
                return "eng"
            elif row.get("morning_star", False):
                return "mstr"
            elif row.get("dragonfly_doji", False):
                return "dfd"
            elif row.get("inside_bar_bullish", False):
                return "ib"
        else:  # short
            # Check bearish patterns in priority order
            if row.get("shooting_star", False):
                return "sstr"
            elif row.get("bearish_engulfing", False):
                return "eng"
            elif row.get("evening_star", False):
                return "estr"
            elif row.get("gravestone_doji", False):
                return "gsd"
            elif row.get("inside_bar_bearish", False):
                return "ib"

        return ""  # No pattern detected

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enter on confirmed setup based on enabled entry types

        Supports all 5 YTC setup types:
        - PB/CPB: Trend-following pullback entries
        - TST: Test of S/R expected to hold
        - BOF: Breakout failure reversal
        - BPB: Breakout pullback continuation

        """
        # Get entry type parameters
        use_pb = self.use_pb_entries.value
        use_cpb = self.use_cpb_entries.value
        use_tst = self.use_tst_entries.value
        use_bof = self.use_bof_entries.value
        use_bpb = self.use_bpb_entries.value

        # ========== TREND-BASED ENTRIES (PB & CPB) ==========

        # Long entry conditions - PB Setup
        if use_pb:
            pb_long_mask = (
                (dataframe["trend"] == "Uptrend")
                & (dataframe["is_pb_long"])
                & (~dataframe["near_resistance"])  # Avoid entering near resistance
                & (dataframe["volume"] > 0)
            )
            # Get pattern names for these entries if tracking is enabled
            if self.track_individual_patterns.value:
                for idx in dataframe[pb_long_mask].index:
                    pattern = self._get_pattern_name(dataframe.loc[idx], "long")
                    tag = f"pb_long_{pattern}" if pattern else "pb_long"
                    dataframe.loc[idx, ["enter_long", "enter_tag"]] = (1, tag)
            else:
                dataframe.loc[pb_long_mask, ["enter_long", "enter_tag"]] = (1, "pb_long")

        # Long entry conditions - CPB Setup
        if use_cpb:
            cpb_long_mask = (
                (dataframe["trend"] == "Uptrend")
                & (dataframe["is_cpb_long"])
                & (~dataframe["near_resistance"])  # Avoid entering near resistance
                & (dataframe["volume"] > 0)
            )
            # Get pattern names for these entries if tracking is enabled
            if self.track_individual_patterns.value:
                for idx in dataframe[cpb_long_mask].index:
                    pattern = self._get_pattern_name(dataframe.loc[idx], "long")
                    tag = f"cpb_long_{pattern}" if pattern else "cpb_long"
                    dataframe.loc[idx, ["enter_long", "enter_tag"]] = (1, tag)
            else:
                dataframe.loc[cpb_long_mask, ["enter_long", "enter_tag"]] = (1, "cpb_long")

        # Short entry conditions - PB Setup
        if use_pb:
            pb_short_mask = (
                (dataframe["trend"] == "Downtrend")
                & (dataframe["is_pb_short"])
                & (dataframe["volume"] > 0)
            )
            # Get pattern names for these entries if tracking is enabled
            if self.track_individual_patterns.value:
                for idx in dataframe[pb_short_mask].index:
                    pattern = self._get_pattern_name(dataframe.loc[idx], "short")
                    tag = f"pb_short_{pattern}" if pattern else "pb_short"
                    dataframe.loc[idx, ["enter_short", "enter_tag"]] = (1, tag)
            else:
                dataframe.loc[pb_short_mask, ["enter_short", "enter_tag"]] = (1, "pb_short")

        # Short entry conditions - CPB Setup
        if use_cpb:
            cpb_short_mask = (
                (dataframe["trend"] == "Downtrend")
                & (dataframe["is_cpb_short"])
                & (dataframe["volume"] > 0)
            )
            # Get pattern names for these entries if tracking is enabled
            if self.track_individual_patterns.value:
                for idx in dataframe[cpb_short_mask].index:
                    pattern = self._get_pattern_name(dataframe.loc[idx], "short")
                    tag = f"cpb_short_{pattern}" if pattern else "cpb_short"
                    dataframe.loc[idx, ["enter_short", "enter_tag"]] = (1, tag)
            else:
                dataframe.loc[cpb_short_mask, ["enter_short", "enter_tag"]] = (1, "cpb_short")

        # ========== S/R-BASED ENTRIES (TST, BOF, BPB) ==========

        # Long entry conditions - TST Setup (Test of Support)
        if use_tst:
            dataframe.loc[
                ((dataframe["is_tst_long"]) & (dataframe["volume"] > 0)),
                ["enter_long", "enter_tag"],
            ] = (1, "tst_long")

        # Short entry conditions - TST Setup (Test of Resistance)
        if use_tst:
            dataframe.loc[
                (
                    (dataframe["is_tst_short"])
                    & (dataframe["near_resistance"])
                    & (dataframe["volume"] > 0)
                ),
                ["enter_short", "enter_tag"],
            ] = (1, "tst_short")

        # Long entry conditions - BOF Setup (Breakout Failure at Support)
        if use_bof:
            dataframe.loc[
                (
                    (dataframe["is_bof_long"])
                    & (dataframe["close"] > dataframe["support"])
                    & (dataframe["volume"] > 0)
                ),
                ["enter_long", "enter_tag"],
            ] = (1, "bof_long")

        # Short entry conditions - BOF Setup (Breakout Failure at Resistance)
        if use_bof:
            dataframe.loc[
                (
                    (dataframe["is_bof_short"])
                    & (dataframe["close"] < dataframe["resistance"])
                    & (dataframe["volume"] > 0)
                ),
                ["enter_short", "enter_tag"],
            ] = (1, "bof_short")

        # Long entry conditions - BPB Setup (Breakout Pullback at Resistance)
        if use_bpb:
            dataframe.loc[
                (
                    (dataframe["is_bpb_long"])
                    & (dataframe["close"] > dataframe["resistance"])
                    & (dataframe["volume"] > 0)
                ),
                ["enter_long", "enter_tag"],
            ] = (1, "bpb_long")

        # Short entry conditions - BPB Setup (Breakout Pullback at Support)
        if use_bpb:
            dataframe.loc[
                (
                    (dataframe["is_bpb_short"])
                    & (dataframe["close"] < dataframe["support"])
                    & (dataframe["volume"] > 0)
                ),
                ["enter_short", "enter_tag"],
            ] = (1, "bpb_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        YTC-compliant exit logic with multiple exit conditions:
        1. S/R target reached (profit taking at key levels)
        2. Trend change (premise invalidated)

        Exit conditions:
        - S/R target hit (if enabled)
        - Long exits: Trend is no longer Uptrend
        - Short exits: Trend is no longer Downtrend
        """
        # Get exit parameters
        use_sr_exit = self.exit_at_sr.value
        sr_threshold = self.sr_exit_threshold.value / 10.0  # Convert to percentage

        # ========== S/R-BASED EXITS (Profit Target) ==========
        if use_sr_exit:
            # Long exit: Price reached resistance (profit target)
            # Exit if close to resistance (use absolute value to catch both sides)
            dataframe.loc[
                (
                    (~dataframe["resistance"].isna())
                    & (dataframe["dist_to_resistance"].abs() <= sr_threshold)
                ),
                ["exit_long", "exit_tag"],
            ] = (1, "sr_target")

            # Short exit: Price reached support (profit target)
            # Exit if close to support (use absolute value to catch both sides)
            dataframe.loc[
                (
                    (~dataframe["support"].isna())
                    & (dataframe["dist_to_support"].abs() <= sr_threshold)
                ),
                ["exit_short", "exit_tag"],
            ] = (1, "sr_target")

        # ========== TREND CHANGE EXITS (Premise Invalidated) ==========

        # Detect when trend changes (transition point only)
        prev_trend = dataframe["trend"].shift(1).fillna("Sideways")

        # Exit long when trend changes FROM Uptrend to something else
        dataframe.loc[
            (prev_trend == "Uptrend") & (dataframe["trend"] != "Uptrend"),
            ["exit_long", "exit_tag"],
        ] = (1, "trend_change")

        # Exit short when trend changes FROM Downtrend to something else
        dataframe.loc[
            (prev_trend == "Downtrend") & (dataframe["trend"] != "Downtrend"),
            ["exit_short", "exit_tag"],
        ] = (1, "trend_change")

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        """
        Custom exit logic for PB, CPB, TST and BOF trades

        PB/CPB: Exit when new highest high (long) or lowest low (short) forms
        TST/BOF: Exit at 1% profit target due to higher risk near S/R levels
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return None

        # ========== TST/BOF TRADES: 1% profit target ==========
        if trade.enter_tag and (
            trade.enter_tag.startswith("tst_") or trade.enter_tag.startswith("bof_")
        ):
            # Exit at 1% profit for TST and BOF trades
            if current_profit >= 0.01:
                return f"{trade.enter_tag}_tp_1pct"

        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """
        Custom stoploss logic for PB, CPB, TST and BOF trades

        TST/BOF: Exit when S/R breaks again (premise invalidated)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1].squeeze()

        # ========== TST/BOF TRADES: S/R break stoploss ==========
        if trade.enter_tag and (
            trade.enter_tag.startswith("tst_") or trade.enter_tag.startswith("bof_")
        ):
            # TST Long / BOF Long: Exit if support breaks down
            if trade.enter_tag in ["tst_long", "bof_long"]:
                # Check both last candle and current rate against support
                support_level = last_candle.get("support")
                if support_level and not pd.isna(support_level):
                    # Exit if current rate breaks below support
                    if current_rate < support_level:
                        return 0.001  # Force immediate exit (0.1% stoploss)
                # Also check if last candle broke support
                elif last_candle.get("broke_support", False):
                    return 0.001  # Force immediate exit (0.1% stoploss)

            # TST Short / BOF Short: Exit if resistance breaks up
            if trade.enter_tag in ["tst_short", "bof_short"]:
                # Check both last candle and current rate against resistance
                resistance_level = last_candle.get("resistance")
                if resistance_level and not pd.isna(resistance_level):
                    # Exit if current rate breaks above resistance
                    if current_rate > resistance_level:
                        return 0.001  # Force immediate exit (0.1% stoploss)
                # Also check if last candle broke resistance
                elif last_candle.get("broke_resistance", False):
                    return 0.001  # Force immediate exit (0.1% stoploss)

        return None

    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Customize leverage for each new trade (futures mode only).

        Returns:
            float: Leverage value between 1.0 and max_leverage
        """
        return 1.0
