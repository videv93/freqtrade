import logging
from datetime import datetime

from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IStrategy, timeframe_to_prev_date
from freqtrade.persistence import Trade


logger = logging.getLogger(__name__)


class MeanReversionRegimeWithKelly(IStrategy):
    """
    Mean Reversion Strategy with Regime Filter and Kelly Position Sizing.

    Converted from Pine Script: "Mean Reversion (Regime Filter Updated)"

    Regime Logic (based on last trade exit):
    - Regime 0: Neutral (allow both long/short on extremes)
    - Regime 1: Uptrend (long only on mean pullback)
    - Regime -1: Downtrend (short only on mean rally)

    Entry Signals:
    - lie: low < min_20 and regime == 0 (long in sideways)
    - lea: low < mean_20 and regime == 1 (long in uptrend)
    - sae: high > max_20 and regime == 0 (short in sideways)
    - sei: high > mean_20 and regime == -1 (short in downtrend)
    - laf: regime shift from 0 to 1 (long on uptrend flip), exit when price < mean_20
    - sif: regime shift from 0 to -1 (short on downtrend flip), exit when price > mean_20

    Leverage Config:
    - Add to config.json: "regime": 1 (or -1, 0) to set target regime
    - If config regime matches current detected regime → 5x leverage
    - Otherwise → 1x leverage (conservative)
    """

    INTERFACE_VERSION = 3

    can_short = True
    use_custom_exit = True
    lookback_period = 23
    atr_period = 14
    atr_multiplier = 2.0

    kelly_mult = 0.5
    max_risk_cap = 0.10
    default_risk = 0.01
    capital_available_ratio = 1.0

    minimal_roi = {"0": 100}
    stoploss = -0.15
    timeframe = "1h"

    plot_config = {
        "main_plot": {
            "min_20": {"color": "red"},
            "max_20": {"color": "green"},
            "mean_20": {"color": "gray"},
            "min_20_buffer": {"color": "darkred"},
            "max_20_buffer": {"color": "darkgreen"},
        },
        "subplots": {
            "Regime": {
                "regime": {"color": "purple", "type": "line"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["lowest_20"] = (
            dataframe["low"].shift(1).rolling(window=self.lookback_period).min()
        )
        dataframe["highest_20"] = (
            dataframe["high"].shift(1).rolling(window=self.lookback_period).max()
        )

        dataframe["min_20"] = dataframe["lowest_20"].rolling(window=5).mean()
        dataframe["max_20"] = dataframe["highest_20"].rolling(window=5).mean()
        dataframe["mean_20"] = (dataframe["min_20"] + dataframe["max_20"]) / 2

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)

        dataframe["min_20_buffer"] = dataframe["min_20"] - dataframe["atr"] * self.atr_multiplier
        dataframe["max_20_buffer"] = dataframe["max_20"] + dataframe["atr"] * self.atr_multiplier

        if "regime" not in dataframe.columns:
            dataframe["regime"] = 0
        if self.config["runmode"].value in ("live", "dry_run"):
            # Update regime based on last trade for current candle
            dataframe.loc[dataframe.index[-1], "regime"] = self.get_regime(metadata["pair"])

        # Track regime shifts for LAF/SIF signals
        dataframe["regime_shift"] = dataframe["regime"].diff()

        return dataframe

    def _get_last_trade_tag(self, pair: str) -> tuple[str | None, bool]:
        """Get the last closed trade's entry tag and whether it was a win for a specific pair."""
        query = Trade.get_trades_query([Trade.is_open.is_(False), Trade.pair == pair])
        query = query.order_by(Trade.close_date.desc()).limit(1)
        trades = Trade.session.scalars(query).all()

        if not trades:
            return None, False

        t = trades[0]
        is_win = t.close_profit > 0
        entry_tag = t.enter_tag

        return entry_tag, is_win

    def get_regime(self, pair: str) -> int:
        """
        Get regime based on last trade exit comment.

        From Pine Script:
        - lie_lose or sei_win → regime = -1
        - sae_lose or lea_win → regime = 1
        - Default → regime = 0
        """
        entry_tag, is_win = self._get_last_trade_tag(pair)

        if entry_tag is None:
            return 0

        postfix = "_win" if is_win else "_lose"
        exit_comment = f"{entry_tag}{postfix}"

        if exit_comment in ("lie_lose", "sei_win"):
            return -1
        elif exit_comment in ("sae_lose", "lea_win"):
            return 1
        else:
            return 0

    def calculate_edge_metrics(self, trades: list[Trade]) -> tuple[float, float, float, float]:
        """
        Calculate edge metrics from trade history.

        Returns:
            tuple: (win_rate, risk_reward_ratio, expectancy, kelly_fraction)

        Formulas:
            - Win Rate (W) = wins / total_trades
            - Risk Reward Ratio (R) = avg_win / avg_loss
            - Expectancy = (W * R) - (1 - W) = (W * R) - L
            - Kelly Fraction = W - [(1 - W) / R]
        """
        if not trades:
            return 0.0, 0.0, 0.0, 0.0

        total_win_amt = 0.0
        total_loss_amt = 0.0
        wins = 0
        losses = 0

        for t in trades:
            profit = t.close_profit_abs or 0
            if profit > 0:
                wins += 1
                total_win_amt += profit
            elif profit < 0:
                losses += 1
                total_loss_amt += abs(profit)

        total_trades = len(trades)
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        avg_win = total_win_amt / wins if wins > 0 else 0.0
        avg_loss = total_loss_amt / losses if losses > 0 else 0.0
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        if risk_reward_ratio > 0:
            expectancy = (win_rate * risk_reward_ratio) - (1 - win_rate)
            kelly = win_rate - ((1 - win_rate) / risk_reward_ratio)
        else:
            expectancy = 0.0
            kelly = 0.0

        return win_rate, risk_reward_ratio, expectancy, kelly

    def calculate_kelly_fraction(self, pair: str) -> float:
        """
        Calculate Kelly fraction based on entire strategy trade history for a specific pair.

        Kelly Formula: W - [(1-W)/R]
        Where W = win rate, R = risk_reward_ratio (avg_win / avg_loss)
        """
        query = Trade.get_trades_query([Trade.is_open.is_(False), Trade.pair == pair])
        trades = Trade.session.scalars(query).all()

        if not trades:
            logger.info(f"Kelly {pair}: No trades, using default_risk={self.default_risk}")
            return self.default_risk

        win_rate, rr_ratio, expectancy, kelly = self.calculate_edge_metrics(trades)

        if rr_ratio == 0:
            logger.info(
                f"Kelly {pair}: trades={len(trades)} W={win_rate:.2%} R=0 (no losses), "
                f"using default_risk={self.default_risk}"
            )
            return self.default_risk

        kelly_adjusted = max(0, kelly * self.kelly_mult)
        result = min(kelly_adjusted, self.max_risk_cap)
        result = result if result > 0 else self.default_risk

        logger.info(
            f"Kelly {pair}: trades={len(trades)} W={win_rate:.2%} R={rr_ratio:.2f} "
            f"E={expectancy:.4f} K={kelly:.4f} -> adjusted={kelly_adjusted:.4f} "
            f"result={result:.4f}"
        )

        return result

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["low"] < dataframe["min_20"]) & (dataframe["regime"] == 0),
            ["enter_long", "enter_tag"],
        ] = (1, "lie")

        dataframe.loc[
            (dataframe["low"] < dataframe["mean_20"]) & (dataframe["regime"] == 1),
            ["enter_long", "enter_tag"],
        ] = (1, "lea")

        dataframe.loc[
            (dataframe["high"] > dataframe["max_20"]) & (dataframe["regime"] == 0),
            ["enter_short", "enter_tag"],
        ] = (1, "sae")

        dataframe.loc[
            (dataframe["high"] > dataframe["mean_20"]) & (dataframe["regime"] == -1),
            ["enter_short", "enter_tag"],
        ] = (1, "sei")

        # LAF: Long only on regime flip from 0 to 1
        dataframe.loc[
            (dataframe["regime_shift"] == 1) & (dataframe["regime"].shift(1) == 0),
            ["enter_long", "enter_tag"],
        ] = (1, "laf")

        # SIF: Short only on regime flip from 0 to -1
        dataframe.loc[
            (dataframe["regime_shift"] == -1) & (dataframe["regime"].shift(1) == 0),
            ["enter_short", "enter_tag"],
        ] = (1, "sif")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False

        last = dataframe.iloc[-1]
        entry_tag = trade.enter_tag or ""
        entry_candle = dataframe.loc[
            dataframe["date"] == timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        ]
        if entry_candle.empty:
            logger.debug(f"Exit {pair} [{entry_tag}]: entry candle not found")
            return False

        entry = entry_candle.iloc[0]
        is_profit = current_profit > 0
        postfix = "_win" if is_profit else "_lose"

        if trade.is_short:
            if entry_tag == "sae":
                if current_rate <= last["mean_20"] or current_rate > entry["max_20_buffer"]:
                    return f"sae{postfix}"
            elif entry_tag == "sei":
                if current_rate < last["min_20"] or current_rate > entry["max_20"]:
                    return f"sei{postfix}"
            elif entry_tag == "sif":
                # SIF: Exit when price closes above mean_20
                if current_rate > last["mean_20"]:
                    return f"sif{postfix}"
        else:
            if entry_tag == "lie":
                if current_rate >= last["mean_20"] or current_rate < entry["min_20_buffer"]:
                    return f"lie{postfix}"
            elif entry_tag == "lea":
                if current_rate > last["max_20"] or current_rate < entry["min_20"]:
                    return f"lea{postfix}"
            elif entry_tag == "laf":
                # LAF: Exit when price closes below mean_20
                if current_rate < last["mean_20"]:
                    return f"laf{postfix}"

        return False

    def calculate_stoploss(self, entry_tag: str | None, current_rate: float, last: dict) -> float:
        """
        Calculate stoploss distance based on entry tag and indicators.

        Returns stoploss as a ratio (e.g., 0.02 for 2% stop).
        """

        if entry_tag == "lie":
            stop_price = last["min_20_buffer"]
            stop_dist = current_rate - stop_price
        elif entry_tag == "lea":
            stop_price = last["min_20"]
            stop_dist = current_rate - stop_price
        elif entry_tag == "sae":
            stop_price = last["max_20_buffer"]
            stop_dist = stop_price - current_rate
        elif entry_tag == "sei":
            stop_price = last["max_20"]
            stop_dist = stop_price - current_rate
        else:
            stop_dist = current_rate * self.default_risk  # Default 1% stoploss

        stoploss = stop_dist / current_rate

        return stoploss

    def calculate_position_size(
        self,
        pair: str,
        available_capital: float,
        allowed_risk: float,
        stoploss: float,
    ) -> float:
        """
        Calculate position size using Edge-style risk management.

        Edge Formula:
            - capital_at_risk = available_capital * allowed_risk
            - position_size = capital_at_risk / stoploss

        Args:
            pair: Trading pair
            available_capital: Capital available for trading
            allowed_risk: Risk per trade as ratio (e.g., 0.01 for 1%)
            stoploss: Stoploss as ratio (e.g., 0.02 for 2%)

        Returns:
            Position size (stake amount)
        """
        capital_at_risk = available_capital * allowed_risk
        position_size = capital_at_risk / stoploss

        logger.info(
            f"Position {pair}: available={available_capital:.2f} "
            f"risk={allowed_risk:.4f} stoploss={stoploss:.4%} "
            f"capital_at_risk={capital_at_risk:.2f} position_size={position_size:.2f}"
        )

        return position_size

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Calculate stake amount using Edge-style position sizing with Kelly risk.

        Edge Position Sizing Logic:
            1. Get available capital = total_balance * capital_available_ratio
            2. Get allowed risk per trade (from Kelly or default)
            3. Calculate stoploss from entry conditions
            4. Position size = (available_capital * allowed_risk) / stoploss
            5. Stake = position_size / leverage
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake

        last = dataframe.iloc[-1]

        stoploss = self.calculate_stoploss(entry_tag, current_rate, last)

        if self.config["runmode"].value in ("live", "dry_run"):
            allowed_risk = self.calculate_kelly_fraction(pair)
        else:
            allowed_risk = self.default_risk

        stake_currency = self.config["stake_currency"]
        if self.wallets:
            total_balance = self.wallets.get_total(stake_currency)
        else:
            total_balance = proposed_stake

        available_capital = total_balance * self.capital_available_ratio

        stake = self.calculate_position_size(pair, available_capital, allowed_risk, stoploss)

        final_stake = max(min(stake, max_stake), min_stake)

        logger.info(
            f"Stake {pair} [{entry_tag}] {side}: "
            f"stake={stake:.2f} -> final={final_stake:.2f} "
            f"(min={min_stake:.2f} max={max_stake:.2f})"
        )

        return final_stake

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Calculate leverage based on regime config match.

        If regime config in config.json matches current detected regime:
            leverage = 5.0
        Otherwise:
            leverage = 1.0

        Add to config.json: "regime": 1 (or -1, 0) to enable regime matching
        """
        regime_config = self.config.get("regime")
        
        if regime_config is None:
            # No config set, use default 5x
            return 5.0
        
        current_regime = self.get_regime(pair)
        
        if regime_config == current_regime:
            # Config regime matches detected regime → boost to 5x
            logger.info(
                f"Leverage {pair}: regime_config={regime_config} matches detected={current_regime} -> 5.0x"
            )
            return 5.0
        else:
            # Config regime doesn't match → conservative 1x
            logger.info(
                f"Leverage {pair}: regime_config={regime_config} != detected={current_regime} -> 1.0x"
            )
            return 1.0
