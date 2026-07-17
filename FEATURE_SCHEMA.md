# Feature Schema

Concise schema for every dimension of `MarketStructureEngine.feature_vector()`, **post feature-optimization pass** -- see `FEATURE_OPTIMIZATION_REPORT.md` for the change log and `FEATURE_REFERENCE.md` for full per-feature detail.

**Total dimensions: 185**

| Index | Feature Name | Data Type | Description | Module |
|---|---|---|---|---|
| 1 | `n_candles` | float64 (integer-valued count) | Number of validated candles currently loaded in the engine. | Other |
| 2 | `trend_direction` | float64 (categorical: -1/0/1) | Overall trend classification from the net HH/HL vs LH/LL score over the last `trend_swing_lookback` (default 6) swing highs/lows. | Trend Engine |
| 3 | `trend_higher_highs` | float64 (integer-valued count) | Count of up-moves among the last `trend_swing_lookback` swing highs. | Trend Engine |
| 4 | `trend_higher_lows` | float64 (integer-valued count) | Count of up-moves among the last `trend_swing_lookback` swing lows. | Trend Engine |
| 5 | `trend_lower_highs` | float64 (integer-valued count) | Count of down-moves among the last `trend_swing_lookback` swing highs. | Trend Engine |
| 6 | `trend_lower_lows` | float64 (integer-valued count) | Count of down-moves among the last `trend_swing_lookback` swing lows. | Trend Engine |
| 7 | `trend_strength` | float64 (bounded ratio) | Magnitude of the net directional score; 1.0 = perfectly one-sided HH/HL or LH/LL structure. | Trend Engine |
| 8 | `trend_duration_bars` | float64 (integer-valued count) | Bars elapsed since the current directional leg (or, if sideways, the last swing) began. | Trend Engine |
| 9 | `trend_momentum` | float64 (continuous) | Least-squares slope of recent swing prices per bar, normalized by the last close. | Trend Engine |
| 10 | `trend_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when total (hh+lh+hl+ll) > 0, i.e. >= 2 swing highs or >= 2 swing lows existed; 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 11 | `structure_last_bos_direction` | float64 (categorical: -1/0/1) | Direction of the most recent Break of Structure (close beyond the active swing level). | BOS Engine |
| 12 | `structure_last_bos_strength` | float64 (continuous, ≥0) | ATR-normalized breach distance of the most recent BOS. | BOS Engine |
| 13 | `structure_bars_since_bos` | float64 (integer-valued count, sentinel -1) | Bars since the last confirmed BOS; -1 sentinel if none has occurred. | BOS Engine |
| 14 | `structure_bullish_bos_count` | float64 (integer-valued count) | Total bullish BOS events detected across the whole loaded series. | BOS Engine |
| 15 | `structure_bearish_bos_count` | float64 (integer-valued count) | Total bearish BOS events detected across the whole loaded series. | BOS Engine |
| 16 | `structure_last_choch_direction` | float64 (categorical: -1/0/1) | Direction of the new character after the most recent Change of Character (first break against the prior BOS run). | CHOCH Engine |
| 17 | `structure_last_choch_strength` | float64 (continuous, ≥0) | ATR-normalized strength inherited from the structure break that caused the CHOCH. | CHOCH Engine |
| 18 | `structure_bars_since_choch` | float64 (integer-valued count, sentinel -1) | Bars since the last CHOCH; -1 sentinel if none has occurred. | CHOCH Engine |
| 19 | `structure_choch_count` | float64 (integer-valued count) | Total number of direction flips (CHOCH events) in the break sequence. | CHOCH Engine |
| 20 | `pa_current_close` | float64 (raw price) | Close price of the most recent candle. | Price Action |
| 21 | `pa_previous_close` | float64 (raw price) | Close price of the second-to-last candle. | Price Action |
| 22 | `pa_price_change` | float64 (continuous) | Absolute close-to-close change. | Price Action |
| 23 | `pa_price_return` | float64 (continuous) | Fractional close-to-close return. (Canonical return representation -- pa_percentage_return was removed as an exact x100 duplicate.) | Price Action |
| 24 | `pa_body_size` | float64 (continuous, ≥0) | Absolute candle body size. | Price Action |
| 25 | `pa_upper_wick` | float64 (continuous, ≥0) | Upper wick length. | Price Action |
| 26 | `pa_lower_wick` | float64 (continuous, ≥0) | Lower wick length. | Price Action |
| 27 | `pa_body_ratio` | float64 (bounded ratio) | Body size as a fraction of the candle's full range. | Price Action |
| 28 | `pa_candle_range` | float64 (continuous, >0) | Full high-low range of the last candle. | Price Action |
| 29 | `pa_atr_ratio` | float64 (continuous, ≥0) | Last candle's range relative to current ATR(14). Inherits validity from ind_atr_valid. | Price Action |
| 30 | `pa_gap_up` | float64 (binary 0/1) | Whether the candle gapped up over the prior candle's high (no overlap). | Price Action |
| 31 | `pa_gap_down` | float64 (binary 0/1) | Whether the candle gapped down under the prior candle's low. | Price Action |
| 32 | `pa_distance_from_ema20` | float64 (continuous) | Close distance from EMA(20), ATR-normalized. Inherits validity from ind_ema_20_valid and ind_atr_valid. | Price Action |
| 33 | `pa_distance_from_ema50` | float64 (continuous) | Close distance from EMA(50), ATR-normalized. Inherits validity from ind_ema_50_valid and ind_atr_valid. | Price Action |
| 34 | `pa_distance_from_ema200` | float64 (continuous) | Close distance from EMA(200), ATR-normalized. Inherits validity from ind_ema_200_valid and ind_atr_valid. | Price Action |
| 35 | `pa_distance_from_vwap` | float64 (continuous) | Close distance from series-cumulative VWAP, ATR-normalized. Inherits validity from ind_vwap_valid and ind_atr_valid. | Price Action |
| 36 | `vol_historical_volatility` | float64 (continuous, ≥0) | Stdev of log-returns over the last `hist_vol_window` (20) bars. | Other |
| 37 | `vol_average_candle_size` | float64 (continuous, ≥0) | Mean high-low range over the last 20 bars. | Other |
| 38 | `vol_average_wick_size` | float64 (continuous, ≥0) | Mean total wick size (upper+lower) over the last 20 bars. | Other |
| 39 | `vol_expansion` | float64 (binary 0/1) | Whether current ATR exceeds `expansion_ratio` (1.5x) times its 20-bar mean. | Other |
| 40 | `vol_compression` | float64 (binary 0/1) | Whether current ATR is below `compression_ratio` (0.66x) times its 20-bar mean. | Other |
| 41 | `vol_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= hist_vol_window (20); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 42 | `micro_impulse_length` | float64 (continuous, ≥0) | Mean absolute size of the last up-to-3 swing legs that moved WITH the current trend direction. | Other |
| 43 | `micro_correction_length` | float64 (continuous, ≥0) | Mean absolute size of the last up-to-3 swing legs that moved AGAINST the current trend direction. | Other |
| 44 | `micro_impulse_ratio` | float64 (continuous, ≥0) | Ratio of impulse to correction leg size. | Other |
| 45 | `micro_retracement_pct` | float64 (continuous, bounded) | Size of the last swing leg relative to the prior leg, capped at 100%. | Other |
| 46 | `micro_extension_pct` | float64 (continuous, ≥0) | Amount by which the last leg exceeds 100% of the prior leg (0 if it does not). | Other |
| 47 | `micro_swing_velocity` | float64 (continuous) | Size of the last swing leg per bar elapsed. | Other |
| 48 | `micro_swing_acceleration` | float64 (continuous) | Change in swing velocity vs. the previous leg. | Other |
| 49 | `micro_time_between_swings` | float64 (continuous, ≥0) | Mean bar-count between the last up-to-3 consecutive swing pairs. | Other |
| 50 | `micro_average_swing_length` | float64 (continuous, ≥0) | Mean absolute size of ALL swing legs across the loaded series. | Other |
| 51 | `micro_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when len(swings) >= 3; 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 52 | `session_is_sydney` | float64 (binary 0/1) | Whether the last candle's UTC hour falls in the configured Sydney session [21:00-06:00). | Session Features |
| 53 | `session_is_asian` | float64 (binary 0/1) | Whether the last candle's UTC hour falls in the configured Asian session [00:00-09:00). | Session Features |
| 54 | `session_is_london` | float64 (binary 0/1) | Whether the last candle's UTC hour falls in the configured London session [07:00-16:00). | Session Features |
| 55 | `session_is_newyork` | float64 (binary 0/1) | Whether the last candle's UTC hour falls in the configured New York session [12:00-21:00). | Session Features |
| 56 | `session_hour` | float64 (integer-valued) | UTC hour of the last candle's timestamp. | Session Features |
| 57 | `session_minute` | float64 (integer-valued) | UTC minute of the last candle's timestamp. | Session Features |
| 58 | `session_day_of_week` | float64 (categorical/integer) | Day of week of the last candle (Monday=0). | Session Features |
| 59 | `session_month` | float64 (categorical/integer) | Calendar month of the last candle. | Session Features |
| 60 | `spread_current` | float64 (raw price, ≥0) | Broker spread (ask-bid) of the last candle. | Liquidity |
| 61 | `spread_rolling_avg` | float64 (continuous, ≥0) | Mean spread over the trailing `spread_window` (20) candles. | Liquidity |
| 62 | `spread_spike` | float64 (binary 0/1) | Whether current spread is >= `spread_spike_multiple` (2.0x) its rolling average. | Liquidity |
| 63 | `spread_atr_ratio` | float64 (continuous, ≥0) | Current spread relative to current ATR(14) -- spread's share of typical candle range. | Liquidity |
| 64 | `spread_percentile` | float64 (bounded ratio) | Fraction of the trailing `spread_window` spreads at or below the current spread. | Liquidity |
| 65 | `spread_volatility` | float64 (continuous, ≥0) | Rolling standard deviation of spread over `spread_window` (20) candles. | Liquidity |
| 66 | `spread_distance_from_avg` | float64 (continuous) | Signed relative deviation of current spread from its rolling average. | Liquidity |
| 67 | `spread_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when the input DataFrame has a 'spread' column AND n_candles >= spread_window (20); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 68 | `sr_support_zone_count` | float64 (integer-valued count) | Number of clustered support zones kept (strongest first, capped at `max_zones`=10). | Support & Resistance |
| 69 | `sr_resistance_zone_count` | float64 (integer-valued count) | Number of clustered resistance zones kept (capped at `max_zones`=10). | Support & Resistance |
| 70 | `sr_nearest_support_strength` | float64 (continuous, ≥0) | Strength score of the nearest support zone below/at price (0 if none exists -- check sr_support_valid). | Support & Resistance |
| 71 | `sr_nearest_support_touches` | float64 (integer-valued count) | Number of swing lows clustered into the nearest support zone. | Support & Resistance |
| 72 | `sr_nearest_support_width` | float64 (continuous, ≥0) | Price width of the nearest support zone. | Support & Resistance |
| 73 | `sr_nearest_resistance_strength` | float64 (continuous, ≥0) | Strength score of the nearest resistance zone at/above price (0 if none exists -- check sr_resistance_valid). | Support & Resistance |
| 74 | `sr_nearest_resistance_touches` | float64 (integer-valued count) | Number of swing highs clustered into the nearest resistance zone. | Support & Resistance |
| 75 | `sr_nearest_resistance_width` | float64 (continuous, ≥0) | Price width of the nearest resistance zone. | Support & Resistance |
| 76 | `sr_distance_to_support` | float64 (continuous) | ATR-normalized distance from close down to the nearest support zone center. | Support & Resistance |
| 77 | `sr_distance_to_resistance` | float64 (continuous) | ATR-normalized distance from close up to the nearest resistance zone center. | Support & Resistance |
| 78 | `sr_support_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when a nearest support zone currently exists; 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 79 | `sr_resistance_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when a nearest resistance zone currently exists; 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 80 | `liq_equal_highs` | float64 (integer-valued count) | Total swing-high touches across all detected buy-side (equal-highs) pools. | Liquidity |
| 81 | `liq_equal_lows` | float64 (integer-valued count) | Total swing-low touches across all detected sell-side (equal-lows) pools. | Liquidity |
| 82 | `liq_pool_count` | float64 (integer-valued count) | Total number of liquidity pools detected (both sides). | Liquidity |
| 83 | `liq_sweep_count` | float64 (integer-valued count) | Total number of liquidity sweeps detected across the series. | Liquidity |
| 84 | `liq_buy_side` | float64 (continuous, ≥0) | Total strength of unswept buy-side pools currently above price. | Liquidity |
| 85 | `liq_sell_side` | float64 (continuous, ≥0) | Total strength of unswept sell-side pools currently below price. | Liquidity |
| 86 | `liq_last_sweep_direction` | float64 (categorical: -1/0/1) | Direction of the most recent liquidity sweep. | Liquidity |
| 87 | `liq_last_sweep_size` | float64 (continuous, ≥0) | ATR-normalized wick penetration of the most recent sweep beyond its pool. | Liquidity |
| 88 | `fvg_bullish_count` | float64 (integer-valued count) | Total bullish FVGs (low[i] > high[i-2]) detected across the series. | Fair Value Gap |
| 89 | `fvg_bearish_count` | float64 (integer-valued count) | Total bearish FVGs (high[i] < low[i-2]) detected across the series. | Fair Value Gap |
| 90 | `fvg_unfilled_count` | float64 (integer-valued count) | Number of detected gaps not yet fully filled by later price action. | Fair Value Gap |
| 91 | `fvg_nearest_direction` | float64 (categorical: -1/0/1) | Direction of the nearest unfilled FVG to the current close. | Fair Value Gap |
| 92 | `fvg_nearest_size_atr` | float64 (continuous, ≥0) | ATR-normalized height of the nearest unfilled gap. | Fair Value Gap |
| 93 | `fvg_nearest_age` | float64 (integer-valued count) | Bars elapsed since the nearest unfilled gap formed. | Fair Value Gap |
| 94 | `fvg_distance_to_nearest` | float64 (continuous, ≥0) | ATR-normalized distance from close to the nearest unfilled gap's closest edge. | Fair Value Gap |
| 95 | `ob_bullish_count` | float64 (integer-valued count) | Total bullish (demand) order blocks detected across the series. | Order Blocks |
| 96 | `ob_bearish_count` | float64 (integer-valued count) | Total bearish (supply) order blocks detected across the series. | Order Blocks |
| 97 | `ob_unmitigated_count` | float64 (integer-valued count) | Number of order blocks not yet mitigated (price hasn't traded back through the origin side). | Order Blocks |
| 98 | `ob_nearest_direction` | float64 (categorical: -1/0/1) | Direction of the nearest unmitigated order block. | Order Blocks |
| 99 | `ob_nearest_strength` | float64 (continuous, ≥0) | ATR-normalized size of the displacement that validated the nearest order block. | Order Blocks |
| 100 | `ob_nearest_freshness` | float64 (continuous, bounded) | Freshness decay of the nearest order block based on retest count. | Order Blocks |
| 101 | `ob_nearest_age` | float64 (integer-valued count) | Bars since the nearest order block formed. | Order Blocks |
| 102 | `ob_distance_to_nearest` | float64 (continuous, ≥0) | ATR-normalized absolute distance from close to the nearest unmitigated block's edge. | Order Blocks |
| 103 | `ind_adx` | float64 (continuous) | Average Directional Index (Wilder), trend-strength oscillator. | Indicators |
| 104 | `ind_adx_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (adx_period (14)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 105 | `ind_atr` | float64 (continuous) | Average True Range (Wilder smoothing, period 14). Canonical ATR used by every ATR-normalized feature engine-wide (vol_atr/ind_true_range duplicates removed). | Indicators |
| 106 | `ind_atr_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (atr_period (14)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 107 | `ind_bb_lower` | float64 (continuous) | Bollinger lower band (SMA(20) - 2σ). | Indicators |
| 108 | `ind_bb_lower_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (bollinger_period (20)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 109 | `ind_bb_upper` | float64 (continuous) | Bollinger upper band (SMA(20) + 2σ). | Indicators |
| 110 | `ind_bb_upper_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (bollinger_period (20)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 111 | `ind_cci` | float64 (continuous) | Commodity Channel Index (period 20). | Indicators |
| 112 | `ind_cci_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (cci_period (20)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 113 | `ind_ema_100` | float64 (continuous) | Exponential moving average, span 100. | Indicators |
| 114 | `ind_ema_100_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (100); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 115 | `ind_ema_20` | float64 (continuous) | Exponential moving average, span 20. | Indicators |
| 116 | `ind_ema_20_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (20); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 117 | `ind_ema_200` | float64 (continuous) | Exponential moving average, span 200. | Indicators |
| 118 | `ind_ema_200_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (200); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 119 | `ind_ema_50` | float64 (continuous) | Exponential moving average, span 50. | Indicators |
| 120 | `ind_ema_50_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (50); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 121 | `ind_macd` | float64 (continuous) | MACD line: fast EMA(12) minus slow EMA(26). | Indicators |
| 122 | `ind_macd_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (max(macd_fast, macd_slow) = 26); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 123 | `ind_macd_signal` | float64 (continuous) | 9-period EMA of the MACD line. | Indicators |
| 124 | `ind_macd_signal_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (max(macd_fast,macd_slow)+macd_signal = 35); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 125 | `ind_momentum` | float64 (continuous) | Raw price momentum over 10 bars. | Indicators |
| 126 | `ind_momentum_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (momentum_period (10)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 127 | `ind_relative_volume` | float64 (continuous) | Current volume relative to its 20-bar SMA. | Volume |
| 128 | `ind_relative_volume_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (volume_ma_period (20)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 129 | `ind_roc` | float64 (continuous) | Rate of change over 10 bars, percent. | Indicators |
| 130 | `ind_roc_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (roc_period (10)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 131 | `ind_rolling_median` | float64 (continuous) | Rolling median of close over 20 bars. | Indicators |
| 132 | `ind_rolling_median_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (rolling_window (20)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 133 | `ind_rsi` | float64 (continuous) | Relative Strength Index (Wilder, period 14). | Indicators |
| 134 | `ind_rsi_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (rsi_period (14)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 135 | `ind_sma` | float64 (continuous) | Simple moving average of close, period 20. Canonical moving-average feature -- ind_rolling_mean and ind_bb_middle removed as exact duplicates under default config. | Indicators |
| 136 | `ind_sma_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (sma_period (20)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 137 | `ind_std_dev` | float64 (continuous) | Rolling standard deviation of close over 20 bars. Canonical dispersion feature -- ind_rolling_variance removed (deterministic sqrt transform of this). | Indicators |
| 138 | `ind_std_dev_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (rolling_window (20)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 139 | `ind_stoch_d` | float64 (continuous) | Stochastic %D: 3-bar SMA of %K. | Indicators |
| 140 | `ind_stoch_d_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (stoch_k+stoch_d = 17); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 141 | `ind_stoch_k` | float64 (continuous) | Stochastic %K (period 14). | Indicators |
| 142 | `ind_stoch_k_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (stoch_k (14)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 143 | `ind_tick_volume` | float64 (continuous) | Raw tick_volume column if the input DataFrame provides one, else zeros (now paired with ind_tick_volume_valid -- see Task 5). | Volume |
| 144 | `ind_tick_volume_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when the input DataFrame has a 'tick_volume' column; 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 145 | `ind_true_range` | float64 (continuous) | Current bar's True Range. Canonical TR feature -- vol_true_range removed as an exact duplicate. | Indicators |
| 146 | `ind_true_range_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (1 (defined from bar 1 via prev_close fallback)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 147 | `ind_volatility` | float64 (continuous) | Rolling stdev of pct-change returns over 20 bars. Canonical return-volatility feature -- vol_rolling_volatility removed as an exact duplicate. | Indicators |
| 148 | `ind_volatility_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (rolling_window+1 = 21); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 149 | `ind_volume_delta` | float64 (continuous) | 20-bar rolling sum of signed volume (volume * sign(close-open)). | Volume |
| 150 | `ind_volume_delta_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (volume_ma_period (20)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 151 | `ind_volume_ma` | float64 (continuous) | Simple moving average of volume, period 20. | Volume |
| 152 | `ind_volume_ma_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (volume_ma_period (20)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 153 | `ind_volume_ratio` | float64 (continuous) | Current volume relative to the previous bar's volume. | Volume |
| 154 | `ind_volume_ratio_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (2); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 155 | `ind_volume_spike` | float64 (binary 0/1) | Whether relative volume exceeds `volume_spike_multiple` (2.0x). | Volume |
| 156 | `ind_volume_spike_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (volume_ma_period (20)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 157 | `ind_volume_trend` | float64 (continuous) | 20-bar-over-20-bar change in the volume moving average. | Volume |
| 158 | `ind_volume_trend_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (volume_ma_period*2 = 40); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 159 | `ind_vwap` | float64 (continuous) | Cumulative Volume-Weighted Average Price from the start of the loaded series. | Indicators |
| 160 | `ind_vwap_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (1); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 161 | `ind_williams_r` | float64 (continuous) | Williams %R (period 14). | Indicators |
| 162 | `ind_williams_r_valid` | float64 (binary 0/1, data-quality flag) | Data-quality flag: 1.0 only when n_candles >= warm-up requirement (williams_period (14)); 0.0 means every other field in this group is a placeholder, not a real reading (see Missing-Value Strategy). | Other |
| 163 | `pat_bearish_engulfing` | float64 (binary 0/1) | Bearish candle whose body fully engulfs the prior bullish candle's body. | Candle Patterns |
| 164 | `pat_bearish_harami` | float64 (binary 0/1) | Small bearish candle contained within the prior bullish candle's body. | Candle Patterns |
| 165 | `pat_bearish_marubozu` | float64 (binary 0/1) | Bearish candle with almost no wicks (body ≥95% of range). | Candle Patterns |
| 166 | `pat_bearish_pin_bar` | float64 (binary 0/1) | Long upper wick, minimal lower wick -- rejection from highs. | Candle Patterns |
| 167 | `pat_bullish_engulfing` | float64 (binary 0/1) | Bullish candle whose body fully engulfs the prior bearish candle's body. | Candle Patterns |
| 168 | `pat_bullish_harami` | float64 (binary 0/1) | Small bullish candle contained within the prior bearish candle's body. | Candle Patterns |
| 169 | `pat_bullish_marubozu` | float64 (binary 0/1) | Bullish candle with almost no wicks (body ≥95% of range). | Candle Patterns |
| 170 | `pat_bullish_pin_bar` | float64 (binary 0/1) | Long lower wick, minimal upper wick -- rejection from lows. | Candle Patterns |
| 171 | `pat_dark_cloud_cover` | float64 (binary 0/1) | Bearish candle opening above the prior bullish candle's high and closing below its midpoint. | Candle Patterns |
| 172 | `pat_doji` | float64 (binary 0/1) | Body is a negligible fraction (≤10%) of the candle's range. | Candle Patterns |
| 173 | `pat_dragonfly_doji` | float64 (binary 0/1) | Doji with a long lower wick and negligible upper wick. | Candle Patterns |
| 174 | `pat_evening_star` | float64 (binary 0/1) | 3-bar top reversal: large bullish candle, small-bodied middle candle, bearish close below the first candle's midpoint. | Candle Patterns |
| 175 | `pat_gravestone_doji` | float64 (binary 0/1) | Doji with a long upper wick and negligible lower wick. | Candle Patterns |
| 176 | `pat_hammer` | float64 (binary 0/1) | Small body near the top of the range with a long lower wick (≥2x body) and minimal upper wick. | Candle Patterns |
| 177 | `pat_inside_bar` | float64 (binary 0/1) | Current candle's range is fully contained within the prior candle's range. | Candle Patterns |
| 178 | `pat_inverted_hammer` | float64 (binary 0/1) | Small body near the bottom of the range with a long upper wick (≥2x body) and minimal lower wick. | Candle Patterns |
| 179 | `pat_morning_star` | float64 (binary 0/1) | 3-bar bottom reversal: large bearish candle, small-bodied middle candle, bullish close above the first candle's midpoint. | Candle Patterns |
| 180 | `pat_outside_bar` | float64 (binary 0/1) | Current candle's range fully contains the prior candle's range. | Candle Patterns |
| 181 | `pat_piercing_line` | float64 (binary 0/1) | Bullish candle opening below the prior bearish candle's low and closing above its midpoint. | Candle Patterns |
| 182 | `pat_shooting_star` | float64 (binary 0/1) | Inverted hammer geometry immediately following a bullish candle. | Candle Patterns |
| 183 | `pat_spinning_top` | float64 (binary 0/1) | Small body with both wicks ≥25% of range (but not a doji). | Candle Patterns |
| 184 | `pat_three_black_crows` | float64 (binary 0/1) | Three consecutive long-bodied bearish candles, each closing lower than the last. | Candle Patterns |
| 185 | `pat_three_white_soldiers` | float64 (binary 0/1) | Three consecutive long-bodied bullish candles, each closing higher than the last. | Candle Patterns |
