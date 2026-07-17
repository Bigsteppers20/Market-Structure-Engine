# Feature Diagnostics Report

Phase 2 of the Linear Regression Engine diagnostics suite -- per-target feature audit (`linear_regression.feature_diagnostics`): Pearson/Spearman/mutual information vs. each target, Variance Inflation Factor, near-constant features, and highly correlated pairs. Recommendations only -- nothing here removes a feature.

## Target: `next_close`

- Near-constant features (all targets, same list): 55
- High-VIF features (VIF > 10): 72
- Highly correlated pairs (|Pearson| >= 0.95): 35

**Top 10 features by mutual information:**

| Feature | MI (nats) | Pearson | Spearman | VIF | Importance |
|---|---:|---:|---:|---:|---:|
| `pa_current_close` | 1.4477 | 0.9672 | 0.9611 | 1000000028.28 | 0.0002 |
| `pa_previous_close` | 1.3813 | 0.9596 | 0.9505 | 1000000028.28 | 0.0002 |
| `ind_ema_20` | 1.1976 | 0.9361 | 0.9237 | 20177.54 | 0.0002 |
| `ind_rolling_median` | 1.1521 | 0.9171 | 0.9065 | 615.47 | 0.0002 |
| `ind_sma` | 1.1453 | 0.9240 | 0.9113 | 1000000028.28 | 0.0002 |
| `ind_bb_upper` | 1.1021 | 0.9186 | 0.8994 | 1000000028.28 | 0.0002 |
| `ind_bb_lower` | 1.0844 | 0.8744 | 0.8733 | 1000000028.28 | 0.0002 |
| `ind_ema_50` | 0.9516 | 0.8865 | 0.8684 | 32971.72 | 0.0001 |
| `ind_ema_100` | 0.8854 | 0.8165 | 0.7985 | 8013.13 | 0.0001 |
| `structure_last_choch_strength` | 0.8693 | -0.1235 | -0.1741 | 3.83 | 0.0001 |

**Recommendations:**

- 55 near-constant feature(s) carry almost no information for ANY target (same list regardless of target): n_candles, trend_momentum, trend_valid, pa_upper_wick, pa_lower_wick, pa_gap_up, pa_gap_down, vol_historical_volatility, vol_average_wick_size, vol_compression, .... Candidates for removal only if this holds across ALL targets, not just this one.
- 'structure_last_bos_direction' and 'structure_last_choch_direction' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'structure_last_choch_direction' has lower mutual information with this target (0.0379 vs 0.0379 for 'structure_last_bos_direction') -- if only one is kept, prefer 'structure_last_bos_direction'.
- 'ind_stoch_k' and 'ind_williams_r' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_stoch_k' has lower mutual information with this target (0.0077 vs 0.0084 for 'ind_williams_r') -- if only one is kept, prefer 'ind_williams_r'.
- 'vol_historical_volatility' and 'ind_volatility' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'vol_historical_volatility' has lower mutual information with this target (0.3001 vs 0.3003 for 'ind_volatility') -- if only one is kept, prefer 'ind_volatility'.
- 'pa_price_change' and 'pa_price_return' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'pa_price_return' has lower mutual information with this target (0.0115 vs 0.0184 for 'pa_price_change') -- if only one is kept, prefer 'pa_price_change'.
- 'ind_momentum' and 'ind_roc' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_momentum' has lower mutual information with this target (0.0525 vs 0.0579 for 'ind_roc') -- if only one is kept, prefer 'ind_roc'.
- 'pa_candle_range' and 'ind_true_range' are highly correlated (Pearson r=+0.999) -- carry near-duplicate information. 'ind_true_range' has lower mutual information with this target (0.0468 vs 0.0486 for 'pa_candle_range') -- if only one is kept, prefer 'pa_candle_range'.
- 'ind_ema_20' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (1.1453 vs 1.1976 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'ind_rolling_median' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (1.1453 vs 1.1521 for 'ind_rolling_median') -- if only one is kept, prefer 'ind_rolling_median'.
- 'ind_ema_20' and 'ind_rolling_median' are highly correlated (Pearson r=+0.995) -- carry near-duplicate information. 'ind_rolling_median' has lower mutual information with this target (1.1521 vs 1.1976 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'sr_nearest_support_strength' and 'sr_nearest_support_touches' are highly correlated (Pearson r=+0.993) -- carry near-duplicate information. 'sr_nearest_support_touches' has lower mutual information with this target (0.1640 vs 0.3708 for 'sr_nearest_support_strength') -- if only one is kept, prefer 'sr_nearest_support_strength'.
- 'pa_current_close' and 'pa_previous_close' are highly correlated (Pearson r=+0.992) -- carry near-duplicate information. 'pa_previous_close' has lower mutual information with this target (1.3813 vs 1.4477 for 'pa_current_close') -- if only one is kept, prefer 'pa_current_close'.
- 'trend_higher_highs' and 'trend_lower_highs' are highly correlated (Pearson r=-0.981) -- carry near-duplicate information. 'trend_higher_highs' has lower mutual information with this target (0.2162 vs 0.2543 for 'trend_lower_highs') -- if only one is kept, prefer 'trend_lower_highs'.
- 'pa_distance_from_ema20' and 'ind_rsi' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_rsi' has lower mutual information with this target (0.0623 vs 0.0827 for 'pa_distance_from_ema20') -- if only one is kept, prefer 'pa_distance_from_ema20'.
- 'ind_ema_20' and 'ind_ema_50' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_ema_50' has lower mutual information with this target (0.9516 vs 1.1976 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'ind_ema_50' and 'ind_sma' are highly correlated (Pearson r=+0.977) -- carry near-duplicate information. 'ind_ema_50' has lower mutual information with this target (0.9516 vs 1.1453 for 'ind_sma') -- if only one is kept, prefer 'ind_sma'.
- 72 feature(s) have VIF > 10.0 (severe multicollinearity): structure_last_bos_direction (VIF=1000000028.3), structure_last_choch_direction (VIF=1000000028.3), pa_current_close (VIF=1000000028.3), pa_previous_close (VIF=1000000028.3), pa_price_change (VIF=1000000028.3), pa_body_size (VIF=1000000028.3), pa_upper_wick (VIF=1000000028.3), pa_lower_wick (VIF=1000000028.3), pa_candle_range (VIF=1000000028.3), ind_bb_lower (VIF=1000000028.3), .... A linear model's coefficients on these are unstable (small data changes swing them sharply) even if overall R^2 looks fine -- consider regularization (ridge/lasso, already supported) over removal, since removal changes what the coefficients mean.
- 8 feature(s) show both near-zero linear correlation (|Pearson| < 0.02) and near-zero mutual information (< 0.01) with this specific target -- likely low-information FOR THIS TARGET specifically (may still matter for other targets or for the Logistic Regression Engine): pa_body_ratio, spread_percentile, pat_doji, pat_inside_bar, pat_morning_star, pat_outside_bar, pat_spinning_top, pat_three_white_soldiers.

## Target: `next_high`

- Near-constant features (all targets, same list): 55
- High-VIF features (VIF > 10): 72
- Highly correlated pairs (|Pearson| >= 0.95): 35

**Top 10 features by mutual information:**

| Feature | MI (nats) | Pearson | Spearman | VIF | Importance |
|---|---:|---:|---:|---:|---:|
| `pa_current_close` | 1.5280 | 0.9703 | 0.9648 | 1000000028.28 | 0.0002 |
| `pa_previous_close` | 1.4103 | 0.9628 | 0.9542 | 1000000028.28 | 0.0002 |
| `ind_ema_20` | 1.2523 | 0.9376 | 0.9255 | 20177.54 | 0.0002 |
| `ind_rolling_median` | 1.1779 | 0.9183 | 0.9077 | 615.47 | 0.0002 |
| `ind_bb_upper` | 1.1710 | 0.9231 | 0.9039 | 1000000028.28 | 0.0002 |
| `ind_sma` | 1.1493 | 0.9251 | 0.9126 | 1000000028.28 | 0.0002 |
| `ind_bb_lower` | 1.1012 | 0.8722 | 0.8723 | 1000000028.28 | 0.0002 |
| `ind_ema_50` | 1.0013 | 0.8863 | 0.8686 | 32971.72 | 0.0001 |
| `ind_ema_100` | 0.8789 | 0.8145 | 0.7974 | 8013.13 | 0.0001 |
| `ind_ema_200` | 0.8750 | 0.7005 | 0.6796 | 1868.22 | 0.0001 |

**Recommendations:**

- 55 near-constant feature(s) carry almost no information for ANY target (same list regardless of target): n_candles, trend_momentum, trend_valid, pa_upper_wick, pa_lower_wick, pa_gap_up, pa_gap_down, vol_historical_volatility, vol_average_wick_size, vol_compression, .... Candidates for removal only if this holds across ALL targets, not just this one.
- 'structure_last_bos_direction' and 'structure_last_choch_direction' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'structure_last_choch_direction' has lower mutual information with this target (0.0613 vs 0.0613 for 'structure_last_bos_direction') -- if only one is kept, prefer 'structure_last_bos_direction'.
- 'ind_stoch_k' and 'ind_williams_r' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_williams_r' has lower mutual information with this target (0.0288 vs 0.0289 for 'ind_stoch_k') -- if only one is kept, prefer 'ind_stoch_k'.
- 'vol_historical_volatility' and 'ind_volatility' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_volatility' has lower mutual information with this target (0.2982 vs 0.2984 for 'vol_historical_volatility') -- if only one is kept, prefer 'vol_historical_volatility'.
- 'pa_price_change' and 'pa_price_return' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'pa_price_return' has lower mutual information with this target (0.0308 vs 0.0355 for 'pa_price_change') -- if only one is kept, prefer 'pa_price_change'.
- 'ind_momentum' and 'ind_roc' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_roc' has lower mutual information with this target (0.0392 vs 0.0405 for 'ind_momentum') -- if only one is kept, prefer 'ind_momentum'.
- 'pa_candle_range' and 'ind_true_range' are highly correlated (Pearson r=+0.999) -- carry near-duplicate information. 'pa_candle_range' has lower mutual information with this target (0.0691 vs 0.0761 for 'ind_true_range') -- if only one is kept, prefer 'ind_true_range'.
- 'ind_ema_20' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (1.1493 vs 1.2523 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'ind_rolling_median' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (1.1493 vs 1.1779 for 'ind_rolling_median') -- if only one is kept, prefer 'ind_rolling_median'.
- 'ind_ema_20' and 'ind_rolling_median' are highly correlated (Pearson r=+0.995) -- carry near-duplicate information. 'ind_rolling_median' has lower mutual information with this target (1.1779 vs 1.2523 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'sr_nearest_support_strength' and 'sr_nearest_support_touches' are highly correlated (Pearson r=+0.993) -- carry near-duplicate information. 'sr_nearest_support_touches' has lower mutual information with this target (0.1740 vs 0.3511 for 'sr_nearest_support_strength') -- if only one is kept, prefer 'sr_nearest_support_strength'.
- 'pa_current_close' and 'pa_previous_close' are highly correlated (Pearson r=+0.992) -- carry near-duplicate information. 'pa_previous_close' has lower mutual information with this target (1.4103 vs 1.5280 for 'pa_current_close') -- if only one is kept, prefer 'pa_current_close'.
- 'trend_higher_highs' and 'trend_lower_highs' are highly correlated (Pearson r=-0.981) -- carry near-duplicate information. 'trend_higher_highs' has lower mutual information with this target (0.2123 vs 0.2620 for 'trend_lower_highs') -- if only one is kept, prefer 'trend_lower_highs'.
- 'pa_distance_from_ema20' and 'ind_rsi' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_rsi' has lower mutual information with this target (0.0170 vs 0.0772 for 'pa_distance_from_ema20') -- if only one is kept, prefer 'pa_distance_from_ema20'.
- 'ind_ema_20' and 'ind_ema_50' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_ema_50' has lower mutual information with this target (1.0013 vs 1.2523 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'ind_ema_50' and 'ind_sma' are highly correlated (Pearson r=+0.977) -- carry near-duplicate information. 'ind_ema_50' has lower mutual information with this target (1.0013 vs 1.1493 for 'ind_sma') -- if only one is kept, prefer 'ind_sma'.
- 72 feature(s) have VIF > 10.0 (severe multicollinearity): structure_last_bos_direction (VIF=1000000028.3), structure_last_choch_direction (VIF=1000000028.3), pa_current_close (VIF=1000000028.3), pa_previous_close (VIF=1000000028.3), pa_price_change (VIF=1000000028.3), pa_body_size (VIF=1000000028.3), pa_upper_wick (VIF=1000000028.3), pa_lower_wick (VIF=1000000028.3), pa_candle_range (VIF=1000000028.3), ind_bb_lower (VIF=1000000028.3), .... A linear model's coefficients on these are unstable (small data changes swing them sharply) even if overall R^2 looks fine -- consider regularization (ridge/lasso, already supported) over removal, since removal changes what the coefficients mean.
- 5 feature(s) show both near-zero linear correlation (|Pearson| < 0.02) and near-zero mutual information (< 0.01) with this specific target -- likely low-information FOR THIS TARGET specifically (may still matter for other targets or for the Logistic Regression Engine): pat_doji, pat_morning_star, pat_outside_bar, pat_spinning_top, pat_three_white_soldiers.

## Target: `next_low`

- Near-constant features (all targets, same list): 55
- High-VIF features (VIF > 10): 72
- Highly correlated pairs (|Pearson| >= 0.95): 35

**Top 10 features by mutual information:**

| Feature | MI (nats) | Pearson | Spearman | VIF | Importance |
|---|---:|---:|---:|---:|---:|
| `pa_current_close` | 1.4732 | 0.9717 | 0.9653 | 1000000028.28 | 0.0002 |
| `pa_previous_close` | 1.4210 | 0.9643 | 0.9557 | 1000000028.28 | 0.0002 |
| `ind_ema_20` | 1.2091 | 0.9418 | 0.9298 | 20177.54 | 0.0002 |
| `ind_rolling_median` | 1.1292 | 0.9232 | 0.9126 | 615.47 | 0.0002 |
| `ind_bb_lower` | 1.1260 | 0.8830 | 0.8808 | 1000000028.28 | 0.0002 |
| `ind_bb_upper` | 1.1182 | 0.9215 | 0.9030 | 1000000028.28 | 0.0002 |
| `ind_sma` | 1.1175 | 0.9299 | 0.9174 | 1000000028.28 | 0.0002 |
| `ind_ema_50` | 0.9755 | 0.8930 | 0.8758 | 32971.72 | 0.0001 |
| `ind_ema_100` | 0.8978 | 0.8238 | 0.8065 | 8013.13 | 0.0001 |
| `ind_ema_200` | 0.8745 | 0.7123 | 0.6883 | 1868.22 | 0.0001 |

**Recommendations:**

- 55 near-constant feature(s) carry almost no information for ANY target (same list regardless of target): n_candles, trend_momentum, trend_valid, pa_upper_wick, pa_lower_wick, pa_gap_up, pa_gap_down, vol_historical_volatility, vol_average_wick_size, vol_compression, .... Candidates for removal only if this holds across ALL targets, not just this one.
- 'structure_last_bos_direction' and 'structure_last_choch_direction' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'structure_last_choch_direction' has lower mutual information with this target (0.0269 vs 0.0269 for 'structure_last_bos_direction') -- if only one is kept, prefer 'structure_last_bos_direction'.
- 'ind_stoch_k' and 'ind_williams_r' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_stoch_k' has lower mutual information with this target (0.0634 vs 0.0637 for 'ind_williams_r') -- if only one is kept, prefer 'ind_williams_r'.
- 'vol_historical_volatility' and 'ind_volatility' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'vol_historical_volatility' has lower mutual information with this target (0.2661 vs 0.2661 for 'ind_volatility') -- if only one is kept, prefer 'ind_volatility'.
- 'pa_price_change' and 'pa_price_return' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'pa_price_change' has lower mutual information with this target (0.0334 vs 0.0365 for 'pa_price_return') -- if only one is kept, prefer 'pa_price_return'.
- 'ind_momentum' and 'ind_roc' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_momentum' has lower mutual information with this target (0.0235 vs 0.0253 for 'ind_roc') -- if only one is kept, prefer 'ind_roc'.
- 'pa_candle_range' and 'ind_true_range' are highly correlated (Pearson r=+0.999) -- carry near-duplicate information. 'ind_true_range' has lower mutual information with this target (0.0557 vs 0.0673 for 'pa_candle_range') -- if only one is kept, prefer 'pa_candle_range'.
- 'ind_ema_20' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (1.1175 vs 1.2091 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'ind_rolling_median' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (1.1175 vs 1.1292 for 'ind_rolling_median') -- if only one is kept, prefer 'ind_rolling_median'.
- 'ind_ema_20' and 'ind_rolling_median' are highly correlated (Pearson r=+0.995) -- carry near-duplicate information. 'ind_rolling_median' has lower mutual information with this target (1.1292 vs 1.2091 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'sr_nearest_support_strength' and 'sr_nearest_support_touches' are highly correlated (Pearson r=+0.993) -- carry near-duplicate information. 'sr_nearest_support_touches' has lower mutual information with this target (0.1454 vs 0.3655 for 'sr_nearest_support_strength') -- if only one is kept, prefer 'sr_nearest_support_strength'.
- 'pa_current_close' and 'pa_previous_close' are highly correlated (Pearson r=+0.992) -- carry near-duplicate information. 'pa_previous_close' has lower mutual information with this target (1.4210 vs 1.4732 for 'pa_current_close') -- if only one is kept, prefer 'pa_current_close'.
- 'trend_higher_highs' and 'trend_lower_highs' are highly correlated (Pearson r=-0.981) -- carry near-duplicate information. 'trend_higher_highs' has lower mutual information with this target (0.2081 vs 0.2444 for 'trend_lower_highs') -- if only one is kept, prefer 'trend_lower_highs'.
- 'pa_distance_from_ema20' and 'ind_rsi' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_rsi' has lower mutual information with this target (0.0493 vs 0.0918 for 'pa_distance_from_ema20') -- if only one is kept, prefer 'pa_distance_from_ema20'.
- 'ind_ema_20' and 'ind_ema_50' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_ema_50' has lower mutual information with this target (0.9755 vs 1.2091 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'ind_ema_50' and 'ind_sma' are highly correlated (Pearson r=+0.977) -- carry near-duplicate information. 'ind_ema_50' has lower mutual information with this target (0.9755 vs 1.1175 for 'ind_sma') -- if only one is kept, prefer 'ind_sma'.
- 72 feature(s) have VIF > 10.0 (severe multicollinearity): structure_last_bos_direction (VIF=1000000028.3), structure_last_choch_direction (VIF=1000000028.3), pa_current_close (VIF=1000000028.3), pa_previous_close (VIF=1000000028.3), pa_price_change (VIF=1000000028.3), pa_body_size (VIF=1000000028.3), pa_upper_wick (VIF=1000000028.3), pa_lower_wick (VIF=1000000028.3), pa_candle_range (VIF=1000000028.3), ind_bb_lower (VIF=1000000028.3), .... A linear model's coefficients on these are unstable (small data changes swing them sharply) even if overall R^2 looks fine -- consider regularization (ridge/lasso, already supported) over removal, since removal changes what the coefficients mean.
- 6 feature(s) show both near-zero linear correlation (|Pearson| < 0.02) and near-zero mutual information (< 0.01) with this specific target -- likely low-information FOR THIS TARGET specifically (may still matter for other targets or for the Logistic Regression Engine): pat_bullish_engulfing, pat_doji, pat_inside_bar, pat_morning_star, pat_outside_bar, pat_three_white_soldiers.

## Target: `next_return`

- Near-constant features (all targets, same list): 55
- High-VIF features (VIF > 10): 72
- Highly correlated pairs (|Pearson| >= 0.95): 35

**Top 10 features by mutual information:**

| Feature | MI (nats) | Pearson | Spearman | VIF | Importance |
|---|---:|---:|---:|---:|---:|
| `micro_correction_length` | 0.1143 | -0.0783 | -0.0592 | 10.46 | 0.0000 |
| `ind_ema_50` | 0.1116 | -0.1396 | -0.1279 | 32971.72 | 0.0000 |
| `session_hour` | 0.1056 | -0.0577 | -0.0422 | 12.30 | 0.0000 |
| `micro_impulse_length` | 0.0937 | -0.0550 | -0.0477 | 16.93 | 0.0000 |
| `ind_atr` | 0.0895 | -0.0685 | -0.0875 | 81.41 | 0.0000 |
| `ind_ema_100` | 0.0875 | -0.1279 | -0.1110 | 8013.13 | 0.0000 |
| `session_is_sydney` | 0.0871 | 0.0495 | 0.0701 | 7.81 | 0.0000 |
| `ind_volume_ma` | 0.0862 | -0.0357 | -0.0536 | 43.35 | 0.0000 |
| `spread_atr_ratio` | 0.0838 | 0.0079 | 0.0828 | 53.08 | 0.0001 |
| `micro_retracement_pct` | 0.0718 | -0.0061 | -0.0210 | 1.72 | 0.0000 |

**Recommendations:**

- 55 near-constant feature(s) carry almost no information for ANY target (same list regardless of target): n_candles, trend_momentum, trend_valid, pa_upper_wick, pa_lower_wick, pa_gap_up, pa_gap_down, vol_historical_volatility, vol_average_wick_size, vol_compression, .... Candidates for removal only if this holds across ALL targets, not just this one.
- 'structure_last_bos_direction' and 'structure_last_choch_direction' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'structure_last_choch_direction' has lower mutual information with this target (0.0081 vs 0.0120 for 'structure_last_bos_direction') -- if only one is kept, prefer 'structure_last_bos_direction'.
- 'ind_stoch_k' and 'ind_williams_r' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_williams_r' has lower mutual information with this target (0.0224 vs 0.0226 for 'ind_stoch_k') -- if only one is kept, prefer 'ind_stoch_k'.
- 'vol_historical_volatility' and 'ind_volatility' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'vol_historical_volatility' has lower mutual information with this target (0.0637 vs 0.0641 for 'ind_volatility') -- if only one is kept, prefer 'ind_volatility'.
- 'pa_price_change' and 'pa_price_return' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'pa_price_return' has lower mutual information with this target (0.0000 vs 0.0000 for 'pa_price_change') -- if only one is kept, prefer 'pa_price_change'.
- 'ind_momentum' and 'ind_roc' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_roc' has lower mutual information with this target (0.0000 vs 0.0000 for 'ind_momentum') -- if only one is kept, prefer 'ind_momentum'.
- 'pa_candle_range' and 'ind_true_range' are highly correlated (Pearson r=+0.999) -- carry near-duplicate information. 'ind_true_range' has lower mutual information with this target (0.0036 vs 0.0112 for 'pa_candle_range') -- if only one is kept, prefer 'pa_candle_range'.
- 'ind_ema_20' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_ema_20' has lower mutual information with this target (0.0191 vs 0.0442 for 'ind_sma') -- if only one is kept, prefer 'ind_sma'.
- 'ind_rolling_median' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_rolling_median' has lower mutual information with this target (0.0356 vs 0.0442 for 'ind_sma') -- if only one is kept, prefer 'ind_sma'.
- 'ind_ema_20' and 'ind_rolling_median' are highly correlated (Pearson r=+0.995) -- carry near-duplicate information. 'ind_ema_20' has lower mutual information with this target (0.0191 vs 0.0356 for 'ind_rolling_median') -- if only one is kept, prefer 'ind_rolling_median'.
- 'sr_nearest_support_strength' and 'sr_nearest_support_touches' are highly correlated (Pearson r=+0.993) -- carry near-duplicate information. 'sr_nearest_support_touches' has lower mutual information with this target (0.0255 vs 0.0567 for 'sr_nearest_support_strength') -- if only one is kept, prefer 'sr_nearest_support_strength'.
- 'pa_current_close' and 'pa_previous_close' are highly correlated (Pearson r=+0.992) -- carry near-duplicate information. 'pa_current_close' has lower mutual information with this target (0.0042 vs 0.0286 for 'pa_previous_close') -- if only one is kept, prefer 'pa_previous_close'.
- 'trend_higher_highs' and 'trend_lower_highs' are highly correlated (Pearson r=-0.981) -- carry near-duplicate information. 'trend_lower_highs' has lower mutual information with this target (0.0042 vs 0.0066 for 'trend_higher_highs') -- if only one is kept, prefer 'trend_higher_highs'.
- 'pa_distance_from_ema20' and 'ind_rsi' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_rsi' has lower mutual information with this target (0.0000 vs 0.0190 for 'pa_distance_from_ema20') -- if only one is kept, prefer 'pa_distance_from_ema20'.
- 'ind_ema_20' and 'ind_ema_50' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_ema_20' has lower mutual information with this target (0.0191 vs 0.1116 for 'ind_ema_50') -- if only one is kept, prefer 'ind_ema_50'.
- 'ind_ema_50' and 'ind_sma' are highly correlated (Pearson r=+0.977) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (0.0442 vs 0.1116 for 'ind_ema_50') -- if only one is kept, prefer 'ind_ema_50'.
- 72 feature(s) have VIF > 10.0 (severe multicollinearity): structure_last_bos_direction (VIF=1000000028.3), structure_last_choch_direction (VIF=1000000028.3), pa_current_close (VIF=1000000028.3), pa_previous_close (VIF=1000000028.3), pa_price_change (VIF=1000000028.3), pa_body_size (VIF=1000000028.3), pa_upper_wick (VIF=1000000028.3), pa_lower_wick (VIF=1000000028.3), pa_candle_range (VIF=1000000028.3), ind_bb_lower (VIF=1000000028.3), .... A linear model's coefficients on these are unstable (small data changes swing them sharply) even if overall R^2 looks fine -- consider regularization (ridge/lasso, already supported) over removal, since removal changes what the coefficients mean.
- 25 feature(s) show both near-zero linear correlation (|Pearson| < 0.02) and near-zero mutual information (< 0.01) with this specific target -- likely low-information FOR THIS TARGET specifically (may still matter for other targets or for the Logistic Regression Engine): structure_bullish_bos_count, pa_body_ratio, pa_atr_ratio, pa_distance_from_ema50, vol_expansion, micro_swing_acceleration, micro_time_between_swings, spread_spike, liq_equal_highs, liq_equal_lows, ....

## Target: `expected_pip_movement`

- Near-constant features (all targets, same list): 55
- High-VIF features (VIF > 10): 72
- Highly correlated pairs (|Pearson| >= 0.95): 35

**Top 10 features by mutual information:**

| Feature | MI (nats) | Pearson | Spearman | VIF | Importance |
|---|---:|---:|---:|---:|---:|
| `session_hour` | 0.1193 | -0.0577 | -0.0425 | 12.30 | 0.0955 |
| `ind_ema_50` | 0.1117 | -0.1395 | -0.1284 | 32971.72 | 0.1902 |
| `micro_correction_length` | 0.1101 | -0.0783 | -0.0580 | 10.46 | 0.0457 |
| `ind_atr` | 0.0878 | -0.0687 | -0.0869 | 81.41 | 0.0057 |
| `micro_impulse_length` | 0.0877 | -0.0552 | -0.0472 | 16.93 | 0.0410 |
| `ind_volume_ma` | 0.0831 | -0.0359 | -0.0531 | 43.35 | 0.0020 |
| `spread_atr_ratio` | 0.0822 | 0.0081 | 0.0822 | 53.08 | 0.8664 |
| `ind_ema_100` | 0.0818 | -0.1278 | -0.1115 | 8013.13 | 0.1703 |
| `micro_retracement_pct` | 0.0633 | -0.0059 | -0.0206 | 1.72 | 0.0002 |
| `vol_historical_volatility` | 0.0611 | -0.0403 | -0.0649 | 7970406.96 | 0.0001 |

**Recommendations:**

- 55 near-constant feature(s) carry almost no information for ANY target (same list regardless of target): n_candles, trend_momentum, trend_valid, pa_upper_wick, pa_lower_wick, pa_gap_up, pa_gap_down, vol_historical_volatility, vol_average_wick_size, vol_compression, .... Candidates for removal only if this holds across ALL targets, not just this one.
- 'structure_last_bos_direction' and 'structure_last_choch_direction' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'structure_last_bos_direction' has lower mutual information with this target (0.0012 vs 0.0115 for 'structure_last_choch_direction') -- if only one is kept, prefer 'structure_last_choch_direction'.
- 'ind_stoch_k' and 'ind_williams_r' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_williams_r' has lower mutual information with this target (0.0287 vs 0.0289 for 'ind_stoch_k') -- if only one is kept, prefer 'ind_stoch_k'.
- 'vol_historical_volatility' and 'ind_volatility' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_volatility' has lower mutual information with this target (0.0607 vs 0.0611 for 'vol_historical_volatility') -- if only one is kept, prefer 'vol_historical_volatility'.
- 'pa_price_change' and 'pa_price_return' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'pa_price_return' has lower mutual information with this target (0.0000 vs 0.0000 for 'pa_price_change') -- if only one is kept, prefer 'pa_price_change'.
- 'ind_momentum' and 'ind_roc' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_momentum' has lower mutual information with this target (0.0003 vs 0.0009 for 'ind_roc') -- if only one is kept, prefer 'ind_roc'.
- 'pa_candle_range' and 'ind_true_range' are highly correlated (Pearson r=+0.999) -- carry near-duplicate information. 'ind_true_range' has lower mutual information with this target (0.0175 vs 0.0217 for 'pa_candle_range') -- if only one is kept, prefer 'pa_candle_range'.
- 'ind_ema_20' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_ema_20' has lower mutual information with this target (0.0245 vs 0.0465 for 'ind_sma') -- if only one is kept, prefer 'ind_sma'.
- 'ind_rolling_median' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_rolling_median' has lower mutual information with this target (0.0389 vs 0.0465 for 'ind_sma') -- if only one is kept, prefer 'ind_sma'.
- 'ind_ema_20' and 'ind_rolling_median' are highly correlated (Pearson r=+0.995) -- carry near-duplicate information. 'ind_ema_20' has lower mutual information with this target (0.0245 vs 0.0389 for 'ind_rolling_median') -- if only one is kept, prefer 'ind_rolling_median'.
- 'sr_nearest_support_strength' and 'sr_nearest_support_touches' are highly correlated (Pearson r=+0.993) -- carry near-duplicate information. 'sr_nearest_support_touches' has lower mutual information with this target (0.0361 vs 0.0521 for 'sr_nearest_support_strength') -- if only one is kept, prefer 'sr_nearest_support_strength'.
- 'pa_current_close' and 'pa_previous_close' are highly correlated (Pearson r=+0.992) -- carry near-duplicate information. 'pa_current_close' has lower mutual information with this target (0.0090 vs 0.0300 for 'pa_previous_close') -- if only one is kept, prefer 'pa_previous_close'.
- 'trend_higher_highs' and 'trend_lower_highs' are highly correlated (Pearson r=-0.981) -- carry near-duplicate information. 'trend_higher_highs' has lower mutual information with this target (0.0036 vs 0.0095 for 'trend_lower_highs') -- if only one is kept, prefer 'trend_lower_highs'.
- 'pa_distance_from_ema20' and 'ind_rsi' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_rsi' has lower mutual information with this target (0.0000 vs 0.0212 for 'pa_distance_from_ema20') -- if only one is kept, prefer 'pa_distance_from_ema20'.
- 'ind_ema_20' and 'ind_ema_50' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_ema_20' has lower mutual information with this target (0.0245 vs 0.1117 for 'ind_ema_50') -- if only one is kept, prefer 'ind_ema_50'.
- 'ind_ema_50' and 'ind_sma' are highly correlated (Pearson r=+0.977) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (0.0465 vs 0.1117 for 'ind_ema_50') -- if only one is kept, prefer 'ind_ema_50'.
- 72 feature(s) have VIF > 10.0 (severe multicollinearity): structure_last_bos_direction (VIF=1000000028.3), structure_last_choch_direction (VIF=1000000028.3), pa_current_close (VIF=1000000028.3), pa_previous_close (VIF=1000000028.3), pa_price_change (VIF=1000000028.3), pa_body_size (VIF=1000000028.3), pa_upper_wick (VIF=1000000028.3), pa_lower_wick (VIF=1000000028.3), pa_candle_range (VIF=1000000028.3), ind_bb_lower (VIF=1000000028.3), .... A linear model's coefficients on these are unstable (small data changes swing them sharply) even if overall R^2 looks fine -- consider regularization (ridge/lasso, already supported) over removal, since removal changes what the coefficients mean.
- 23 feature(s) show both near-zero linear correlation (|Pearson| < 0.02) and near-zero mutual information (< 0.01) with this specific target -- likely low-information FOR THIS TARGET specifically (may still matter for other targets or for the Logistic Regression Engine): pa_body_ratio, pa_atr_ratio, pa_distance_from_ema50, vol_expansion, micro_swing_acceleration, micro_time_between_swings, spread_spike, liq_equal_highs, liq_equal_lows, liq_pool_count, ....

## Target: `future_volatility`

- Near-constant features (all targets, same list): 55
- High-VIF features (VIF > 10): 72
- Highly correlated pairs (|Pearson| >= 0.95): 35

**Top 10 features by mutual information:**

| Feature | MI (nats) | Pearson | Spearman | VIF | Importance |
|---|---:|---:|---:|---:|---:|
| `session_hour` | 0.1923 | -0.1334 | -0.2066 | 12.30 | 0.0000 |
| `pa_candle_range` | 0.1612 | 0.4040 | 0.4799 | 1000000028.28 | 0.0000 |
| `vol_average_candle_size` | 0.1608 | 0.3569 | 0.4722 | 177.51 | 0.0000 |
| `vol_average_wick_size` | 0.1490 | 0.3255 | 0.4430 | 43.74 | 0.0000 |
| `ind_atr` | 0.1474 | 0.3872 | 0.4872 | 81.41 | 0.0000 |
| `ind_volatility` | 0.1468 | 0.3283 | 0.4401 | 7962860.98 | 0.0000 |
| `vol_historical_volatility` | 0.1465 | 0.3284 | 0.4401 | 7970406.96 | 0.0000 |
| `ind_true_range` | 0.1435 | 0.4027 | 0.4776 | 1732.32 | 0.0000 |
| `ind_volume_ma` | 0.1348 | 0.3282 | 0.4459 | 43.35 | 0.0000 |
| `ind_tick_volume` | 0.1326 | 0.4455 | 0.5051 | 27.68 | 0.0000 |

**Recommendations:**

- 55 near-constant feature(s) carry almost no information for ANY target (same list regardless of target): n_candles, trend_momentum, trend_valid, pa_upper_wick, pa_lower_wick, pa_gap_up, pa_gap_down, vol_historical_volatility, vol_average_wick_size, vol_compression, .... Candidates for removal only if this holds across ALL targets, not just this one.
- 'structure_last_bos_direction' and 'structure_last_choch_direction' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'structure_last_choch_direction' has lower mutual information with this target (0.0000 vs 0.0000 for 'structure_last_bos_direction') -- if only one is kept, prefer 'structure_last_bos_direction'.
- 'ind_stoch_k' and 'ind_williams_r' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_williams_r' has lower mutual information with this target (0.0290 vs 0.0297 for 'ind_stoch_k') -- if only one is kept, prefer 'ind_stoch_k'.
- 'vol_historical_volatility' and 'ind_volatility' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'vol_historical_volatility' has lower mutual information with this target (0.1465 vs 0.1468 for 'ind_volatility') -- if only one is kept, prefer 'ind_volatility'.
- 'pa_price_change' and 'pa_price_return' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'pa_price_return' has lower mutual information with this target (0.0310 vs 0.0314 for 'pa_price_change') -- if only one is kept, prefer 'pa_price_change'.
- 'ind_momentum' and 'ind_roc' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_roc' has lower mutual information with this target (0.0000 vs 0.0000 for 'ind_momentum') -- if only one is kept, prefer 'ind_momentum'.
- 'pa_candle_range' and 'ind_true_range' are highly correlated (Pearson r=+0.999) -- carry near-duplicate information. 'ind_true_range' has lower mutual information with this target (0.1435 vs 0.1612 for 'pa_candle_range') -- if only one is kept, prefer 'pa_candle_range'.
- 'ind_ema_20' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (0.0934 vs 0.0972 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'ind_rolling_median' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_rolling_median' has lower mutual information with this target (0.0559 vs 0.0934 for 'ind_sma') -- if only one is kept, prefer 'ind_sma'.
- 'ind_ema_20' and 'ind_rolling_median' are highly correlated (Pearson r=+0.995) -- carry near-duplicate information. 'ind_rolling_median' has lower mutual information with this target (0.0559 vs 0.0972 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'sr_nearest_support_strength' and 'sr_nearest_support_touches' are highly correlated (Pearson r=+0.993) -- carry near-duplicate information. 'sr_nearest_support_strength' has lower mutual information with this target (0.0064 vs 0.0223 for 'sr_nearest_support_touches') -- if only one is kept, prefer 'sr_nearest_support_touches'.
- 'pa_current_close' and 'pa_previous_close' are highly correlated (Pearson r=+0.992) -- carry near-duplicate information. 'pa_current_close' has lower mutual information with this target (0.0334 vs 0.0673 for 'pa_previous_close') -- if only one is kept, prefer 'pa_previous_close'.
- 'trend_higher_highs' and 'trend_lower_highs' are highly correlated (Pearson r=-0.981) -- carry near-duplicate information. 'trend_higher_highs' has lower mutual information with this target (0.0308 vs 0.0469 for 'trend_lower_highs') -- if only one is kept, prefer 'trend_lower_highs'.
- 'pa_distance_from_ema20' and 'ind_rsi' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_rsi' has lower mutual information with this target (0.0508 vs 0.0610 for 'pa_distance_from_ema20') -- if only one is kept, prefer 'pa_distance_from_ema20'.
- 'ind_ema_20' and 'ind_ema_50' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_ema_50' has lower mutual information with this target (0.0776 vs 0.0972 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'ind_ema_50' and 'ind_sma' are highly correlated (Pearson r=+0.977) -- carry near-duplicate information. 'ind_ema_50' has lower mutual information with this target (0.0776 vs 0.0934 for 'ind_sma') -- if only one is kept, prefer 'ind_sma'.
- 72 feature(s) have VIF > 10.0 (severe multicollinearity): structure_last_bos_direction (VIF=1000000028.3), structure_last_choch_direction (VIF=1000000028.3), pa_current_close (VIF=1000000028.3), pa_previous_close (VIF=1000000028.3), pa_price_change (VIF=1000000028.3), pa_body_size (VIF=1000000028.3), pa_upper_wick (VIF=1000000028.3), pa_lower_wick (VIF=1000000028.3), pa_candle_range (VIF=1000000028.3), ind_bb_lower (VIF=1000000028.3), .... A linear model's coefficients on these are unstable (small data changes swing them sharply) even if overall R^2 looks fine -- consider regularization (ridge/lasso, already supported) over removal, since removal changes what the coefficients mean.
- 7 feature(s) show both near-zero linear correlation (|Pearson| < 0.02) and near-zero mutual information (< 0.01) with this specific target -- likely low-information FOR THIS TARGET specifically (may still matter for other targets or for the Logistic Regression Engine): trend_higher_lows, trend_duration_bars, session_minute, pat_bearish_engulfing, pat_bullish_harami, pat_hammer, pat_three_white_soldiers.

## Target: `maximum_favorable_excursion`

- Near-constant features (all targets, same list): 55
- High-VIF features (VIF > 10): 72
- Highly correlated pairs (|Pearson| >= 0.95): 35

**Top 10 features by mutual information:**

| Feature | MI (nats) | Pearson | Spearman | VIF | Importance |
|---|---:|---:|---:|---:|---:|
| `spread_volatility` | 0.0803 | -0.1278 | -0.0392 | 28.08 | 0.0000 |
| `liq_equal_highs` | 0.0802 | 0.1164 | 0.1058 | 20.98 | 0.0000 |
| `pa_distance_from_ema200` | 0.0695 | 0.0353 | -0.0325 | 713.15 | 0.0000 |
| `ind_ema_20` | 0.0670 | -0.1345 | -0.1502 | 20177.54 | 0.0000 |
| `vol_historical_volatility` | 0.0661 | 0.1692 | 0.1705 | 7970406.96 | 0.0000 |
| `ind_volatility` | 0.0660 | 0.1692 | 0.1704 | 7962860.98 | 0.0000 |
| `ind_roc` | 0.0657 | 0.0968 | 0.0058 | 982096.15 | 0.0003 |
| `pa_distance_from_ema50` | 0.0651 | 0.0741 | 0.0005 | 455.91 | 0.0001 |
| `liq_buy_side` | 0.0630 | 0.0115 | 0.0372 | 8.88 | 0.0000 |
| `ind_momentum` | 0.0554 | 0.0966 | 0.0056 | 981477.95 | 0.0000 |

**Recommendations:**

- 55 near-constant feature(s) carry almost no information for ANY target (same list regardless of target): n_candles, trend_momentum, trend_valid, pa_upper_wick, pa_lower_wick, pa_gap_up, pa_gap_down, vol_historical_volatility, vol_average_wick_size, vol_compression, .... Candidates for removal only if this holds across ALL targets, not just this one.
- 'structure_last_bos_direction' and 'structure_last_choch_direction' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'structure_last_choch_direction' has lower mutual information with this target (0.0000 vs 0.0007 for 'structure_last_bos_direction') -- if only one is kept, prefer 'structure_last_bos_direction'.
- 'ind_stoch_k' and 'ind_williams_r' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_williams_r' has lower mutual information with this target (0.0519 vs 0.0526 for 'ind_stoch_k') -- if only one is kept, prefer 'ind_stoch_k'.
- 'vol_historical_volatility' and 'ind_volatility' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_volatility' has lower mutual information with this target (0.0660 vs 0.0661 for 'vol_historical_volatility') -- if only one is kept, prefer 'vol_historical_volatility'.
- 'pa_price_change' and 'pa_price_return' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'pa_price_return' has lower mutual information with this target (0.0000 vs 0.0000 for 'pa_price_change') -- if only one is kept, prefer 'pa_price_change'.
- 'ind_momentum' and 'ind_roc' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_momentum' has lower mutual information with this target (0.0554 vs 0.0657 for 'ind_roc') -- if only one is kept, prefer 'ind_roc'.
- 'pa_candle_range' and 'ind_true_range' are highly correlated (Pearson r=+0.999) -- carry near-duplicate information. 'pa_candle_range' has lower mutual information with this target (0.0133 vs 0.0170 for 'ind_true_range') -- if only one is kept, prefer 'ind_true_range'.
- 'ind_ema_20' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (0.0315 vs 0.0670 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'ind_rolling_median' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_rolling_median' has lower mutual information with this target (0.0137 vs 0.0315 for 'ind_sma') -- if only one is kept, prefer 'ind_sma'.
- 'ind_ema_20' and 'ind_rolling_median' are highly correlated (Pearson r=+0.995) -- carry near-duplicate information. 'ind_rolling_median' has lower mutual information with this target (0.0137 vs 0.0670 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'sr_nearest_support_strength' and 'sr_nearest_support_touches' are highly correlated (Pearson r=+0.993) -- carry near-duplicate information. 'sr_nearest_support_touches' has lower mutual information with this target (0.0116 vs 0.0308 for 'sr_nearest_support_strength') -- if only one is kept, prefer 'sr_nearest_support_strength'.
- 'pa_current_close' and 'pa_previous_close' are highly correlated (Pearson r=+0.992) -- carry near-duplicate information. 'pa_previous_close' has lower mutual information with this target (0.0000 vs 0.0383 for 'pa_current_close') -- if only one is kept, prefer 'pa_current_close'.
- 'trend_higher_highs' and 'trend_lower_highs' are highly correlated (Pearson r=-0.981) -- carry near-duplicate information. 'trend_higher_highs' has lower mutual information with this target (0.0173 vs 0.0412 for 'trend_lower_highs') -- if only one is kept, prefer 'trend_lower_highs'.
- 'pa_distance_from_ema20' and 'ind_rsi' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'pa_distance_from_ema20' has lower mutual information with this target (0.0276 vs 0.0306 for 'ind_rsi') -- if only one is kept, prefer 'ind_rsi'.
- 'ind_ema_20' and 'ind_ema_50' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_ema_50' has lower mutual information with this target (0.0511 vs 0.0670 for 'ind_ema_20') -- if only one is kept, prefer 'ind_ema_20'.
- 'ind_ema_50' and 'ind_sma' are highly correlated (Pearson r=+0.977) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (0.0315 vs 0.0511 for 'ind_ema_50') -- if only one is kept, prefer 'ind_ema_50'.
- 72 feature(s) have VIF > 10.0 (severe multicollinearity): structure_last_bos_direction (VIF=1000000028.3), structure_last_choch_direction (VIF=1000000028.3), pa_current_close (VIF=1000000028.3), pa_previous_close (VIF=1000000028.3), pa_price_change (VIF=1000000028.3), pa_body_size (VIF=1000000028.3), pa_upper_wick (VIF=1000000028.3), pa_lower_wick (VIF=1000000028.3), pa_candle_range (VIF=1000000028.3), ind_bb_lower (VIF=1000000028.3), .... A linear model's coefficients on these are unstable (small data changes swing them sharply) even if overall R^2 looks fine -- consider regularization (ridge/lasso, already supported) over removal, since removal changes what the coefficients mean.
- 12 feature(s) show both near-zero linear correlation (|Pearson| < 0.02) and near-zero mutual information (< 0.01) with this specific target -- likely low-information FOR THIS TARGET specifically (may still matter for other targets or for the Logistic Regression Engine): micro_swing_acceleration, session_is_newyork, sr_distance_to_resistance, fvg_nearest_direction, fvg_distance_to_nearest, ob_nearest_strength, ob_nearest_freshness, ob_distance_to_nearest, ind_volume_delta, pat_bullish_harami, ....

## Target: `maximum_adverse_excursion`

- Near-constant features (all targets, same list): 55
- High-VIF features (VIF > 10): 72
- Highly correlated pairs (|Pearson| >= 0.95): 35

**Top 10 features by mutual information:**

| Feature | MI (nats) | Pearson | Spearman | VIF | Importance |
|---|---:|---:|---:|---:|---:|
| `ind_atr` | 0.0745 | 0.3718 | 0.3348 | 81.41 | 0.0000 |
| `vol_average_candle_size` | 0.0703 | 0.3458 | 0.3195 | 177.51 | 0.0000 |
| `ind_tick_volume` | 0.0701 | 0.2960 | 0.2779 | 27.68 | 0.0000 |
| `session_hour` | 0.0619 | -0.0139 | -0.0499 | 12.30 | 0.0000 |
| `spread_atr_ratio` | 0.0619 | -0.1810 | -0.3178 | 53.08 | 0.0000 |
| `ind_vwap` | 0.0610 | -0.0287 | 0.0172 | 351.68 | 0.0000 |
| `session_is_london` | 0.0595 | 0.2511 | 0.2395 | 7.28 | 0.0001 |
| `vol_average_wick_size` | 0.0566 | 0.3476 | 0.2935 | 43.74 | 0.0000 |
| `sr_distance_to_resistance` | 0.0562 | -0.1000 | -0.0779 | 2.19 | 0.0000 |
| `ind_macd` | 0.0560 | 0.1278 | 0.0827 | 1227.37 | 0.0000 |

**Recommendations:**

- 55 near-constant feature(s) carry almost no information for ANY target (same list regardless of target): n_candles, trend_momentum, trend_valid, pa_upper_wick, pa_lower_wick, pa_gap_up, pa_gap_down, vol_historical_volatility, vol_average_wick_size, vol_compression, .... Candidates for removal only if this holds across ALL targets, not just this one.
- 'structure_last_bos_direction' and 'structure_last_choch_direction' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'structure_last_bos_direction' has lower mutual information with this target (0.0000 vs 0.0018 for 'structure_last_choch_direction') -- if only one is kept, prefer 'structure_last_choch_direction'.
- 'ind_stoch_k' and 'ind_williams_r' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_williams_r' has lower mutual information with this target (0.0000 vs 0.0000 for 'ind_stoch_k') -- if only one is kept, prefer 'ind_stoch_k'.
- 'vol_historical_volatility' and 'ind_volatility' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_volatility' has lower mutual information with this target (0.0503 vs 0.0517 for 'vol_historical_volatility') -- if only one is kept, prefer 'vol_historical_volatility'.
- 'pa_price_change' and 'pa_price_return' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'pa_price_change' has lower mutual information with this target (0.0023 vs 0.0080 for 'pa_price_return') -- if only one is kept, prefer 'pa_price_return'.
- 'ind_momentum' and 'ind_roc' are highly correlated (Pearson r=+1.000) -- carry near-duplicate information. 'ind_roc' has lower mutual information with this target (0.0000 vs 0.0000 for 'ind_momentum') -- if only one is kept, prefer 'ind_momentum'.
- 'pa_candle_range' and 'ind_true_range' are highly correlated (Pearson r=+0.999) -- carry near-duplicate information. 'pa_candle_range' has lower mutual information with this target (0.0377 vs 0.0408 for 'ind_true_range') -- if only one is kept, prefer 'ind_true_range'.
- 'ind_ema_20' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_ema_20' has lower mutual information with this target (0.0000 vs 0.0270 for 'ind_sma') -- if only one is kept, prefer 'ind_sma'.
- 'ind_rolling_median' and 'ind_sma' are highly correlated (Pearson r=+0.998) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (0.0270 vs 0.0515 for 'ind_rolling_median') -- if only one is kept, prefer 'ind_rolling_median'.
- 'ind_ema_20' and 'ind_rolling_median' are highly correlated (Pearson r=+0.995) -- carry near-duplicate information. 'ind_ema_20' has lower mutual information with this target (0.0000 vs 0.0515 for 'ind_rolling_median') -- if only one is kept, prefer 'ind_rolling_median'.
- 'sr_nearest_support_strength' and 'sr_nearest_support_touches' are highly correlated (Pearson r=+0.993) -- carry near-duplicate information. 'sr_nearest_support_strength' has lower mutual information with this target (0.0033 vs 0.0163 for 'sr_nearest_support_touches') -- if only one is kept, prefer 'sr_nearest_support_touches'.
- 'pa_current_close' and 'pa_previous_close' are highly correlated (Pearson r=+0.992) -- carry near-duplicate information. 'pa_current_close' has lower mutual information with this target (0.0000 vs 0.0068 for 'pa_previous_close') -- if only one is kept, prefer 'pa_previous_close'.
- 'trend_higher_highs' and 'trend_lower_highs' are highly correlated (Pearson r=-0.981) -- carry near-duplicate information. 'trend_higher_highs' has lower mutual information with this target (0.0287 vs 0.0350 for 'trend_lower_highs') -- if only one is kept, prefer 'trend_lower_highs'.
- 'pa_distance_from_ema20' and 'ind_rsi' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'pa_distance_from_ema20' has lower mutual information with this target (0.0000 vs 0.0108 for 'ind_rsi') -- if only one is kept, prefer 'ind_rsi'.
- 'ind_ema_20' and 'ind_ema_50' are highly correlated (Pearson r=+0.979) -- carry near-duplicate information. 'ind_ema_20' has lower mutual information with this target (0.0000 vs 0.0311 for 'ind_ema_50') -- if only one is kept, prefer 'ind_ema_50'.
- 'ind_ema_50' and 'ind_sma' are highly correlated (Pearson r=+0.977) -- carry near-duplicate information. 'ind_sma' has lower mutual information with this target (0.0270 vs 0.0311 for 'ind_ema_50') -- if only one is kept, prefer 'ind_ema_50'.
- 72 feature(s) have VIF > 10.0 (severe multicollinearity): structure_last_bos_direction (VIF=1000000028.3), structure_last_choch_direction (VIF=1000000028.3), pa_current_close (VIF=1000000028.3), pa_previous_close (VIF=1000000028.3), pa_price_change (VIF=1000000028.3), pa_body_size (VIF=1000000028.3), pa_upper_wick (VIF=1000000028.3), pa_lower_wick (VIF=1000000028.3), pa_candle_range (VIF=1000000028.3), ind_bb_lower (VIF=1000000028.3), .... A linear model's coefficients on these are unstable (small data changes swing them sharply) even if overall R^2 looks fine -- consider regularization (ridge/lasso, already supported) over removal, since removal changes what the coefficients mean.
- 12 feature(s) show both near-zero linear correlation (|Pearson| < 0.02) and near-zero mutual information (< 0.01) with this specific target -- likely low-information FOR THIS TARGET specifically (may still matter for other targets or for the Logistic Regression Engine): structure_bars_since_bos, micro_time_between_swings, spread_distance_from_avg, ob_nearest_freshness, ind_stoch_d, ind_stoch_k, ind_williams_r, pat_bearish_engulfing, pat_bullish_engulfing, pat_morning_star, ....

