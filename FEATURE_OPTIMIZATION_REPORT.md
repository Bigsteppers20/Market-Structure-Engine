# Feature Optimization Report

Lead-engineer pass over the Market Structure Engine's feature-engineering layer to
prepare it for **Linear Regression** and **Logistic Regression** training. Scope was
strictly feature engineering: no model was built, no architecture was redesigned, and
the public API (`engine.load(data).analyze()` → `MarketState` / `feature_vector()`) is
unchanged. This document is the change log and ML-readiness reference for that pass;
`FEATURE_REFERENCE.md` holds the full per-feature table (now including Category,
Predictive Value, and Suggested Scaler columns) and `FEATURE_SCHEMA.md` holds the
concise index.

## Summary

| Metric | Value |
|---|---|
| Current (pre-optimization) feature count | 156 |
| Optimized feature count | **185** |
| Features removed | 14 |
| Features added | 43 (7 new spread signals + 36 data-quality/validity flags) |
| Features with modified semantics (same name) | 0 -- every change is an add/remove, not a silent redefinition |
| Structural (dataclass/config) modifications | 9 (see below) |
| Tests passing after change | 54 passed, 1 skipped (was 48 passed, 1 skipped, 5 pre-existing failures fixed to match new contract) |
| Live verification | Ran against real OANDA EUR/USD M5 + H1 candles (practice account) -- vector length 185 confirmed, name/order cross-checked against source exactly |

---

## Task 1 -- Removed Exact Duplicates

| Removed feature | Duplicate of | Why |
|---|---|---|
| `vol_atr` | `ind_atr` | Both read the same `IndicatorPanel.snapshot['atr']` value -- byte-for-byte identical |
| `vol_true_range` | `ind_true_range` | Same underlying `true_range()` array, read twice |
| `vol_rolling_volatility` | `ind_volatility` | Same rolling-return-stdev array, read twice |

**Canonical implementation kept**: `IndicatorEngine` (`indicators.py`) is now the single
source of truth for ATR, True Range, and rolling return-volatility. `VolatilityFeatures`
(`feature_vector.py`) no longer recomputes or re-exposes them -- it only holds the
volatility-*regime* features that don't exist anywhere else (`historical_volatility`,
`average_candle_size`, `average_wick_size`, `expansion`, `compression`).

## Task 2 -- Removed Redundant Features

| Removed feature | Redundant with | Relationship |
|---|---|---|
| `ind_macd_histogram` | `ind_macd`, `ind_macd_signal` | Exact linear combination: `histogram = macd - macd_signal`. Both inputs remain in the vector, so a linear model already spans this feature; a tree model can split on the difference implicitly. |
| `ind_rolling_mean` | `ind_sma` | Identical formula (SMA of close) and identical default period (20, from `sma_period`/`rolling_window`) -- exact duplicate under default `EngineConfig`. |
| `ind_bb_middle` | `ind_sma` | Same as above; the Bollinger middle band *is* `sma(close, bollinger_period)`. Still computed internally to derive `bb_upper`/`bb_lower`, just not re-exposed as its own feature. |
| `ind_rolling_variance` | `ind_std_dev` | `std_dev = sqrt(rolling_variance)` -- a deterministic monotonic transform carrying identical information in more directly interpretable (price) units. |
| `pa_percentage_return` | `pa_price_return` | Exact scaling: `percentage_return = price_return * 100`. `pa_price_return` (the fraction) is kept as canonical since it's the more standard ML representation. |
| `pat_harami` | `pat_bullish_harami`, `pat_bearish_harami` | Exact union (`bullish_harami OR bearish_harami`), zero information beyond those two. |
| `pat_marubozu` | `pat_bullish_marubozu`, `pat_bearish_marubozu` | Exact union. |
| `pat_pin_bar` | `pat_bullish_pin_bar`, `pat_bearish_pin_bar` | Exact union. |
| `pat_bullish_score` | 10 individual bullish `pat_*` flags | Exact sum; carries no information a linear model can't reconstruct as a weighted combination of the flags it summed. |
| `pat_bearish_score` | 9 individual bearish `pat_*` flags | Exact sum, same reasoning. |
| `session_session_overlap` | `session_is_sydney/asian/london/newyork` | Exact deterministic function (`>= 2 active`) of the 4 flags above -- zero new information. |

All 14 removals were re-derived and confirmed against the running engine (not just
theorized) -- see `tests/test_indicators_patterns.py` and `tests/test_engine.py` for the
updated assertions that pin this contract down.

### New duplicates discovered by Task 9 (not removed -- see Correlation Analysis)

Running the actual correlation analysis (Task 9) surfaced **two more exact duplicates
that this pass did not originally target** because they weren't in the task's example
list. They are documented here and in the Correlation Analysis section below, but were
**not removed**, since removing features beyond the explicitly scoped list is a decision
for you to confirm first:

- **`ind_stoch_k` ≡ `ind_williams_r` + 100`** exactly (`Williams %R = %K - 100` is an
  algebraic identity of the two formulas -- confirmed both symbolically and empirically,
  Pearson r = 1.000 on live data).
- **`structure_last_bos_direction` ≡ `structure_last_choch_direction`** whenever at
  least one CHOCH has occurred -- structurally guaranteed: every break between two CHOCH
  events shares the CHOCH's direction by definition, so the *last* BOS direction always
  equals the *last* CHOCH direction. Confirmed empirically (Pearson r = 1.000).

## Task 3 -- Missing Value Handling

**Decision: Option B (value + `_valid` companion flag), not NaN.**

**Why not NaN (Option A):** the engine's stated purpose is to hand a "flat float64
array" to downstream linear/logistic regression. `sklearn` estimators reject NaN by
default, so shipping NaN would just move the missing-value problem onto every
consumer instead of solving it. A `_valid` companion keeps `to_vector()`'s output
finite and immediately usable, while making "is this real" a first-class, inspectable
signal instead of an implicit contract.

**Rule applied:** a feature gets a `_valid` companion wherever `0.0` was previously used
as a placeholder for "not enough history/structure exists yet" **and** `0.0` is also a
plausible genuine reading (so the ambiguity is real). Features where `0.0` is already
self-describing were deliberately *not* given a companion flag, to avoid inflating the
vector with redundant flags:

| Already unambiguous (no flag added) | Why |
|---|---|
| `structure_last_bos_direction`, `structure_last_choch_direction`, `liq_last_sweep_direction`, `fvg_nearest_direction`, `ob_nearest_direction` | `0.0` is reserved exclusively for "no event exists"; a real event is always `±1.0` |
| `structure_bars_since_bos`, `structure_bars_since_choch` | Already use a `-1` sentinel, distinct from any real non-negative bar count |
| `*_count` fields (e.g. `sr_support_zone_count`, `liq_pool_count`) | `0` is a true, meaningful count of zero -- not a placeholder |
| `fvg_nearest_size_atr/age/distance`, `ob_nearest_strength/freshness/age/distance`, `liq_last_sweep_size` | Co-gated by their group's direction sentinel above -- check direction `!= 0` first |

| New `_valid` flag | Gates | Condition |
|---|---|---|
| `trend_valid` | all 8 `trend_*` fields | `>= 2` swing highs or `>= 2` swing lows existed (mirrors the exact code path that defaults `trend_*` to zero/SIDEWAYS) |
| `vol_valid` | `vol_historical_volatility`, `vol_average_candle_size`, `vol_average_wick_size`, `vol_expansion`, `vol_compression` | `n_candles >= hist_vol_window` (20) |
| `micro_valid` | all 9 `micro_*` fields | `len(swings) >= 3` |
| `sr_support_valid` | `sr_nearest_support_strength/touches/width`, `sr_distance_to_support` | a nearest support zone currently exists |
| `sr_resistance_valid` | `sr_nearest_resistance_strength/touches/width`, `sr_distance_to_resistance` | a nearest resistance zone currently exists |
| `ind_{name}_valid` (30, one per remaining indicator) | that indicator's snapshot value | `n_candles >= warm-up requirement` (documented per-indicator in `indicators.py::_warmup_requirements`, e.g. `ema_200` needs 200 bars, `rsi`/`atr` need 14, `macd_signal` needs 35) |
| `spread_valid` | all 7 `spread_*` fields | input has a `spread` column **and** `n_candles >= spread_window` (20) |

**Known simplification (documented, not hidden):** `trend_valid`/`vol_valid`/`micro_valid`
are single shared flags covering several fields at once, rather than one flag per field.
This trades a small amount of precision (e.g. `trend_duration_bars` and `trend_momentum`
have slightly different exact gating conditions in the source) for avoiding a further
~15-dimension explosion. Documented explicitly here and in each field's docstring so it's
never silently wrong.

**pa_\* inherits validity, no new flags:** `pa_atr_ratio`, `pa_distance_from_ema20/50/200`,
and `pa_distance_from_vwap` are direct functions of `close`, `atr`, and one EMA/VWAP value.
Rather than duplicating flags, they **inherit** validity from the already-present
`ind_atr_valid`, `ind_ema_20_valid`, `ind_ema_50_valid`, `ind_ema_200_valid`,
`ind_vwap_valid` -- check those before trusting the corresponding `pa_*` field.

## Task 4 -- Spread Features (new)

`spread` was validated by `DataLoader` since before this pass, but no feature ever read
it. Added `market_structure/spread.py::SpreadEngine`, wired into `engine.py` exactly
like every other sub-engine (no architectural change -- same `analyze(df) -> State`
pattern as `TrendEngine`, `LiquidityEngine`, etc.):

| Feature | Formula | Notes |
|---|---|---|
| `spread_current` | `spread[-1]` | Raw spread, price units |
| `spread_rolling_avg` | `mean(spread[-20:])` | `spread_window` config, default 20 |
| `spread_spike` | `float(current/rolling_avg >= 2.0)` | `spread_spike_multiple` config, default 2.0 |
| `spread_atr_ratio` | `current / atr_now` | Spread's share of typical candle range |
| `spread_percentile` | `mean(window <= current)` | Rank of current spread within its trailing window |
| `spread_volatility` | `std(spread[-20:], ddof=0)` | |
| `spread_distance_from_avg` | `(current - rolling_avg) / rolling_avg` | Signed relative deviation |
| `spread_valid` | see Task 3 table | 0.0 (all fields placeholder) whenever no `spread` column was supplied |

Two new `EngineConfig` fields back this (`spread_window=20`, `spread_spike_multiple=2.0`),
additive with defaults -- fully backward compatible.

## Task 5 -- Tick Volume

`ind_tick_volume` still defaults to `zeros_like(volume)` when the input has no
`tick_volume` column (unchanged behavior -- there is nothing else it *could* be without
inventing data), but it is now always paired with `ind_tick_volume_valid`, which is
`1.0` only when the input DataFrame actually supplied a `tick_volume` column. This uses
the same mechanism as every other indicator's `_valid` flag (Task 3), just gated on
column presence rather than a warm-up bar count.

## Task 6 -- Normalization Metadata

Every one of the 185 features now has **Range**, **Normalization Recommendation**, and
**Suggested Scaler** columns in `FEATURE_REFERENCE.md` (the engine itself performs no
scaling -- this is guidance only, per your instruction). Rules applied, in priority order:

| Rule | Scaler |
|---|---|
| Categorical (`{-1,0,1}` direction codes) | **No Scaling Required** -- one-hot encode for linear/logistic models instead |
| Binary flags (`{0,1}`, including every `_valid` flag) | **No Scaling Required** |
| Already bounded `[0,1]` ratios (e.g. `trend_strength`, `pa_body_ratio`, `ob_nearest_freshness`, `spread_percentile`) | **No Scaling Required** |
| Bounded oscillators with fixed theoretical range (`ind_rsi`, `ind_adx`, `ind_stoch_k/d`, `ind_williams_r`) | **MinMaxScaler** |
| ATR-normalized distances/ratios/strengths (most `sr_*`, `liq_*`, `fvg_*`, `ob_*`, `pa_atr_ratio`, `pa_distance_from_ema*`) | **RobustScaler** -- already scale-normalized, but volatility spikes still create outliers/heavy tails |
| Cyclical calendar fields (`session_hour/minute/day_of_week/month`) | **No Scaling Required** -- recommend a sin/cos transform instead of any linear scaler |
| Count features (`*_count`, `*_touches`) | **RobustScaler** -- right-skewed, consider `log1p` before scaling (see the cumulative-counter caveat under Correlation Analysis) |
| Volume/tick-volume-unit features | **RobustScaler** -- heavy-tailed with spikes |
| Raw price levels (`pa_current_close`, `pa_previous_close`) | **StandardScaler**, but consider excluding entirely -- non-stationary across time/instruments |
| Everything else (roughly symmetric, unbounded continuous) | **StandardScaler** |

## Task 7 -- Feature Categories

Every feature is assigned exactly one of the 14 requested categories in
`FEATURE_REFERENCE.md`. Distribution (185 total):

| Category | Count |
|---|---|
| Statistical | 40 |
| Pattern Recognition | 23 |
| Market Structure | 19 |
| Trend | 13 |
| Liquidity | 15 |
| Institutional Concepts | 15 |
| Price Action | 16 |
| Momentum | 10 |
| Volatility | 10 |
| Microstructure | 9 |
| Volume | 7 |
| Session | 4 |
| Time | 4 |

Note: **Statistical (40)** is large mostly because every `_valid` flag and `n_candles`
were categorized there (they're data-quality/meta features, not market-descriptive
features) -- see the per-feature table for the exact assignment logic.

## Task 8 -- Feature Importance Preparation

Every feature has a **Predictive Value** (High/Medium/Low) + **Reason** in
`FEATURE_REFERENCE.md`. This is a domain-informed prior based on established TA/ICT
usage and the feature's statistical character (bounded vs. unbounded, sparse vs. dense) --
**not** derived from any trained model, per your explicit instruction not to build one.
Headline calls:

- **High**: `trend_direction`, `trend_strength`, `structure_last_bos_direction`,
  `structure_last_choch_direction`, `ind_rsi`, `ind_adx`, `ind_atr`, `liq_buy_side`,
  `liq_sell_side`, `fvg_nearest_direction`, `ob_nearest_direction`,
  `sr_distance_to_support/resistance`, `spread_spike`, `vol_expansion/compression`,
  `ind_volume_spike`.
- **Low**: raw price levels (`pa_current_close`, `pa_previous_close` -- non-stationary),
  `n_candles`, `session_minute`/`session_month` (weak/high-cardinality calendar signal),
  sparse multi-candle patterns (`pat_dragonfly_doji`, `pat_gravestone_doji`,
  `pat_morning_star`, `pat_evening_star`, `pat_piercing_line`, `pat_dark_cloud_cover`,
  `pat_three_white_soldiers`, `pat_three_black_crows`), and every `_valid` flag (a
  data-quality gate, not a market signal -- though essential for correct training).
- **Medium / Medium-High**: everything else, weighted toward Medium-High for the
  engine's structural/liquidity/institutional-concept features (its actual
  differentiator) and Medium for generic oscillators/statistics.

## Task 9 -- Feature Correlation

`scripts/feature_correlation_analysis.py` computes Pearson correlation, Spearman
correlation, a manual VIF (no `statsmodels` dependency -- implemented via OLS R²,
consistent with the engine's "no external stats/TA library" convention), and a variance
threshold, then reports highly-correlated pairs, low-variance features, and removal
candidates. It does not train or evaluate a model.

**This was actually run against real data**, not left as a hypothetical: 2,499 live
EUR/USD M5 candles from the OANDA practice account, sampled by re-running
`engine.load(df.iloc[:i+1]).analyze()` at every 5th bar from bar 300 onward (**440
snapshots x 185 features**). Re-slicing and re-running per sample (rather than reusing
one `analyze()` call's internal arrays) keeps the sampling itself leakage-safe, per the
Data-Leakage Analysis in `FEATURE_REFERENCE.md`. Full CSVs are in `analysis_output/`
(`pearson_correlation_matrix.csv`, `spearman_correlation_matrix.csv`,
`vif_report.csv`, `variance_report.csv`, `high_correlation_pairs.csv`).

### Methodology caveat (read before trusting the numbers below)

Sampling via a **growing window** (`df.iloc[:i+1]` with `i` increasing) means `n_candles`
increases monotonically across the 440 samples. Several features are **cumulative
counters since the start of the loaded series** by construction (`structure_bullish_bos_count`,
`structure_bearish_bos_count`, `structure_choch_count`, `fvg_bullish_count`,
`fvg_bearish_count`, and similarly `ob_*_count`, `liq_pool_count`, `liq_sweep_count`,
`liq_equal_highs/lows` -- confirmed by reading `bos.py`/`fvg.py`/`liquidity.py`: they sum
over the *entire* loaded `breaks`/`gaps`/`pools` list, not a rolling recent window). These
necessarily grow with `n_candles` and therefore show inflated correlation/VIF against each
other and against `n_candles` in this sampling scheme -- **that is a property of the
features' cumulative design, amplified by the growing-window sampling, not a coincidence
of market conditions.** A fixed-length rolling-window resample would shrink but not
eliminate this (the features would still encode "how much history exists," just bounded).

**Actionable recommendation (not implemented in this pass -- out of the Task 1-5 scope,
flagged for a follow-up decision):** consider converting `structure_bullish_bos_count`,
`structure_bearish_bos_count`, `structure_choch_count`, `fvg_bullish_count`,
`fvg_bearish_count`, `ob_bullish_count`, `ob_bearish_count`, `liq_pool_count`,
`liq_sweep_count`, `liq_equal_highs`, `liq_equal_lows` to either (a) a rolling count over
a fixed recent lookback, or (b) a rate (`count / n_candles`), so they describe *recent
activity* instead of *how long the engine has been running*.

### Genuine duplicates confirmed (not sampling artifacts)

| Pair | Pearson | Why it's real, not an artifact |
|---|---|---|
| `ind_stoch_k` ↔ `ind_williams_r` | **+1.000** | Algebraic identity (`Williams %R = %K - 100`), true for any data, any instrument |
| `structure_last_bos_direction` ↔ `structure_last_choch_direction` | **+1.000** | Structurally guaranteed once >=1 CHOCH has occurred (see Task 2 note above) |

### Near-perfect correlations specific to low-volatility FX (not exact duplicates)

| Pair | Pearson | Why |
|---|---|---|
| `vol_historical_volatility` ↔ `ind_volatility` | +1.000 | log-return vs pct-return stdev; `log(1+x) ≈ x` at EUR/USD's ~10⁻⁴ return scale. Would diverge more on a higher-volatility instrument (crypto). |
| `ind_momentum` ↔ `ind_roc` | +1.000 | `roc = momentum / close[t-10] * 100`; `close[t-10]` barely moves relative to itself over 10 bars on FX |
| `pa_price_change` ↔ `pa_price_return` | +1.000 | `price_return = price_change / prev_close`; `prev_close` ≈ constant across nearby samples |
| `pa_candle_range` ↔ `ind_true_range` | +1.000 (Spearman +0.999) | `TR ≈ high-low` whenever there's no gap vs. the prior close -- true for continuous FX, would diverge around weekend gaps/news |

### Residual design collinearity (already documented in FEATURE_REFERENCE.md, confirmed empirically)

`ind_sma` ↔ `ind_rolling_median` (+0.999), `ind_ema_20` ↔ `ind_sma` (+0.998),
`ind_ema_20` ↔ `ind_ema_50` (+0.986), `sr_nearest_support_strength` ↔
`sr_nearest_support_touches` (+0.999), `sr_nearest_resistance_strength` ↔
`sr_nearest_resistance_touches` (+0.999), `liq_pool_count` ↔ `liq_sweep_count` (+0.979).
These are expected given each feature's formula (e.g. zone `strength` is partly built
from `touches`) and were intentionally kept distinct in Tasks 1-2 because each adds
*some* independent information (e.g. `strength` also folds in swing strength and
recency, not just touch count) -- flagged here for visibility, not removed.

### Low-variance / near-constant features (51 flagged, two different causes)

1. **36 `_valid` flags showed zero variance** in this sample -- because the sample
   deliberately started at bar 300 (past every indicator's warm-up requirement, the
   longest being `ema_200`'s 200 bars), all `_valid` flags were already `1.0` for the
   entire window. **This is expected, not a flaw**: these flags exist precisely to
   distinguish the *cold-start* period (bars 0-300, excluded here on purpose to avoid
   contaminating correlation stats with placeholder zeros) from the warmed-up period.
   `tests/test_indicators_patterns.py::test_indicator_validity_warms_up` verifies the
   0.0 -> 1.0 transition directly on a short series.
2. **Genuinely near-constant on this instrument/timeframe** (real finding, matches the
   design-time predictions in `FEATURE_REFERENCE.md`): `pa_gap_up` (99.77% zero),
   `pa_gap_down` (99.55% zero), `pat_dragonfly_doji` (99.55% zero), `pat_gravestone_doji`
   (99.55% zero), `pat_dark_cloud_cover` (100% zero in this sample), `pat_piercing_line`
   (100% zero in this sample). Confirmed: rare-event/rare-pattern features are sparse on
   EUR/USD M5 as predicted, not just in theory.
3. **False positives from the raw-variance threshold** (methodology caveat): several
   price-unit features (`pa_upper_wick`, `pa_lower_wick`, `vol_average_wick_size`,
   `micro_average_swing_length`, `spread_current`, `spread_rolling_avg`,
   `spread_volatility`, `trend_momentum`) were flagged by the absolute variance
   threshold (`1e-8`) purely because EUR/USD trades near 1.09 and these are priced in
   that unit (wick sizes ~0.0001-0.0005). **This is a threshold-scale artifact, not
   evidence of low information content** -- their ATR-normalized/percentage siblings
   (`pa_atr_ratio`, `spread_atr_ratio`, etc.) don't trip this flag. Recommendation: run
   variance-threshold screening on standardized (z-scored) features, not raw units.

### Candidate features for removal (Task 9 output, for your review -- none removed in this pass)

1. `ind_williams_r` **or** `ind_stoch_k` (exact duplicate pair -- keep whichever your
   downstream convention prefers).
2. `structure_last_bos_direction` **or** `structure_last_choch_direction` (exact
   duplicate whenever `structure_choch_count >= 1`; they only theoretically diverge
   before the first CHOCH, where `structure_last_choch_direction` is `0.0`/invalid-by-sentinel
   anyway).
3. Convert cumulative `*_count` features (listed above) to rolling/rate versions --
   highest-leverage fix for the VIF blow-up, but a behavior change, not a pure removal.

---

## Task 10 -- Feature Stability Under Regime Change

| Condition | Features most affected | Why |
|---|---|---|
| **High volatility / news spikes** | All ATR-normalized features (`pa_atr_ratio`, every `sr_*`/`liq_*`/`fvg_*`/`ob_*` distance/strength/size field) | ATR itself spikes, so the *normalizer* moves, not just the numerator -- these can swing sharply even with modest raw price movement. `spread_spike` and `vol_expansion` are specifically designed to flag this regime. |
| **News events** | `spread_*` (this is their intended purpose -- `spread_spike`/`spread_atr_ratio` should fire), `vol_expansion`, `ind_volume_spike`, `pa_gap_up/down` if the news causes a genuine price gap | These features exist specifically to be unstable/reactive here; that's a feature, not a bug -- but downstream models should treat them as regime indicators, not stationary inputs. |
| **Weekend gaps** | `pa_gap_up`/`pa_gap_down`, `fvg_*` (a weekend gap can register as a fair value gap), `ind_vwap`/`pa_distance_from_vwap` (cumulative VWAP absorbs the jump permanently), swing/trend features (a gap can register as an extreme swing prominence) | The engine has no explicit weekend-gap handling; a large weekend gap is indistinguishable from an intra-week news gap in the current feature set. |
| **Session changes (session boundary crossings)** | `session_is_*` flags change discontinuously at exact UTC boundaries; `spread_*` and `ind_volume_*` typically shift at the London/NY open/close; `vol_expansion`/`compression` often trip right at session transitions | These are expected step-function discontinuities, not noise -- but a model naively trained on session-boundary-adjacent bars should account for the discontinuity rather than smoothing over it. |
| **Low liquidity (Asian session, holidays)** | `spread_*` (spread widens -- `spread_atr_ratio`/`spread_percentile` should reflect this), `liq_*` (fewer genuine liquidity pools form), swing/structure detection generally (thin markets produce noisier, less reliable swings, degrading everything downstream of `SwingDetector`) | Low liquidity increases the *relative* noise-to-signal ratio for the whole swing-dependent feature family, not just liquidity features directly. |
| **Instrument/volatility-regime portability** | The "near-perfect correlation, not exact duplicate" pairs from Task 9 (`vol_historical_volatility`/`ind_volatility`, `ind_momentum`/`ind_roc`, `pa_candle_range`/`ind_true_range`) | These were measured as near-1.0 correlated *on EUR/USD M5 specifically*; the underlying formulas diverge more on higher-volatility instruments (crypto, exotic FX pairs) or on gappy markets. Don't assume this correlation structure transfers to a different instrument without re-running Task 9's analysis there. |

## Task 11 -- Look-Ahead Bias Re-Audit

Full analysis lives in `FEATURE_REFERENCE.md`'s Data-Leakage Analysis section
(re-verified after this pass, not just carried over). Conclusion unchanged:
**no feature depends on data beyond the last loaded candle.** Two pre-existing,
already-documented nuances still apply (swing confirmation lag; full-series
fill/mitigation scans, safe only for single-snapshot use). The new `spread_*` features
were re-audited specifically for this pass: they use only `pandas.Series.rolling(...)`
with default trailing alignment (never `center=True`), so they introduce **no new
look-ahead risk**.

## ML Readiness Checklist

- [x] No feature vector dimension changed meaning silently -- every change is a
      documented add or remove (`git diff` / this report is the full change log).
- [x] Exact duplicates removed (Task 1) and re-verified by test + live cross-check.
- [x] Redundant/derivable features removed (Task 2); two more found and **flagged, not
      silently removed** (Task 9) -- awaiting your call.
- [x] No feature silently encodes "missing" as `0.0` anymore -- every gated feature has
      a `_valid` companion (Task 3), and the vector stays fully finite (no NaN) for
      direct `sklearn` compatibility.
- [x] Spread data (validated but previously unused) now produces 7 real features + 1
      validity flag (Task 4).
- [x] Tick volume never silently reads as a real zero -- paired with
      `ind_tick_volume_valid` (Task 5).
- [x] Every feature has a documented range, normalization recommendation, and suggested
      scaler (Task 6) -- **the engine itself still performs zero normalization**, exactly
      as instructed.
- [x] Every feature has exactly one of the 14 requested categories (Task 7).
- [x] Every feature has a predictive-value tag + reason (Task 8) -- domain prior only,
      no model was trained.
- [x] Correlation/VIF/variance analysis was written **and actually executed** against
      live OANDA data, not left as unexercised code (Task 9); outputs are in
      `analysis_output/*.csv` for direct loading into a notebook.
- [x] Feature stability under 5 regime-change scenarios documented (Task 10).
- [x] Look-ahead bias re-audited; conclusion unchanged, new features specifically
      checked (Task 11).
- [x] `pytest tests -q` passes (54 passed, 1 skipped) after every change.
- [x] Live-verified against real broker data (OANDA EUR/USD M5 + H1), not just synthetic
      data -- vector shape and every feature name/position cross-checked programmatically
      against the running engine (zero mismatches).
- [ ] **Not done, and intentionally out of scope**: building/training the Linear or
      Logistic Regression model itself, per your explicit instruction.
- [ ] **Follow-up recommended, not implemented**: convert cumulative `*_count` features
      to rolling/rate versions (see Task 9); resolve the two newly-found exact
      duplicates (`stoch_k`/`williams_r`, BOS/CHOCH direction) once you've reviewed them.
