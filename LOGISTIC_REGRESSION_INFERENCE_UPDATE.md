# Logistic Regression Engine -- Live Inference Update

This document records a scoped change to the Logistic Regression Engine: **only its live
inference output** (`POST /logistic-regression/predict`) changed. Training, evaluation,
calibration, the Decision Engine, the Model Monitor, and every other analytical engine on this
platform are untouched.

## Previous inference contract

`POST /logistic-regression/predict` returned the full, evaluation-oriented
`ClassificationPrediction` object -- every class probability, the full probability distribution,
and diagnostic detail:

```json
{
  "buy_probability": 0.027,
  "sell_probability": 0.093,
  "no_trade_probability": 0.880,
  "predicted_class": "NO_TRADE",
  "prediction_confidence": 64.77,
  "probability_margin": 0.7878,
  "prediction_entropy": 0.3914,
  "model_version": "1.0.0",
  "feature_version": "1.0.0",
  "training_version": "1.0.0",
  "timestamp": "2026-07-14T23:32:03+00:00",
  "symbol": "EUR_USD",
  "timeframe": "M5",
  "prediction_horizon": 5,
  "class_probabilities": {"SELL": 0.093, "NO_TRADE": 0.880, "BUY": 0.027},
  "confidence_breakdown": {
    "probability_separation": 78.78, "historical_accuracy": 33.33,
    "distribution_distance": 100.0, "feature_completeness": 100.0,
    "prediction_stability": 83.33, "calibration_quality": 71.2, "overall": 64.77
  },
  "explanation": ["Predicted class: NO_TRADE", "SELL probability: 9.3%", "..."],
  "warnings": []
}
```

This is a reasonable shape for **evaluating a model**, but it exposes a full class-probability
distribution and internal diagnostic detail to every live caller -- more than a production
prediction service should return, and not what a Decision Engine consuming a single directional
signal needs.

## New production inference contract

`POST /logistic-regression/predict` now returns exactly one object,
`logistic_regression.live_inference.LiveInferenceResponse`:

```json
{
  "prediction": "BUY | SELL | WAIT",
  "prediction_confidence": 91,
  "prediction_horizon": 5,
  "model_version": "1.0.0",
  "feature_version": "1.0.0",
  "training_version": "1.0.0",
  "model_health": 94,
  "timestamp": "2026-07-15T00:00:00+00:00"
}
```

Nothing else. No `classes`, no `buy_probability`/`sell_probability`/`no_trade_probability`, no
`class_probabilities`, no confusion matrix, no accuracy/precision/recall/F1/ROC-AUC/PR-AUC/log-loss/
Brier score. `tests/test_lgr_live_inference.py::test_to_dict_has_exactly_the_contract_keys_and_nothing_else`
asserts this shape directly against a real trained model's output, not just against the formula.

### `prediction`

One of `"BUY"`, `"SELL"`, `"WAIT"` -- the engine's internal class set stays exactly
`("SELL", "NO_TRADE", "BUY")` everywhere it already was (label generation, model training, the
model registry, evaluation reports, the Decision Engine's own consumption of the full prediction
object). Only `logistic_regression/live_inference.py`'s `to_live_inference()` maps
`"NO_TRADE" -> "WAIT"` for this one display layer; `"BUY"`/`"SELL"` pass through unchanged. This
keeps the internal class name a single source of truth while giving live callers the
production-facing vocabulary the spec asks for.

### `prediction_confidence`

Unchanged computation -- still `logistic_regression/confidence.py`'s existing six-factor engine
(probability separation, historical accuracy, distribution distance, feature completeness,
prediction stability, calibration quality), still structurally independent of which class was
predicted (`compute_confidence()`'s signature never sees the predicted class). Only the
*presentation* changed: the live contract rounds it to an integer 0-100.

### `model_health`

New. A single, per-**model** (not per-prediction) 0-100 score -- distinct from
`prediction_confidence`, which varies per input. Computed once per training run in
`logistic_regression/trainer.py::fit_model()` via the new `logistic_regression/model_health.py`,
blending two statistics the trainer already computes on an internal held-out slice:
65% historical (held-out) balanced accuracy, 35% calibration quality (inverted Brier score, neutral
60 if calibration wasn't run). This mirrors the idea behind `linear_regression`'s Phase 8
`model_health_score` for a classifier, written independently for this engine (nothing was imported
from or changed in `linear_regression/`).

## Separation of evaluation vs. inference

Training and evaluation are completely unaffected and still produce every metric the spec
requires -- confirmed live via `POST /logistic-regression/train`:

```json
{
  "symbol": "EUR_USD", "timeframe": "M5", "classes": ["SELL", "NO_TRADE", "BUY"],
  "test_metrics": {
    "accuracy": 0.879, "precision": 0.293, "recall": 0.333, "f1": 0.312,
    "specificity": 0.667, "balanced_accuracy": 0.333,
    "confusion_matrix": [[0, 6, 0], [0, 102, 0], [0, 8, 0]],
    "roc_auc": 0.630, "pr_auc": 0.478, "log_loss": 0.468, "brier_score": 0.073
  }
}
```

None of `evaluator.py`, `metrics.py`, `calibration.py`, or `trainer.py`'s evaluation-report
augmentation were modified. `tests/test_lgr_live_inference.py::test_training_evaluation_metrics_untouched`
is a regression guard confirming `record.testing_metrics`/`validation_metrics` still contain every
evaluation key. These outputs remain available **only** from:

- Training responses (`POST /logistic-regression/train`'s `test_metrics`).
- `ExperimentRecord.testing_metrics`/`validation_metrics`/`training_metrics`.
- `evaluation_report.json` (calibration curves, coefficient diagnostics, feature importance,
  prediction/probability distributions -- via `ClassificationEvaluator`, unmodified).

They are never returned by `POST /logistic-regression/predict` anymore.

## Why only the predicted direction is exposed

A live trading signal consumer (ultimately the Decision Engine) needs one thing: *what does this
model think happens next, and how much should that be trusted*. A full probability distribution,
margin, and entropy are model-introspection detail useful for evaluation and debugging, not for a
production decision -- and returning them over a live API invites a caller to (incorrectly) re-derive
its own decision logic from raw probabilities instead of treating the model's own argmax decision
(already threshold-aware via `probability_engine.predicted_class()`) as authoritative. Collapsing to
a single class + one confidence score is also symmetric with how `linear_regression`'s live
predictions are consumed -- one directional estimate, one trust score, nothing else -- keeping the
two engines' live contracts consistent for whatever consumes them next.

## Decision Engine integration

**Unaffected.** `api/pipeline.py::make_decision()` (which powers `POST /decision/predict`) and
`monitor_health()` (which powers `POST /monitor/health`) both call
`classification_bundle.engine.predict()` **directly** and pass the full `ClassificationPrediction`
object onward -- neither ever goes through `predict_classification()`, the function whose return
shape changed. `LogisticRegressionEngine.predict()` and `ClassificationPredictor.predict()` still
return the complete `ClassificationPrediction` (now with one additive `model_health` field, backward
compatible with every existing consumer since it's a new optional trailing field, never a removed or
renamed one). Verified live: `POST /decision/predict` was called end-to-end after this change and
its `logistic_regression` section still reports the full
`predicted_class`/`buy_probability`/`sell_probability`/`no_trade_probability`/`classification_confidence`
detail the Decision Engine's reasoning depends on, unchanged.

Per the design principle in the spec, the Logistic Regression Engine still makes its prediction from
`MarketState` alone -- it never reads the Linear Regression Engine, the Strategy Engine, or the
Decision Engine. Agreement or disagreement across engines remains the Decision Engine's job alone.

## Example responses

Two of these are real, captured live from `POST /logistic-regression/predict` against OANDA practice
EUR_USD/GBP_USD M5 data on 2026-07-15 (`NO_TRADE`/`WAIT` is this label generator's majority class by
design -- it only fires BUY/SELL when a 5-bar-ahead move clears both `min_pip_movement=5.0` pips and
a 1.5 risk/reward threshold, so most windows land on WAIT). BUY and SELL are shown with the same real
model's version/health metadata for a session where the market move cleared that threshold.

**WAIT (real, EUR_USD M5):**

```json
{
  "prediction": "WAIT",
  "prediction_confidence": 65,
  "prediction_horizon": 5,
  "model_version": "1.0.0",
  "feature_version": "1.0.0",
  "training_version": "1.0.0",
  "model_health": 57,
  "timestamp": "2026-07-14T23:32:03+00:00"
}
```

**WAIT (real, GBP_USD M5, a second independently-trained model):**

```json
{
  "prediction": "WAIT",
  "prediction_confidence": 53,
  "prediction_horizon": 5,
  "model_version": "1.0.0",
  "feature_version": "1.0.0",
  "training_version": "1.0.0",
  "model_health": 37,
  "timestamp": "2026-07-14T23:35:00+00:00"
}
```

**BUY (constructed -- same contract shape, illustrating a session where BUY wins the argmax):**

```json
{
  "prediction": "BUY",
  "prediction_confidence": 78,
  "prediction_horizon": 5,
  "model_version": "1.0.0",
  "feature_version": "1.0.0",
  "training_version": "1.0.0",
  "model_health": 57,
  "timestamp": "2026-07-15T00:00:00+00:00"
}
```

**SELL (constructed -- same contract shape, illustrating a session where SELL wins the argmax):**

```json
{
  "prediction": "SELL",
  "prediction_confidence": 82,
  "prediction_horizon": 5,
  "model_version": "1.0.0",
  "feature_version": "1.0.0",
  "training_version": "1.0.0",
  "model_health": 57,
  "timestamp": "2026-07-15T00:00:00+00:00"
}
```

All four shapes are produced and asserted directly (not fabricated ad hoc) by
`tests/test_lgr_live_inference.py`: `test_no_trade_maps_to_wait`, `test_buy_passes_through`,
`test_sell_passes_through`, and `test_live_inference_from_real_trained_model` (which runs the real
contract end-to-end and asserts every prediction lands in `{"BUY", "SELL", "WAIT"}`).

## Files changed

| File | Change |
|---|---|
| `logistic_regression/model_health.py` | New -- per-model 0-100 health score. |
| `logistic_regression/trainer.py` | One additive line: sets `model.model_health_`. |
| `logistic_regression/predictor.py` | One new optional field (`model_health`) on `ClassificationPrediction`, backward compatible. |
| `logistic_regression/live_inference.py` | New -- `LiveInferenceResponse` + `to_live_inference()`. |
| `logistic_regression/__init__.py` | Exports the three new symbols above. |
| `api/pipeline.py` | `predict_classification()` now returns `to_live_inference(prediction).to_dict()`. |
| `api/main.py` | Updated the `/logistic-regression/predict` description string only. |
| `tests/test_lgr_live_inference.py` | New -- 10 tests covering the contract shape, class mapping, and two regression guards. |

Nothing else changed. Full test suite: 875 passed, 1 skipped, 0 failed (up from 865/1 before this
change -- the 10 new tests).
