"""Phase 6 -- target suitability: does a linear model actually fit this
target well, and if not, why?

Pure aggregation over diagnostics already computed by the other three
phases (``target_diagnostics.py``, ``feature_diagnostics.py``,
``residual_diagnostics.py``) plus the cross-validated R^2 already produced
by ``cross_validation.py`` -- this module computes nothing new from raw
data, it only reasons about numbers those modules already produced. Every
verdict is accompanied by concrete, checkable reasons (never a bare "R^2 is
low") -- the same "recommendations, never silent judgments" discipline
``feature_diagnostics.py`` already established.

Verdicts are based on the walk-forward cross-validated mean R^2, not a
single train/test split's R^2, per the spec's own framing: "The goal is NOT
to artificially maximize R^2. The goal is to improve generalization."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .feature_diagnostics import FeatureAuditReport
from .residual_diagnostics import ResidualDiagnostics
from .target_diagnostics import TargetDiagnostics

#: Below this walk-forward CV mean R^2, a target is flagged "poor" --
#: a linear model is explaining almost none of its out-of-sample variance.
POOR_R2_THRESHOLD = 0.05

#: Below this (and at/above POOR_R2_THRESHOLD), a target is "marginal" --
#: some genuine signal, but too weak to be a primary decision input.
MARGINAL_R2_THRESHOLD = 0.15

#: A price-level target whose top correlated feature exceeds this |r| is
#: almost certainly just echoing the current close, not predicting anything.
TRIVIAL_CORRELATION_THRESHOLD = 0.999

#: Raw price-level targets -- a near-1.0 R^2 on these is expected
#: autocorrelation with the current close, not genuine directional skill.
PRICE_LEVEL_TARGETS = frozenset({"next_close", "next_open", "average_future_price", "future_midpoint"})

#: Targets already expressed as a return/movement (rather than a raw level).
RETURN_LIKE_TARGETS = frozenset({
    "next_return", "next_log_return", "expected_pip_movement", "expected_percentage_change",
})

#: Targets measuring dispersion/spread rather than direction.
DISPERSION_TARGETS = frozenset({
    "future_volatility", "future_atr", "future_range",
    "maximum_favorable_excursion", "maximum_adverse_excursion",
})


@dataclass(slots=True)
class TargetSuitability:
    """Complete Phase 6 verdict for one regression target."""

    target: str
    verdict: str  # "suitable" | "marginal" | "poor"
    r2_before: float
    cv_mean_r2: Optional[float]
    cv_std_r2: Optional[float]
    max_abs_pearson: float
    max_mutual_information: float
    is_stationary: bool
    residual_issues: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    suggested_reformulation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target, "verdict": self.verdict,
            "r2_before": round(self.r2_before, 4),
            "cv_mean_r2": None if self.cv_mean_r2 is None else round(self.cv_mean_r2, 4),
            "cv_std_r2": None if self.cv_std_r2 is None else round(self.cv_std_r2, 4),
            "max_abs_pearson": round(self.max_abs_pearson, 4),
            "max_mutual_information": round(self.max_mutual_information, 4),
            "is_stationary": self.is_stationary,
            "residual_issues": self.residual_issues,
            "reasons": self.reasons,
            "suggested_reformulation": self.suggested_reformulation,
        }


def _verdict(r2: Optional[float]) -> str:
    if r2 is None or r2 < POOR_R2_THRESHOLD:
        return "poor"
    if r2 < MARGINAL_R2_THRESHOLD:
        return "marginal"
    return "suitable"


def _suggest_reformulation(target: str, verdict: str, trivial_r2: bool) -> Optional[str]:
    if trivial_r2:
        return (
            f"'{target}' is a raw price-level target whose near-perfect R^2 mostly reflects "
            "autocorrelation with the current close, not genuine predictive skill -- prefer a "
            "return-based target (e.g. 'next_return' or 'expected_pip_movement') for actual "
            "trading decisions."
        )
    if verdict == "suitable":
        return None
    if target in RETURN_LIKE_TARGETS:
        return (
            f"'{target}' is already return/movement-based (the correct formulation for this kind "
            "of target); if R^2 remains poor after regularization, the relationship may be "
            "genuinely non-linear or regime-dependent -- consider a directional classification "
            "framing (up/down) rather than regressing the exact magnitude."
        )
    if target in DISPERSION_TARGETS:
        return (
            f"'{target}' measures dispersion/spread, which is typically right-skewed and "
            "heteroscedastic -- consider fitting a log-transformed version of this target rather "
            "than the raw value."
        )
    if target in PRICE_LEVEL_TARGETS:
        return (
            f"'{target}' is a raw price level -- consider a return-based reformulation "
            "(e.g. the corresponding 'next_return'-style target) instead of the raw level."
        )
    return "No specific reformulation suggested; the poor fit appears feature-driven (low signal density) rather than a formulation artifact."


def assess_target_suitability(
    target: str, *, r2_before: float, cv_mean_r2: Optional[float], cv_std_r2: Optional[float],
    target_diag: TargetDiagnostics, feature_audit: FeatureAuditReport, residual_diag: ResidualDiagnostics,
) -> TargetSuitability:
    """Aggregate Phases 1/2/3/4 evidence into one suitability verdict for
    ``target``. Every argument is an already-computed diagnostics object
    from the corresponding phase -- this function performs no new
    statistical computation of its own."""
    effective_r2 = cv_mean_r2 if cv_mean_r2 is not None else r2_before
    verdict = _verdict(effective_r2)

    abs_pearson = {k: abs(v) for k, v in feature_audit.pearson.items()}
    max_abs_pearson = max(abs_pearson.values()) if abs_pearson else 0.0
    max_mi = max(feature_audit.mutual_information.values()) if feature_audit.mutual_information else 0.0
    top_feature = max(abs_pearson, key=abs_pearson.get) if abs_pearson else None

    trivial_r2 = (
        target in PRICE_LEVEL_TARGETS
        and top_feature is not None
        and max_abs_pearson > TRIVIAL_CORRELATION_THRESHOLD
    )

    reasons: List[str] = []
    if trivial_r2:
        reasons.append(
            f"Top correlated feature '{top_feature}' has |Pearson r|={max_abs_pearson:.4f} with "
            f"'{target}' -- an R^2 this high on a raw price-level target is expected autocorrelation "
            "with the current close, not genuine directional predictive skill."
        )

    if verdict != "suitable":
        cv_desc = (
            f"walk-forward CV mean R^2={cv_mean_r2:.4f} (+/- {cv_std_r2:.4f})"
            if cv_mean_r2 is not None else f"single-split R^2={r2_before:.4f} (no CV run)"
        )
        reasons.append(f"{cv_desc} is {'below' if verdict == 'poor' else 'near'} the {'poor' if verdict == 'poor' else 'marginal'} threshold.")

        if max_abs_pearson < 0.05 and max_mi < 0.01:
            reasons.append(
                f"No individual feature carries strong signal for this target: max |Pearson r|={max_abs_pearson:.4f}, "
                f"max mutual information={max_mi:.4f} -- consistent with the poor R^2 (weak feature-driven fit, "
                "not obviously a bug)."
            )

        high_vif = [n for n, v in feature_audit.vif.items() if v == v and v > 10.0]
        if high_vif and max_abs_pearson >= 0.05:
            reasons.append(
                f"{len(high_vif)} feature(s) have VIF > 10 despite some real per-feature signal -- "
                "multicollinearity may be masking the true relationship; consider ridge/lasso "
                "(already supported by this engine) before concluding the target itself is unusable."
            )

        if not target_diag.stationarity.is_stationary_5pct:
            reasons.append(
                f"Target series is non-stationary at 5% (ADF stat={target_diag.stationarity.adf_statistic}) -- "
                "likely trending/drifting over the sample, which can distort a single split's R^2 depending on "
                "which regime the split lands in."
            )

    residual_issues: List[str] = []
    if residual_diag.heteroscedasticity.is_heteroscedastic_5pct:
        residual_issues.append("heteroscedastic")
        reasons.append(
            f"Residuals are heteroscedastic (Breusch-Pagan p={residual_diag.heteroscedasticity.lm_pvalue:.4f}) -- "
            "error variance is not constant, so a single global linear fit underserves some regimes "
            "(see the regime breakdown in RESIDUAL_ANALYSIS_REPORT.md)."
        )
    if not residual_diag.normality.is_normal_5pct:
        residual_issues.append("non-normal")

    if not reasons:
        reasons.append(f"No suitability issues detected at the configured thresholds for '{target}'.")

    return TargetSuitability(
        target=target, verdict=verdict, r2_before=r2_before, cv_mean_r2=cv_mean_r2, cv_std_r2=cv_std_r2,
        max_abs_pearson=max_abs_pearson, max_mutual_information=max_mi,
        is_stationary=target_diag.stationarity.is_stationary_5pct, residual_issues=residual_issues,
        reasons=reasons, suggested_reformulation=_suggest_reformulation(target, verdict, trivial_r2),
    )


def assess_all_targets(
    *, r2_before: Dict[str, float], cv_results: Dict[str, Any],
    target_diagnostics: Dict[str, TargetDiagnostics], feature_audits: Dict[str, FeatureAuditReport],
    residual_diagnostics: Dict[str, ResidualDiagnostics],
) -> Dict[str, TargetSuitability]:
    """Convenience wrapper mirroring ``target_diagnostics.analyze_all_targets``'s
    dict-in/dict-out shape. ``cv_results`` maps target -> an object with
    ``mean_r2``/``std_r2`` attributes (``cross_validation.CrossValidationResult``),
    or ``None`` if CV wasn't run for that target."""
    out: Dict[str, TargetSuitability] = {}
    for target in r2_before:
        cv = cv_results.get(target)
        out[target] = assess_target_suitability(
            target, r2_before=r2_before[target],
            cv_mean_r2=cv.mean_r2 if cv is not None else None,
            cv_std_r2=cv.std_r2 if cv is not None else None,
            target_diag=target_diagnostics[target], feature_audit=feature_audits[target],
            residual_diag=residual_diagnostics[target],
        )
    return out
