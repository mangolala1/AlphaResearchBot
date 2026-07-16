"""Unit tests for the V4 composite scoring system (core.decision.score_alpha)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.decision import (
    IC_MEAN_HARD, IC_MEAN_SOFT, ICIR_HARD, ICIR_SOFT, MONO_HARD, MONO_SOFT,
    PROMISING_MIN, REVISE_MIN, SHARPE_HARD, SHARPE_SOFT,
    _ramp, _ramp2, score_alpha,
)
from core.formula_validator import formula_complexity

SIMPLE_FORMULA = "rank(ADJUSTED_PRICE) * -1"


def make_metrics(ic=0.03, icir=0.5, sharpe=0.8, dsh=0.7, mono=0.5, to=0.4, dd=-0.15):
    return dict(
        IC_mean=ic, ICIR=icir, Sharpe=sharpe, deflated_sharpe=dsh,
        monotonicity=mono, turnover=to, max_drawdown=dd,
        Q5_Q1_return=sharpe / 10, noise_risk="low",
    )


NEUTRAL_ROBUSTNESS = dict(
    sector_stability={}, subperiod_stability=0.5,
    market_regime_sharpe={}, placebo_score=0.5,
)


# ── Ramp anchors: continuity with the old V3 thresholds ─────────────────────

@pytest.mark.parametrize("hard,soft,hi", [
    (IC_MEAN_HARD, IC_MEAN_SOFT, 0.05),
    (ICIR_HARD, ICIR_SOFT, 1.0),
    (SHARPE_HARD, SHARPE_SOFT, 1.5),
    (MONO_HARD, MONO_SOFT, 0.8),
])
def test_ramp2_anchors_at_old_thresholds(hard, soft, hi):
    assert _ramp2(soft, hard, soft, hi) == pytest.approx(0.5)
    assert _ramp2(hard, hard, soft, hi) == pytest.approx(0.0)
    assert _ramp2(hi, hard, soft, hi) == pytest.approx(1.0)


def test_ramp_clips():
    assert _ramp(-1.0, 0.0, 1.0) == 0.0
    assert _ramp(2.0, 0.0, 1.0) == 1.0


# ── Monotonicity: raising any single metric never lowers the total ──────────

@pytest.mark.parametrize("field,lo,hi", [
    ("ic", 0.0, 0.06), ("icir", 0.0, 1.2), ("sharpe", 0.1, 2.0),
    ("mono", -0.1, 0.9), ("dd", -0.39, -0.05),
])
def test_total_monotone_in_each_metric(field, lo, hi):
    kwargs_lo = {field: lo}
    kwargs_hi = {field: hi}
    s_lo = score_alpha(make_metrics(**kwargs_lo), NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.3)
    s_hi = score_alpha(make_metrics(**kwargs_hi), NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.3)
    assert s_hi.total >= s_lo.total


def test_lower_turnover_never_hurts():
    s_lo = score_alpha(make_metrics(to=0.2), NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.3)
    s_hi = score_alpha(make_metrics(to=0.85), NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.3)
    assert s_lo.total >= s_hi.total


# ── Fatal gates ──────────────────────────────────────────────────────────────

def test_dead_signal_is_fatal():
    m = make_metrics(ic=0.001, sharpe=0.05, icir=0.01, mono=0.0)
    s = score_alpha(m, NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.3)
    assert s.fatal
    assert s.verdict == "failed"
    assert "dead signal" in s.failure_reason


def test_catastrophic_drawdown_is_fatal():
    m = make_metrics(dd=-0.45)
    s = score_alpha(m, NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.3)
    assert s.fatal
    assert s.verdict == "failed"


# ── Direction handling ────────────────────────────────────────────────────────

def test_contrarian_alpha_fails_with_direction_note():
    """A real-but-inverted signal is not a separate outcome — it fails with a note."""
    m = make_metrics(ic=-0.04, icir=-0.9, sharpe=-1.4, dsh=-1.2, mono=-0.7, dd=-0.3)
    s = score_alpha(m, NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.2)
    assert not s.fatal  # strong |Sharpe| — the bidirectional dead-signal gate passes
    assert s.verdict == "failed"
    assert s.direction_status == "contradicted"
    assert s.predictive_magnitude > s.total  # magnitude sees the signal, total doesn't
    assert "direction contradicted" in s.failure_reason
    assert "dead signal" not in s.failure_reason


def test_contrarian_with_catastrophic_drawdown_is_fatal():
    m = make_metrics(ic=-0.04, icir=-0.9, sharpe=-1.4, dsh=-1.2, mono=-0.7, dd=-0.5)
    s = score_alpha(m, NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.2)
    assert s.fatal
    assert s.verdict == "failed"
    assert s.direction_status == "contradicted"
    assert "direction contradicted" in s.failure_reason  # note survives the fatal gate


def test_strong_positive_alpha_is_supported():
    m = make_metrics(ic=0.06, icir=1.2, sharpe=1.8, dsh=1.5, mono=0.9, to=0.2, dd=-0.08)
    s = score_alpha(m, NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.2)
    assert s.verdict == "promising"
    assert s.failure_reason is None
    assert s.direction_status == "supported"
    assert s.predictive_magnitude == s.total  # all-positive metrics: abs is identity


def test_mixed_signs_are_uncertain():
    m = make_metrics(ic=0.03, sharpe=-0.6)
    s = score_alpha(m, NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.2)
    assert s.direction_status == "uncertain"
    assert "direction contradicted" not in (s.failure_reason or "")


def test_dead_signal_is_uncertain():
    m = make_metrics(ic=0.001, sharpe=0.05, icir=0.01, mono=0.0)
    s = score_alpha(m, NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.3)
    assert s.direction_status == "uncertain"


# ── Verdict bands ─────────────────────────────────────────────────────────────

def test_all_soft_thresholds_lands_in_revise():
    m = make_metrics(ic=IC_MEAN_SOFT, icir=ICIR_SOFT, sharpe=SHARPE_SOFT,
                     dsh=SHARPE_SOFT, mono=MONO_SOFT, to=0.5, dd=-0.25)
    s = score_alpha(m, NEUTRAL_ROBUSTNESS, SIMPLE_FORMULA, 0.3)
    assert REVISE_MIN <= s.total < PROMISING_MIN
    assert s.verdict == "revise"


def test_all_strong_is_promising():
    m = make_metrics(ic=0.06, icir=1.2, sharpe=1.8, dsh=1.5, mono=0.9, to=0.2, dd=-0.08)
    rob = dict(sector_stability={}, subperiod_stability=0.8,
               market_regime_sharpe={"bull": 1.5, "bear": 0.8}, placebo_score=0.9)
    s = score_alpha(m, rob, SIMPLE_FORMULA, 0.1)
    assert s.total >= PROMISING_MIN
    assert s.verdict == "promising"
    assert s.failure_reason is None


def test_failure_reason_names_weakest_subscore():
    # Weak on simplicity: very complex formula
    complex_formula = (
        "rank(ts_mean(rank(REVENUE_LTM / COGS_LTM), 20))"
        " + rank(ts_std(rank(CFO_LTM / NET_INCOME_LTM), 60))"
        " * rank(decay_linear(GROSS_PROFIT_LTM / SGA_EXPENSE_LTM, 10))"
        " - rank(ts_corr(DA_LTM, CFI_LTM, 30))"
    )
    m = make_metrics()
    s = score_alpha(m, NEUTRAL_ROBUSTNESS, complex_formula, 0.3)
    if s.verdict not in ("promising",):
        assert "weakest component" in (s.failure_reason or "")


# ── Formula complexity ────────────────────────────────────────────────────────

def test_formula_complexity_calibration():
    # 1 call + depth 2 + 1 column = 4 (X alone isn't a known column → use a real one)
    assert formula_complexity("rank(ADJUSTED_PRICE) * -1") == 4
    quality_value = (
        "rank((OPERATING_INCOME_LTM + DA_LTM) / REVENUE_LTM)"
        " + rank(ADJUSTED_PRICE / (REVENUE_LTM / SHARES_DILUTED)) * -1"
    )
    assert 10 <= formula_complexity(quality_value) <= 13


def test_formula_complexity_syntax_error_fallback():
    assert formula_complexity("rank(ADJUSTED_PRICE * ") >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
