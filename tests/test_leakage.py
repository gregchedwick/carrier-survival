"""Tests for the leakage detector.

The point of these is the planted leak. A check that has never been shown to
fail is not evidence of anything — it may simply be incapable of firing. So the
central test builds a feature deliberately derived from the outcome and asserts
the detector catches it, and its counterpart asserts that honest features of
similar shape are left alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carrier_survival.leakage import (  # noqa: E402
    IMPLAUSIBLE_SOLO_AUC,
    MEANINGFUL_DECAY,
    check,
    single_feature_auc,
)

RNG = np.random.default_rng(0)
N = 20_000


def sample(base_rate: float = 0.06) -> np.ndarray:
    return (RNG.random(N) < base_rate).astype(int)


def test_planted_leak_is_caught():
    """A feature built from the outcome must be flagged.

    This is the control. If this test ever passes trivially — because the
    detector cannot fire at all — every other leakage result is meaningless.
    """
    y = sample()
    # The outcome with 10% of labels flipped: obviously leaking, but not perfect,
    # so it cannot be caught by a naive "identical to y" check.
    noisy = np.where(RNG.random(N) < 0.10, 1 - y, y).astype(float)
    frame = pd.DataFrame({"leaky": noisy})

    report = check(frame, ["leaky"], y)

    assert not report.passed
    assert "leaky" in report.suspicious
    assert report.suspicious["leaky"] > IMPLAUSIBLE_SOLO_AUC


def test_inverted_leak_is_caught():
    """A feature that predicts the outcome backwards leaks just as badly."""
    y = sample()
    frame = pd.DataFrame({"inverted": 1.0 - y})

    report = check(frame, ["inverted"], y)

    assert "inverted" in report.suspicious


def test_honest_features_are_left_alone():
    """Weak, genuinely predictive features must not trip the detector."""
    y = sample()
    frame = pd.DataFrame({
        # Correlated with the outcome but far from determining it — the shape a
        # real feature has.
        "weak_signal": y * 0.6 + RNG.normal(0, 1.0, N),
        "pure_noise": RNG.normal(0, 1, N),
        "constant": np.ones(N),
        "all_null": np.full(N, np.nan),
    })

    report = check(frame, list(frame.columns), y)

    assert report.passed, f"false positives: {report.suspicious}"


def test_degenerate_features_are_skipped_not_scored():
    """Constant and all-null columns cannot be scored and must not raise."""
    y = sample()
    frame = pd.DataFrame({"constant": np.ones(N), "all_null": np.full(N, np.nan)})

    scores = single_feature_auc(frame, list(frame.columns), y)

    assert scores == {}


def test_improving_with_staleness_is_a_failure_not_a_pass():
    """A model that gets BETTER as information ages must not pass.

    Regression test. The verdict compared ``abs(decay)`` against the threshold,
    so a probe whose AUC rose from 0.302 to 0.323 over six months was reported
    as "degrades as expected" — the exact opposite of what happened.
    """
    y = sample()
    frame = pd.DataFrame({"x": RNG.normal(0, 1, N)})

    report = check(frame, ["x"], y, decay=-MEANINGFUL_DECAY * 5)

    assert not report.passed
    assert not report.decay_conclusive
    assert "SUSPECT" in report.describe()
    assert "IMPROVED" in report.describe()


def test_an_inverted_probe_is_reported_as_broken():
    """A probe scoring below 0.5 is measuring the wrong thing entirely."""
    y = sample()
    frame = pd.DataFrame({"x": RNG.normal(0, 1, N)})

    report = check(frame, ["x"], y, decay=0.05, probe_auc=0.302)

    assert not report.passed
    assert report.probe_inverted
    assert "BROKEN" in report.describe()


def test_a_healthy_probe_does_not_trip_the_inversion_check():
    y = sample()
    frame = pd.DataFrame({"x": RNG.normal(0, 1, N)})

    report = check(frame, ["x"], y, decay=0.05, probe_auc=0.74)

    assert report.passed
    assert not report.probe_inverted


def test_small_decay_is_reported_as_inconclusive():
    """A decay within noise must not be reported as a pass.

    The earlier version printed "OK" off a 0.004 gap, which manufactures
    confidence rather than testing for anything.
    """
    y = sample()
    frame = pd.DataFrame({"x": RNG.normal(0, 1, N)})

    marginal = check(frame, ["x"], y, decay=MEANINGFUL_DECAY / 2)
    assert not marginal.decay_conclusive
    assert "inconclusive" in marginal.describe()

    clear = check(frame, ["x"], y, decay=MEANINGFUL_DECAY * 5)
    assert clear.decay_conclusive
    assert "degrades as expected" in clear.describe()
