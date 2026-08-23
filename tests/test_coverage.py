"""Tests for snapshot coverage and quality detection.

The staleness detector is the guardrail standing between a frozen export and
two trajectory features that compute to exactly zero for every carrier while
looking fully populated. It failed silently once already; these pin the failure
mode down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carrier_survival.census_history import coverage_report  # noqa: E402

N = 5_000


def snapshot(period: str, units: np.ndarray, source_kind: str = "reporting") -> pd.DataFrame:
    return pd.DataFrame({
        "dot_number": pd.array(np.arange(N), dtype="Int64"),
        "period": period,
        "source_kind": source_kind,
        "power_units": units,
        "driver_count": np.ones(N),
    })


def test_frozen_export_is_flagged_even_when_many_values_are_null():
    """A republished export must be caught despite nulls on both sides.

    Regression test. The detector compared with ``!=``, and because NaN != NaN
    is True in pandas, carriers null in *both* exports counted as changed. At
    the real 5.6% null rate that alone cleared the 0.1% threshold, so six
    consecutive frozen months were reported as safe to difference.
    """
    units = np.arange(N, dtype=float)
    units[:400] = np.nan                      # 8% null, null in both periods

    panel = pd.concat([snapshot("202407", units), snapshot("202408", units.copy())])
    report = coverage_report(panel).set_index("period")

    assert report.loc["202408", "stale_export"]
    assert not report.loc["202408", "safe_to_diff"]


def test_a_genuinely_refreshed_export_is_not_flagged():
    """Real month-over-month movement must survive the check."""
    units = np.arange(N, dtype=float)
    units[:400] = np.nan

    moved = units.copy()
    moved[1000:1500] += 1                     # 10% of carriers changed fleet size

    panel = pd.concat([snapshot("202407", units), snapshot("202408", moved)])
    report = coverage_report(panel).set_index("period")

    assert not report.loc["202408", "stale_export"]


def test_staleness_is_not_compared_across_vintages():
    """Different export types are different populations, not a stale republish."""
    units = np.arange(N, dtype=float)

    panel = pd.concat([
        snapshot("202406", units, source_kind="reporting"),
        snapshot("202506", units.copy(), source_kind="datahub"),
    ])
    report = coverage_report(panel).set_index("period")

    assert not report.loc["202506", "stale_export"]
