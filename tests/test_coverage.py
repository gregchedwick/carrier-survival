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
import pytest

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


# --------------------------------------------------------------------------
# date parsing across vintages
# --------------------------------------------------------------------------


def test_each_vintage_date_format_parses():
    """Three formats appear across the archive and all must survive ingest.

    Regression test. The parser hard-coded ``format="%Y%m%d"`` with
    ``errors="coerce"``, so the November 2023 export's ``M/D/YYYY`` dates all
    became NaT. Downstream that read as "the source did not carry this field"
    rather than "the parser did not understand it".
    """
    from carrier_survival.census_history import _parse_dates

    text_ymd = _parse_dates(pd.Series(["20220609", "20231023"]), "t")
    slashed = _parse_dates(pd.Series(["9/25/2023", "7/5/2023"]), "t")
    integers = _parse_dates(pd.Series([20220609, 20231023]), "t")

    assert list(text_ymd) == [pd.Timestamp("2022-06-09"), pd.Timestamp("2023-10-23")]
    assert list(slashed) == [pd.Timestamp("2023-09-25"), pd.Timestamp("2023-07-05")]
    assert list(integers) == [pd.Timestamp("2022-06-09"), pd.Timestamp("2023-10-23")]


def test_blanks_stay_null_without_tripping_the_guard():
    """Genuinely empty values are missing data, not a format failure."""
    from carrier_survival.census_history import _parse_dates

    out = _parse_dates(pd.Series([None, "", "20220609"]), "t")

    assert out.isna().sum() == 2
    assert out.notna().sum() == 1


def test_an_unparseable_date_column_raises_instead_of_nulling():
    """A format nobody recognises must fail loudly, not vanish."""
    from carrier_survival.census_history import _parse_dates

    with pytest.raises(ValueError, match="Unrecognised format"):
        _parse_dates(pd.Series(["banana"] * 2000), "some_file:add_date")
