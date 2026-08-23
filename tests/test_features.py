"""Unit tests for point-in-time feature construction.

These exist because every defect this project has hit was silent. A population
break, a grand total read as fleet size, a pandas dtype change that zeroed a
column — none raised an error, and each was found by staring at aggregate
output and guessing. Small inputs with known answers find them in seconds.

Each test builds a panel of a few rows where the correct result can be worked
out by hand, so a failure says which function is wrong rather than that the
numbers look off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carrier_survival.features import (  # noqa: E402
    HORIZON_END_MONTHS,
    HORIZON_START_MONTHS,
    build_spine,
    fleet_features,
    static_features,
)


def make_panel(rows: list[dict]) -> pd.DataFrame:
    """Build a panel with the columns the real one has."""
    frame = pd.DataFrame(rows)
    frame["as_of"] = pd.PeriodIndex(frame["period"], freq="M").to_timestamp(how="end").normalize()
    frame["dot_number"] = frame["dot_number"].astype("Int64")
    # Fill rather than only add: naming a column on one row leaves the others
    # null, and a null vintage matches no filter — which silently empties the
    # panel and makes the test fail for a reason that has nothing to do with
    # what it is testing.
    for column, default in (
        ("source_kind", "reporting"),
        ("status", "A"),
        ("safety_rating", None),
        ("state", "NV"),
        ("driver_count", 1.0),
    ):
        if column not in frame.columns:
            frame[column] = default
        elif default is not None:
            frame[column] = frame[column].fillna(default)
    return frame


# --------------------------------------------------------------------------
# fleet trajectory
# --------------------------------------------------------------------------


def test_fleet_delta_detects_growth_and_shrinkage():
    """The delta must be a real proportion, not zero for everyone.

    This is the regression test for the bug that made both trajectory features
    compute to exactly 0.0 for every carrier while looking populated.
    """
    panel = make_panel([
        {"dot_number": 1, "period": "202407", "power_units": 10.0},
        {"dot_number": 1, "period": "202408", "power_units": 12.0},   # +20%
        {"dot_number": 2, "period": "202407", "power_units": 8.0},
        {"dot_number": 2, "period": "202408", "power_units": 4.0},    # -50%
        {"dot_number": 3, "period": "202407", "power_units": 5.0},
        {"dot_number": 3, "period": "202408", "power_units": 5.0},    # unchanged
    ])

    out = fleet_features(
        panel,
        pd.Timestamp("2024-08-31"),
        diffable_periods={"202407", "202408"},
    ).set_index("dot_number")

    assert out.loc[1, "power_units_chg_3m"] == pytest.approx(0.2)
    assert out.loc[2, "power_units_chg_3m"] == pytest.approx(-0.5)
    assert out.loc[3, "power_units_chg_3m"] == pytest.approx(0.0)


def test_fleet_delta_skips_periods_flagged_unusable():
    """A stale or broken export must not be used as the comparison point."""
    panel = make_panel([
        {"dot_number": 1, "period": "202406", "power_units": 10.0},
        {"dot_number": 1, "period": "202407", "power_units": 10.0},   # stale copy
        {"dot_number": 1, "period": "202408", "power_units": 15.0},
    ])

    # 202407 excluded: the comparison should reach back to 202406 instead, so
    # the change is 10 -> 15, not 10 -> 15 measured against a duplicate.
    out = fleet_features(
        panel,
        pd.Timestamp("2024-08-31"),
        diffable_periods={"202406", "202408"},
    ).set_index("dot_number")

    assert out.loc[1, "power_units_chg_3m"] == pytest.approx(0.5)


def test_fleet_level_survives_when_current_period_is_not_diffable():
    """Fleet size must still be reported when its period cannot be differenced.

    Filtering the level and the delta together emptied the trajectory features
    whenever the newest period happened to be flagged.
    """
    panel = make_panel([
        {"dot_number": 1, "period": "202407", "power_units": 9.0},
        {"dot_number": 1, "period": "202408", "power_units": 11.0},
    ])

    out = fleet_features(
        panel,
        pd.Timestamp("2024-08-31"),
        diffable_periods={"202407"},          # current period excluded
    ).set_index("dot_number")

    assert out.loc[1, "power_units_now"] == 11.0


def test_fleet_features_never_cross_vintages():
    """A delta across vintages measures which file a carrier was in."""
    panel = make_panel([
        {"dot_number": 1, "period": "202402", "power_units": 100.0, "source_kind": "mcmis"},
        {"dot_number": 1, "period": "202407", "power_units": 10.0},
        {"dot_number": 1, "period": "202408", "power_units": 10.0},
    ])

    out = fleet_features(
        panel,
        pd.Timestamp("2024-08-31"),
        diffable_periods={"202402", "202407", "202408"},
    ).set_index("dot_number")

    # Comparing against the mcmis row would show a 90% collapse that never happened.
    assert out.loc[1, "power_units_chg_3m"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# point-in-time discipline
# --------------------------------------------------------------------------


def test_fleet_features_ignore_the_future():
    """Nothing after the prediction date may influence a feature."""
    panel = make_panel([
        {"dot_number": 1, "period": "202407", "power_units": 6.0},
        {"dot_number": 1, "period": "202409", "power_units": 60.0},   # after the cut-off
    ])

    out = fleet_features(
        panel,
        pd.Timestamp("2024-08-31"),
        diffable_periods={"202407", "202409"},
    ).set_index("dot_number")

    assert out.loc[1, "power_units_now"] == 6.0


def test_days_since_mcs150_is_measured_to_the_prediction_date():
    """The filing-recency feature counts from the last filing, not the snapshot.

    MCS-150 is mandatory every 24 months, so a carrier well past that has
    stopped filing paperwork. The feature must be anchored to the prediction
    date or it just re-encodes which vintage the value came from.
    """
    panel = make_panel([
        {"dot_number": 1, "period": "202312", "power_units": 5.0, "source_kind": "mcmis",
         "mcs150_date": pd.Timestamp("2023-06-01")},
        {"dot_number": 2, "period": "202312", "power_units": 5.0, "source_kind": "mcmis",
         "mcs150_date": pd.Timestamp("2019-01-01")},   # long overdue
        {"dot_number": 1, "period": "202408", "power_units": 5.0},
        {"dot_number": 2, "period": "202408", "power_units": 5.0},
    ])

    date = pd.Timestamp("2024-08-31")
    out = static_features(panel, date).set_index("dot_number")

    assert out.loc[1, "days_since_mcs150"] == (date - pd.Timestamp("2023-06-01")).days
    assert out.loc[2, "days_since_mcs150"] == (date - pd.Timestamp("2019-01-01")).days
    assert out.loc[2, "days_since_mcs150"] > out.loc[1, "days_since_mcs150"]


def test_a_filing_after_the_prediction_date_is_not_used():
    """A future filing date must never produce a negative age.

    If a later vintage ever leaks into the lookup, this is where it shows up —
    as a carrier that filed in the future.
    """
    panel = make_panel([
        {"dot_number": 1, "period": "202312", "power_units": 5.0, "source_kind": "mcmis",
         "mcs150_date": pd.Timestamp("2025-06-01")},
        {"dot_number": 1, "period": "202408", "power_units": 5.0},
    ])

    out = static_features(panel, pd.Timestamp("2024-08-31")).set_index("dot_number")

    assert pd.isna(out.loc[1, "days_since_mcs150"])


def test_mileage_report_age_rejects_impossible_years():
    """A mileage year in the future, or decades stale, is bad data not signal."""
    reported = {1: 2021.0, 2: 2099.0, 3: 1900.0}
    panel = make_panel(
        [
            {"dot_number": n, "period": "202312", "power_units": 5.0,
             "source_kind": "mcmis", "mcs150_mileage_year": year}
            for n, year in reported.items()
        ]
        + [{"dot_number": n, "period": "202408", "power_units": 5.0} for n in reported]
    )

    out = static_features(panel, pd.Timestamp("2024-08-31")).set_index("dot_number")

    assert out.loc[1, "mileage_report_age_years"] == 3
    assert pd.isna(out.loc[2, "mileage_report_age_years"])
    assert pd.isna(out.loc[3, "mileage_report_age_years"])


def test_static_attributes_come_from_any_earlier_vintage():
    """Slow-moving fields may be read from an older vintage, never a newer one."""
    panel = make_panel([
        {"dot_number": 1, "period": "202402", "power_units": 5.0, "source_kind": "mcmis",
         "add_date": pd.Timestamp("2020-01-01")},
        {"dot_number": 1, "period": "202408", "power_units": 5.0, "add_date": pd.NaT},
    ])

    out = static_features(panel, pd.Timestamp("2024-08-31")).set_index("dot_number")

    # Registration date is immutable, so a February reading is valid in August.
    expected = (pd.Timestamp("2024-08-31") - pd.Timestamp("2020-01-01")).days
    assert out.loc[1, "carrier_age_days"] == expected


# --------------------------------------------------------------------------
# the risk set
# --------------------------------------------------------------------------


def _labels(rows: list[dict]) -> pd.DataFrame:
    columns = ["dot_number", "exit_date", "datable", "is_failure", "exclude_from_training"]
    frame = pd.DataFrame(rows, columns=columns) if not rows else pd.DataFrame(rows)
    frame["dot_number"] = frame["dot_number"].astype("Int64")
    for column, default in (
        ("datable", True),
        ("is_failure", True),
        ("exclude_from_training", False),
    ):
        if column not in frame.columns:
            frame[column] = default
    return frame


def test_carriers_already_gone_are_not_at_risk():
    """A carrier that exited before the prediction date is not a survivor."""
    panel = make_panel([
        {"dot_number": 1, "period": "202408", "power_units": 5.0},
        {"dot_number": 2, "period": "202408", "power_units": 5.0},
    ])
    labels = _labels([{"dot_number": 1, "exit_date": pd.Timestamp("2024-01-15")}])

    spine = build_spine(panel, labels, [pd.Timestamp("2024-08-31")]).frame

    assert set(spine["dot_number"]) == {2}


def test_failure_is_only_inside_the_outcome_window():
    """Exits before the blackout ends, or after the horizon, are not failures."""
    date = pd.Timestamp("2024-08-31")
    panel = make_panel([
        {"dot_number": n, "period": "202408", "power_units": 5.0} for n in (1, 2, 3)
    ])
    labels = _labels([
        # inside the blackout — too close to the prediction date to count
        {"dot_number": 1, "exit_date": date + pd.DateOffset(months=HORIZON_START_MONTHS - 2)},
        # squarely inside the window
        {"dot_number": 2, "exit_date": date + pd.DateOffset(months=HORIZON_START_MONTHS + 3)},
        # beyond the horizon
        {"dot_number": 3, "exit_date": date + pd.DateOffset(months=HORIZON_END_MONTHS + 2)},
    ])

    spine = build_spine(panel, labels, [date]).frame.set_index("dot_number")

    assert not spine.loc[1, "failed"]
    assert spine.loc[2, "failed"]
    assert not spine.loc[3, "failed"]


def test_inactive_carriers_are_excluded_from_the_risk_set():
    """An inactive carrier has already stopped; it cannot fail."""
    panel = make_panel([
        {"dot_number": 1, "period": "202408", "power_units": 5.0, "status": "A"},
        {"dot_number": 2, "period": "202408", "power_units": 5.0, "status": "I"},
    ])
    spine = build_spine(panel, _labels([]), [pd.Timestamp("2024-08-31")]).frame

    assert set(spine["dot_number"]) == {1}
