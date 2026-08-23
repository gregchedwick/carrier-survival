"""Build the point-in-time feature table.

    python scripts/build_features.py

Reads data/raw/ and data/interim/exit_labels.parquet, writes
data/interim/features.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carrier_survival.census_history import coverage_report  # noqa: E402
from carrier_survival.config import DATA_DIR, RAW_DIR  # noqa: E402
from carrier_survival.features import (  # noqa: E402
    HORIZON_END_MONTHS,
    HORIZON_START_MONTHS,
    assemble,
    build_spine,
    prediction_dates,
)

INTERIM_DIR = DATA_DIR / "interim"


def main() -> None:
    panel = pd.read_parquet(RAW_DIR / "carrier_panel.parquet")
    labels = pd.read_parquet(INTERIM_DIR / "exit_labels.parquet")
    crash = pd.read_parquet(
        RAW_DIR / "crash.parquet",
        columns=["dot_number", "report_date", "fatalities", "injuries", "tow_away"],
    )
    authhist = pd.read_parquet(
        RAW_DIR / "authhist.parquet",
        columns=[
            "dot_number",
            "docket_number",
            "original_action_desc",
            "disp_action_desc",
            "disp_served_date",
        ],
    )

    coverage = coverage_report(panel)
    diffable = set(coverage.loc[coverage["safe_to_diff"], "period"])
    print(f"panel periods usable for differencing: {len(diffable)} of {len(coverage)}")

    dates = prediction_dates(panel, labels)
    print(
        f"prediction dates: {len(dates)}  "
        f"({dates[0].date()} .. {dates[-1].date()}), "
        f"outcome window months {HORIZON_START_MONTHS}-{HORIZON_END_MONTHS}\n"
    )

    spine = build_spine(panel, labels, dates)
    features = assemble(spine, crash, authhist, panel, diffable)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / "features.parquet"
    features.to_parquet(out, index=False)

    print(f"\n  rows              : {len(features):,}")
    print(f"  distinct carriers : {features['dot_number'].nunique():,}")
    print(f"  failure rate      : {features['failed'].mean() * 100:.2f}%")
    print(f"  feature columns   : {features.shape[1]}")
    print(f"  -> {out}  ({out.stat().st_size / 1024 / 1024:,.1f} MB)")


if __name__ == "__main__":
    main()
