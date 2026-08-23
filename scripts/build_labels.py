"""Build the carrier exit label table.

    python scripts/build_labels.py

Reads data/raw/{authhist,revocation}.parquet and writes
data/interim/exit_labels.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carrier_survival.config import DATA_DIR, RAW_DIR  # noqa: E402
from carrier_survival.labels import build_labels, summarize  # noqa: E402

INTERIM_DIR = DATA_DIR / "interim"


def main() -> None:
    for name in ("authhist", "revocation"):
        if not (RAW_DIR / f"{name}.parquet").exists():
            raise SystemExit(f"Missing {name}.parquet — run scripts/fetch_fmcsa.py first")

    authhist = pd.read_parquet(
        RAW_DIR / "authhist.parquet",
        columns=["dot_number", "original_action_desc", "disp_served_date"],
    )
    revocation = pd.read_parquet(RAW_DIR / "revocation.parquet")

    labels = build_labels(authhist, revocation)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / "exit_labels.parquet"
    labels.to_parquet(out, index=False)

    print(summarize(labels))

    dated = labels[labels["is_failure"]]
    by_year = dated.groupby(dated["exit_date"].dt.year).size()
    recent = by_year[(by_year.index >= 2015) & (by_year.index <= 2025)]
    print("\n  failures per year (usable modelling window):")
    for year, count in recent.items():
        print(f"    {year}  {count:>8,}")

    print(f"\n  -> {out}  ({out.stat().st_size / 1024 / 1024:,.1f} MB)")


if __name__ == "__main__":
    main()
