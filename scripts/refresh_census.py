"""Pull the public FMCSA company census and keep the copy.

    python scripts/refresh_census.py

FMCSA overwrites this dataset in place and publishes no archive, so today's
snapshot is unrecoverable tomorrow. Point-in-time features need what was true
*then*. Keeping a dated copy of each pull is the entire mechanism by which a
daily-overwritten source becomes usable history — there is no cleverer trick
available, and the value compounds only if the pull actually runs.

Writes ``census_YYYYMMDD.parquet`` into the snapshot archive, where
``census_history.discover()`` picks it up as a ``datahub`` vintage alongside
whatever is already there.

**This is deliberately not tied to any private warehouse.** The project's claim
is that it reproduces from public sources, and a dependency on a private
workspace would quietly make that false — as well as coupling the model's
refresh to a schedule it does not control.

Cadence
-------
Monthly is enough. Fleet size and the MCS-150 filing date both move on the
biennial MCS-150 cycle (roughly 4% of carriers in any month), so a faster pull
buys resolution the source does not have. Run it on a fixed day:

    # crontab, 06:00 on the 1st
    0 6 1 * * cd /path/to/carrier-survival && python scripts/refresh_census.py

    # Windows Task Scheduler
    schtasks /create /tn "Carrier census" /sc monthly /d 1 /st 06:00 ^
      /tr "python C:\\path\\to\\carrier-survival\\scripts\\refresh_census.py"

After a refresh, rebuild downstream:

    python scripts/build_panel.py && python scripts/build_features.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_panel import snapshot_root  # noqa: E402
from carrier_survival.census_history import assert_no_proprietary_columns  # noqa: E402
from carrier_survival.config import Dataset  # noqa: E402
from carrier_survival.fmcsa import iter_pages  # noqa: E402

#: Public company census. Column names already match the canonical keys in
#: ``COLUMN_PREFERENCES``, so no translation layer is needed — the loader
#: resolves them the same way it resolves the archived CSV vintages.
#:
#: Projected server-side. The full table is 53 columns wide over ~4.5M rows, and
#: most of it is contact details this project has no business holding: phone,
#: email, officer names, D&B number. Narrowing the request is a privacy decision
#: before it is a performance one.
CENSUS_COLUMNS = [
    "dot_number",
    "status_code",
    "add_date",
    "mcs150_date",
    "mcs150_mileage",
    "mcs150_mileage_year",
    "power_units",
    "total_drivers",
    "business_org_desc",
    "carrier_operation",
    "phy_state",
    "phy_city",
    "phy_cnty",
    "phy_zip",
    "undeliv_phy",
]

CENSUS = Dataset(
    name="census_snapshot",
    dataset_id="az4n-8mr2",
    key_column="dot_number",
    columns=CENSUS_COLUMNS,
    date_columns=[],
    notes="Daily-overwritten public census, archived per pull to build history.",
)

#: A pull returning far less than this is a partial response, not a shrinking
#: industry. Writing it would plant a fake population collapse in the panel that
#: `coverage_report()` would then have to explain away.
MIN_EXPECTED_ROWS = 3_000_000


def main() -> None:
    root = snapshot_root()
    stamp = date.today().strftime("%Y%m%d")
    out = root / f"census_{stamp}.parquet"

    if out.exists():
        print(f"{out.name} already exists — nothing to do.")
        return

    print(f"Fetching {CENSUS.dataset_id} ({len(CENSUS_COLUMNS)} columns)…")
    frames = []
    rows = 0
    for page in iter_pages(CENSUS):
        frames.append(pd.DataFrame(page))
        rows += len(page)
        print(f"\r  {rows:,} rows", end="", flush=True)
    print()

    if not frames:
        raise SystemExit("No rows returned — refusing to write an empty snapshot.")

    frame = pd.concat(frames, ignore_index=True)

    # The API can hand back a column of nulls rather than omitting it, so check
    # what actually arrived instead of what was asked for.
    missing = [c for c in ("dot_number", "power_units", "mcs150_date")
               if c not in frame.columns]
    if missing:
        raise SystemExit(f"Response is missing required columns: {missing}")

    if len(frame) < MIN_EXPECTED_ROWS:
        raise SystemExit(
            f"Only {len(frame):,} rows returned, expected at least "
            f"{MIN_EXPECTED_ROWS:,}. Refusing to archive a partial pull — it would "
            "read downstream as a population collapse."
        )

    assert_no_proprietary_columns(frame)

    frame = frame.drop_duplicates("dot_number")
    frame.to_parquet(out, index=False)

    filled = frame["mcs150_date"].notna().mean() * 100
    print(f"\n  rows            : {len(frame):,}")
    print(f"  mcs150_date      : {filled:.1f}% populated")
    print(f"  -> {out}")
    print("\nNext: python scripts/build_panel.py && python scripts/build_features.py")


if __name__ == "__main__":
    main()
