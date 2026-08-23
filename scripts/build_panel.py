"""Build the carrier-period panel from the local monthly snapshots.

    python scripts/build_panel.py

Writes data/raw/carrier_panel.parquet plus a coverage report. Proprietary score
columns are excluded at read time and the result is asserted clean before it is
written.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carrier_survival.census_history import (  # noqa: E402
    assert_no_proprietary_columns,
    build_panel,
    coverage_report,
)
from carrier_survival.config import RAW_DIR  # noqa: E402

#: Directory holding the archived monthly FMCSA census pulls, one folder or file
#: per period. Supplied by environment variable rather than hardcoded: the
#: archive is a local artefact, its location differs per machine, and a personal
#: path does not belong in a public repository.
ENV_VAR = "CARRIER_SURVIVAL_SNAPSHOT_DIR"


def snapshot_root() -> Path:
    configured = os.environ.get(ENV_VAR)
    if not configured:
        raise SystemExit(
            f"{ENV_VAR} is not set.\n\n"
            "Point it at the directory holding the archived monthly FMCSA census\n"
            "pulls, then re-run:\n\n"
            f'    setx {ENV_VAR} "D:\\path\\to\\snapshots"     # Windows, persistent\n'
            f'    export {ENV_VAR}=/path/to/snapshots          # macOS / Linux\n'
        )
    root = Path(configured)
    if not root.exists():
        raise SystemExit(f"{ENV_VAR} points at a missing directory: {root}")
    return root


def main() -> None:
    SNAPSHOT_ROOT = snapshot_root()

    print(f"Reading snapshots from {SNAPSHOT_ROOT}\n")
    panel = build_panel(SNAPSHOT_ROOT)
    assert_no_proprietary_columns(panel)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "carrier_panel.parquet"
    panel.to_parquet(out, index=False)

    print(f"\n  periods           : {panel['period'].nunique()}")
    print(f"  rows              : {len(panel):,}")
    print(f"  distinct carriers : {panel['dot_number'].nunique():,}")
    print(f"  columns           : {', '.join(panel.columns)}")
    print(f"  -> {out}  ({out.stat().st_size / 1024 / 1024:,.1f} MB)")

    report = coverage_report(panel)
    report.to_csv(RAW_DIR / "panel_coverage.csv", index=False)

    print("\ncoverage — safe_to_diff marks periods usable for period-over-period features:\n")
    print(report.to_string(index=False))

    unusable = report.loc[~report["safe_to_diff"], "period"].tolist()
    if unusable:
        print(f"\n  excluded from differencing: {', '.join(unusable)}")


if __name__ == "__main__":
    main()
