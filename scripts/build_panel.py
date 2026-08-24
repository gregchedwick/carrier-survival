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

#: Directories holding archived monthly FMCSA census pulls, one folder or file
#: per period. Supplied by environment variable rather than hardcoded: the
#: archive is a local artefact, its location differs per machine, and a personal
#: path does not belong in a public repository.
#:
#: Accepts several directories separated by the platform path separator (``;``
#: on Windows, ``:`` elsewhere), because the archive legitimately lives in more
#: than one place. Historical exports are often held wherever they were
#: originally collected — frequently shared or managed storage — while snapshots
#: written by ``refresh_census.py`` are public data belonging to this project
#: alone and have no reason to be added to it.
ENV_VAR = "CARRIER_SURVIVAL_SNAPSHOT_DIR"

#: Where ``refresh_census.py`` writes. Defaults to the first configured root.
REFRESH_ENV_VAR = "CARRIER_SURVIVAL_REFRESH_DIR"


def snapshot_roots() -> list[Path]:
    configured = os.environ.get(ENV_VAR)
    if not configured:
        raise SystemExit(
            f"{ENV_VAR} is not set.\n\n"
            "Point it at the directory holding the archived monthly FMCSA census\n"
            "pulls, then re-run:\n\n"
            f'    setx {ENV_VAR} "D:\\path\\to\\snapshots"     # Windows, persistent\n'
            f'    export {ENV_VAR}=/path/to/snapshots          # macOS / Linux\n\n'
            "Several directories may be given, separated by "
            f"{os.pathsep!r}.\n"
        )
    roots = [Path(part) for part in configured.split(os.pathsep) if part.strip()]
    missing = [r for r in roots if not r.exists()]
    if missing:
        raise SystemExit(
            f"{ENV_VAR} points at missing "
            f"{'directories' if len(missing) > 1 else 'directory'}: "
            + ", ".join(str(m) for m in missing)
        )
    return roots


def refresh_dir() -> Path:
    """Where a fresh pull is written. Explicit setting wins; else the first root."""
    configured = os.environ.get(REFRESH_ENV_VAR)
    if configured:
        target = Path(configured)
        target.mkdir(parents=True, exist_ok=True)
        return target
    return snapshot_roots()[0]


def main() -> None:
    roots = snapshot_roots()

    for root in roots:
        print(f"Reading snapshots from {root}")
    print()
    panel = build_panel(roots)
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
