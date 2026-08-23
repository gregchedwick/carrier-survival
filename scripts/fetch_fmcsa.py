"""Download the FMCSA source datasets to data/raw/.

    python scripts/fetch_fmcsa.py                 # all defaults
    python scripts/fetch_fmcsa.py crash           # one dataset
    python scripts/fetch_fmcsa.py --list          # show what is available

Re-running overwrites in place. The public files are refreshed daily, so a
re-fetch is how you pick up new events.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carrier_survival.config import DATASETS, DEFAULT_FETCH, RAW_DIR  # noqa: E402
from carrier_survival.fmcsa import fetch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=None,
                        help=f"names to fetch (default: {' '.join(DEFAULT_FETCH)})")
    parser.add_argument("--list", action="store_true", help="list datasets and exit")
    args = parser.parse_args()

    if args.list:
        for name, ds in DATASETS.items():
            marker = "*" if name in DEFAULT_FETCH else " "
            print(f" {marker} {name:12s} {ds.dataset_id}  {len(ds.columns) or 'all'} cols")
        print("\n * = fetched by default")
        return

    names = args.datasets or DEFAULT_FETCH
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}")

    print(f"Writing to {RAW_DIR}\n")
    for name in names:
        started = time.time()
        path = fetch(DATASETS[name])
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"    -> {path.name}  {size_mb:,.1f} MB  in {time.time() - started:,.0f}s\n")


if __name__ == "__main__":
    main()
