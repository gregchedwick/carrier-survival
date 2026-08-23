"""Extract FMCSA datasets from the DOT Public Data Portal into local Parquet.

The public API is Socrata's SODA endpoint. Two details matter more than they
look:

**Paging must be ordered.** SODA's ``$offset`` walks an unordered result set, so
without an explicit ``$order`` the same row can arrive twice across pages while
another never arrives at all. Ordering by ``:id`` — Socrata's stable internal row
identifier — makes the walk deterministic. Nothing here would raise an error if
it were omitted; the row count would simply be right and the contents subtly
wrong.

**Keys and dates are inconsistent across files.** The USDOT number is
``dot_number`` in some datasets and ``usdot_number`` in others, zero-padded in
one and bare in the next, and every date column is stored as free text in at
least two different formats. Both are normalised on the way in so downstream
code can join and filter without special cases.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from .config import RAW_DIR, SODA_BASE, Dataset

PAGE_SIZE = 50_000
MAX_RETRIES = 4
USER_AGENT = "carrier-survival/0.1 (portfolio research project)"

#: Dates arrive as MM/DD/YYYY in some files and YYYYMMDD in others. AuthHist also
#: contains transcription errors — observed years include 0189, 2044 and 2517 —
#: which are dropped rather than clamped, since a wrong date is worse than a
#: missing one when the whole design depends on knowing what was true when.
_MIN_YEAR = 1970
_MAX_YEAR = date.today().year + 1


def parse_fmcsa_date(raw: object) -> date | None:
    """Parse the several date shapes FMCSA uses. Returns None for junk."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    parsed: date | None = None
    if "/" in text:
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                parsed = datetime.strptime(text, fmt).date()
                break
            except ValueError:
                continue
    elif len(text) >= 8 and text[:8].isdigit():
        try:
            parsed = datetime.strptime(text[:8], "%Y%m%d").date()
        except ValueError:
            parsed = None
    else:
        try:
            parsed = datetime.fromisoformat(text[:10]).date()
        except ValueError:
            parsed = None

    if parsed is None or not (_MIN_YEAR <= parsed.year <= _MAX_YEAR):
        return None
    return parsed


def parse_dot_number(raw: object) -> int | None:
    """Normalise a USDOT number to an int, dropping padding and placeholders.

    AuthHist stores these zero-padded ("00000000") while the insurance files
    store them bare ("293258"). Joining the text forms silently matches nothing.
    Zero is a placeholder for "no carrier", not a real DOT number.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or not text.isdigit():
        return None
    value = int(text)
    return value or None


def _request(url: str) -> list[dict]:
    """GET with retry. Socrata rate-limits anonymous callers under load."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {url}") from last_error


def count_rows(dataset: Dataset) -> int:
    params = urllib.parse.urlencode({"$select": "count(*) as n"})
    rows = _request(f"{SODA_BASE}/{dataset.dataset_id}.json?{params}")
    return int(rows[0]["n"])


def iter_pages(dataset: Dataset) -> Iterator[list[dict]]:
    """Yield pages of raw rows, ordered so offset paging is stable."""
    offset = 0
    while True:
        query = {
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$order": ":id",  # see module docstring — not optional
        }
        if dataset.columns:
            query["$select"] = ",".join(dataset.columns)

        page = _request(f"{SODA_BASE}/{dataset.dataset_id}.json?{urllib.parse.urlencode(query)}")
        if not page:
            return
        yield page
        if len(page) < PAGE_SIZE:
            return
        offset += PAGE_SIZE


def normalize(frame: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """Apply key and date normalisation in place of the raw text columns."""
    if dataset.key_column in frame.columns:
        frame["dot_number"] = (
            frame[dataset.key_column].map(parse_dot_number).astype("Int64")
        )

    for column in dataset.date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(
                frame[column].map(parse_fmcsa_date), errors="coerce"
            )

    # Numeric-looking measures arrive as text; make them numbers once, here.
    for column in frame.columns:
        if column == "dot_number" or column in dataset.date_columns:
            continue
        if any(k in column for k in ("_total", "fatalities", "injuries", "_id", "vehicles_in")):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def fetch(dataset: Dataset, out_dir: Path = RAW_DIR, verbose: bool = True) -> Path:
    """Download one dataset to Parquet and write a manifest beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = count_rows(dataset)
    if verbose:
        print(f"  {dataset.name}: {expected:,} rows expected")

    chunks: list[pd.DataFrame] = []
    fetched = 0
    for page in iter_pages(dataset):
        chunks.append(pd.DataFrame(page))
        fetched += len(page)
        if verbose:
            pct = fetched / expected * 100 if expected else 0
            print(f"\r    {fetched:>10,} / {expected:,}  ({pct:5.1f}%)", end="", flush=True)
    if verbose:
        print()

    frame = normalize(pd.concat(chunks, ignore_index=True), dataset)
    path = out_dir / f"{dataset.name}.parquet"
    frame.to_parquet(path, index=False)

    manifest = {
        "dataset": dataset.name,
        "dataset_id": dataset.dataset_id,
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows_reported_by_api": expected,
        "rows_written": int(len(frame)),
        "columns": list(frame.columns),
        "null_dot_number": int(frame["dot_number"].isna().sum())
        if "dot_number" in frame
        else None,
        "distinct_dot_number": int(frame["dot_number"].nunique())
        if "dot_number" in frame
        else None,
        "notes": dataset.notes,
    }
    (out_dir / f"{dataset.name}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return path
