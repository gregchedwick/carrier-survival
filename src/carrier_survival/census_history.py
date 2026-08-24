"""Load the local monthly carrier snapshots into a point-in-time panel.

FMCSA publishes the Company Census as a daily snapshot and keeps no public
archive, so fleet size and safety rating cannot be reconstructed for any past
date from the public API. These local monthly pulls are the only source of that
history. Every field taken from them is public FMCSA data that happened to be
captured monthly; nothing derived is used.

**The score columns are dropped at read time, deliberately.** The same files
carry proprietary risk scores from a separate commercial product. Reading them
would make this model a derivative of that product, entangle intellectual
property that is meant to stay separate, and make the project impossible for
anyone else to reproduce from public sources. The exclusion is enforced here
rather than left to discipline further down the pipeline, and
:func:`assert_no_proprietary_columns` fails loudly if one ever slips through.

Coverage is irregular — roughly 20 of the 26 months between Nov 2023 and Dec
2025, with several gaps. That is workable: discrete-time survival models tolerate
uneven observation, and deltas are computed across whatever gap actually exists
rather than assuming a fixed monthly step.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

#: Any column whose name matches this is proprietary and never read.
PROPRIETARY_PATTERN = re.compile(r"score", re.IGNORECASE)

#: Canonical name -> candidate source columns, **most preferred first**.
#:
#: Order is load-bearing. Several exports are Power BI extracts that repeat a
#: grand total on every row: "Sum of Nbr Power Unit" is column 0 of the 2025
#: files and holds 5,726,224 in all 713,276 rows, while the real per-carrier
#: "Power Units" sits at column 78. Matching on whichever column appeared first
#: silently swapped a fleet-size feature for a constant. The plain per-carrier
#: names are therefore always preferred over "Sum of ..." aggregates, and
#: :func:`_reject_constant_columns` catches any that still slip through.
#: Three source vintages use three vocabularies for the same FMCSA fields:
#: the older MCMIS CSV exports (TOT_PWR, ACT_STAT), the intermediate reporting
#: exports (Power Units, Driver Count), and the modern DataHub/Lakehouse schema
#: (POWER_UNITS, TOTAL_DRIVERS, STATUS_CODE). Resolution is by preference, not
#: by whichever appears first in the header — see the note above.
COLUMN_PREFERENCES: dict[str, tuple[str, ...]] = {
    "dot_number": ("dotnumber",),
    "power_units": ("powerunits", "totpwr", "sumofnbrpowerunit"),
    "driver_count": ("totaldrivers", "drivercount", "totdrs", "sumofdrivercount"),
    "safety_rating": ("safetyrating", "rating"),
    "status": ("statuscode", "status", "actstat"),
    "state": ("phystate", "physt", "state"),
    "mcs150_mileage_year": ("mcs150mileageyear",),
    "mcs150_date": ("mcs150date",),
    # --- available in the MCMIS and DataHub vintages only ---
    "add_date": ("adddate",),
    "city": ("phycity",),
    "zip": ("phyzip",),
    "county": ("phycnty",),
    "street": ("phystreet", "phystr"),
    #: Mail returned undeliverable — about as direct a distress signal as public
    #: data offers.
    "undeliverable_address": ("undelivphy",),
    #: Sole proprietor / LLC / corporation. These fail at very different rates.
    "business_org": ("businessorgdesc",),
    #: FMCSA's own link from a carrier to a previously revoked DOT number — the
    #: reincarnated-carrier signal, stated rather than inferred.
    "prior_revoke_dot": ("priorrevokedotnumber",),
    "mileage": ("mcs150mileage",),
    "crash_rate": ("recordablecrashrate",),
}

#: Columns that must vary between carriers. A zero-variance value in any of
#: these means an aggregate was picked up instead of a per-row measure.
MUST_VARY = ("power_units", "driver_count")

#: Categorical fields, forced to string on the way in.
#:
#: The vintages disagree about types for the same field — county arrives numeric
#: in one export and text in another, ZIP loses leading zeros when read as a
#: number. Neither is used arithmetically, and a mixed-type column survives
#: every per-file check only to fail when the concatenated panel is written.
TEXT_COLUMNS = (
    "state", "status", "safety_rating", "city", "zip", "county", "street",
    "undeliverable_address", "business_org",
)


@dataclass(frozen=True)
class Snapshot:
    """One monthly carrier snapshot on disk."""

    period: str  # YYYYMM
    path: Path
    #: Several files can cover one period. The 2024 exports are split into three
    #: parts that together make up the ~1.86M active carriers; reading only one
    #: silently models a non-random 39% of the population.
    part: int = 0

    @property
    def as_of(self) -> pd.Timestamp:
        """Month end — the snapshot describes the state as at that date."""
        return pd.Period(self.period, freq="M").to_timestamp(how="end").normalize()

    @property
    def source_kind(self) -> str:
        """Which export produced this month — each covers a different population.

        Three vintages, three populations, and the differences are large enough
        to dominate any feature built by differencing across them:

        ``mcmis``      the older Census.csv exports, ~2.17M registered entities
        ``reporting``  activity-filtered exports, ~0.72M carriers
        ``datahub``    the modern Company Census, ~4.47M rows

        A carrier present in one vintage and absent from the next has usually
        not exited; it was never in scope. Diffing presence across the
        mcmis/reporting boundary alone manufactures roughly 1.46M phantom exits,
        so the vintage is recorded here rather than left to be rediscovered.
        """
        name = self.path.name.lower()
        if (
            re.fullmatch(r"census_20\d{6}(\.parquet)?", name)
            or name.startswith("company_census_file")
        ):
            return "datahub"
        if name.startswith("census"):
            return "mcmis"
        return "reporting"


def _canonical_key(column: str) -> str:
    return re.sub(r"[^a-z0-9]", "", column.lower())


def assert_no_proprietary_columns(frame: pd.DataFrame) -> None:
    """Raise if any proprietary score column survived into the frame."""
    leaked = [c for c in frame.columns if PROPRIETARY_PATTERN.search(c)]
    if leaked:
        raise ValueError(
            "Proprietary score columns must never reach the feature store. "
            f"Found: {leaked}"
        )


#: Filename patterns tried inside a ``YYYYMM/`` directory, most preferred first.
#: Preference order rather than header order matters: several exports may cover
#: the same month and the raw census carries the most columns. Override with
#: ``CARRIER_SURVIVAL_SNAPSHOT_PATTERNS`` (comma-separated) for a differently
#: named archive.
SNAPSHOT_PATTERNS = tuple(
    pattern.strip()
    for pattern in os.environ.get(
        "CARRIER_SURVIVAL_SNAPSHOT_PATTERNS",
        "*Census*.csv,*Combined*.csv,*Scored*.csv",
    ).split(",")
    if pattern.strip()
)


def discover(root: Path | Iterable[Path]) -> list[Snapshot]:
    """Find every monthly snapshot under ``root``, newest last.

    Accepts one directory or several. The archive legitimately spans more than
    one location: historical exports usually sit wherever they were originally
    collected, while snapshots written by ``refresh_census.py`` are public data
    belonging to this project alone. Where two roots both cover a period, the
    earlier root wins — configuration order states the precedence rather than
    leaving it to whichever happened to be scanned last.

    Four layouts accumulate here, reflecting how the archive was collected over
    time rather than any design:

    * ``YYYYMM/`` directories holding a census or reporting export
    * flat scored exports carrying ``YYYYMM`` in the filename
    * ``census_YYYYMMDD/`` directories of Parquet part files, exported from a
      Delta table where each version is a full snapshot
    * flat ``Company_Census_File_YYYYMMDD.csv`` exports

    Where several *kinds* cover the same month, the first pattern to match wins
    — the raw census carries the most columns. But every file matching that
    winning pattern is kept, because an export can be split across parts.

    The distinction matters. A 2024 month directory holds three files that
    together make up ~1.86M active carriers; taking only the first models a
    non-random 39% slice, biased by fleet size (the three parts have mean fleet
    19.9, 3.2 and 97.7). A 2023 month directory holds a full ``Census.csv``
    *and* a derived extract of the same month — there, matching one pattern
    rather than everything is what stops two vintages being stacked as if they
    were one population.
    """
    roots = [root] if isinstance(root, Path) else list(root)

    found: dict[str, list[Path]] = {}
    for one in roots:
        # Whole periods, not individual files. Merging file-by-file across roots
        # would let a one-file pull in a later root attach itself to a
        # three-part export in an earlier one, producing a period that is
        # part-duplicated and part-fresh.
        for period, paths in _scan(one).items():
            found.setdefault(period, paths)

    return [
        Snapshot(period, path, part)
        for period, paths in sorted(found.items())
        for part, path in enumerate(paths)
    ]


def _scan(root: Path) -> dict[str, list[Path]]:
    """Every period found under one root, resolved independently of any other."""
    found: dict[str, list[Path]] = {}

    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if re.fullmatch(r"20\d{4}", directory.name):
            for pattern in SNAPSHOT_PATTERNS:
                matches = sorted(directory.glob(pattern))
                if matches:
                    found[directory.name] = matches
                    break
        # Parquet snapshot directories, named for the day they were captured.
        elif re.fullmatch(r"census_20\d{6}", directory.name):
            if any(directory.glob("*.parquet")):
                found.setdefault(directory.name[7:13], [directory])

    for path in sorted(root.glob("*Scored*.csv")):
        match = re.search(r"(20\d{4})", path.name)
        if match:
            found.setdefault(match.group(1), []).append(path)

    for path in sorted(root.glob("Company_Census_File_*.csv")):
        match = re.search(r"(20\d{4})\d{2}", path.name)
        if match:
            found.setdefault(match.group(1), []).append(path)

    # Snapshots written by scripts/refresh_census.py, one Parquet file per pull.
    # This is how the daily-overwritten public census becomes history: keep the
    # copies. Point-in-time features need what was true then, and FMCSA publishes
    # no archive of that.
    for path in sorted(root.glob("census_20??????.parquet")):
        found.setdefault(path.stem[7:13], []).append(path)

    return found

    return [
        Snapshot(period, path, part)
        for period, paths in sorted(found.items())
        for part, path in enumerate(paths)
    ]


#: Refuse a date column that parses for almost nobody. A vintage using an
#: unrecognised format coerces to null silently and looks like a field the
#: source simply did not populate.
MIN_DATE_PARSE_RATE = 0.10


def _parse_dates(values: pd.Series, source: str) -> pd.Series:
    """Parse a date column that arrives in a different format per vintage.

    Three formats appear across the archive: ``YYYYMMDD`` as text in the later
    CSVs, the same as an integer in the Parquet vintage, and ``M/D/YYYY`` in the
    November 2023 export. A hard-coded ``format="%Y%m%d"`` with
    ``errors="coerce"`` turned every date in that third vintage into ``NaT``
    without complaint, which read downstream as "the source did not carry this
    field" rather than "the parser did not understand it".

    So: try the dominant format, then fall back to a general parse for whatever
    is left, and raise if the result is still mostly empty.
    """
    text = values.astype("string").str.strip()
    populated = text.notna() & (text != "")
    if not populated.any():
        return pd.to_datetime(pd.Series([pd.NaT] * len(values), index=values.index))

    parsed = pd.to_datetime(text.str.slice(0, 8), format="%Y%m%d", errors="coerce")

    missed = populated & parsed.isna()
    if missed.any():
        parsed.loc[missed] = pd.to_datetime(text[missed], errors="coerce", format="mixed")

    rate = parsed.notna().sum() / populated.sum()
    if rate < MIN_DATE_PARSE_RATE:
        example = text[populated].iloc[0]
        raise ValueError(
            f"{source}: only {rate:.1%} of populated dates parsed. "
            f"Unrecognised format (first value: {example!r}). "
            "Refusing to emit a column of nulls that looks like missing data."
        )
    return parsed


def _reject_constant_columns(frame: pd.DataFrame, source: str) -> None:
    """Raise if a per-carrier measure has no variance across the file.

    A grand total repeated on every row is indistinguishable from a real column
    by name alone, and produces a feature that is silently useless rather than
    obviously broken. Checking variance catches it whatever it is called.
    """
    for column in MUST_VARY:
        if column not in frame.columns:
            continue
        values = frame[column].dropna()
        if len(values) > 1 and values.nunique() == 1:
            raise ValueError(
                f"{source}: '{column}' is constant at {values.iloc[0]!r} across "
                f"{len(values):,} rows — an aggregate was read instead of a "
                "per-carrier measure. Check COLUMN_PREFERENCES ordering."
            )


def _is_parquet(path: Path) -> bool:
    """A snapshot may be a Parquet directory, a single Parquet file, or a CSV.

    The directory form comes from a Delta export (many part files); the single
    file from :mod:`scripts.refresh_census`, which writes one snapshot per pull.
    """
    return path.is_dir() or path.suffix.lower() == ".parquet"


def _header_columns(path: Path) -> list[str]:
    """Column names only, without reading the body."""
    if path.is_dir():
        part = next(iter(sorted(path.glob("*.parquet"))))
        return list(pq.ParquetFile(part).schema.names)
    if path.suffix.lower() == ".parquet":
        return list(pq.ParquetFile(path).schema.names)
    return list(pd.read_csv(path, nrows=0, encoding="latin-1").columns)


def _read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read the named columns from a CSV, a Parquet file, or a Parquet directory."""
    if _is_parquet(path):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, encoding="latin-1", low_memory=False)


def load_snapshot(snapshot: Snapshot) -> pd.DataFrame:
    """Read one snapshot, keeping only the public census fields."""
    # Index the header by canonical key, ignoring anything proprietary.
    available: dict[str, str] = {}
    for column in _header_columns(snapshot.path):
        if PROPRIETARY_PATTERN.search(column):
            continue  # never even read into memory
        available.setdefault(_canonical_key(column), column)

    # Resolve by preference order rather than by header order.
    keep: dict[str, str] = {}
    for canonical, candidates in COLUMN_PREFERENCES.items():
        for candidate in candidates:
            if candidate in available:
                keep[available[candidate]] = canonical
                break

    if "dot_number" not in keep.values():
        raise ValueError(f"{snapshot.path.name} has no DOT number column")

    frame = _read_columns(snapshot.path, list(keep)).rename(columns=keep)

    frame["dot_number"] = pd.to_numeric(frame["dot_number"], errors="coerce").astype("Int64")
    frame = frame[frame["dot_number"].notna() & (frame["dot_number"] > 0)]

    for column in ("power_units", "driver_count", "mcs150_mileage_year",
                   "mileage", "crash_rate", "prior_revoke_dot"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in TEXT_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()

    for column in ("add_date", "mcs150_date"):
        if column in frame.columns:
            frame[column] = _parse_dates(frame[column], f"{snapshot.path.name}:{column}")

    frame["as_of"] = snapshot.as_of
    frame["period"] = snapshot.period
    frame["source_kind"] = snapshot.source_kind

    # One row per carrier per period; later duplicates are re-filings.
    frame = frame.drop_duplicates(subset=["dot_number"], keep="last")

    assert_no_proprietary_columns(frame)
    _reject_constant_columns(frame, snapshot.path.name)
    return frame


def build_panel(root: Path | Iterable[Path], verbose: bool = True) -> pd.DataFrame:
    """Stack every snapshot into a carrier-period panel. See :func:`discover`."""
    snapshots = discover(root)
    if not snapshots:
        raise SystemExit(f"No monthly snapshots found under {root}")

    frames = []
    for snapshot in snapshots:
        frame = load_snapshot(snapshot)
        frames.append(frame)
        if verbose:
            print(f"  {snapshot.period}  {len(frame):>9,} carriers  <- {snapshot.path.name}")

    panel = pd.concat(frames, ignore_index=True).sort_values(["dot_number", "as_of"])

    # A multi-part export can list the same carrier twice at the boundary
    # between parts (a few hundred rows in practice). One row per carrier per
    # period is what everything downstream assumes.
    before = len(panel)
    panel = panel.drop_duplicates(["dot_number", "period"], keep="first")
    if verbose and before != len(panel):
        print(f"  dropped {before - len(panel):,} duplicate carrier-periods across parts")

    assert_no_proprietary_columns(panel)
    return panel


#: A period whose carrier count falls this far below the median for its own
#: source kind is treated as a truncated export rather than real attrition.
TRUNCATION_THRESHOLD = 0.85


def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-period coverage, flagging periods that should not be diffed.

    Two distinct problems show up here and neither raises an error on its own:

    ``population_break`` marks the first period after the source kind changes.
    Carrier counts differ ~3x across that boundary, so a presence diff over it
    is meaningless.

    ``suspect_truncation`` marks a period well below the median for its own
    source kind — observed in 202501, which lost 206k carriers of which 96%
    returned the following period. Real exits do not come back.
    """
    counts = (
        panel.groupby(["period", "source_kind"], as_index=False)["dot_number"]
        .nunique()
        .rename(columns={"dot_number": "carriers"})
        .sort_values("period")
        .reset_index(drop=True)
    )

    medians = counts.groupby("source_kind")["carriers"].transform("median")
    counts["pct_of_median"] = counts["carriers"] / medians
    counts["suspect_truncation"] = counts["pct_of_median"] < TRUNCATION_THRESHOLD
    counts["population_break"] = counts["source_kind"] != counts["source_kind"].shift()
    counts.loc[0, "population_break"] = False

    counts["stale_export"] = _stale_periods(panel, counts)

    counts["safe_to_diff"] = ~(
        counts["population_break"]
        | counts["suspect_truncation"]
        | counts["suspect_truncation"].shift(fill_value=False)
        | counts["stale_export"]
    )
    return counts


#: A period is treated as stale if fewer than this share of overlapping carriers
#: show any change in fleet size against the previous period. Real month-to-month
#: churn runs 2-20%; a genuinely refreshed export never sits at zero.
STALE_CHANGE_THRESHOLD = 0.001


def _stale_periods(panel: pd.DataFrame, counts: pd.DataFrame) -> pd.Series:
    """Flag periods whose measures are unchanged from the previous period.

    Some exports are new files containing old data — the timestamp advances but
    the fleet numbers are identical to the month before. Differencing across one
    produces a delta of exactly zero for every carrier, which is indistinguishable
    from a stable fleet and quietly destroys the trajectory features rather than
    erroring.

    Detected by measuring how many carriers changed at all, which needs no
    knowledge of why a particular export went stale.
    """
    if "power_units" not in panel.columns:
        return pd.Series(False, index=counts.index)

    flags = []
    previous: pd.Series | None = None
    previous_kind: str | None = None

    for row in counts.itertuples():
        current = (
            panel.loc[panel["period"] == row.period, ["dot_number", "power_units"]]
            .dropna(subset=["dot_number"])
            .drop_duplicates("dot_number")
            .set_index("dot_number")["power_units"]
        )
        stale = False
        if previous is not None and previous_kind == row.source_kind:
            overlap = current.index.intersection(previous.index)
            # Compare only where both sides have a value. NaN != NaN is True in
            # pandas, so carriers null in both exports otherwise count as
            # "changed" — and at a 5% null rate that alone clears the threshold,
            # reporting a completely frozen column as fresh.
            pair = pd.concat(
                [current.loc[overlap].rename("now"), previous.loc[overlap].rename("before")],
                axis=1,
            ).dropna()
            if len(pair) > 1000:
                changed = (pair["now"] != pair["before"]).mean()
                stale = changed < STALE_CHANGE_THRESHOLD
        flags.append(stale)
        previous, previous_kind = current, row.source_kind

    return pd.Series(flags, index=counts.index)
