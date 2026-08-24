"""Tests for snapshot discovery.

Discovery has to satisfy two requirements that pull against each other: pick up
every part of a split export, while refusing to stack two different vintages of
the same month. Getting the first wrong silently models a fraction of the
population; getting the second wrong invents a population break.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carrier_survival.census_history import discover  # noqa: E402


def touch(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("Dot Number,Power Units\n1,5\n")


def test_every_part_of_a_split_export_is_found(tmp_path: Path):
    """A month split across three files must yield all three.

    Regression test. Discovery took the first match and stopped, so the 2024
    exports contributed 717K of their 1.86M carriers — and because the parts
    differ in mean fleet size (19.9, 3.2, 97.7), the 39% that survived was not
    a random sample.
    """
    touch(
        tmp_path / "202407",
        "Data File Scored.csv",
        "Data File Not Scored 1.csv",
        "Data File Not Scored 2.csv",
    )

    found = discover(tmp_path)

    assert len(found) == 3
    assert {s.period for s in found} == {"202407"}
    assert {s.part for s in found} == {0, 1, 2}
    assert {s.source_kind for s in found} == {"reporting"}


def test_two_vintages_of_one_month_are_not_stacked(tmp_path: Path):
    """A full census and a derived extract of the same month are not parts.

    Reading both would double-count the month and mix a 2.17M-carrier registry
    with a 0.7M activity-filtered slice.
    """
    touch(tmp_path / "202312", "Census.csv", "Combined.csv", "Extract.csv")

    found = discover(tmp_path)

    assert len(found) == 1
    assert found[0].path.name == "Census.csv"
    assert found[0].source_kind == "mcmis"


def test_single_file_months_still_resolve(tmp_path: Path):
    """The flat exports and parquet directories keep working."""
    touch(tmp_path, "Scored - 202412.csv")
    touch(tmp_path, "Company_Census_File_20250609.csv")
    (tmp_path / "census_20260503").mkdir()
    (tmp_path / "census_20260503" / "part-0.parquet").write_bytes(b"")

    found = {s.period: s for s in discover(tmp_path)}

    assert found["202412"].source_kind == "reporting"
    assert found["202506"].source_kind == "datahub"
    assert found["202605"].source_kind == "datahub"


def test_a_refreshed_census_parquet_is_found_and_read(tmp_path: Path):
    """Snapshots written by refresh_census.py must join the panel.

    A single Parquet file, not a directory of part files — the helpers branched
    on ``is_dir()``, so a flat ``.parquet`` fell through to the CSV reader.
    """
    import pandas as pd

    from carrier_survival.census_history import _header_columns, _read_columns, load_snapshot

    pd.DataFrame({
        "dot_number": [1, 2, 3],
        "power_units": [4, 9, 2],
        "total_drivers": [5, 11, 3],
        "status_code": ["A", "A", "I"],
        "mcs150_date": ["20231020 2057", "20220218 0000", "20210105 0000"],
        "phy_state": ["NV", "CA", "TX"],
    }).to_parquet(tmp_path / "census_20260823.parquet", index=False)

    found = discover(tmp_path)

    assert len(found) == 1
    snapshot = found[0]
    assert snapshot.period == "202608"
    assert snapshot.source_kind == "datahub"

    assert "mcs150_date" in _header_columns(snapshot.path)
    assert len(_read_columns(snapshot.path, ["dot_number", "power_units"])) == 3

    frame = load_snapshot(snapshot)
    assert list(frame["power_units"]) == [4, 9, 2]
    # The API appends a time to the date; it must still parse.
    assert frame["mcs150_date"].notna().all()
    assert frame["mcs150_date"].iloc[0] == pd.Timestamp("2023-10-20")


def test_a_refreshed_snapshot_sits_alongside_the_archived_vintages(tmp_path: Path):
    """A new pull must not displace or duplicate what is already there."""
    import pandas as pd

    touch(tmp_path / "202407", "Data File Scored.csv")
    pd.DataFrame({"dot_number": [1], "power_units": [4]}).to_parquet(
        tmp_path / "census_20260823.parquet", index=False
    )

    periods = {s.period for s in discover(tmp_path)}

    assert periods == {"202407", "202608"}
