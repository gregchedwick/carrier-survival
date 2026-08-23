"""Dataset definitions for the FMCSA sources this project pulls.

Every source is a public dataset on the DOT Public Data Portal (DataHub). The
older catalog.data.gov listings point at the same data but are no longer the
maintained surface, so the SODA endpoints on data.transportation.gov are what
we call.

Columns are projected server-side. The raw files carry 59-63 columns and tens of
millions of rows; naming what we need up front turns a multi-gigabyte download
into a few hundred megabytes and costs nothing in fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

SODA_BASE = "https://data.transportation.gov/resource"


@dataclass(frozen=True)
class Dataset:
    """One FMCSA source and the slice of it we care about."""

    name: str
    dataset_id: str
    columns: list[str]
    #: Column holding the carrier's USDOT number. Spelling varies by file.
    key_column: str
    #: Columns to parse into dates. Format varies by file — see parse_fmcsa_date.
    date_columns: list[str] = field(default_factory=list)
    notes: str = ""


DATASETS: dict[str, Dataset] = {
    # --- label sources -----------------------------------------------------
    "authhist": Dataset(
        name="authhist",
        dataset_id="9mw4-x3tu",
        key_column="dot_number",
        columns=[
            "dot_number",
            "docket_number",
            "original_action_desc",
            "disp_action_desc",
            "orig_served_date",
            "disp_decided_date",
            "disp_served_date",
        ],
        date_columns=["orig_served_date", "disp_decided_date", "disp_served_date"],
        notes=(
            "Authority lifecycle. disp_action_desc distinguishes REVOKED from "
            "DISCONTINUED REVOCATION — the latter outnumbers the former, so a label "
            "built on 'a revocation happened' would be majority noise. Also carries "
            "REINSTATED, which is how a temporary lapse is told apart from an exit. "
            "Contains corrupt dates (years 0189, 2044, 2517); see parse_fmcsa_date."
        ),
    ),
    "revocation": Dataset(
        name="revocation",
        dataset_id="sa6p-acbp",
        key_column="dot_number",
        columns=[
            "dot_number",
            "docket_number",
            "type_license",
            "order1_serve_date",
            "order2_type_desc",
            "order2_effective_date",
        ],
        date_columns=["order1_serve_date", "order2_effective_date"],
        notes=(
            "order2_type_desc splits INVOLUNTARY from VOLUNTARY revocation. A "
            "voluntary surrender is an orderly exit, not a failure, and belongs in "
            "the competing-risks arm rather than the positive class."
        ),
    ),
    # --- feature sources ---------------------------------------------------
    "crash": Dataset(
        name="crash",
        dataset_id="aayw-vxb3",
        key_column="dot_number",
        columns=[
            "dot_number",
            "crash_id",
            "report_date",
            "report_state",
            "fatalities",
            "injuries",
            "tow_away",
            "vehicles_in_accident",
            "hazmat_released",
            "federal_recordable",
            "state_recordable",
            "truck_bus_ind",
        ],
        date_columns=["report_date"],
        notes=(
            "Full history back to 1982 with no null report dates — the only feature "
            "source that reaches far enough back to support point-in-time "
            "reconstruction across many prediction dates."
        ),
    ),
    "inspection": Dataset(
        name="inspection",
        dataset_id="fx4q-ay7w",
        key_column="dot_number",
        columns=[
            "dot_number",
            "inspection_id",
            "insp_date",
            "report_state",
            "insp_level_id",
            "viol_total",
            "oos_total",
            "driver_viol_total",
            "driver_oos_total",
            "vehicle_viol_total",
            "vehicle_oos_total",
            "hazmat_viol_total",
            "hazmat_oos_total",
        ],
        date_columns=["insp_date"],
        notes=(
            "Only reaches back to 2023 — a rolling window, not an archive. This caps "
            "how far back inspection-derived features can go and is the reason the "
            "modelling splits into a deep-history model and a recent, richer one."
        ),
    ),
    # --- current-state only ------------------------------------------------
    "census": Dataset(
        name="census",
        dataset_id="az4n-8mr2",
        key_column="dot_number",
        columns=[],  # full width; it is a snapshot we keep for reference
        date_columns=[],
        notes=(
            "Daily snapshot with no published archive, so it cannot be used for "
            "point-in-time features. Pulled only to describe the present. Historical "
            "fleet trajectory comes from locally archived monthly census pulls "
            "instead — see census_history.py."
        ),
    ),
}

#: Datasets fetched by default. Census is excluded — it is a snapshot, large, and
#: not usable as history.
DEFAULT_FETCH = ["authhist", "revocation", "crash", "inspection"]
