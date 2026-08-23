"""Point-in-time feature construction.

Every feature answers the same question: *what was knowable about this carrier on
this date?* Nothing else. A model built on features that quietly include future
information scores beautifully and is worthless, and the failure is invisible —
there is no error, just an implausibly good number.

So the rules here are mechanical rather than a matter of care:

* every source is filtered to ``date <= prediction_date`` before any aggregation
* the fleet snapshot is the most recent period **at or before** the prediction
  date, never the nearest one
* period-over-period fleet deltas skip the periods flagged unusable by
  ``census_history.coverage_report`` (population breaks, truncated exports)
* :func:`leakage_check` refits at a later prediction date and asserts the signal
  moves the way information theory says it must

The prediction window is bounded by the label horizon, not by feature coverage.
Outcomes are measured over months 6-18, and revocation data runs to roughly
2026-05, so the last usable prediction date is about 2024-11 — extra feature
history beyond that helps scoring, not training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Outcome window, in months after the prediction date. The six-month blackout
#: keeps the model out of the mechanical insurance-lapse cascade: a lapse
#: triggers revocation roughly a month later, so predicting the near term
#: mostly rediscovers FMCSA's published procedure rather than carrier distress.
HORIZON_START_MONTHS = 6
HORIZON_END_MONTHS = 18

#: Trailing windows for event-rate features, in months.
LOOKBACK_MONTHS = (6, 12, 24)


@dataclass(frozen=True)
class Spine:
    """The (carrier, prediction_date) pairs a model is trained on."""

    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)


#: Which panel vintage defines the population at risk.
#:
#: The two vintages are not the same population: census exports list every
#: registered entity (~2.17M, most of them dormant), while the later exports are
#: filtered to carriers with actual safety activity (~0.72M). Mixing them makes
#: the failure rate jump from ~2% to ~6% purely because the denominator changed,
#: and pools two populations into one model.
#:
#: Restricting to the activity-filtered vintage costs prediction dates but buys a
#: consistent risk set, which matters more. It also matches the commercial
#: question: a broker is choosing among carriers that actually operate.
COHORT_SOURCE_KIND = "reporting"


def prediction_dates(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    freq: str = "ME",
    source_kind: str | None = COHORT_SOURCE_KIND,
) -> list[pd.Timestamp]:
    """Month ends where both features and outcomes are observable.

    The upper bound is the binding constraint: a prediction date is only usable
    if the full outcome window has already elapsed within the label data. The
    lower bound is where the chosen cohort vintage begins.

    Monthly rather than quarterly, since restricting to one vintage would
    otherwise leave too few dates. Outcome windows overlap between adjacent
    dates, which is expected in a discrete-time survival panel — it makes
    carrier-level grouping essential when splitting for validation.
    """
    eligible = panel if source_kind is None else panel[panel["source_kind"] == source_kind]
    first_feature_date = eligible["as_of"].min()
    last_label_date = labels.loc[labels["datable"], "exit_date"].max()
    last_usable = last_label_date - pd.DateOffset(months=HORIZON_END_MONTHS)
    return list(pd.date_range(first_feature_date, last_usable, freq=freq))


def build_spine(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    dates: list[pd.Timestamp],
    source_kind: str | None = COHORT_SOURCE_KIND,
) -> Spine:
    """One row per carrier observed at each prediction date, with its outcome.

    A carrier enters the spine for date *P* if it appears in the most recent
    panel period at or before *P*. Carriers that had already exited by *P* are
    dropped — they are no longer at risk and would otherwise be counted as
    surviving.
    """
    exits = labels[labels["is_failure"] & labels["exit_date"].notna()]
    first_exit = exits.groupby("dot_number")["exit_date"].min()

    # Carriers whose exit could not be dated cannot be placed on the timeline;
    # leaving them in would label a departed carrier as a survivor.
    excluded = set(labels.loc[labels["exclude_from_training"], "dot_number"])

    cohort_panel = (
        panel if source_kind is None else panel[panel["source_kind"] == source_kind]
    )

    rows = []
    for date in dates:
        eligible = cohort_panel[cohort_panel["as_of"] <= date]
        if eligible.empty:
            continue
        latest_period = eligible["as_of"].max()
        current = eligible[eligible["as_of"] == latest_period]

        # An inactive carrier is not at risk — it has already stopped. Leaving
        # them in the risk set counts a completed exit as a survivor and dilutes
        # the base rate. Status is missing in some vintages; treat that as
        # active rather than dropping the row.
        if "status" in current.columns:
            status = current["status"].astype("string").str.strip().str.upper()
            current = current[status.isna() | status.eq("A")]

        cohort = current[["dot_number"]].copy()

        cohort = cohort[~cohort["dot_number"].isin(excluded)]
        cohort["prediction_date"] = date
        cohort["observed_as_of"] = latest_period

        # Force the datetime dtype: mapping against an empty or all-missing
        # series yields floats, and comparing those to a Timestamp raises a
        # TypeError several steps later rather than here.
        exit_date = pd.to_datetime(cohort["dot_number"].map(first_exit), errors="coerce")
        cohort["exit_date"] = exit_date

        # Already gone before the prediction date: not at risk.
        cohort = cohort[exit_date.isna() | (exit_date > date)]

        window_open = date + pd.DateOffset(months=HORIZON_START_MONTHS)
        window_close = date + pd.DateOffset(months=HORIZON_END_MONTHS)
        cohort["failed"] = (
            cohort["exit_date"].notna()
            & (cohort["exit_date"] > window_open)
            & (cohort["exit_date"] <= window_close)
        )
        rows.append(cohort)

    frame = pd.concat(rows, ignore_index=True)
    return Spine(frame)


def _window_counts(
    events: pd.DataFrame,
    date_col: str,
    prediction_date: pd.Timestamp,
    prefix: str,
    value_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Per-carrier event counts over trailing windows ending at the cut-off."""
    past = events[events[date_col] <= prediction_date]
    if past.empty:
        return pd.DataFrame(columns=["dot_number"])

    frames = []
    for months in LOOKBACK_MONTHS:
        window_start = prediction_date - pd.DateOffset(months=months)
        recent = past[past[date_col] > window_start]
        agg = recent.groupby("dot_number").agg(
            **{f"{prefix}_{months}m": ("dot_number", "size")},
            **{
                f"{prefix}_{col}_{months}m": (col, "sum")
                for col in value_cols
                if col in recent.columns
            },
        )
        frames.append(agg)

    # Time since the most recent event, and lifetime count.
    lifetime = past.groupby("dot_number").agg(
        **{
            f"{prefix}_lifetime": ("dot_number", "size"),
            f"{prefix}_last_date": (date_col, "max"),
        }
    )
    lifetime[f"{prefix}_days_since_last"] = (
        prediction_date - lifetime[f"{prefix}_last_date"]
    ).dt.days
    lifetime = lifetime.drop(columns=[f"{prefix}_last_date"])
    frames.append(lifetime)

    return pd.concat(frames, axis=1).reset_index()


def as_numeric_flag(series: pd.Series) -> pd.Series:
    """Coerce an FMCSA flag column to 0/1.

    These arrive inconsistently: ``tow_away`` is ``Y``/``N`` text while
    ``fatalities`` is already numeric.

    The test is "is it numeric?", not "is it object?". Pandas 3 gives text
    columns dtype ``str`` rather than ``object``, so an object check silently
    routes ``Y``/``N`` into ``to_numeric``, which produces NaN, which
    ``fillna(0)`` turns into a column of zeros. The feature then survives every
    null check while carrying no information at all.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0)
    return (
        series.astype("string").str.strip().str.upper().isin({"Y", "YES", "1", "TRUE"})
    ).astype("int8")


def crash_features(crash: pd.DataFrame, prediction_date: pd.Timestamp) -> pd.DataFrame:
    """Crash history as at the prediction date."""
    usable = crash[crash["dot_number"].notna() & crash["report_date"].notna()].copy()
    for column in ("fatalities", "injuries", "tow_away"):
        if column in usable.columns:
            usable[column] = as_numeric_flag(usable[column])

    return _window_counts(
        usable,
        "report_date",
        prediction_date,
        prefix="crash",
        value_cols=("fatalities", "injuries", "tow_away"),
    )


def authority_features(
    authhist: pd.DataFrame, prediction_date: pd.Timestamp
) -> pd.DataFrame:
    """Authority lifecycle as at the prediction date.

    Prior revocations and reinstatements are the interesting part: a carrier
    that has been revoked and come back before is in a materially different
    position from one that never has, and neither is visible in a current-state
    snapshot.
    """
    usable = authhist[
        authhist["dot_number"].notna() & authhist["disp_served_date"].notna()
    ]
    past = usable[usable["disp_served_date"] <= prediction_date]
    if past.empty:
        return pd.DataFrame(columns=["dot_number"])

    grants = past[past["original_action_desc"].eq("GRANTED")]
    revokes = past[past["disp_action_desc"].eq("REVOKED")]
    reinstates = past[past["original_action_desc"].eq("REINSTATED")]
    discontinued = past[past["disp_action_desc"].eq("DISCONTINUED REVOCATION")]

    features = pd.DataFrame({"dot_number": past["dot_number"].unique()})

    first_grant = grants.groupby("dot_number")["disp_served_date"].min()
    features["authority_age_days"] = (
        prediction_date - features["dot_number"].map(first_grant)
    ).dt.days

    for name, subset in (
        ("prior_revocations", revokes),
        ("prior_reinstatements", reinstates),
        ("prior_discontinued", discontinued),
        ("authority_actions", past),
    ):
        features[name] = features["dot_number"].map(
            subset.groupby("dot_number").size()
        ).fillna(0).astype(int)

    features["distinct_dockets"] = features["dot_number"].map(
        past.groupby("dot_number")["docket_number"].nunique()
    ).fillna(0).astype(int)

    last_action = past.groupby("dot_number")["disp_served_date"].max()
    features["days_since_authority_action"] = (
        prediction_date - features["dot_number"].map(last_action)
    ).dt.days

    return features


def fleet_features(
    panel: pd.DataFrame,
    prediction_date: pd.Timestamp,
    diffable_periods: set[str] | None = None,
    source_kind: str | None = COHORT_SOURCE_KIND,
) -> pd.DataFrame:
    """Fleet level and trajectory as at the prediction date.

    Level comes from the most recent period at or before the cut-off. Trajectory
    compares that against the closest earlier period that is safe to difference
    against — periods spanning a population break or a truncated export are
    skipped, since the apparent change there is an artefact of the export rather
    than the carrier.
    """
    history = panel[panel["as_of"] <= prediction_date]

    # Compare like with like. The panel interleaves vintages covering 0.7M, 2.2M
    # and 4.5M carriers; a delta taken across that boundary measures which file
    # the carrier appeared in, not how its fleet changed.
    if source_kind is not None:
        history = history[history["source_kind"] == source_kind]

    if history.empty:
        return pd.DataFrame(columns=["dot_number"])

    # The level comes from the latest period regardless of whether it is safe to
    # difference; only the comparison period has to be. Filtering both together
    # discards the current fleet size whenever the newest period happens to be
    # flagged, which silently emptied the trajectory features entirely.
    periods = sorted(history["as_of"].unique())
    current = history[history["as_of"] == periods[-1]]

    if diffable_periods is not None:
        comparable = history[history["period"].isin(diffable_periods)]
        periods = sorted(comparable["as_of"].unique())
        history = comparable if not comparable.empty else history

    wanted = [
        "dot_number", "power_units", "driver_count", "safety_rating", "status",
        "state", "mileage",
    ]
    features = current[[c for c in wanted if c in current.columns]].copy()
    features = features.rename(
        columns={"power_units": "power_units_now", "driver_count": "drivers_now"}
    )


    for label, offset in (("3m", 1), ("12m", 4)):
        earlier = [p for p in periods if p < current["as_of"].iloc[0]]
        if len(earlier) < offset:
            features[f"power_units_chg_{label}"] = np.nan
            continue
        prior = history[history["as_of"] == earlier[-offset]]
        prior_units = prior.set_index("dot_number")["power_units"]
        before = features["dot_number"].map(prior_units)
        features[f"power_units_chg_{label}"] = (
            features["power_units_now"] - before
        ) / before.replace(0, np.nan)

    return features


#: Attributes carried by the census vintages but not the reporting exports.
#: All are effectively static: a registration date never changes, a prior
#: revoked DOT number and a business-organisation type almost never do.
STATIC_ATTRIBUTES = (
    "add_date",
    "business_org",
    "prior_revoke_dot",
    "undeliverable_address",
)


def static_features(
    panel: pd.DataFrame, prediction_date: pd.Timestamp
) -> pd.DataFrame:
    """Slow-moving attributes, taken from the latest vintage that carries them.

    These fields are absent from the reporting exports that define the cohort,
    so restricting the lookup to that vintage yields a column of nulls. Reading
    them from *any* snapshot at or before the prediction date is still
    point-in-time correct — a February census is knowable in April — and for a
    registration date, staleness is irrelevant because the value cannot change.

    The one that does drift is the undeliverable-address flag, which is read
    as-of rather than as-now for the same reason.
    """
    have = [c for c in STATIC_ATTRIBUTES if c in panel.columns]
    if not have:
        return pd.DataFrame(columns=["dot_number"])

    history = panel.loc[panel["as_of"] <= prediction_date, ["dot_number", "as_of", *have]]
    history = history[history[have].notna().any(axis=1)]
    if history.empty:
        return pd.DataFrame(columns=["dot_number"])

    # groupby.last() skips nulls per column, so each attribute takes its most
    # recent observed value rather than whatever the final row happened to hold.
    latest = (
        history.sort_values("as_of").groupby("dot_number", as_index=False)[have].last()
    )

    if "add_date" in latest.columns:
        latest["carrier_age_days"] = (
            prediction_date - pd.to_datetime(latest["add_date"], errors="coerce")
        ).dt.days
        latest = latest.drop(columns=["add_date"])

    if "prior_revoke_dot" in latest.columns:
        latest["has_prior_revoked_dot"] = (
            pd.to_numeric(latest["prior_revoke_dot"], errors="coerce").fillna(0) > 0
        ).astype("int8")
        latest = latest.drop(columns=["prior_revoke_dot"])

    if "undeliverable_address" in latest.columns:
        latest["undeliverable_address"] = as_numeric_flag(latest["undeliverable_address"])

    return latest


def assemble(
    spine: Spine,
    crash: pd.DataFrame,
    authhist: pd.DataFrame,
    panel: pd.DataFrame,
    diffable_periods: set[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Join every feature family onto the spine, one prediction date at a time."""
    out = []
    for date, group in spine.frame.groupby("prediction_date"):
        block = group.copy()
        for builder in (
            lambda: crash_features(crash, date),
            lambda: authority_features(authhist, date),
            lambda: fleet_features(panel, date, diffable_periods),
            lambda: static_features(panel, date),
        ):
            feats = builder()
            if not feats.empty:
                block = block.merge(feats, on="dot_number", how="left")

        # A carrier with no crash record has zero crashes, not unknown crashes.
        for column in block.columns:
            if column.startswith(("crash_", "prior_", "authority_actions", "distinct_")):
                if not column.endswith("days_since_last"):
                    block[column] = block[column].fillna(0)

        out.append(block)
        if verbose:
            rate = block["failed"].mean() * 100
            print(f"  {date.date()}  {len(block):>8,} carriers  {rate:5.2f}% failed")

    return pd.concat(out, ignore_index=True)
