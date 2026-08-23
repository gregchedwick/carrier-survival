"""Construct the carrier exit label.

The question is whether a carrier stopped operating, which is not the same as
whether FMCSA started a revocation proceeding against it. Two filters separate
those, and together they discard roughly 60% of what a naive label would count.

**Discontinued proceedings are not exits.** ``AuthHist`` records 2,151,687
``DISCONTINUED REVOCATION`` dispositions against 1,525,785 executed ``REVOKED``
ones. Most revocations begin with an insurance lapse; the carrier files a new
policy and trades on. Only executed revocations are candidates.

**Reinstated carriers are not exits either.** A further 8.8% of executed
revocations are followed by the carrier returning within a year. Those are
censored, not counted as failures.

**Voluntary surrender is a competing risk, not a failure.** A carrier that wound
itself down — sold, merged, retired — left the risk pool rather than failing in
it. Keeping those in the positive class would teach the model to predict orderly
exits, which is not what anyone is paying to see coming.

Exit events come from the ``Revocation`` file rather than ``AuthHist`` because
only it carries the voluntary/involuntary split. ``AuthHist`` supplies the
return events used for censoring.
"""

from __future__ import annotations

import pandas as pd

#: A carrier that regains authority within this many months of a revocation was
#: having a bad quarter, not exiting.
RETURN_WINDOW_MONTHS = 12

#: Actions that put a carrier back in business.
RETURN_ACTIONS = ("REINSTATED", "GRANTED")

EXIT_TYPES = {
    "INVOLUNTARY REVOCATION": "involuntary",
    "VOLUNTARY REVOCATION": "voluntary",
    "ADMINISTRATIVE REVOCATION": "administrative",
}


def load_return_events(authhist: pd.DataFrame) -> pd.DataFrame:
    """Dates on which carriers regained authority."""
    events = authhist[
        authhist["original_action_desc"].isin(RETURN_ACTIONS)
        & authhist["dot_number"].notna()
        & authhist["disp_served_date"].notna()
    ]
    return (
        events[["dot_number", "disp_served_date"]]
        .rename(columns={"disp_served_date": "return_date"})
        .drop_duplicates()
        .sort_values(["dot_number", "return_date"])
        .reset_index(drop=True)
    )


def load_exit_candidates(revocation: pd.DataFrame) -> pd.DataFrame:
    """Executed revocations, typed as voluntary or involuntary.

    The two types are dated in different columns, which is not documented
    anywhere and is easy to miss: ``order2_effective_date`` is present on 96.6%
    of involuntary revocations but on 92 of 200,017 voluntary ones (0.05%).
    ``order1_serve_date`` covers 99.7% and 29.3% respectively. Keying only on
    the effective date drops almost every voluntary exit — and, worse, leaves
    those carriers in the data looking like survivors.

    So the exit date coalesces the two, preferring the effective date where it
    exists. Voluntary exits that remain undatable are still returned, with a
    null ``exit_date`` and ``datable=False``, so callers can exclude those
    carriers rather than silently mislabel them.
    """
    events = revocation[revocation["dot_number"].notna()].copy()
    events["exit_type"] = events["order2_type_desc"].map(EXIT_TYPES)
    events = events[events["exit_type"].notna()]

    events["exit_date"] = events["order2_effective_date"].fillna(
        events["order1_serve_date"]
    )
    events["datable"] = events["exit_date"].notna()

    return (
        events[["dot_number", "exit_date", "exit_type", "type_license", "datable"]]
        .drop_duplicates(subset=["dot_number", "exit_date", "exit_type"])
        .sort_values(["dot_number", "exit_date"])
        .reset_index(drop=True)
    )


def build_labels(authhist: pd.DataFrame, revocation: pd.DataFrame) -> pd.DataFrame:
    """One row per revocation event, marked permanent or censored.

    Returns columns:
      dot_number, exit_date, exit_type, type_license,
      returned_within_window, is_permanent_exit, is_failure
    """
    candidates = load_exit_candidates(revocation)
    returns = load_return_events(authhist)

    # Undatable events cannot be placed on a timeline, so they cannot be
    # censored or used as outcomes. They are carried through unmatched so the
    # carriers involved can be excluded from training rather than counted as
    # survivors — see the `datable` flag.
    datable = candidates[candidates["datable"]]
    undatable = candidates[~candidates["datable"]].copy()

    # merge_asof finds, for each revocation, the first return on or after it —
    # far cheaper than the full cross join, which is ~10^11 pairs here.
    merged = pd.merge_asof(
        datable.sort_values("exit_date"),
        returns.sort_values("return_date"),
        left_on="exit_date",
        right_on="return_date",
        by="dot_number",
        direction="forward",
        allow_exact_matches=False,
    )
    if not undatable.empty:
        undatable["return_date"] = pd.NaT
        merged = pd.concat([merged, undatable], ignore_index=True)

    cutoff = merged["exit_date"] + pd.DateOffset(months=RETURN_WINDOW_MONTHS)
    merged["returned_within_window"] = (
        merged["return_date"].notna() & (merged["return_date"] <= cutoff)
    )
    merged["is_permanent_exit"] = merged["datable"] & ~merged["returned_within_window"]

    # The positive class: datable, permanent, and not a voluntary wind-down.
    merged["is_failure"] = merged["is_permanent_exit"] & merged["exit_type"].eq(
        "involuntary"
    )

    # Carriers whose only recorded exit cannot be dated must be dropped from
    # training entirely. We know they left; we just cannot say when, and leaving
    # them in would label a departed carrier as a survivor.
    merged["exclude_from_training"] = ~merged["datable"]

    return merged[
        [
            "dot_number",
            "exit_date",
            "exit_type",
            "type_license",
            "return_date",
            "datable",
            "returned_within_window",
            "is_permanent_exit",
            "is_failure",
            "exclude_from_training",
        ]
    ].sort_values(["dot_number", "exit_date"]).reset_index(drop=True)


def summarize(labels: pd.DataFrame) -> str:
    """A short human-readable account of what the filters removed."""
    total = len(labels)
    undatable = int((~labels["datable"]).sum())
    datable = total - undatable
    returned = int(labels["returned_within_window"].sum())
    permanent = int(labels["is_permanent_exit"].sum())
    failures = int(labels["is_failure"].sum())

    by_type = (
        labels[labels["is_permanent_exit"]]["exit_type"].value_counts().to_dict()
    )

    lines = [
        f"  revocation events                   : {total:>10,}",
        f"  undatable (no usable date)          : {undatable:>10,}"
        f"  ({undatable / total * 100:.1f}%)  -> carriers excluded",
        f"  datable                             : {datable:>10,}",
        f"  returned within {RETURN_WINDOW_MONTHS} months            : {returned:>10,}"
        f"  ({returned / datable * 100:.1f}% of datable)  -> censored",
        f"  permanent exits                     : {permanent:>10,}"
        f"  ({permanent / datable * 100:.1f}% of datable)",
        "",
        "  permanent exits by type:",
    ]
    for name, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {name:<16s} {count:>10,}  ({count / permanent * 100:.1f}%)")
    lines += [
        "",
        f"  POSITIVE CLASS (involuntary, permanent): {failures:,}",
        f"  distinct carriers with a failure        : {labels[labels.is_failure].dot_number.nunique():,}",
    ]
    return "\n".join(lines)
