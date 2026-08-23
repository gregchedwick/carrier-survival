"""Fit the baseline and a full model, and check for leakage.

    python scripts/train_baseline.py

The baseline is deliberately trivial — carrier age and fleet size, two columns
anyone can compute in a single query. It exists because "the model gets 0.78
AUC" means nothing on its own. If the full feature set cannot clearly beat two
obvious variables, there is no product, and that is worth discovering now rather
than after building a dashboard on top of it.

Splitting is by carrier, not by row. A carrier appears at up to seven prediction
dates with overlapping outcome windows, so a random row split would put the same
carrier on both sides and inflate every score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carrier_survival import leakage  # noqa: E402
from carrier_survival.config import DATA_DIR  # noqa: E402

INTERIM_DIR = DATA_DIR / "interim"

BASELINE_FEATURES = ["carrier_age_days", "power_units_now"]

#: Columns that are identifiers, outcomes, or unavailable during the training
#: window — see the README on vintage coverage.
NOT_FEATURES = {
    "dot_number", "prediction_date", "observed_as_of", "exit_date", "failed",
    "state", "status", "safety_rating", "business_org",
    "has_prior_revoked_dot", "undeliverable_address",
}

FLEET_BANDS = [(-1, 2, "1-2"), (2, 5, "3-5"), (5, 20, "6-20"),
               (20, 100, "21-100"), (100, np.inf, "100+")]


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    base = float(np.mean(y_true))
    ap = average_precision_score(y_true, scores)
    return {
        "auc": roc_auc_score(y_true, scores),
        "pr_auc": ap,
        "base_rate": base,
        "lift": ap / base if base else float("nan"),
    }


def main() -> None:
    frame = pd.read_parquet(INTERIM_DIR / "features.parquet")

    candidates = [
        c for c in frame.columns
        if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(frame[c])
    ]

    # A column that is entirely null, or takes one value, carries no signal —
    # and an all-null column crashes the binner rather than being ignored.
    features, dropped = [], []
    for column in candidates:
        values = frame[column].dropna()
        (features if len(values) and values.nunique() > 1 else dropped).append(column)
    if dropped:
        print(f"dropped {len(dropped)} degenerate features: {', '.join(dropped)}\n")

    # Pandas nullable integer columns (Int64) reach sklearn's binner as masked
    # arrays and fail there with an error about window shapes that says nothing
    # about the cause. Plain float64 with NaN is what it expects.
    frame[features] = frame[features].astype("float64")
    print(f"rows {len(frame):,}   carriers {frame['dot_number'].nunique():,}   "
          f"features {len(features)}   base rate {frame['failed'].mean() * 100:.2f}%\n")

    # Split on carriers so no carrier appears in both train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_idx, test_idx = next(
        splitter.split(frame, frame["failed"], groups=frame["dot_number"])
    )
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    print(f"train {len(train):,} rows / {train['dot_number'].nunique():,} carriers")
    print(f"test  {len(test):,} rows / {test['dot_number'].nunique():,} carriers\n")

    y_train, y_test = train["failed"].to_numpy(), test["failed"].to_numpy()

    results = {}

    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    baseline.fit(train[BASELINE_FEATURES].fillna(-1), y_train)
    results["baseline (age + fleet size)"] = evaluate(
        y_test, baseline.predict_proba(test[BASELINE_FEATURES].fillna(-1))[:, 1]
    )

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=0
    )
    model.fit(train[features], y_train)
    test_scores = model.predict_proba(test[features])[:, 1]
    results["full feature set"] = evaluate(y_test, test_scores)

    print(f"{'model':<30} {'AUC':>7} {'PR-AUC':>8} {'base':>7} {'lift':>6}")
    for name, m in results.items():
        print(f"{name:<30} {m['auc']:7.3f} {m['pr_auc']:8.3f} "
              f"{m['base_rate']:7.3f} {m['lift']:6.1f}x")

    gain = results["full feature set"]["auc"] - results["baseline (age + fleet size)"]["auc"]
    print(f"\n  AUC gained over the trivial baseline: {gain:+.3f}")

    # --- performance where the money is -----------------------------------
    print("\nby fleet size (test set):")
    print(f"  {'band':>8} {'carriers':>10} {'base':>7} {'AUC':>7} {'PR-AUC':>8}")
    units = test["power_units_now"].fillna(-1)
    for low, high, label in FLEET_BANDS:
        mask = (units > low) & (units <= high)
        if mask.sum() < 500 or y_test[mask.to_numpy()].sum() < 20:
            continue
        seg_y, seg_s = y_test[mask.to_numpy()], test_scores[mask.to_numpy()]
        m = evaluate(seg_y, seg_s)
        print(f"  {label:>8} {mask.sum():>10,} {m['base_rate']:7.3f} "
              f"{m['auc']:7.3f} {m['pr_auc']:8.3f}")

    # --- leakage check -----------------------------------------------------
    #
    # Train on the earliest prediction date only, then score the latest. The
    # features are staler and the outcome further away, so performance must
    # fall. If it holds up, something in the feature set is seeing the future.
    dates = sorted(frame["prediction_date"].unique())
    early, late = frame[frame.prediction_date == dates[0]], frame[frame.prediction_date == dates[-1]]
    shared = set(early.dot_number) & set(late.dot_number)
    early_only = early[~early.dot_number.isin(shared)]

    probe = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    probe.fit(early_only[features], early_only["failed"])

    same = evaluate(
        early[early.dot_number.isin(shared)]["failed"].to_numpy(),
        probe.predict_proba(early[early.dot_number.isin(shared)][features])[:, 1],
    )["auc"]
    later = evaluate(
        late[late.dot_number.isin(shared)]["failed"].to_numpy(),
        probe.predict_proba(late[late.dot_number.isin(shared)][features])[:, 1],
    )["auc"]

    print(f"\nleakage check  (trained on {pd.Timestamp(dates[0]).date()})")
    print(f"  scored at the same date : AUC {same:.3f}")
    print(f"  scored {len(dates) - 1} months later : AUC {later:.3f}")

    report = leakage.check(test, features, y_test, decay=same - later)
    print(report.describe())

    strongest = sorted(
        leakage.single_feature_auc(test, features, y_test).items(),
        key=lambda kv: -kv[1],
    )[:5]
    print("\n  strongest single features (solo AUC):")
    for name, auc in strongest:
        print(f"    {name:<32} {auc:.3f}")

    if not report.passed:
        raise SystemExit("Leakage detected — refusing to report these results as valid.")


if __name__ == "__main__":
    main()
