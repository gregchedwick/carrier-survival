"""Fit the model once, then write the evaluation charts and a metrics file.

    python scripts/make_charts.py

Produces `docs/charts/*.svg` and `docs/metrics.json`. Both are committed, so a
reader can inspect the results without the private snapshot archive the panel
build needs.

Charts render on an explicit white background rather than a transparent one.
GitHub serves README images unchanged to both themes, and a transparent chart
with dark axis labels is invisible to anyone browsing in dark mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carrier_survival.config import DATA_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHART_DIR = ROOT / "docs" / "charts"
INTERIM_DIR = DATA_DIR / "interim"

BASELINE_FEATURES = ["carrier_age_days", "power_units_now"]
NOT_FEATURES = {
    "dot_number", "prediction_date", "observed_as_of", "exit_date", "failed",
    "state", "status", "safety_rating", "business_org",
    "has_prior_revoked_dot", "undeliverable_address",
}
FLEET_BANDS = [(-1, 2, "1-2"), (2, 5, "3-5"), (5, 20, "6-20"),
               (20, 100, "21-100"), (100, np.inf, "100+")]

INK = "#1a1a1a"
ACCENT = "#0b6e4f"
MUTED = "#9aa0a6"
SECOND = "#b45309"


def style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color="#e8eaed", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / name
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 20):
    """Observed failure rate against predicted risk, by equal-count bucket.

    Equal-count rather than equal-width: predictions cluster hard at the low end
    when the base rate is 2%, so equal-width buckets leave the top ones almost
    empty and the curve becomes noise.
    """
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    predicted, observed, counts = [], [], []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        predicted.append(p[chunk].mean())
        observed.append(y[chunk].mean())
        counts.append(len(chunk))
    return np.array(predicted), np.array(observed), np.array(counts)


def expected_calibration_error(y, p, bins: int = 20) -> float:
    predicted, observed, counts = calibration_bins(y, p, bins)
    return float(np.sum(counts * np.abs(predicted - observed)) / counts.sum())


def main() -> None:
    frame = pd.read_parquet(INTERIM_DIR / "features.parquet")

    candidates = [c for c in frame.columns
                  if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(frame[c])]
    features = []
    for column in candidates:
        values = frame[column].dropna()
        if len(values) and values.nunique() > 1:
            features.append(column)
    frame[features] = frame[features].astype("float64")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_idx, test_idx = next(
        splitter.split(frame, frame["failed"], groups=frame["dot_number"])
    )
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    y_train, y_test = train["failed"].to_numpy(), test["failed"].to_numpy()
    print(f"train {len(train):,} rows / test {len(test):,} rows / "
          f"{len(features)} features / base {y_test.mean():.4f}\n")

    baseline = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    baseline.fit(train[BASELINE_FEATURES].fillna(-1), y_train)
    base_scores = baseline.predict_proba(test[BASELINE_FEATURES].fillna(-1))[:, 1]

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=0
    )
    model.fit(train[features], y_train)
    scores = model.predict_proba(test[features])[:, 1]

    base_rate = float(y_test.mean())
    metrics = {
        "rows": int(len(frame)),
        "carriers": int(frame["dot_number"].nunique()),
        "features": len(features),
        "base_rate": base_rate,
        "baseline": {
            "auc": float(roc_auc_score(y_test, base_scores)),
            "pr_auc": float(average_precision_score(y_test, base_scores)),
        },
        "model": {
            "auc": float(roc_auc_score(y_test, scores)),
            "pr_auc": float(average_precision_score(y_test, scores)),
            "brier": float(brier_score_loss(y_test, scores)),
            "ece": expected_calibration_error(y_test, scores),
        },
    }
    metrics["model"]["lift"] = metrics["model"]["pr_auc"] / base_rate
    metrics["baseline"]["lift"] = metrics["baseline"]["pr_auc"] / base_rate

    # --- ROC and precision-recall ------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax in (left, right):
        style(ax)

    for label, s, colour in (("full model", scores, ACCENT), ("baseline", base_scores, SECOND)):
        fpr, tpr, _ = roc_curve(y_test, s)
        left.plot(fpr, tpr, color=colour, linewidth=1.8,
                  label=f"{label} (AUC {roc_auc_score(y_test, s):.3f})")
    left.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="chance")
    left.set_xlabel("false positive rate"); left.set_ylabel("true positive rate")
    left.set_title("ROC", color=INK, fontsize=11)
    left.legend(frameon=False, fontsize=8.5, loc="lower right")

    for label, s, colour in (("full model", scores, ACCENT), ("baseline", base_scores, SECOND)):
        precision, recall, _ = precision_recall_curve(y_test, s)
        right.plot(recall, precision, color=colour, linewidth=1.8,
                   label=f"{label} (PR-AUC {average_precision_score(y_test, s):.3f})")
    right.axhline(base_rate, color=MUTED, linewidth=1, linestyle="--",
                  label=f"base rate ({base_rate:.3f})")
    right.set_xlabel("recall"); right.set_ylabel("precision")
    right.set_title("Precision–recall — the honest view at a 2% base rate",
                    color=INK, fontsize=11)
    right.legend(frameon=False, fontsize=8.5)
    save(fig, "roc_pr.svg")

    # --- cumulative gains ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style(ax)
    order = np.argsort(-scores)
    captured = np.cumsum(y_test[order]) / y_test.sum()
    targeted = np.arange(1, len(order) + 1) / len(order)
    ax.plot(targeted, captured, color=ACCENT, linewidth=2, label="full model")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="random")
    for mark in (0.05, 0.10, 0.20):
        hit = captured[int(mark * len(order)) - 1]
        ax.plot([mark, mark], [0, hit], color=SECOND, linewidth=0.9, linestyle=":")
        ax.annotate(f"{hit:.0%}", (mark, hit), textcoords="offset points",
                    xytext=(5, -10), fontsize=8.5, color=SECOND)
        metrics.setdefault("gains", {})[f"top_{int(mark * 100)}pct"] = float(hit)
    ax.set_xlabel("share of carriers reviewed, highest risk first")
    ax.set_ylabel("share of all failures captured")
    ax.set_title("Cumulative gains — what a review queue actually buys",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "gains.svg")

    # --- calibration --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    style(ax)
    predicted, observed, _ = calibration_bins(y_test, scores)
    limit = max(predicted.max(), observed.max()) * 1.05
    ax.plot([0, limit], [0, limit], color=MUTED, linewidth=1, linestyle="--",
            label="perfect calibration")
    ax.plot(predicted, observed, color=ACCENT, linewidth=1.6, marker="o",
            markersize=4, label="model, by 5% bucket")
    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_xlabel("mean predicted risk"); ax.set_ylabel("observed failure rate")
    ax.set_title(f"Calibration  (Brier {metrics['model']['brier']:.4f}, "
                 f"ECE {metrics['model']['ece']:.4f})", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "calibration.svg")

    # --- performance by fleet band -----------------------------------------
    units = test["power_units_now"].fillna(-1)
    labels, aucs, praucs = [], [], []
    for low, high, label in FLEET_BANDS:
        mask = ((units > low) & (units <= high)).to_numpy()
        if mask.sum() < 500 or y_test[mask].sum() < 20:
            continue
        labels.append(label)
        aucs.append(roc_auc_score(y_test[mask], scores[mask]))
        praucs.append(average_precision_score(y_test[mask], scores[mask]))
    metrics["segments"] = [
        {"band": b, "auc": float(a), "pr_auc": float(p)}
        for b, a, p in zip(labels, aucs, praucs)
    ]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    style(ax)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, aucs, width=0.4, color=ACCENT, label="AUC")
    ax.bar(x + 0.2, praucs, width=0.4, color=SECOND, label="PR-AUC")
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
    for i, value in enumerate(aucs):
        ax.annotate(f"{value:.3f}", (i - 0.2, value), ha="center",
                    textcoords="offset points", xytext=(0, 3), fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("power units"); ax.set_ylim(0, 1)
    ax.set_title("Discrimination by fleet size", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    save(fig, "segments.svg")

    (ROOT / "docs" / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n  wrote docs/metrics.json")
    print(f"\n  Brier {metrics['model']['brier']:.5f}   "
          f"ECE {metrics['model']['ece']:.5f}")
    print(f"  gains: {metrics['gains']}")


if __name__ == "__main__":
    main()
