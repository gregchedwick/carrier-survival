"""Leakage detection.

A model trained on features that quietly contain future information scores
beautifully and is worthless. There is no error and no warning — just a number
that looks like success. The only defence is a check that can fail, and that has
been shown to fail on a known-bad input.

Two independent signals here, because either alone is easy to fool:

**Solo AUC.** A single feature that separates the outcome far better than the
whole model plausibly could is almost always derived from the outcome. For
carrier failure, nothing knowable in advance should reach 0.90 on its own.

**Temporal decay.** A model scored on staler features should do worse. If
performance holds up as information ages, the model is reading something it
should not have.

The second is the weaker test and is reported with an explicit margin rather
than a verdict — a difference of a few thousandths is noise, and calling that
"passed" is worse than not checking, because it manufactures confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

#: A feature scoring above this alone is treated as leaking. Carrier failure is
#: not that predictable from anything legitimately knowable in advance; the best
#: honest single feature here reaches roughly 0.65.
IMPLAUSIBLE_SOLO_AUC = 0.90

#: How much a model must lose when scored on features six months staler before
#: the decay is distinguishable from noise.
MEANINGFUL_DECAY = 0.01


@dataclass(frozen=True)
class LeakageReport:
    """What the checks found. ``passed`` is false if anything looks wrong."""

    suspicious: dict[str, float]
    decay: float | None
    decay_conclusive: bool

    @property
    def passed(self) -> bool:
        return not self.suspicious

    def describe(self) -> str:
        # ASCII only: the Windows console codepage mangles em-dashes, and a
        # leakage warning that renders as mojibake is a leakage warning nobody
        # reads.
        lines = []
        if self.suspicious:
            lines.append("  LEAKING - these features separate the outcome implausibly well:")
            for name, auc in sorted(self.suspicious.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {name:<32} solo AUC {auc:.3f}")
        else:
            lines.append(f"  no feature exceeds solo AUC {IMPLAUSIBLE_SOLO_AUC:.2f}")

        if self.decay is None:
            lines.append("  temporal decay: not evaluated")
        elif self.decay_conclusive:
            lines.append(f"  temporal decay: {self.decay:+.3f} - degrades as expected")
        else:
            lines.append(
                f"  temporal decay: {self.decay:+.3f} - within noise, inconclusive "
                f"(needs |decay| > {MEANINGFUL_DECAY:.2f})"
            )
        return "\n".join(lines)


def single_feature_auc(
    frame: pd.DataFrame, features: list[str], y: np.ndarray
) -> dict[str, float]:
    """AUC of each feature used alone, direction-agnostic.

    A feature that predicts the outcome perfectly *inverted* is leaking just as
    badly as one that predicts it directly, so the score is folded about 0.5.
    """
    scores: dict[str, float] = {}
    for name in features:
        values = pd.to_numeric(frame[name], errors="coerce")
        usable = values.notna().to_numpy()
        # Needs both classes present and some variation to mean anything.
        if usable.sum() < 1000 or len(np.unique(y[usable])) < 2:
            continue
        if values[usable].nunique() < 2:
            continue
        auc = roc_auc_score(y[usable], values[usable].to_numpy())
        scores[name] = max(auc, 1.0 - auc)
    return scores


def check(
    frame: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    decay: float | None = None,
    threshold: float = IMPLAUSIBLE_SOLO_AUC,
) -> LeakageReport:
    """Run the checks and report. See the module docstring for what each means."""
    solo = single_feature_auc(frame, features, y)
    suspicious = {name: auc for name, auc in solo.items() if auc >= threshold}
    return LeakageReport(
        suspicious=suspicious,
        decay=decay,
        decay_conclusive=decay is not None and abs(decay) > MEANINGFUL_DECAY,
    )
