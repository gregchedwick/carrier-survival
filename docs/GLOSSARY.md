# Glossary

Terms used throughout this repository, defined in the context of this project
rather than in the abstract. Roughly ordered from data to modelling to
evaluation.

---

## Data shape

**Snapshot** — one export file, describing the carrier population as it stood on
one date. FMCSA overwrites its census daily and publishes no archive, so a
snapshot kept on disk is the only way to know what was true in the past.

**Vintage** — a *family* of exports sharing a schema and a population
definition. This archive has three: `mcmis` (~2.17M registered entities),
`reporting` (~1.86M active carriers), and `datahub` (~4.47M rows including
inactive). They are not interchangeable. A carrier present in one and absent
from the next has usually not exited — it was never in scope. Comparing across a
vintage boundary measures *which file you are looking at*, not what changed.

**Panel** — the stacked result: one row per carrier per period. 44.8M rows here.
The BI analogue is a monthly snapshot fact table, and it exists for the same
reason: so you can ask what a carrier looked like at a past date instead of only
today.

**Period** — one month of the panel, `YYYYMM`.

**Part** — one file of a multi-file export. The 2024 monthly exports are split
across three parts that must all be read; reading one gives 39% of the
population.

**Cohort** — the set of carriers eligible for prediction at a given date. Here:
present in the panel, status active, and not already exited.

**Risk set** — the carriers still "at risk" of the outcome at a given date. A
carrier that already failed cannot fail again, so it leaves the risk set. This
is the core bookkeeping idea in survival analysis.

---

## The outcome

**Label** — the thing being predicted, as a concrete column. Choosing it is
usually the hardest judgment in a project. Here the obvious label ("was
revoked") was wrong: more revocation proceedings are discontinued than executed,
so it would mostly have labelled carriers having a bad month.

**Base rate** — how often the label is true. 2.25% here. Every other metric has
to be read against it; a model that is right 97.75% of the time by always
predicting "survives" is worthless.

**Class imbalance** — when the base rate is far from 50%. Rare outcomes make
accuracy meaningless and make precision-style metrics essential.

**Censoring** — knowing a carrier survived *at least* this long, but not how
long in total, because the observation window ended. Ignoring censored records
biases survival estimates downward. Here, a carrier revoked but reinstated
within 12 months is censored rather than counted as a failure.

**Competing risk** — a different ending that removes a carrier from the risk set
without being the outcome you are modelling. Voluntary surrender is the example:
the carrier is gone, but it did not fail. Counting it as a failure teaches the
model the wrong thing; counting it as a survivor is equally wrong.

**Outcome window** — the span in which the outcome is measured. Months 6–18
after the prediction date here.

**Blackout** — the deliberate gap between the prediction date and the start of
the outcome window (months 0–6). Insurance lapse mechanically triggers
revocation about a month later, so predicting the near term would just
rediscover FMCSA's published procedure. The blackout forces the model to find
signal that *precedes* the administrative cascade.

---

## Features

**Feature** — one input column the model learns from. `days_since_mcs150` is a
feature; so is `power_units_now`.

**Feature engineering** — turning raw fields into inputs that carry signal. A
raw MCS-150 filing date is not directly useful; *days since* that date,
measured to the prediction date, is.

**Point-in-time** — the discipline that every feature on a row dated
2024-08-31 must be computable using only information that existed on
2024-08-31. Closest BI analogue: a slowly-changing dimension, where you need the
value *as it was then*, not as it is now.

**Degenerate feature** — a column that is entirely null or takes one value.
Carries no signal, and an all-null column crashes scikit-learn's binner with an
error that says nothing about the cause.

---

## Leakage

**Leakage** — when a feature contains information that would not have been
available at prediction time. The model scores beautifully in testing and fails
completely in production. There is no error and no warning; it simply looks like
success. This is the single most common way a predictive project dies quietly.

**Planted leak** — a deliberately-leaking feature, built from the outcome, used
to prove a leakage detector can actually fire. A check that has never been shown
to fail is not evidence — it may be incapable of failing. See defects 11 and 12
in the README for what happens when you skip this.

**Solo AUC** — the AUC of one feature used alone. A single feature separating
the outcome far better than the whole model plausibly could is almost always
derived from it. Nothing here should exceed 0.90; the strongest honest feature
reaches 0.743.

**Temporal decay** — training at one date and scoring the same carriers six
months later on staler information. Performance should fall. If it holds up, or
improves, something is seeing the future.

**Group split** — splitting train and test by *carrier* rather than by row. A
carrier appears at up to seven prediction dates with overlapping outcome
windows, so a random row split would put the same carrier on both sides and
inflate every score. A subtler cousin of leakage.

---

## Modelling

**Survival analysis** — modelling *when* something ends, not just whether. The
complication is that most subjects have not ended when you look, which is what
censoring handles.

**Discrete-time survival** — reframing survival as ordinary classification by
building one row per subject per period, each asking: *given this carrier is
alive now, does it fail in the window ahead?* Sidesteps the heavier survival
mathematics and is what this project uses.

**Hazard** — the probability of failing in the next interval, given survival up
to now. What the model actually estimates on each row.

**Baseline model** — a deliberately trivial model that a real one must beat.
Here: carrier age and fleet size, two columns anyone can get in one query. It
exists because "the model gets 0.89 AUC" means nothing on its own.

**Gradient boosting** — the model type used here
(`HistGradientBoostingClassifier`). Builds many small decision trees in
sequence, each correcting the previous ones' errors. Handles missing values and
non-linear relationships without manual preparation, which suits messy
regulatory data.

**Logistic regression** — the simpler linear model used for the baseline.

---

## Evaluation

**AUC** (Area Under the ROC Curve) — take one carrier that failed and one that
did not; how often does the model rank the failure as riskier? 0.5 is a coin
flip, 1.0 is perfect, below 0.5 means the model is *inverted* — ranking failures
as safer. 0.889 here.

**PR-AUC** (Precision–Recall AUC) — the same idea but focused on the rare class.
Far more informative than AUC when the base rate is low, because AUC flatters
rare-event models by rewarding correct rankings among the vast majority of
negatives.

**Lift** — PR-AUC divided by the base rate. The commercially meaningful number:
**6.7x** means acting on the model's top-ranked carriers finds failures 6.7
times more densely than picking at random. This is the figure to put in front of
an underwriter.

**Precision** — of the carriers flagged as risky, how many actually failed.
**Recall** — of the carriers that actually failed, how many were flagged. They
trade against each other; where you sit on that curve is a business decision
about the relative cost of a missed failure versus a false alarm.

**Segmentation at evaluation** — measuring performance separately by fleet band
rather than filtering small carriers out of training. Dropping them would change
the base rate rather than improve the model.

**Calibration** — whether a predicted 5% risk actually fails 5% of the time.
Distinct from ranking: a model can rank perfectly and still be badly calibrated.
Not yet assessed here, and it would be required before any pricing use.

---

## Pipeline

**Ingestion** — pulling source data into local storage.

**SODA / Socrata** — the API serving the DOT Public Data Portal. Its `$offset`
paging walks an unordered set, so an explicit `$order` is mandatory; without it
rows silently duplicate while others never arrive.

**Manifest** — a small record written beside each extract logging row counts,
distinct carriers, null keys and fetch time, so a bad pull is visible after the
fact.

**Guardrail** — an automated check that fails loudly on a condition that would
otherwise pass silently. Most of this repository's value is in these; see the
thirteen-defect catalogue in the README.

**Parquet** — the columnar file format used for the panel and features.
Compresses well and reads selected columns without loading the whole file.
