# Carrier Survival

Predicting whether a motor carrier will still be operating twelve months from
now, from public FMCSA data.

Freight brokers, insurers, factoring companies and equipment lessors all carry
exposure to carriers that may not exist next year, and the tools available to
them are backward-looking: you can check whether a carrier's authority is active
*today*, but nothing tells you the odds it survives the length of a contract.

**Status: working end-to-end.** Ingestion, labels, point-in-time features,
models and leakage controls all run. Results below.

---

## Results

617k carriers, 3.8M carrier-months, seven monthly prediction dates in 2024,
6.08% twelve-month failure rate. Split by **carrier**, not by row — a carrier
appears at up to seven dates with overlapping outcome windows, so a random row
split would put the same carrier on both sides and inflate every score.

| Model | AUC | PR-AUC | Base rate | Lift |
|---|---|---|---|---|
| Baseline — carrier age + fleet size | 0.668 | 0.098 | 0.061 | 1.6x |
| **Full feature set** (24 features) | **0.767** | **0.165** | 0.061 | **2.7x** |

The baseline is deliberately trivial: two columns anyone can compute in a single
query. It exists because "the model gets 0.77 AUC" means nothing on its own. The
full feature set beats it by **+0.099 AUC**, which is the number actually worth
reporting.

### By fleet size

Segmented at evaluation rather than filtered at training — micro-carriers are
most of the population and dropping them would change the base rate rather than
improve the model.

| Fleet | Carriers | Base rate | AUC | PR-AUC |
|---|---|---|---|---|
| 1–2 | 613,684 | 6.7% | 0.741 | 0.167 |
| 3–5 | 144,218 | 4.3% | 0.824 | 0.170 |
| 6–20 | 105,161 | 3.5% | 0.837 | 0.151 |
| 21–100 | 34,469 | 2.7% | 0.848 | 0.140 |
| 100+ | 6,739 | 1.5% | 0.806 | 0.103 |

Discrimination is best in the 3–100 unit band — the carriers a broker or insurer
actually underwrites. One-truck operators are the hardest to call, which is
intuitive: their failure is often a personal circumstance that leaves no trace in
any federal file until it has already happened.

### Leakage controls

A model that has seen the future scores beautifully and is worthless, with no
error to warn you. Two checks run on every training run:

**Solo AUC.** Any single feature separating the outcome implausibly well
(≥ 0.90) is almost certainly derived from it. The strongest honest feature here
reaches 0.671:

```
authority_age_days             0.671
carrier_age_days               0.665
days_since_authority_action    0.654
crash_days_since_last          0.635
distinct_dockets               0.630
```

**Temporal decay.** Train on the earliest prediction date, score six months
later on staler features. Performance should fall.

The decay check currently reports `+0.004`, and the run labels that
**inconclusive** rather than passing it — four thousandths is noise, and calling
it a pass would manufacture confidence the data doesn't support. An earlier
version of this check printed "OK" off exactly that number.

The controls are themselves tested against a **planted leak** — a feature built
from the outcome with 10% of labels flipped — and the suite asserts the detector
catches it (`tests/test_leakage.py`). A check that has never been shown to fail
is not evidence; it may simply be incapable of firing. Training aborts rather
than reporting results if anything trips.

---

## What this is not

This is not a safety score. It does not estimate crash risk, and it is not a
competitor to CSA percentiles or any commercial risk product. It answers one
different question: *will this business still be here?*

Nothing in this repository derives from any proprietary scoring model. See
[Boundaries](#boundaries).

---

## The label

The obvious label — "did FMCSA revoke this carrier's authority?" — is wrong, and
the data says so plainly. `AuthHist` records how each revocation proceeding
resolved:

| Disposition | Count |
|---|---|
| **DISCONTINUED REVOCATION** | **2,208,586** |
| REVOKED | 1,600,771 |
| FAILURE TO REAPPLY | 9,743 |
| ADMINISTRATIVE INACTIVATION | 3,572 |

More proceedings are discontinued than executed. Most revocations begin with an
insurance lapse; the carrier files a new policy and continues trading. A further
**498,866** `REINSTATED` actions confirm it. Training on "a revocation happened"
means training mostly on carriers having a bad month.

So exit is defined as:

> `disp_action_desc = REVOKED` **and** no `REINSTATED` or `GRANTED` action for
> that carrier within the following 12 months.

**Exit type is separated**, because the `Revocation` file distinguishes
involuntary (1,329,063) from voluntary (200,017) surrender. A carrier that
voluntarily wound down did not fail — it left the risk pool, which makes it a
competing risk rather than a negative case to learn from.

### Result

```
revocation events           1,373,059
  undatable                   114,164   ( 8.3%)  carriers excluded
  returned within 12 months   308,699   (24.5% of datable)  censored
  permanent exits             950,196   (75.5% of datable)

POSITIVE CLASS (involuntary, permanent)   949,982
distinct carriers                         772,553
```

Between 31k and 85k failures per year across 2015–2025 — ample for the modelling
window.

### The two exit types are dated in different columns

Undocumented, and quietly destructive if missed:

| Type | `order2_effective_date` | `order1_serve_date` |
|---|---|---|
| Involuntary | **96.6%** | 99.7% |
| Voluntary | **0.05%** (92 of 200,017) | 29.3% |

Keying the exit date off `order2_effective_date` alone discards 99.95% of
voluntary exits — and leaves those carriers in the data looking like survivors,
which is a mislabel rather than a gap. The exit date therefore coalesces both
columns, and anything still undatable is flagged `exclude_from_training` so the
carrier is dropped instead of counted as surviving.

### Voluntary exit cannot be modelled as a competing risk here

Even after coalescing, only 212 voluntary revocations resolve to a permanent
exit. Almost all datable voluntary surrenders are followed by a new grant of
authority within a year — restructuring, re-registration or transfer rather than
departure. The competing-risks arm is therefore effectively empty, and the model
is scoped to involuntary exit alone. This is a limitation of the source, not a
modelling choice.

### Prediction window

Features are frozen at month 0 and the outcome is measured over **months 6–18**.
The blackout matters: insurance lapse mechanically triggers revocation about a
month later, so a model predicting the near term would mostly rediscover FMCSA's
own published procedure. Pushing the horizon out forces it to find signal that
precedes the administrative cascade.

### What this cannot see

Authority status is a legal proxy for operating status. A carrier that keeps its
authority and insurance current while hauling nothing is invisible here. No
public source resolves that.

---

## Features

24 features survive the degenerate-column filter, all constructed strictly from
information available at the prediction date:

| Group | Features |
|---|---|
| **Crash history** | Counts, fatalities, injuries and tow-aways over 6/12/24 months and lifetime; days since last crash |
| **Authority lifecycle** | Age of authority, prior revocations, prior reinstatements, prior discontinued proceedings, total actions, distinct dockets, days since last action |
| **Registration** | Carrier age, undeliverable address flag, prior revoked DOT number flag |
| **Fleet** | Power units, driver count |

Three columns are dropped automatically as degenerate: `mileage` (100% null in
every vintage) and both fleet-trajectory features — see below.

### The fleet-trajectory features cannot be built from this archive

This is the significant negative result of the project, and it undercuts the
original rationale for choosing a panel-limited design.

`power_units_chg_3m` and `power_units_chg_12m` were meant to capture fleet
momentum — a carrier shedding trucks before it fails. Both compute to **exactly
zero for every carrier**, because the monthly reporting exports do not refresh
that column. Per-carrier comparison across the archive:

```
202404 -> 202405   702,589 carriers in both   0 changed
202405 -> 202406   689,092                    0 changed
202406 -> 202407   678,132                    0 changed
202407 -> 202408   674,047                    0 changed
202408 -> 202409   667,154                    0 changed
202409 -> 202410   656,517                    0 changed
202410 -> 202412   631,131               78,501 changed  (12.4%)
202412 -> 202505   627,792                    0 changed
```

`power_units` moves at exactly one boundary in the reporting vintage and is
frozen everywhere else, including five straight months from December 2024 to May
2025. **15 of 23 archived periods are republished copies of earlier data.** The
timestamp advances; the fleet numbers do not.

The features are now correctly null rather than falsely zero, and the training
script drops them. Model scores are unchanged at 0.767 — they were contributing
nothing all along.

Genuine trajectory needs either the FOIA'd internal snapshots or forward
collection at the boundaries where the column actually refreshes.

---

## Sources

All from the [DOT Public Data Portal](https://data.transportation.gov/). The
older `catalog.data.gov` listings mirror the same data but are no longer the
maintained surface.

| Dataset | ID | Rows | Coverage | Role |
|---|---|---|---|---|
| Crash File | `aayw-vxb3` | 4.97M | **1982–2026** | Features — full history, no null dates |
| AuthHist – All With History | `9mw4-x3tu` | 4.94M | 1996–2026 | Label + authority lifecycle |
| Revocation – All With History | `sa6p-acbp` | 1.53M | 1996–2026 | Label, exit type |
| Vehicle Inspection File | `fx4q-ay7w` | 8.30M | **2023–2026 only** | Features — recent model only |
| Company Census | `az4n-8mr2` | 4.48M | current snapshot | Reference only |

### Sources evaluated and rejected

**`Motus InsHist`** (`3uet-3z4i`) is described as previous insurance policies but
is not a history. Of 53,490 rows, 34,858 have no cancellation date, and of the
18,632 that do, **18,572 fall in 2026** — earlier years are single digits. It
cannot support historical insurance features.

**Company Census** is overwritten daily with no published archive, so fleet size
and safety rating cannot be reconstructed for any past date from the API. A FOIA
request for internal historical snapshots was filed; FMCSA quoted 9–12 months.
Local monthly captures are used instead (below).

---

## Archived monthly census snapshots

FMCSA's census gap is filled by monthly pulls of the same public Company Census
file, archived locally before the collection was automated. They cover 23
periods between Nov 2023 and Jul 2026.

They are ordinary FMCSA public data — the only thing special about them is that
someone kept copies. Point `CARRIER_SURVIVAL_SNAPSHOT_DIR` at your own archive;
no path is hardcoded.

`coverage_report()` detects three defects automatically rather than leaving them
to be found later:

**Population break.** Three export types are interleaved — 0.7M, 2.2M and 4.5M
carriers. **1,464,357 carriers appear to vanish** at the Feb→Apr 2024 boundary.
They did not exit; they were never in scope for the second file. Features are
never differenced across a vintage boundary, because such a delta measures which
file a carrier appeared in.

**Truncated export.** `202501` holds 507,916 carriers against a ~700K median. It
drops 206,085 from December, and **198,603 of them (96%) reappear in February.**
Real exits do not come back.

**Stale republish.** Detailed above — 15 of 23 periods repeat the previous
period's values.

All three exclude the affected periods from differencing. The label is unaffected
either way, because exit is derived from dated event streams rather than from a
carrier's absence from a file.

---

## Nine silent defects

None of these raised an error. Each produced plausible-looking output that was
wrong, and each now has an automated guardrail and, where testable, a regression
test. This catalogue is the part of the project most worth reading.

| # | Defect | How it presented | Guardrail |
|---|---|---|---|
| 1 | Socrata `$offset` paging without `$order` walks an unordered set | Row counts correct, contents quietly duplicated and missing | Explicit `$order=:id` on every page request |
| 2 | `Sum of Nbr Power Unit` sat at column 0 — a grand total, not a fleet size | `5,726,224` on all 713,276 rows | Preference-ordered column resolution + `_reject_constant_columns()` |
| 3 | Exit types dated in different columns | 99.95% of voluntary exits silently relabelled as survivors | Coalesce both date columns; flag the remainder `exclude_from_training` |
| 4 | pandas 3.0 gives text columns dtype `str`, not `object` | `Y`/`N` flags fell through to `to_numeric` → NaN → `fillna(0)` → all zeros | Test `is_numeric_dtype`, not `dtype == object` |
| 5 | Type collisions across vintages (`add_date` int vs text, `county` numeric vs text) | Passed every per-file check, failed only at panel write | `TEXT_COLUMNS` normalisation + explicit date parsing |
| 6 | Population break between export vintages | 1.46M carriers "vanish"; a naive delta reads as mass fleet collapse | `population_break` flag; vintage-scoped differencing |
| 7 | Truncated `202501` export | 206K carriers "exit", 96% of them return in February | `suspect_truncation` flag |
| 8 | Stale republished exports masked by `NaN != NaN` | Detector saw a 5.6% null rate as 5.6% "changed", clearing its 0.1% threshold — six frozen months reported as fresh | Compare only rows non-null on both sides; regression test with nulls on both sides |
| 9 | An all-null column reaches sklearn's binner | `ValueError: window shape cannot be larger than input array shape` — says nothing about the cause | Degenerate-column filter before fitting |

Two more worth noting, though they are judgment rather than defects: revocation
is not exit (see [The label](#the-label)), and AuthHist carries corrupt dates —
observed years include 0189, 2044 and 2517 — which are dropped rather than
clamped, since a wrong date is worse than a missing one when the entire design
depends on knowing what was true when.

**Defect 8 is the instructive one.** The staleness detector was written
specifically to catch republished exports, was per-carrier and correct in
outline, and still failed — because `NaN != NaN` is `True` in pandas, so
carriers null in *both* exports counted as changed. That alone cleared the
threshold. The guardrail existed, ran on every build, and reported the opposite
of the truth for months.

---

## Boundaries

The local snapshot files also contain proprietary risk scores from a separate
commercial product. **They are excluded at read time**, not filtered later:
`census_history.py` never reads a column matching `/score/i`, and
`assert_no_proprietary_columns()` raises if one reaches a frame.

Only public FMCSA census fields are used — power units, driver count, safety
rating, status, state, MCS-150 date and mileage year. This keeps the model
reproducible from public sources by anyone, and keeps the commercial product's
intellectual property out of it.

---

## Layout

```
src/carrier_survival/
  config.py          Dataset definitions and column projections
  fmcsa.py           Paged SODA extraction, key and date normalisation
  census_history.py  Snapshot loader, coverage and quality checks
  labels.py          Exit label construction with the 12-month return window
  features.py        Point-in-time feature construction and the risk set
  leakage.py         Solo-AUC and temporal-decay leakage detection
scripts/
  fetch_fmcsa.py     Download FMCSA sources to data/raw/
  build_panel.py     Build the carrier-period panel from local snapshots
  build_labels.py    Build the exit label table
  build_features.py  Build the modelling frame
  train_baseline.py  Fit baseline + full model, evaluate, check for leakage
tests/               17 tests: feature construction, coverage, leakage
data/                Parquet + manifests (gitignored; re-fetchable)
```

## Running it

```bash
pip install -r requirements.txt

python scripts/fetch_fmcsa.py --list     # available datasets
python scripts/fetch_fmcsa.py            # fetch all defaults (~20 min)
python scripts/build_labels.py           # build the exit label table

# The panel needs the local census archive; everything else runs from the API.
export CARRIER_SURVIVAL_SNAPSHOT_DIR=/path/to/snapshots
python scripts/build_panel.py            # carrier-period panel + coverage report
python scripts/build_features.py         # point-in-time modelling frame
python scripts/train_baseline.py         # models, segments, leakage checks

pytest                                   # 17 tests
```

Each fetch writes a manifest recording row counts, distinct carriers, null keys
and fetch time beside its Parquet file.

### Two implementation details that are load-bearing

**Paging is ordered by `:id`.** Socrata's `$offset` walks an unordered result
set, so without an explicit `$order` a row can arrive twice while another never
arrives. Nothing errors — the row count comes out right and the contents are
quietly wrong.

**Keys and dates are normalised on ingest.** The USDOT number is `dot_number` in
some files and `usdot_number` in others, zero-padded in one and bare in the next;
dates appear as `MM/DD/YYYY` and `YYYYMMDD`, stored as text throughout.

---

## Limitations

- **No fleet trajectory.** The archive cannot support it; see above. This is the
  largest gap, because fleet momentum is the most intuitively predictive signal
  available.
- **Seven prediction dates in one year.** Point-in-time features need the
  archived panel, which limits the modelling window to 2024. The model has not
  been tested across a freight cycle.
- **The temporal-decay check is inconclusive** at `+0.004`. A wider window would
  be needed to make it informative, which again needs more panel.
- **Involuntary exit only.** The competing-risks arm is empty for the reasons
  above.
- **Authority status is a proxy for operating status.** A dormant carrier holding
  active authority is indistinguishable from a working one.
