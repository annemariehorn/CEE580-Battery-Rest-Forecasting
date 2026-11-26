# Forecasting Battery Capacity Fade and End of Life: The Role of Rest Periods

CEE 580 (Data Science and Machine Learning for Engineers), Duke University, Fall 2026
**Lithium Learners:** Andrew Geyser, Annemarie Horn, Emi Sohi

## The project in plain language

A lithium-ion battery holds less charge each time it is used. NASA repeatedly charged and
fully discharged four commercial battery cells (labeled **B0005, B0006, B0007 and B0018**)
until they wore out. Each full discharge is one **discharge measurement** (one cycle) and
records the cell's **capacity**: how much charge it delivered, in ampere-hours (Ah).
New cells hold about 2.0 Ah.

Capacity mostly declines, but after the cells sat unused for a while (a **rest**), capacity
temporarily jumps back up. Most forecasting models ignore this.

**Question.** When forecasting a battery's capacity 20–50 measurements ahead, does giving
the model information about rest periods make the forecast more accurate on a battery the
model has never seen?

**Hypothesis.** Supplying the actual future rest times (an *oracle* setting, as if the test
schedule were known in advance) reduces root-mean-square error (RMSE) 20–50 measurements
ahead by at least 10% for both ridge regression and boosted trees. The competing
explanation: with only three training batteries, or because recent capacity already
reflects past rests, the rest information adds nothing.

## How the forecasting works

```
measurements 1 ... 30 | 31 ........................ 80
  model sees these    |  model predicts these (horizons 1-50)
                      ^ forecast origin = 30
```

- Forecasts are made at origins 30, 45 and 60 on the **held-out battery B0018**, the only
  battery tested on its own schedule. Models train on B0005–B0007.
- Every model is run in three **information settings**:
  `capacity_only`, `past_rest` (rest times up to the origin) and
  `future_rest_oracle` (plus the actual future rest times).
- The primary score is RMSE in mAh over horizons 20–50 (31 predictions per origin).
- Measurements are never split randomly: neighbouring measurements are nearly identical,
  so a random split would leak the answer.

## Repository layout

```
data/
  raw/                  the four NASA .mat files (do not edit) + source and checksums
  processed/            per_cycle_table.csv (generated)
src/battery_forecast/
  config.py             shared settings: batteries, origins, horizons, thresholds
  data.py               build/load the per-discharge table, rest-time estimates
  forecasting.py        forecast examples, hold-out splits, scoring
  baselines.py          persistence, linear trend, exponential trend
  models.py             ridge regression and boosted trees (untuned starting points)
  uncertainty.py        block bootstrap (specification, to be implemented)
  plotting.py           report figures
scripts/                runnable steps (see below)
tests/                  pytest checks, including no-leakage tests
docs/                   proposal, feedback, decisions log, report drafts
notebooks/              scratch exploration (not part of the pipeline)
outputs/                figures and results (generated, not committed)
```

## Setup

Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running everything

```bash
python scripts/build_table.py              # 1. per-discharge table  -> data/processed/
python scripts/make_figures.py             # 2. Figures 1-2          -> outputs/figures/
python scripts/run_baselines.py            # 3. baselines on B0018   -> outputs/results/
python scripts/tune_ridge.py               # 4. choose ridge settings on B0005-B0007 (slow, ~1-2 min)
python scripts/run_models.py               # 5. ridge (tuned) + boosted trees (untuned) on B0018
python scripts/run_models.py --all-holdouts  #   + robustness: hold out each battery
python scripts/bootstrap_comparisons.py    # 6. 95% intervals: does rest information help?
python -m pytest                           # tests
```

Or, with `make`: `make all` and `make test`.

## Current results (as of Nov 26)

RMSE (mAh) on B0018, horizons 20–50, pooled over origins 30, 45 and 60.
"Change" is relative to the same model with capacity only; the 95% interval is from a
block bootstrap (block length 10).

| Method | Capacity only | + past rest | + future rest (oracle) |
|---|---|---|---|
| Persistence | 154.5 | | |
| Linear trend (last 20 measurements) | 94.9 | | |
| Exponential trend (last 20 measurements) | 87.4 | | |
| Ridge, raw rest hours (first version) | 35.5 | 35.5 (no change) | 36.7 (3.3% worse; interval includes 0) |
| **Ridge, log rest hours (final)** | **34.7** | 33.5 (−3.7%; interval includes 0) | **25.1 (−27.9%; 95% interval 11 to 50% lower)** |
| Boosted trees, **untuned** | 44.4 | 43.7 | 31.6 |

Ridge settings were chosen on B0005–B0007 only (`docs/decisions.md`). B0018 was scored
first with raw rest hours and then with log rest hours; both are reported. Boosted trees
are not results to report until tuned the same way.

## Key dates

| Deliverable | Due |
|---|---|
| Progress report | Mon Nov 16, 11:59 PM |
| Final report | Wed Dec 9, 11:59 PM |
| Final-presentation slides | Thu Dec 10, 11:59 PM |
| Final presentation | Fri Dec 11, 9:00 AM–12:00 PM |

## Data source

Saha, B., & Goebel, K. (2007). *Battery data set*. NASA Prognostics Data Repository,
NASA Ames Research Center. Details and checksums: [`data/raw/README.md`](data/raw/README.md).
