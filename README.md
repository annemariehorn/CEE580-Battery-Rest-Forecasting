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
  models.py             ridge regression (tuned) and boosted trees (untuned starting point)
  uncertainty.py        paired block bootstraps for comparing settings
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
python scripts/make_ridge_figure.py        # 7. ridge forecasts with/without future rest
python scripts/ridge_robustness.py         # 8. robustness: other batteries, origins, horizons (~2 min)
python scripts/rest_duration_sensitivity.py  # 9. assumed impedance-test length 0/15/32 min
python -m pytest                           # tests
```

Or, with `make`: `make all` (steps 1–3 and 5–7), `make checks` (steps 4, 8 and 9; slower)
and `make test`.

## Current results (as of Nov 26)

RMSE (mAh) on B0018, horizons 20–50, pooled over origins 30, 45 and 60.

| Method | Capacity only | + past rest | + future rest (oracle) |
|---|---|---|---|
| Persistence | 154.5 | | |
| Linear trend (last 20 measurements) | 94.9 | | |
| Exponential trend (last 20 measurements) | 87.4 | | |
| Ridge, raw rest hours (initial specification) | 35.5 | 35.5 | 36.7 |
| **Ridge, log rest hours (refinement)** | **34.7** | 33.5 | **25.1** |
| Boosted trees, **untuned** | 44.4 | 43.7 | 31.6 |

Improvement for ridge with log rest hours, with two approximate 95% bootstrap sensitivity intervals
(`scripts/bootstrap_comparisons.py`, block length 10):

| Comparison | Reduction | Within-origin interval | Target-block interval* |
|---|---|---|---|
| Past rest vs capacity only | 3.7% | −0.4 to 9.5% | −4.6 to 8.7% |
| Future rest vs capacity only | 27.9% | 10.8 to 49.6% | −1.7 to 52.5% |
| Future rest vs past rest | 25.1% | 10.7 to 44.2% | 2.7 to 48.4% |

\*Keeps forecasts of the same measurement from overlapping origins together. It accounts for
more of the dependence than the within-origin version and is wider here, but it is not
guaranteed to be conservative. Its lower end moves between about −6% and +6% with block length
(5–15). The comparison is favoured in about 97% of resamples, which is not the probability that
the improvement is real; its size is uncertain with one independent test battery.

Rest-duration sensitivity (`scripts/rest_duration_sensitivity.py`): rest times are upper
bounds because impedance-test durations were not recorded. The primary result uses those
original estimates (0 minutes). As scenarios, each test is assumed to last 15 or 32 minutes,
capped at its own timestamp window (time until the next operation; typically 32 minutes,
minimum 14.7). Future rest then improves ridge by 27.9% / 23.2% / 15.4% over capacity only
(and 25.1% / 18.6% / 8.8% over past rest) for 0 / 15 / 32 minutes. The point estimates stay
positive but shrink, and the intervals widen: the result is sensitive to how rest is measured.
The robustness checks below use the original rest estimates.

Robustness (`scripts/ridge_robustness.py`): future rest also lowers ridge RMSE with each of
B0005–B0007 as the test battery (8–19%), at 9 of 13 origins from 20 to 80 on B0018 (worse
at one, origin 65), and for every horizon window (28–56%).

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
