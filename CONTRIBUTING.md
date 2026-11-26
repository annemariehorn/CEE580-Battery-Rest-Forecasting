# Working on this project

## Who owns what

| Person | Required | Exploratory | Do first |
|---|---|---|---|
| **Annemarie** | Ridge regression in all three settings; bootstrap intervals (`uncertainty.py`); rest-duration check — **all done** | Ridge 100-measurement forecasts for end of life (lower priority); extra signals (lower priority) | Bootstrap is ready: `scripts/bootstrap_comparisons.py` covers every model |
| **Emi** | Boosted trees in all three settings | Same extra signals added to boosted trees; tree 100-measurement forecasts for end of life | Tune trees on B0005–B0007 the same way as ridge (copy `scripts/tune_ridge.py`) |
| **Andrew** | Baselines (`baselines.py`, done; check trend window) | How long post-rest gains last; end-of-life prediction from the models' 100-measurement forecasts | Post-rest analysis, which informs the rest features |

Everyone reads over one other person's code before each deliverable, and everyone
should be able to explain the whole project.

## Milestones (from the progress report)

| By | What |
|---|---|
| Nov 23 | Check how uncertain rest durations affect results (done, Annemarie); trend baselines (done); ridge (done) and boosted trees tuned on B0005–B0007 |
| Dec 5 | Ridge and boosted trees compared in all three settings on B0018, with bootstrap intervals; robustness check holding out each battery |
| Dec 9 | Final report. End of life and extra signals only if time allows |

## Where files go

| What | Where |
|---|---|
| Reusable code | `src/battery_forecast/` (e.g. ridge settings in `models.py`) |
| A script that runs one step | `scripts/` (e.g. `scripts/tune_ridge.py`) |
| Tests | `tests/` |
| Scratch exploration | `notebooks/`, with your name in the filename |
| Decisions and the reason for them | a new row in `docs/decisions.md` |
| Submitted reports | `docs/` |
| Figures and results | not committed; the scripts write them to `outputs/` |
| Raw data | do not edit `data/raw/` |

## Rules that keep results valid

1. **Never tune on B0018.** It is the test battery. Choose settings by training on two of
   B0005–B0007 and validating on the third, then score B0018 once.
2. **Never split measurements randomly.** Split by battery only.
3. **The three settings must differ only in rest features.** Same model, same settings,
   same training data.
4. **Inputs come from measurements up to the origin,** except the oracle future-rest
   features. `tests/test_forecasting.py::test_no_future_leakage` checks this; keep it passing.
5. **Fit scalers and transforms on training batteries only.** The pipelines in
   `models.py` already do this.
6. **Results holding out B0005–B0007 are optimistic** (they share a rest schedule).
   Report them as robustness checks, not the main test.

## Git workflow

We each work on different files, so we commit straight to `main`.

```bash
git pull                                 # before you start: get everyone's changes
# ...edit, then run the tests...
python -m pytest
git add scripts/tune_ridge.py            # the files you changed
git commit -m "Tune ridge alpha on B0005-B0007"
git pull                                 # just before pushing, in case someone else pushed
git push
```

- Only push when the tests pass, so `main` always runs.
- Tell the group before changing shared code (`forecasting.py`, `data.py`, `config.py`);
  everyone's results depend on it. For a big change there, use a branch and a pull request.
- Do not commit generated files (`outputs/`, `data/processed/`); the scripts rebuild them.
- Record every design choice and its reason in `docs/decisions.md`.

## Adding a model

Write a factory in `src/battery_forecast/models.py`, register it in `MODEL_FACTORIES`,
and `scripts/run_models.py` runs it in all three settings automatically. Use
`fit_predict` so every model is trained and scored the same way.
