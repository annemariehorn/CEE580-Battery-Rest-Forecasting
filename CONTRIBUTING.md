# Working on this project

## Who owns what

| Person | Required | Exploratory | Do first |
|---|---|---|---|
| **Annemarie** | Ridge regression in all three settings; block-bootstrap intervals (`uncertainty.py`) | Temperature, charging-time and impedance signals added to ridge; ridge 100-measurement forecasts for end of life | Finish `block_bootstrap_rmse_ratio` early; everyone uses it |
| **Emi** | Boosted trees in all three settings | Same extra signals added to boosted trees; tree 100-measurement forecasts for end of life | Tune trees on B0005–B0007 before the extra signals |
| **Andrew** | Baselines (`baselines.py`, done; check trend window) | How long post-rest gains last; end-of-life prediction from the models' 100-measurement forecasts | Post-rest analysis, which informs the rest features |

Everyone reviews one other person's pull request before each deliverable, and everyone
should be able to explain the whole project.

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

```bash
git pull                                 # start from the latest main
git switch -c ridge-tuning               # one branch per piece of work
# ...edit, then run the tests...
python -m pytest
git add -A && git commit -m "Tune ridge alpha on B0005-B0007"
git push -u origin ridge-tuning          # open a pull request; a teammate reviews it
```

- Keep `main` runnable: scripts and tests should work after every merge.
- Do not commit generated files (`outputs/`, `data/processed/`); they are rebuilt by the scripts.
- Put exploratory notebooks in `notebooks/` with your name in the filename.
- Record every design choice and its reason in `docs/decisions.md`.

## Adding a model

Write a factory in `src/battery_forecast/models.py`, register it in `MODEL_FACTORIES`,
and `scripts/run_models.py` runs it in all three settings automatically. Use
`fit_predict` so every model is trained and scored the same way.
