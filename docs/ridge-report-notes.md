# Ridge and uncertainty: notes for the final report

What to include in the ridge and uncertainty sections, collected from the analysis
and three external reviews. Numbers come from the scripts named in brackets.

## Suggested order
1. Ridge regression and its inputs (capacity features; past-rest and future-rest features).
2. How alpha was chosen: hold out each of B0005–B0007 in turn, one alpha for all three
   settings, B0018 never used [`tune_ridge.py`].
3. Why rest is log-transformed, and the raw-vs-log comparison (a finding, not a footnote).
4. Evaluation: battery held out, time-ordered forecasts, origins 30/45/60, horizons 20–50.
5. Primary results, overall and **by origin** [`run_models.py`; figure from `make_ridge_figure.py`].
6. Bootstrap sensitivity intervals [`bootstrap_comparisons.py`].
7. Rest-duration sensitivity [`rest_duration_sensitivity.py`] and robustness [`ridge_robustness.py`].
8. Limitations.

## Points to make explicitly
- **Unit of evaluation:** each origin–horizon forecast (93 per setting). Origins overlap, so
  some measurements are scored up to three times; this is not 93 independent observations.
- **Results differ by origin:** capacity only → future rest RMSE is 34 → 32 (origin 30),
  16 → 16 (origin 45), 47 → 24 mAh (origin 60). Most of the pooled 27.9% comes from origin 60.
- **Raw vs log rest:** with raw hours future rest did not help (35.5 → 36.7 mAh); with log hours
  it did (34.7 → 25.1). Alpha also changed (3000 → 1000), so the B0018 comparison does not isolate
  the transform; the validation batteries do: log beats raw at the same alpha (55.1 vs 57.9 mAh at
  1000; 56.0 vs 57.8 at 3000). Raw hours impose a straight-line relationship between each rest feature
  and the predicted capacity change; the log allows diminishing effects as rests get longer and
  reduces the influence of exceptionally long rests.
- **How independent B0018 is:** it was held out from model fitting, but the log transform was
  suggested by exploratory analysis that used B0018 (proposal: log rest explained 79% of B0018's
  cycle-to-cycle change) and was adopted after a first B0018 evaluation. Describe the log-rest
  result as an exploratory refinement; avoid "unbiased" or "confirmed out-of-sample".
- **Predictive, not causal:** the future-rest features (rest before the target; mean and
  maximum future rest) test whether knowing the future rest schedule improves forecasts, not
  how much a given rest causes the battery to recover. They may also partly encode
  experimental scheduling differences (supported by the rest-duration results).
- **Oracle:** future rest times were supplied after the fact; it is an idealized scenario showing
  the potential predictive value of knowing future rests, not a deployable forecasting system
  (and not a mathematical upper bound on what any model could gain).
- **What the bootstrap does and does not cover:** it shows how sensitive the estimated RMSE
  reduction is to resampling B0018's dependent forecast errors. It does not cover which battery is tested, model-development choices, or
  the true impedance-test durations; discuss those separately. Call the intervals approximate
  sensitivity intervals, not confidence intervals.
- **Rest-duration sensitivity:** primary result uses the original upper-bound rests (0 min).
  Assuming 15 / 32 min per impedance test (each capped at its own timestamp window), the gain
  is 23.2% / 15.4%. The remaining gain may come from genuinely long rests (plausible, not shown).
- **Robustness checks** used the original rest estimates. Origin counts use a 5% threshold:
  future rest lowers RMSE by more than 5% at 9 of 13 origins, raises it by more than 5% at one
  (65), and changes it by less than 5% at three.
- **Against the 10% hypothesis:** the log-rest point estimates exceed 10% under all three
  rest-duration assumptions (27.9 / 23.2 / 15.4%), but the intervals do not establish that the
  improvement reliably exceeds 10%, and the raw-rest specification did not support it.
- **Where the gain comes from:** the forecast trajectories *suggest* part of it comes from better
  prediction of capacity increases after long rests; this was not formally isolated.
- **Title:** if end of life is not completed, consider retitling (e.g. "Forecasting Lithium-Ion
  Battery Capacity Fade: The Predictive Value of Rest Periods").
- **Boosted trees:** compare only after they are tuned the same way; do not compare tuned
  ridge with untuned trees.

## Conclusion paragraph
See the running results doc (Lithium Learners: Results and Report Notes) for the latest
version; an earlier version is the last "Conclusion wording" row in `docs/decisions.md`.
